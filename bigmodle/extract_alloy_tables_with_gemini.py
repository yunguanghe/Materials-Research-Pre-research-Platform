#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_URL = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
CHAT_COMPLETIONS_URL = f"{BASE_URL.rstrip('/')}/chat/completions"
MODEL_NAME = os.environ.get("YUNWU_MODEL", "gemini-3.1-flash-lite-preview")
API_KEY = os.environ.get("YUNWU_API_KEY", "")

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "PNG"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "bigmodle" / "output_json"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_IMPURITY_KEYS = [
    "Fe", "C", "N", "H", "O", "其他元素-单一", "其他元素-总和",
]


def list_images(image_dir: Path) -> List[Path]:
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _guess_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    return mime_type or "image/png"


def _load_pil_image():
    try:
        from PIL import Image  # type: ignore
        return Image
    except Exception:
        return None


def _encode_image_bytes(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_image_variants(image_path: Path) -> List[Tuple[str, str]]:
    raw_bytes = image_path.read_bytes()
    variants: List[Tuple[str, str]] = [("raw", _encode_image_bytes(raw_bytes, _guess_mime_type(image_path)))]

    Image = _load_pil_image()
    if Image is None:
        return variants

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = img.convert("RGB")
            max_edges = [1800, 1400, 1100]
            qualities = [88, 76, 65]
            for max_edge, quality in zip(max_edges, qualities):
                copy = img.copy()
                copy.thumbnail((max_edge, max_edge))
                buffer = io.BytesIO()
                copy.save(buffer, format="JPEG", quality=quality, optimize=True)
                variants.append((f"jpeg_{max_edge}_{quality}", _encode_image_bytes(buffer.getvalue(), "image/jpeg")))
    except Exception:
        return variants

    deduped: List[Tuple[str, str]] = []
    seen = set()
    for label, payload in variants:
        if payload in seen:
            continue
        seen.add(payload)
        deduped.append((label, payload))
    return deduped


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[\\\\/:*?\"<>|]+", "_", name.strip())
    cleaned = cleaned.replace("\n", "_").replace("\r", "_")
    return cleaned[:120] or "unknown_alloy"


def extract_json_block(text: str) -> object:
    text = text.strip()
    if not text:
        raise ValueError("模型返回内容为空。")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("无法从模型响应中解析 JSON。")


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("二", "")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_alloy_family(alloy_grade: str) -> str:
    match = re.match(r"([A-Za-z]+)", alloy_grade.strip())
    return match.group(1).upper() if match else "UNKNOWN"


def infer_component_keys(rows: Iterable[Dict[str, object]]) -> Tuple[List[str], List[str]]:
    major_keys: List[str] = []
    impurity_keys = list(DEFAULT_IMPURITY_KEYS)

    for row in rows:
        composition = row.get("化学成分（质量分数）", {}) or {}
        for key in (composition.get("主要成分", {}) or {}).keys():
            normalized = normalize_value(key)
            if normalized and normalized not in major_keys:
                major_keys.append(normalized)
        for key in (composition.get("杂质，不大于", {}) or {}).keys():
            normalized = normalize_value(key)
            if not normalized:
                continue
            if normalized in {"其他元素", "其他元素-单一/总和"}:
                continue
            if normalized not in impurity_keys:
                impurity_keys.append(normalized)

    return major_keys, impurity_keys


def normalize_row(
    row: Dict[str, object],
    source_image: str,
    *,
    major_keys: Iterable[str],
    impurity_keys: Iterable[str],
) -> Dict[str, object]:
    alloy_grade = normalize_value(row.get("合金牌号"))
    nominal = normalize_value(row.get("名义化学成分"))
    composition = row.get("化学成分（质量分数）", {}) or {}
    major = composition.get("主要成分", {}) or {}
    impurities = composition.get("杂质，不大于", {}) or {}

    normalized_major = {
        normalize_value(key): normalize_value(value)
        for key, value in major.items()
        if normalize_value(key)
    }
    normalized_impurities = {
        normalize_value(key): normalize_value(value)
        for key, value in impurities.items()
        if normalize_value(key)
    }

    effective_major_keys = list(major_keys) or list(normalized_major.keys())
    effective_impurity_keys = list(impurity_keys) or list(normalized_impurities.keys())

    completed_major = {key: normalized_major.get(key, "") for key in effective_major_keys}
    completed_impurities = {key: normalized_impurities.get(key, "") for key in effective_impurity_keys}

    if "其他元素" in normalized_impurities and not completed_impurities.get("其他元素-单一"):
        completed_impurities["其他元素-单一"] = normalized_impurities["其他元素"]

    return {
        "合金牌号": alloy_grade,
        "名义化学成分": nominal,
        "化学成分（质量分数）": {
            "主要成分": completed_major,
            "杂质，不大于": completed_impurities,
        },
        "来源图片": source_image,
    }


def build_prompt() -> str:
    return (
        "你是材料表格抽取助手。"
        "请读取图片中的合金成分表，只抽取真实可见内容，不要臆造。"
        "你的任务是按表格的每一行输出结构化 JSON。"
        "每一行代表一个合金牌号。"
        "请严格返回 JSON 对象，不要返回任何额外解释。"
        "JSON 顶层结构必须是："
        "{"
        "\"rows\":["
        "{"
        "\"合金牌号\":\"TA1\","
        "\"名义化学成分\":\"工业纯钛\","
        "\"化学成分（质量分数）\":{"
        "\"主要成分\":{\"Ti\":\"余量\",\"Al\":\"\",\"V\":\"\"},"
        "\"杂质，不大于\":{\"Fe\":\"0.25\",\"C\":\"0.10\",\"N\":\"0.03\",\"H\":\"0.015\",\"O\":\"0.20\",\"其他元素-单一\":\"0.1\",\"其他元素-总和\":\"0.4\"}"
        "}"
        "}"
        "]"
        "}"
        "要求："
        "1. 一个对象只对应表格中的一行。"
        "2. 必须保留“合金牌号”“名义化学成分”“化学成分（质量分数）”。"
        "3. “化学成分（质量分数）”下只允许两个键：“主要成分”和“杂质，不大于”。"
        "4. 所有化学元素和对应数值必须写进对应子对象，尤其主要成分不要漏字段。"
        "5. “其他元素”必须拆成“其他元素-单一”和“其他元素-总和”。"
        "6. 如果某元素在该行为空、横线或未给出，可以保留为空字符串。"
        "7. 如果 OCR 误识别出“二”，请删除这个字符。"
        "8. 只抽取图片里真实存在的行，不要猜测缺失行。"
        "9. 输出必须是合法 JSON。"
        "10. 顶层必须使用 rows 字段包裹数组。"
        "11. 对于主要成分，请尽量完整输出表头中的全部元素键，没有值也保留空字符串。"
    )


def call_yunwu_vision(image_path: Path, temperature: float = 0.0, retries: int = 3) -> List[Dict[str, object]]:
    if not API_KEY:
        raise RuntimeError("未配置 YUNWU_API_KEY。")
    last_error: Optional[Exception] = None
    variants = build_image_variants(image_path)

    for variant_label, image_payload in variants:
        for attempt in range(1, retries + 1):
            payload = {
                "model": MODEL_NAME,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": build_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请抽取这张图片中的全部合金表格行，并按要求返回 rows JSON。"},
                            {"type": "image_url", "image_url": {"url": image_payload}},
                        ],
                    },
                ],
            }

            request_body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                CHAT_COMPLETIONS_URL,
                data=request_body,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                    "User-Agent": "yunwu-gemini-alloy-extractor/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = extract_json_block(content)
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("rows"), list):
                        parsed = parsed["rows"]
                    elif isinstance(parsed.get("data"), list):
                        parsed = parsed["data"]
                    else:
                        parsed = [parsed]
                if not isinstance(parsed, list):
                    raise ValueError("模型返回不是 JSON 数组。")
                return [item for item in parsed if isinstance(item, dict)]
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                http.client.RemoteDisconnected,
                ConnectionResetError,
                TimeoutError,
                KeyError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                print(
                    f"[WARN] variant={variant_label} attempt={attempt}/{retries} failed: {exc}",
                    file=sys.stderr,
                )
                if attempt == retries:
                    break
                time.sleep(2 * attempt)

    raise RuntimeError(f"云雾 API 调用失败: {last_error}")


def merge_rows(existing: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    result = dict(existing)
    if not result.get("名义化学成分") and incoming.get("名义化学成分"):
        result["名义化学成分"] = incoming["名义化学成分"]

    existing_comp = result.setdefault("化学成分（质量分数）", {})
    incoming_comp = incoming.get("化学成分（质量分数）", {}) or {}
    for section in ["主要成分", "杂质，不大于"]:
        target = existing_comp.setdefault(section, {})
        source = incoming_comp.get(section, {}) or {}
        for key, value in source.items():
            if value and (not target.get(key)):
                target[key] = value

    sources = set(result.get("来源图片", [])) if isinstance(result.get("来源图片"), list) else {str(result.get("来源图片", ""))}
    incoming_sources = incoming.get("来源图片", [])
    if isinstance(incoming_sources, str):
        incoming_sources = [incoming_sources]
    for item in incoming_sources:
        if item:
            sources.add(item)
    result["来源图片"] = sorted(item for item in sources if item)
    return result


def save_rows(rows: Iterable[Dict[str, object]], output_dir: Path) -> Tuple[int, Dict[str, Dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged: Dict[str, Dict[str, object]] = {}
    rows = list(rows)
    family_rows: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        alloy_grade = normalize_value(row.get("合金牌号"))
        family = extract_alloy_family(alloy_grade) if alloy_grade else "UNKNOWN"
        family_rows.setdefault(family, []).append(row)

    family_component_keys = {
        family: infer_component_keys(family_items)
        for family, family_items in family_rows.items()
    }
    count = 0
    for row in rows:
        count += 1
        alloy_grade = normalize_value(row.get("合金牌号"))
        if not alloy_grade:
            continue
        family = extract_alloy_family(alloy_grade)
        major_keys, impurity_keys = family_component_keys.get(family, ([], list(DEFAULT_IMPURITY_KEYS)))
        payload = normalize_row(
            row,
            normalize_value(row.get("来源图片")),
            major_keys=major_keys,
            impurity_keys=impurity_keys,
        )
        payload["来源图片"] = [payload["来源图片"]] if payload["来源图片"] else []
        if alloy_grade in merged:
            merged[alloy_grade] = merge_rows(merged[alloy_grade], payload)
        else:
            merged[alloy_grade] = payload

    for alloy_grade, payload in merged.items():
        file_path = output_dir / f"{sanitize_filename(alloy_grade)}.json"
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return count, merged


def process_images(image_dir: Path, output_dir: Path) -> None:
    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"在目录中没有找到图片: {image_dir}")

    all_rows: List[Dict[str, object]] = []
    if _load_pil_image() is None:
        print("[WARN] Pillow 未安装，将直接发送原图，远端断连概率可能更高。", file=sys.stderr)
    for image_path in images:
        print(f"[INFO] Processing {image_path.name}")
        rows = call_yunwu_vision(image_path)
        for row in rows:
            row["来源图片"] = image_path.name
        all_rows.extend(rows)

    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "all_rows.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    total_rows, merged = save_rows(all_rows, output_dir)
    summary = {
        "图片目录": str(image_dir),
        "输出目录": str(output_dir),
        "模型": MODEL_NAME,
        "原始行数": total_rows,
        "去重后合金数": len(merged),
        "合金牌号": sorted(merged.keys()),
    }
    (output_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use Yunwu Gemini vision to extract alloy tables from images.")
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR), help="Image folder to read.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write alloy JSON files.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    image_dir = Path(args.image_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    try:
        process_images(image_dir, output_dir)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
