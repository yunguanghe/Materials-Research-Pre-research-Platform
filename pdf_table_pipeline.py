#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

DEFAULT_REPO_CANDIDATES = [
    Path("/Users/stardust/Documents/New project/TableStructureRec"),
    Path(__file__).resolve().parent / "TableStructureRec",
]
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "PNG"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "bigmodel_structured_json"

LOGGER = logging.getLogger("image_table_pipeline")

PLACEHOLDER_VALUES = {"", "-", "--", "---", "—", "——", "———", "_", "__", "一"}
HEADER_GROUP_IGNORE = {
    "化学成分",
    "化学成分%",
    "化学成分（质量分数）",
    "化学成分(质量分数)",
    "质量分数",
    "%",
}
ELEMENT_ALIASES = {
    "ti": "Ti",
    "t1": "Ti",
    "al": "Al",
    "a1": "Al",
    "si": "Si",
    "v": "V",
    "mn": "Mn",
    "fe": "Fe",
    "ni": "Ni",
    "cu": "Cu",
    "zr": "Zr",
    "nb": "Nb",
    "mo": "Mo",
    "ru": "Ru",
    "pd": "Pd",
    "sn": "Sn",
    "ta": "Ta",
    "nd": "Nd",
    "c": "C",
    "n": "N",
    "h": "H",
    "o": "O",
    "0": "O",
}
IMPURITY_ORDER = ["Fe", "C", "N", "H", "O", "其他元素-单一", "其他元素-总和"]
MAX_CELL_RECOVERY_PER_IMAGE = 18


def bootstrap_repo_path() -> None:
    repo_from_env = os.environ.get("TABLE_STRUCTURE_REC_ROOT")
    candidates = []
    if repo_from_env:
        candidates.append(Path(repo_from_env).expanduser())
    candidates.extend(DEFAULT_REPO_CANDIDATES)

    for candidate in candidates:
        if (candidate / "table_cls").exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return

    raise ModuleNotFoundError("Cannot locate TableStructureRec repo")


bootstrap_repo_path()

from rapidocr_onnxruntime import RapidOCR
from table_cls import TableCls
from wired_table_rec.main import WiredTableInput, WiredTableRecognition
try:
    from lineless_table_rec.main import LinelessTableInput, LinelessTableRecognition
except Exception:
    LinelessTableInput = None
    LinelessTableRecognition = None


@dataclass
class ParsedCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


class TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[ParsedCell]]] = []
        self._current_table: Optional[List[List[ParsedCell]]] = None
        self._current_row: Optional[List[ParsedCell]] = None
        self._cell_text_parts: List[str] = []
        self._cell_attrs: Dict[str, int] = {}
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"}:
            self._in_cell = True
            self._cell_text_parts = []
            self._cell_attrs = {
                "rowspan": int(attr_map.get("rowspan", "1")),
                "colspan": int(attr_map.get("colspan", "1")),
            }

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text_parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._current_row is not None:
            text = normalize_text("".join(self._cell_text_parts))
            self._current_row.append(
                ParsedCell(text, self._cell_attrs["rowspan"], self._cell_attrs["colspan"])
            )
            self._in_cell = False
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            self.tables.append(self._current_table)
            self._current_table = None


def normalize_text(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").replace("\n", " ")).strip()
    return text.replace("≤", "<=")


def compact_text(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class ImageTablePipeline:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.json_root = output_root / "json"
        self.material_root = output_root / "materials"

        for path in (self.output_root, self.json_root, self.material_root):
            path.mkdir(parents=True, exist_ok=True)

        self.ocr_engine = RapidOCR()
        self.wired_engine = None
        self.lineless_engine = None
        self.table_cls = self._build_table_classifier()
        self._image_cache: Dict[str, Any] = {}
        self._cell_ocr_cache: Dict[Tuple[str, Tuple[int, int, int, int]], str] = {}
        self.enable_lineless_fallback = os.environ.get("TABLE_ENABLE_LINELESS", "0") == "1"
        self.max_cell_recovery_per_image = int(
            os.environ.get("TABLE_MAX_CELL_RECOVERY", str(MAX_CELL_RECOVERY_PER_IMAGE))
        )
        self._image_recovery_count: Dict[str, int] = {}
        self.enable_cell_level_ocr = os.environ.get("TABLE_CELL_LEVEL_OCR", "1") == "1"

    def _build_table_classifier(self):
        try:
            return TableCls()
        except Exception:
            return None

    def process_directory(self, image_dir: Path):
        image_paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            image_paths.extend(image_dir.glob(pattern))
        image_paths = sorted(set(image_paths))

        if not image_paths:
            LOGGER.warning("No image files found in %s", image_dir)
            return

        for image_path in image_paths:
            LOGGER.info("Processing %s", image_path.name)
            self.process_image(image_path)

    def process_image(self, image_path: Path):
        self._image_recovery_count[str(image_path)] = 0
        ocr_raw, _ = self.ocr_engine(str(image_path), return_word_box=True)
        ocr_result = self._normalize_ocr_result(ocr_raw)

        table_result, _ = self._recognize_table(image_path, ocr_result)
        tables = self._parse_table_html(table_result.pred_html or "", getattr(table_result, "cell_bboxes", None))
        if not tables:
            LOGGER.warning("No table recognized: %s", image_path.name)
            return

        for table_index, cell_rows in enumerate(tables, start=1):
            if self.enable_cell_level_ocr:
                LOGGER.info("Running cell-level OCR for %s table %s", image_path.name, table_index)
                cell_rows = self._refresh_cells_with_primary_ocr(cell_rows, image_path)
            standardized = self._standardize_rows(cell_rows, image_path)
            if not standardized["records"]:
                LOGGER.warning(
                    "No material rows parsed: %s table %s",
                    image_path.name,
                    table_index,
                )
                continue

            table_dump_path = self.json_root / f"{image_path.stem}_table_{table_index}.json"
            self._write_json(table_dump_path, standardized)

            for record in standardized["records"]:
                grade = record["合金牌号"]
                material_path = self.material_root / f"{self._safe_filename(grade)}.json"
                self._write_json(material_path, record)

    def _recognize_table(self, image_path: Path, ocr_result):
        if self.wired_engine is None:
            self.wired_engine = WiredTableRecognition(WiredTableInput())
        wired_result = self.wired_engine(str(image_path), ocr_result=ocr_result)
        candidates: List[Tuple[object, str]] = [(wired_result, "wired")]

        wired_score = self._score_table_result(wired_result)
        wired_score_value = wired_score[0] if isinstance(wired_score, tuple) else wired_score
        if wired_score_value > 80 or not self.enable_lineless_fallback:
            LOGGER.info("Using wired engine for %s (score=%s)", image_path.name, wired_score)
            return wired_result, "wired"

        if LinelessTableRecognition is not None and LinelessTableInput is not None:
            if self.lineless_engine is None:
                self.lineless_engine = LinelessTableRecognition(LinelessTableInput())
            try:
                LOGGER.info("Wired score is low for %s, trying lineless fallback", image_path.name)
                lineless_result = self.lineless_engine(str(image_path), ocr_result=ocr_result)
                candidates.append((lineless_result, "lineless"))
            except Exception:
                LOGGER.exception("Lineless recognition failed for %s", image_path.name)

        best_result, best_mode = max(candidates, key=lambda item: self._score_table_result(item[0]))
        return best_result, best_mode

    def _score_table_result(self, table_result) -> Tuple[int, int, int]:
        html = getattr(table_result, "pred_html", "") or ""
        score = 0
        score += html.count("<tr")
        score += html.count("<td") + html.count("<th")
        for keyword in ("合金牌号", "牌号", "名义化学成分", "主要成分", "杂质", "Fe", "Ti"):
            if keyword in html:
                score += 5
        return (score, len(html), html.count("</table>"))

    def _normalize_ocr_result(self, ocr_raw):
        if not ocr_raw:
            return []
        return [item[:3] for item in ocr_raw if len(item) >= 3]

    def _parse_table_html(self, html: str, cell_bboxes) -> List[List[List[Dict[str, Any]]]]:
        parser = TableHTMLParser()
        parser.feed(html)
        bbox_list = self._normalize_cell_bboxes(cell_bboxes)
        expanded_tables = []
        for table in parser.tables:
            expanded = self._expand_table(table, bbox_list)
            if expanded:
                expanded_tables.append(expanded)
        return expanded_tables

    def _normalize_cell_bboxes(self, cell_bboxes) -> List[Tuple[int, int, int, int]]:
        if cell_bboxes is None:
            return []
        normalized = []
        for box in cell_bboxes:
            flat = [float(v) for v in np.array(box).reshape(-1).tolist()]
            if len(flat) >= 8:
                xs = flat[0::2]
                ys = flat[1::2]
                normalized.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
            elif len(flat) >= 4:
                normalized.append((int(flat[0]), int(flat[1]), int(flat[2]), int(flat[3])))
        return normalized

    def _expand_table(
        self,
        table: List[List[ParsedCell]],
        bbox_list: Sequence[Tuple[int, int, int, int]],
    ) -> List[List[Dict[str, Any]]]:
        grid: List[List[Dict[str, Any]]] = []
        active_rowspans: Dict[int, Dict[str, Any]] = {}
        bbox_index = 0

        for row in table:
            expanded_row: List[Dict[str, Any]] = []
            col_idx = 0

            while col_idx in active_rowspans:
                span_info = active_rowspans[col_idx]
                expanded_row.append({"text": span_info["text"], "bbox": span_info["bbox"]})
                span_info["remaining"] -= 1
                if span_info["remaining"] <= 0:
                    del active_rowspans[col_idx]
                col_idx += 1

            for cell in row:
                bbox = bbox_list[bbox_index] if bbox_index < len(bbox_list) else None
                bbox_index += 1
                while col_idx in active_rowspans:
                    span_info = active_rowspans[col_idx]
                    expanded_row.append({"text": span_info["text"], "bbox": span_info["bbox"]})
                    span_info["remaining"] -= 1
                    if span_info["remaining"] <= 0:
                        del active_rowspans[col_idx]
                    col_idx += 1

                cell_text = normalize_text(cell.text)
                for offset in range(cell.colspan):
                    expanded_row.append({"text": cell_text, "bbox": bbox})
                    if cell.rowspan > 1:
                        active_rowspans[col_idx + offset] = {
                            "text": cell_text,
                            "bbox": bbox,
                            "remaining": cell.rowspan - 1,
                        }
                col_idx += cell.colspan

            while col_idx in active_rowspans:
                span_info = active_rowspans[col_idx]
                expanded_row.append({"text": span_info["text"], "bbox": span_info["bbox"]})
                span_info["remaining"] -= 1
                if span_info["remaining"] <= 0:
                    del active_rowspans[col_idx]
                col_idx += 1

            grid.append(expanded_row)

        max_cols = max((len(row) for row in grid), default=0)
        return [row + [{"text": "", "bbox": None}] * (max_cols - len(row)) for row in grid]

    def _standardize_rows(self, rows: List[List[Dict[str, Any]]], image_path: Path) -> Dict[str, object]:
        return self._standardize_rows_local(rows, image_path)

    def _refresh_cells_with_primary_ocr(
        self,
        rows: List[List[Dict[str, Any]]],
        image_path: Path,
    ) -> List[List[Dict[str, Any]]]:
        refreshed_rows: List[List[Dict[str, Any]]] = []
        for row in rows:
            refreshed_row: List[Dict[str, Any]] = []
            for cell in row:
                original_text = normalize_text(cell.get("text", ""))
                bbox = cell.get("bbox")
                recovered_text = self._read_cell_text_from_bbox(image_path, bbox) if bbox else ""
                final_text = original_text
                if self._is_better_cell_text(recovered_text, original_text):
                    final_text = recovered_text
                refreshed_row.append({"text": final_text, "bbox": bbox})
            refreshed_rows.append(refreshed_row)
        return refreshed_rows

    def _read_cell_text_from_bbox(
        self,
        image_path: Path,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> str:
        if bbox is None:
            return ""
        cache_key = (str(image_path), tuple(bbox))
        if cache_key in self._cell_ocr_cache:
            return self._cell_ocr_cache[cache_key]

        crop = self._crop_cell_image(image_path, bbox)
        if crop is None:
            self._cell_ocr_cache[cache_key] = ""
            return ""

        candidates: List[str] = []
        for variant in self._generate_cell_variants(crop):
            try:
                ocr_raw, _ = self.ocr_engine(variant, return_word_box=True)
            except Exception:
                continue
            text = self._ocr_result_to_text(ocr_raw)
            if text:
                candidates.append(self._clean_cell_value(text))

        best = max(candidates, key=self._score_cell_text_quality, default="")
        self._cell_ocr_cache[cache_key] = best
        return best

    def _is_better_cell_text(self, candidate: str, current: str) -> bool:
        return self._score_cell_text_quality(candidate) > self._score_cell_text_quality(current)

    def _score_cell_text_quality(self, value: str) -> int:
        if not value:
            return -10
        compact = compact_text(value)
        score = len(compact)
        if compact and compact not in PLACEHOLDER_VALUES:
            score += 4
        if re.search(r"\d", value):
            score += 6
        if re.search(r"[A-Za-z]", value):
            score += 4
        if re.search(r"[\u4e00-\u9fff]", value):
            score += 4
        if "~" in value or "<=" in value:
            score += 2
        if value.count("?") or value.count("'"):
            score -= 4
        return score

    def _standardize_rows_local(self, rows: List[List[Dict[str, Any]]], image_path: Path) -> Dict[str, object]:
        if not rows:
            return {"headers": [], "records": []}

        rows = [self._trim_row(row) for row in rows if any(normalize_text(cell["text"]) for cell in row)]
        if not rows:
            return {"headers": [], "records": []}

        data_start = self._find_data_start(rows)
        header_rows = rows[:data_start]
        data_rows = rows[data_start:]
        column_specs = self._build_column_specs([[cell["text"] for cell in row] for row in header_rows])

        records = []
        for row in data_rows:
            record = self._build_material_json(row, column_specs, image_path)
            if record:
                records.append(record)

        return {
            "headers": [[cell["text"] for cell in row] for row in header_rows],
            "column_specs": column_specs,
            "records": records,
        }

    def _trim_row(self, row: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trimmed = [{"text": normalize_text(cell["text"]), "bbox": cell.get("bbox")} for cell in row]
        while trimmed and not trimmed[-1]["text"]:
            trimmed.pop()
        return trimmed

    def _find_data_start(self, rows: List[List[Dict[str, Any]]]) -> int:
        for idx, row in enumerate(rows):
            first_cell = compact_text(row[0]["text"]) if row else ""
            second_cell = compact_text(row[1]["text"]) if len(row) > 1 else ""
            if self._looks_like_material_grade(first_cell) and second_cell:
                return idx
        return min(3, len(rows))

    def _looks_like_material_grade(self, value: str) -> bool:
        if not value:
            return False
        if any(keyword in value for keyword in ("牌号", "化学成分", "主要成分", "杂质")):
            return False
        return bool(re.match(r"^[A-Za-z]{1,4}[A-Za-z0-9\-\*/\.]*$", value))

    def _build_column_specs(self, header_rows: List[List[str]]) -> List[Dict[str, str]]:
        if not header_rows:
            return []

        max_cols = max(len(row) for row in header_rows)
        column_texts = [
            unique_preserve_order(
                compact_text(row[col_idx]) for row in header_rows if col_idx < len(row) and row[col_idx]
            )
            for col_idx in range(max_cols)
        ]
        grade_idx = self._find_special_column(column_texts, ("合金牌号", "牌号"))
        nominal_idx = self._find_special_column(column_texts, ("名义化学成分",))
        impurity_start_idx = self._find_impurity_start(column_texts, nominal_idx)
        impurity_tail_count = self._count_trailing_impurity_columns(column_texts)
        if impurity_start_idx is None and nominal_idx is not None and impurity_tail_count:
            impurity_start_idx = max(nominal_idx + 1, len(column_texts) - impurity_tail_count)

        specs: List[Dict[str, str]] = []
        for col_idx, texts in enumerate(column_texts):
            specs.append(
                self._infer_column_spec(
                    col_idx,
                    texts,
                    grade_idx=grade_idx,
                    nominal_idx=nominal_idx,
                    impurity_start_idx=impurity_start_idx,
                )
            )

        return specs

    def _find_special_column(self, column_texts: Sequence[Sequence[str]], keywords: Sequence[str]) -> Optional[int]:
        for idx, texts in enumerate(column_texts):
            joined = "".join(texts)
            if any(keyword in joined for keyword in keywords):
                return idx
        return None

    def _find_impurity_start(
        self,
        column_texts: Sequence[Sequence[str]],
        nominal_idx: Optional[int],
    ) -> Optional[int]:
        start_idx = (nominal_idx + 1) if nominal_idx is not None else 0
        for idx in range(start_idx, len(column_texts)):
            joined = "".join(column_texts[idx])
            if "杂质" in joined or "不大于" in joined:
                return idx
        return None

    def _count_trailing_impurity_columns(self, column_texts: Sequence[Sequence[str]]) -> int:
        count = 0
        for texts in reversed(column_texts):
            leaf = self._resolve_leaf_name(texts)
            if self._is_impurity_name(leaf):
                count += 1
            else:
                break
        return count

    def _infer_column_spec(
        self,
        col_idx: int,
        texts: Sequence[str],
        *,
        grade_idx: Optional[int],
        nominal_idx: Optional[int],
        impurity_start_idx: Optional[int],
    ) -> Dict[str, str]:
        if not texts:
            return {"kind": "ignore", "name": ""}

        if grade_idx is not None and col_idx == grade_idx:
            return {"kind": "grade", "name": "合金牌号"}
        if nominal_idx is not None and col_idx == nominal_idx:
            return {"kind": "nominal", "name": "名义化学成分"}

        leaf = self._resolve_leaf_name(texts)
        if not leaf:
            return {"kind": "ignore", "name": ""}
        if leaf in {"合金牌号", "名义化学成分", "主要成分", "杂质，不大于", "杂质", "不大于"}:
            return {"kind": "ignore", "name": leaf}

        if self._is_impurity_name(leaf):
            return {"kind": "impurity", "name": leaf}

        if nominal_idx is not None and col_idx > nominal_idx:
            if impurity_start_idx is not None and col_idx >= impurity_start_idx:
                return {"kind": "impurity", "name": leaf}
            return {"kind": "major", "name": leaf}

        return {"kind": "other", "name": leaf}

    def _resolve_leaf_name(self, texts: Sequence[str]) -> str:
        compact_values = [text for text in texts if text and text not in HEADER_GROUP_IGNORE]
        if not compact_values:
            return ""

        resolved_other = self._resolve_other_elements_name(compact_values)
        if resolved_other:
            return resolved_other

        resolved_element = self._resolve_element_alias(compact_values)
        if resolved_element:
            return resolved_element

        for text in reversed(compact_values):
            if text in {"主要成分", "杂质", "不大于", "杂质不大于"}:
                continue
            return text
        return ""

    def _resolve_other_elements_name(self, texts: Sequence[str]) -> str:
        joined = "".join(texts)
        if "其他元素" not in joined and "其余元素" not in joined:
            return ""
        if any("单一" in text or "单个" in text for text in texts):
            return "其他元素-单一"
        if any("总和" in text or "合计" in text for text in texts):
            return "其他元素-总和"
        return "其他元素"

    def _resolve_element_alias(self, texts: Sequence[str]) -> str:
        for text in reversed(texts):
            alpha = re.sub(r"[^A-Za-z0-9]", "", text).lower()
            if alpha in ELEMENT_ALIASES:
                return ELEMENT_ALIASES[alpha]
        return ""

    def _is_impurity_name(self, name: str) -> bool:
        return name in IMPURITY_ORDER or name == "其他元素"

    def _build_material_json(
        self,
        row: List[Dict[str, Any]],
        column_specs: Sequence[Dict[str, str]],
        image_path: Path,
    ) -> Optional[Dict[str, object]]:
        major_elements = self._initialize_group_fields(column_specs, "major")
        impurity_elements = self._initialize_group_fields(column_specs, "impurity", include_impurity_defaults=True)
        other_fields: Dict[str, str] = {}
        grade = ""
        nominal = ""

        for idx, spec in enumerate(column_specs):
            cell = row[idx] if idx < len(row) else {"text": "", "bbox": None}
            value = self._extract_cell_value(cell, spec, image_path)
            kind = spec["kind"]
            name = spec["name"]

            if kind == "grade":
                grade = value
            elif kind == "nominal":
                nominal = value
            elif kind == "major":
                self._merge_group_value(major_elements, name, value)
            elif kind == "impurity":
                target_name = name
                if name == "其他元素":
                    target_name = self._infer_other_elements_slot(idx, column_specs, impurity_elements)
                self._merge_group_value(impurity_elements, target_name, value)
            elif kind == "other" and self._has_meaningful_value(value):
                other_fields[name] = value

        if not self._looks_like_material_grade(compact_text(grade)):
            return None

        impurity_elements = self._normalize_impurity_elements(impurity_elements)

        result: Dict[str, object] = {
            "合金牌号": grade,
            "名义化学成分": nominal,
            "化学成分（质量分数）": {
                "主要成分": major_elements,
                "杂质，不大于": impurity_elements,
            },
        }
        if other_fields:
            result["其他字段"] = other_fields
        return result

    def _has_meaningful_value(self, value: str) -> bool:
        compact = compact_text(value)
        return bool(compact) and compact not in PLACEHOLDER_VALUES

    def _extract_cell_value(self, cell: Dict[str, Any], spec: Dict[str, str], image_path: Path) -> str:
        value = self._clean_cell_value(normalize_text(cell.get("text", "")))
        if spec["kind"] not in {"major", "impurity"}:
            return value
        if self._needs_value_recovery(value):
            recovered = self._recover_cell_text(image_path, cell.get("bbox"), spec["kind"])
            if self._is_better_value(recovered, value):
                return recovered
        return value

    def _needs_value_recovery(self, value: str) -> bool:
        compact = compact_text(value)
        if not compact or compact in PLACEHOLDER_VALUES:
            return True
        return False

    def _is_better_value(self, candidate: str, current: str) -> bool:
        return self._score_value(candidate) > self._score_value(current)

    def _score_value(self, value: str) -> int:
        if not value:
            return -10
        compact = compact_text(value)
        score = 0
        if compact and compact not in PLACEHOLDER_VALUES:
            score += 2
        if "余量" in value:
            score += 6
        if re.search(r"\d", value):
            score += 8
        if "~" in value or "<=" in value:
            score += 4
        if len(value) > 1:
            score += min(len(value), 12)
        if re.search(r"[A-Za-z\u4e00-\u9fff]{4,}", value) and "余量" not in value:
            score -= 6
        return score

    def _recover_cell_text(
        self,
        image_path: Path,
        bbox: Optional[Tuple[int, int, int, int]],
        value_kind: str,
    ) -> str:
        if bbox is None:
            return ""
        image_key = str(image_path)
        current_count = self._image_recovery_count.get(image_key, 0)
        if current_count >= self.max_cell_recovery_per_image:
            return ""
        cache_key = (str(image_path), tuple(bbox))
        if cache_key in self._cell_ocr_cache:
            return self._cell_ocr_cache[cache_key]

        crop = self._crop_cell_image(image_path, bbox)
        if crop is None:
            self._cell_ocr_cache[cache_key] = ""
            return ""

        self._image_recovery_count[image_key] = current_count + 1
        LOGGER.info(
            "Recovering missing %s value for %s (%s/%s)",
            value_kind,
            image_path.name,
            self._image_recovery_count[image_key],
            self.max_cell_recovery_per_image,
        )

        candidates: List[str] = []
        for variant in self._generate_cell_variants(crop):
            try:
                ocr_raw, _ = self.ocr_engine(variant, return_word_box=True)
            except Exception:
                continue
            text = self._ocr_result_to_text(ocr_raw)
            if text:
                candidates.append(self._clean_cell_value(text))

        best = max(candidates, key=self._score_value, default="")
        self._cell_ocr_cache[cache_key] = best
        return best

    def _crop_cell_image(
        self,
        image_path: Path,
        bbox: Tuple[int, int, int, int],
    ):
        if cv2 is None:
            return None
        image = self._load_image(image_path)
        if image is None:
            return None
        x1, y1, x2, y2 = bbox
        pad_x = max(2, int((x2 - x1) * 0.08))
        pad_y = max(2, int((y2 - y1) * 0.12))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image.shape[1], x2 + pad_x)
        y2 = min(image.shape[0], y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    def _load_image(self, image_path: Path):
        key = str(image_path)
        if key not in self._image_cache:
            if cv2 is None:
                self._image_cache[key] = None
            else:
                self._image_cache[key] = cv2.imread(key)
        return self._image_cache[key]

    def _generate_cell_variants(self, crop):
        variants = [crop]
        if cv2 is None:
            return variants
        enlarged = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        variants.append(enlarged)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(binary)
        return variants

    def _ocr_result_to_text(self, ocr_raw) -> str:
        if not ocr_raw:
            return ""
        texts = []
        for item in ocr_raw:
            if len(item) >= 2 and item[1]:
                texts.append(str(item[1]))
        joined = "".join(texts)
        return normalize_text(joined)

    def _initialize_group_fields(
        self,
        column_specs: Sequence[Dict[str, str]],
        kind: str,
        include_impurity_defaults: bool = False,
    ) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for spec in column_specs:
            if spec["kind"] != kind:
                continue
            name = spec["name"]
            if not name:
                continue
            if kind == "impurity" and name == "其他元素":
                fields.setdefault("其他元素-单一", "")
                fields.setdefault("其他元素-总和", "")
                continue
            fields.setdefault(name, "")
        if include_impurity_defaults:
            for name in IMPURITY_ORDER:
                fields.setdefault(name, "")
        return fields

    def _merge_group_value(self, target: Dict[str, str], name: str, value: str) -> None:
        current = target.get(name, "")
        if self._has_meaningful_value(value):
            if not self._has_meaningful_value(current):
                target[name] = value
            elif len(compact_text(value)) > len(compact_text(current)):
                target[name] = value
        elif name not in target:
            target[name] = value

    def _infer_other_elements_slot(
        self,
        col_idx: int,
        column_specs: Sequence[Dict[str, str]],
        impurity_elements: Dict[str, str],
    ) -> str:
        remaining_other_columns = [
            idx
            for idx, spec in enumerate(column_specs[col_idx:], start=col_idx)
            if spec["kind"] == "impurity" and spec["name"] == "其他元素"
        ]
        if len(remaining_other_columns) >= 2:
            return "其他元素-单一"
        if "其他元素-单一" not in impurity_elements:
            return "其他元素-单一"
        return "其他元素-总和"

    def _normalize_impurity_elements(self, impurity_elements: Dict[str, str]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for key in IMPURITY_ORDER:
            normalized[key] = impurity_elements.get(key, "")
        if "其他元素" in impurity_elements:
            if not normalized["其他元素-单一"]:
                normalized["其他元素-单一"] = impurity_elements["其他元素"]
            elif not normalized["其他元素-总和"]:
                normalized["其他元素-总和"] = impurity_elements["其他元素"]
        return normalized

    def _clean_cell_value(self, value: str) -> str:
        cleaned = normalize_text(value)
        if "二" in cleaned:
            cleaned = cleaned.replace("二", "")
            cleaned = normalize_text(cleaned)
        return cleaned

    def _safe_filename(self, value: str) -> str:
        cleaned = re.sub(r'[\\/:*?"<>|]+', "_", normalize_text(value))
        return cleaned or "unknown_material"

    def _write_json(self, path: Path, data):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    pipeline = ImageTablePipeline(Path(args.output_dir))
    pipeline.process_directory(Path(args.image_dir))


if __name__ == "__main__":
    main()
