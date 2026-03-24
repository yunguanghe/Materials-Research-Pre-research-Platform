#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ase.io import write

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_API_KEY = os.environ.get("MATERIALS_PROJECT_API_KEY", "")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "mp_cif_downloads"


def normalize_chemsys(value: str) -> str:
    raw = value.strip().replace(" ", "")
    parts = [part for part in re.split(r"[-–—]", raw) if part]
    if len(parts) < 2:
        raise ValueError("请输入合法的元素体系，例如 Nb-Ti 或 Nb-Ti-Sn。")
    normalized: List[str] = []
    for part in parts:
        if not re.fullmatch(r"[A-Z][a-z]?", part):
            raise ValueError(f"无法识别元素符号: {part}")
        if part not in normalized:
            normalized.append(part)
    return "-".join(normalized)


def normalize_formula(value: str) -> str:
    raw = value.strip().replace(" ", "")
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*){2,8}", raw):
        raise ValueError(f"无法识别化学式: {value}")
    return raw


def normalize_query(value: str) -> Tuple[str, str]:
    raw = value.strip()
    if re.search(r"[-–—]", raw):
        return "chemsys", normalize_chemsys(raw)
    return "formula", normalize_formula(raw)


def fetch_modern_chemsys(chemsys: str, top_k: int, api_key: str) -> List[Tuple[Dict[str, Any], Any]]:
    from mp_api.client import MPRester

    docs_with_structures: List[Tuple[Dict[str, Any], Any]] = []
    with MPRester(api_key=api_key, mute_progress_bars=True) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            fields=["material_id", "formula_pretty", "energy_above_hull", "is_stable"],
            all_fields=False,
        )
        ranked_docs = sorted(
            docs,
            key=lambda item: float(getattr(item, "energy_above_hull", 999.0) or 999.0),
        )[:top_k]
        for doc in ranked_docs:
            material_id = str(getattr(doc, "material_id", "") or "")
            if not material_id:
                continue
            structure = mpr.materials.get_structure_by_material_id(material_id)
            docs_with_structures.append(
                (
                    {
                        "material_id": material_id,
                        "formula_pretty": str(getattr(doc, "formula_pretty", chemsys)),
                        "energy_above_hull": float(getattr(doc, "energy_above_hull", 999.0) or 999.0),
                        "is_stable": bool(getattr(doc, "is_stable", False)),
                    },
                    structure,
                )
            )
    return docs_with_structures


def fetch_modern_formula(formula: str, top_k: int, api_key: str) -> List[Tuple[Dict[str, Any], Any]]:
    from mp_api.client import MPRester

    docs_with_structures: List[Tuple[Dict[str, Any], Any]] = []
    with MPRester(api_key=api_key, mute_progress_bars=True) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "energy_above_hull", "is_stable"],
            all_fields=False,
        )
        ranked_docs = sorted(
            docs,
            key=lambda item: float(getattr(item, "energy_above_hull", 999.0) or 999.0),
        )[:top_k]
        for doc in ranked_docs:
            material_id = str(getattr(doc, "material_id", "") or "")
            if not material_id:
                continue
            structure = mpr.materials.get_structure_by_material_id(material_id)
            docs_with_structures.append(
                (
                    {
                        "material_id": material_id,
                        "formula_pretty": str(getattr(doc, "formula_pretty", formula)),
                        "energy_above_hull": float(getattr(doc, "energy_above_hull", 999.0) or 999.0),
                        "is_stable": bool(getattr(doc, "is_stable", False)),
                    },
                    structure,
                )
            )
    return docs_with_structures


def fetch_legacy_chemsys(chemsys: str, top_k: int, api_key: str) -> List[Tuple[Dict[str, Any], Any]]:
    from pymatgen.ext.matproj import MPRester

    parts = chemsys.split("-")
    docs_with_structures: List[Tuple[Dict[str, Any], Any]] = []
    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys(parts)
        ranked_entries: List[Tuple[float, str]] = []
        for entry in entries:
            material_id = str(getattr(entry, "entry_id", "") or "")
            if not material_id:
                continue
            energy = float(getattr(entry, "energy_per_atom", 999.0) or 999.0)
            ranked_entries.append((energy, material_id))

        seen: set[str] = set()
        for energy, material_id in sorted(ranked_entries, key=lambda item: item[0]):
            if material_id in seen:
                continue
            try:
                structure = mpr.get_structure_by_material_id(material_id, final=True)
            except Exception:
                continue
            seen.add(material_id)
            docs_with_structures.append(
                (
                    {
                        "material_id": material_id,
                        "formula_pretty": structure.composition.reduced_formula,
                        "energy_above_hull": None,
                        "energy_per_atom": energy,
                        "is_stable": None,
                    },
                    structure,
                )
            )
            if len(docs_with_structures) >= top_k:
                break
    return docs_with_structures


def fetch_legacy_formula(formula: str, top_k: int, api_key: str) -> List[Tuple[Dict[str, Any], Any]]:
    from pymatgen.ext.matproj import MPRester

    docs_with_structures: List[Tuple[Dict[str, Any], Any]] = []
    with MPRester(api_key) as mpr:
        structures = mpr.get_structures(formula, final=True)[:top_k]
        material_ids = mpr.get_materials_ids(formula)[: len(structures)]
        for index, structure in enumerate(structures):
            material_id = material_ids[index] if index < len(material_ids) else ""
            docs_with_structures.append(
                (
                    {
                        "material_id": material_id,
                        "formula_pretty": structure.composition.reduced_formula,
                        "energy_above_hull": None,
                        "energy_per_atom": None,
                        "is_stable": None,
                    },
                    structure,
                )
            )
    return docs_with_structures


def save_results(chemsys: str, rows: List[Tuple[Dict[str, Any], Any]], output_root: Path) -> Path:
    from pymatgen.io.ase import AseAtomsAdaptor

    output_dir = output_root / chemsys
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    for index, (meta, structure) in enumerate(rows, start=1):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        cif_name = f"{index:02d}_{meta['formula_pretty']}_{meta['material_id']}.cif"
        cif_name = re.sub(r"[^A-Za-z0-9._-]+", "_", cif_name)
        cif_path = output_dir / cif_name
        write(cif_path, atoms)
        summary.append(
            {
                "rank": index,
                "material_id": meta["material_id"],
                "formula_pretty": meta["formula_pretty"],
                "energy_above_hull": meta.get("energy_above_hull"),
                "energy_per_atom": meta.get("energy_per_atom"),
                "is_stable": meta.get("is_stable"),
                "cif_file": cif_path.name,
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "query_chemsys": chemsys,
                "count": len(summary),
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def download_query(query: str, top_k: int, output_root: Path, api_key: str) -> Dict[str, Any]:
    query_type, normalized_query = normalize_query(query)
    if len(api_key) == 32:
        if query_type == "chemsys":
            rows = fetch_modern_chemsys(normalized_query, top_k, api_key)
        else:
            rows = fetch_modern_formula(normalized_query, top_k, api_key)
        api_mode = "modern"
    else:
        if query_type == "chemsys":
            rows = fetch_legacy_chemsys(normalized_query, top_k, api_key)
        else:
            rows = fetch_legacy_formula(normalized_query, top_k, api_key)
        api_mode = "legacy"

    if not rows:
        raise ValueError(f"{normalized_query} 没有返回可下载的 CIF 结果。")

    output_dir = save_results(normalized_query, rows, output_root)
    results = []
    for index, (meta, _) in enumerate(rows, start=1):
        results.append(
            {
                "rank": index,
                "material_id": meta["material_id"],
                "formula_pretty": meta["formula_pretty"],
                "energy_above_hull": meta.get("energy_above_hull"),
                "energy_per_atom": meta.get("energy_per_atom"),
                "is_stable": meta.get("is_stable"),
            }
        )
    return {
        "query": normalized_query,
        "query_type": query_type,
        "api_mode": api_mode,
        "count": len(results),
        "output_dir": str(output_dir),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 Materials Project 按元素体系下载前 N 个 CIF 文件。")
    parser.add_argument("--query", default="Nb-Ti", help="元素体系，例如 Nb-Ti 或 Nb-Ti-Sn")
    parser.add_argument("--top-k", type=int, default=5, help="最多下载几个 CIF 文件")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Materials Project API key")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).expanduser()
    api_key = str(args.api_key or "").strip()
    if not api_key:
        raise ValueError("未提供 Materials Project API key。")

    payload = download_query(args.query, args.top_k, output_root, api_key)
    print(f"查询对象: {payload['query']}")
    print(f"查询类型: {payload['query_type']}")
    print(f"API 模式: {payload['api_mode']}")
    print(f"下载数量: {payload['count']}")
    print(f"输出目录: {payload['output_dir']}")
    for item in payload["results"]:
        print(f"{item['rank']}. {item['formula_pretty']}  {item['material_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
