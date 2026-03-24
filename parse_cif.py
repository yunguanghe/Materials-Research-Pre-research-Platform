#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CIF_FILE = str(PROJECT_ROOT / "sim_tasks" / "sample" / "materials" / "example.cif")
DEFAULT_JSON_OUT = str(PROJECT_ROOT / "cif_result.json")


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _format_delta(before: Any, after: Any, digits: int = 6) -> str:
    before_value = _safe_float(before)
    after_value = _safe_float(after)
    if before_value is None or after_value is None:
        return ""
    delta = round(after_value - before_value, digits)
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta}"


def _stringify_space_group(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " / ".join(str(item) for item in value)
    return str(value or "")


def parse_with_ase(cif_path: Path) -> Dict[str, Any]:
    from ase.io import read

    atoms = read(str(cif_path))
    cell = atoms.cell
    lengths = cell.lengths()
    angles = cell.angles()

    sites: List[Dict[str, Any]] = []
    scaled = atoms.get_scaled_positions()
    cart = atoms.get_positions()
    for idx, (symbol, frac, xyz) in enumerate(zip(atoms.get_chemical_symbols(), scaled, cart), start=1):
        sites.append(
            {
                "index": idx,
                "element": symbol,
                "fractional_position": [round(float(v), 8) for v in frac],
                "cartesian_position": [round(float(v), 8) for v in xyz],
            }
        )

    return {
        "source_file": str(cif_path),
        "formula": atoms.get_chemical_formula(),
        "reduced_formula": atoms.get_chemical_formula(mode="reduce"),
        "atom_count": len(atoms),
        "cell": {
            "a": round(float(lengths[0]), 8),
            "b": round(float(lengths[1]), 8),
            "c": round(float(lengths[2]), 8),
            "alpha": round(float(angles[0]), 8),
            "beta": round(float(angles[1]), 8),
            "gamma": round(float(angles[2]), 8),
            "matrix": [[round(float(v), 8) for v in row] for row in cell.tolist()],
            "volume": round(float(cell.volume), 8),
        },
        "pbc": [bool(v) for v in atoms.pbc],
        "sites": sites,
    }


def enrich_with_pymatgen(cif_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from pymatgen.core import Structure
    except Exception:
        return payload

    try:
        structure = Structure.from_file(str(cif_path))
    except Exception:
        return payload

    payload["reduced_formula"] = structure.composition.reduced_formula
    payload["formula"] = str(structure.composition.formula)
    payload["lattice"] = {
        "a": round(float(structure.lattice.a), 8),
        "b": round(float(structure.lattice.b), 8),
        "c": round(float(structure.lattice.c), 8),
        "alpha": round(float(structure.lattice.alpha), 8),
        "beta": round(float(structure.lattice.beta), 8),
        "gamma": round(float(structure.lattice.gamma), 8),
        "volume": round(float(structure.lattice.volume), 8),
    }
    try:
        payload["space_group"] = structure.get_space_group_info()
    except Exception:
        pass
    return payload


def parse_cif(cif_path: Path) -> Dict[str, Any]:
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF 文件不存在: {cif_path}")
    payload = parse_with_ase(cif_path)
    payload = enrich_with_pymatgen(cif_path, payload)
    return payload


def compare_cif_payloads(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_lattice = before.get("lattice") or before.get("cell", {})
    after_lattice = after.get("lattice") or after.get("cell", {})

    rows = [
        {
            "指标": "文件",
            "初始结构": before.get("source_file", ""),
            "弛豫后结构": after.get("source_file", ""),
            "变化": "",
        },
        {
            "指标": "化学式",
            "初始结构": before.get("formula", ""),
            "弛豫后结构": after.get("formula", ""),
            "变化": "",
        },
        {
            "指标": "约化化学式",
            "初始结构": before.get("reduced_formula", ""),
            "弛豫后结构": after.get("reduced_formula", ""),
            "变化": "",
        },
        {
            "指标": "原子数",
            "初始结构": before.get("atom_count", ""),
            "弛豫后结构": after.get("atom_count", ""),
            "变化": _format_delta(before.get("atom_count"), after.get("atom_count"), 0),
        },
        {
            "指标": "空间群",
            "初始结构": _stringify_space_group(before.get("space_group", "")),
            "弛豫后结构": _stringify_space_group(after.get("space_group", "")),
            "变化": "",
        },
    ]

    for key, label in [
        ("a", "晶格常数 a"),
        ("b", "晶格常数 b"),
        ("c", "晶格常数 c"),
        ("alpha", "晶格角 alpha"),
        ("beta", "晶格角 beta"),
        ("gamma", "晶格角 gamma"),
        ("volume", "晶胞体积"),
    ]:
        rows.append(
            {
                "指标": label,
                "初始结构": before_lattice.get(key, ""),
                "弛豫后结构": after_lattice.get(key, ""),
                "变化": _format_delta(before_lattice.get(key), after_lattice.get(key)),
            }
        )

    return {
        "before_file": before.get("source_file", ""),
        "after_file": after.get("source_file", ""),
        "rows": rows,
    }


def build_text_summary(payload: Dict[str, Any]) -> str:
    lines = [
        f"文件: {payload['source_file']}",
        f"化学式: {payload.get('formula', '')}",
        f"约化化学式: {payload.get('reduced_formula', '')}",
        f"原子数: {payload.get('atom_count', '')}",
    ]

    lattice = payload.get("lattice") or payload.get("cell", {})
    lines.extend(
        [
            "晶格参数:",
            f"  a = {lattice.get('a', '')}",
            f"  b = {lattice.get('b', '')}",
            f"  c = {lattice.get('c', '')}",
            f"  alpha = {lattice.get('alpha', '')}",
            f"  beta = {lattice.get('beta', '')}",
            f"  gamma = {lattice.get('gamma', '')}",
            f"  volume = {lattice.get('volume', '')}",
        ]
    )

    if payload.get("space_group"):
        lines.append(f"空间群: {payload['space_group']}")

    lines.append("原子位点:")
    for site in payload.get("sites", []):
        lines.append(
            f"  {site['index']}. {site['element']}  frac={site['fractional_position']}  cart={site['cartesian_position']}"
        )
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a CIF file and print structure information.")
    parser.add_argument("cif_file", nargs="?", default=DEFAULT_CIF_FILE, help="Path to the CIF file.")
    parser.add_argument("--json-out", default="", help="Optional path to save parsed JSON.")
    parser.add_argument("--text", action="store_true", help="Print human-readable text instead of JSON.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    cif_path = Path(args.cif_file).expanduser()
    payload = parse_cif(cif_path)

    json_out = args.json_out or ""
    if json_out == "__default__":
        json_out = DEFAULT_JSON_OUT

    if json_out:
        out_path = Path(json_out).expanduser()
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出: {out_path}")

    if args.text:
        print(build_text_summary(payload))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
