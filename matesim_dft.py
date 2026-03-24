#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from ase.spacegroup import crystal

from mattersim.applications.relax import Relaxer
from mattersim.forcefield.potential import MatterSimCalculator

SIMULATION_STEPS = [
    "接收材料设计指令",
    "解析结构输入",
    "加载计算模型",
    "执行计算",
]

SUPPORTED_FILE_SUFFIXES = {".cif", ".vasp", ".poscar", ".contcar", ".xyz"}
MATERIALS_SUBDIR = "materials"
RUNTIME_SUBDIR = "runtime"


def update_progress(progress_path: Path, step_index: int, note: str, status: str = "running") -> None:
    steps: List[Dict[str, str]] = []
    for idx, label in enumerate(SIMULATION_STEPS):
        if idx < step_index:
            step_status = "done"
        elif idx == step_index:
            step_status = status
        else:
            step_status = "waiting"
        steps.append({"label": label, "status": step_status})

    payload = {
        "current_step": step_index,
        "status": status,
        "note": note,
        "steps": steps,
        "updated_at": time.time(),
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_failure(progress_path: Path, result_path: Path, step_index: int, note: str) -> None:
    result = {
        "summary": note,
        "table": [
            {"字段": "状态", "数值": "失败"},
            {"字段": "原因", "数值": note},
        ],
        "artifacts": {},
        "error": note,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    update_progress(progress_path, step_index, note, status="failed")


def get_materials_dir(task_dir: Path) -> Path:
    path = task_dir / MATERIALS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_runtime_dir(task_dir: Path) -> Path:
    path = task_dir / RUNTIME_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_formula(question: str) -> Optional[str]:
    matches = re.findall(r"[A-Z][a-z]?(?:[0-9]+(?:\.[0-9]+)?)?", question)
    if matches:
        return "".join(matches[:8])
    return None


def find_structure_path(question: str) -> Optional[Path]:
    candidates = re.findall(r"/[^\s]+", question)
    for candidate in candidates:
        path = Path(candidate.strip('\'"'))
        if path.suffix.lower() in SUPPORTED_FILE_SUFFIXES and path.exists():
            return path
    return None


def build_atoms_from_formula(formula: str) -> Atoms:
    clean = formula.strip()
    if clean in {"Al", "Cu", "Ag", "Au", "Ni", "Pd", "Pt"}:
        return bulk(clean, crystalstructure="fcc")
    if clean in {"Fe", "Cr", "Mo", "W", "V", "Nb", "Ta"}:
        return bulk(clean, crystalstructure="bcc")
    if clean in {"Mg", "Ti", "Co", "Zn"}:
        return bulk(clean, crystalstructure="hcp")
    if clean in {"Si", "Ge", "C"}:
        return bulk(clean, crystalstructure="diamond")
    if clean == "NaCl":
        return bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    if clean == "MgO":
        return bulk("MgO", crystalstructure="rocksalt", a=4.21)
    if clean == "SiC":
        return bulk("SiC", crystalstructure="zincblende", a=4.36)
    if clean == "GaAs":
        return bulk("GaAs", crystalstructure="zincblende", a=5.65)
    if clean == "TiO2":
        return crystal(
            symbols=["Ti", "O"],
            basis=[(0, 0, 0), (0.305, 0.305, 0)],
            spacegroup=136,
            cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
        )
    raise ValueError(
        f"暂时不能仅凭公式 {clean} 自动构造可靠初始结构。请在问题里附上 CIF/POSCAR/XYZ 文件路径，"
        "或者使用当前脚本已支持的原型材料（如 Si、Al、Cu、Fe、NaCl、MgO、SiC、GaAs、TiO2）。"
    )


def build_atoms(question: str) -> Tuple[Atoms, Dict[str, str]]:
    structure_path = find_structure_path(question)
    if structure_path is not None:
        atoms = read(structure_path)
        metadata = {
            "source": "file",
            "structure_path": str(structure_path),
            "formula": atoms.get_chemical_formula(),
        }
        return atoms, metadata

    formula = normalize_formula(question)
    if not formula:
        raise ValueError("没有从指令中识别到材料公式，也没有检测到结构文件路径。")

    atoms = build_atoms_from_formula(formula)
    metadata = {
        "source": "formula",
        "formula": formula,
    }
    return atoms, metadata


def load_metadata(metadata_path: Optional[str]) -> Dict[str, object]:
    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_params(params_path: Optional[str]) -> Dict[str, object]:
    if not params_path:
        return {}
    path = Path(params_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_calculator(device: str, model_path: Optional[str], extra_kwargs: Optional[Dict[str, object]] = None) -> MatterSimCalculator:
    kwargs: Dict[str, object] = {"device": device}
    if model_path:
        kwargs["load_path"] = model_path
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return MatterSimCalculator(**kwargs)


def run_relaxation(
    atoms: Atoms,
    *,
    device: str,
    model_path: Optional[str],
    steps: int,
    fmax: float,
    optimizer_name: str,
    filter_name: str,
    constrain_symmetry: bool,
    calculator_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[Atoms, bool]:
    calculator = build_calculator(device=device, model_path=model_path, extra_kwargs=calculator_kwargs)
    atoms = atoms.copy()
    atoms.calc = calculator

    relaxer = Relaxer(
        optimizer=optimizer_name,
        filter=filter_name or None,
        constrain_symmetry=constrain_symmetry,
    )
    converged, relaxed_atoms = relaxer.relax(atoms, steps=steps, fmax=fmax)
    return relaxed_atoms, converged


def make_summary_table(relaxed_atoms: Atoms, converged: bool, metadata: Dict[str, str]) -> List[Dict[str, object]]:
    energy = float(relaxed_atoms.get_potential_energy())
    forces = relaxed_atoms.get_forces()
    max_force = max(math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2) for x, y, z in forces)
    lengths = relaxed_atoms.cell.lengths()
    angles = relaxed_atoms.cell.angles()
    display_formula = (
        metadata.get("original_formula")
        or metadata.get("materials_project_formula")
        or metadata.get("formula")
        or relaxed_atoms.get_chemical_formula()
    )

    return [
        {"字段": "候选材料", "数值": display_formula},
        {"字段": "输入来源", "数值": metadata.get("source", "unknown")},
        {"字段": "是否收敛", "数值": "是" if converged else "否"},
        {"字段": "总能量 (eV)", "数值": round(energy, 6)},
        {"字段": "每原子能量 (eV/atom)", "数值": round(energy / len(relaxed_atoms), 6)},
        {"字段": "最大受力 (eV/Å)", "数值": round(max_force, 6)},
        {"字段": "晶格 a (Å)", "数值": round(float(lengths[0]), 6)},
        {"字段": "晶格 b (Å)", "数值": round(float(lengths[1]), 6)},
        {"字段": "晶格 c (Å)", "数值": round(float(lengths[2]), 6)},
        {"字段": "晶格 α (°)", "数值": round(float(angles[0]), 4)},
        {"字段": "晶格 β (°)", "数值": round(float(angles[1]), 4)},
        {"字段": "晶格 γ (°)", "数值": round(float(angles[2]), 4)},
        {"字段": "原子数", "数值": len(relaxed_atoms)},
    ]


def write_svg(svg_path: Path, rows: List[Dict[str, object]], title: str) -> None:
    chart_rows = [row for row in rows if isinstance(row["数值"], (int, float))]
    if not chart_rows:
        chart_rows = rows[:6]

    numeric_rows = [row for row in chart_rows if isinstance(row["数值"], (int, float))]
    max_value = max(abs(float(row["数值"])) for row in numeric_rows) if numeric_rows else 1.0
    width = 920
    height = max(360, 120 + len(chart_rows) * 42)

    bars: List[str] = []
    for idx, row in enumerate(chart_rows):
        value = row["数值"]
        y = 92 + idx * 38
        if isinstance(value, (int, float)):
            bar_w = 420 * abs(float(value)) / max_value if max_value else 0
            value_label = f"{value}"
            bar = (
                f'<rect x="300" y="{y}" width="430" height="22" rx="11" fill="rgba(255,255,255,0.06)" />'
                f'<rect x="300" y="{y}" width="{bar_w:.1f}" height="22" rx="11" fill="url(#grad)" />'
            )
        else:
            value_label = str(value)
            bar = '<rect x="300" y="{y}" width="430" height="22" rx="11" fill="rgba(255,255,255,0.03)" />'
        bars.append(
            f'''
            <text x="46" y="{y + 16}" fill="#d8f4ff" font-size="13">{row["字段"]}</text>
            {bar}
            <text x="748" y="{y + 16}" fill="#8fdcff" font-size="13">{value_label}</text>
            '''
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#081426"/>
      <stop offset="100%" stop-color="#0f2746"/>
    </linearGradient>
    <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#4cc9ff"/>
      <stop offset="100%" stop-color="#1b73ff"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" rx="24" fill="url(#bg)"/>
  <rect x="18" y="18" width="{width - 36}" height="{height - 36}" rx="22" fill="rgba(255,255,255,0.03)" stroke="rgba(108,193,255,0.22)"/>
  <text x="38" y="54" fill="#eefaff" font-size="24" font-family="Arial">计算报告</text>
  <text x="38" y="80" fill="#84b8dc" font-size="15" font-family="Arial">{title}</text>
  {''.join(bars)}
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "candidate"


def load_candidate_manifest(manifest_path: Optional[str]) -> Dict[str, object]:
    if not manifest_path:
        return {"candidates": [], "query_stats": [], "requested_queries": []}
    path = Path(manifest_path)
    if not path.exists():
        return {"candidates": [], "query_stats": [], "requested_queries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "candidates": payload.get("candidates", []),
        "query_stats": payload.get("query_stats", []),
        "requested_queries": payload.get("requested_queries", []),
    }


def _priority_score(entry: Dict[str, object], question: str) -> float:
    energy = float(entry["energy_per_atom"])
    max_force = float(entry["max_force"])
    score = -energy * 20.0 - max_force * 8.0
    question_lower = question.lower()
    if "稳定" in question or "stable" in question_lower:
        score += 1.5 if entry["converged"] else -1.5
    if "超导" in question or "tc" in question_lower:
        score -= 0.5
    return round(score, 4)


def _recommendation(entry: Dict[str, object], question: str) -> str:
    if "超导" in question or "tc" in question.lower():
        return "按稳定性代理排序；MatterSim 不能直接预测 Tc，建议把靠前候选继续做后续电子结构分析。"
    if float(entry["max_force"]) < 0.08 and entry["converged"]:
        return "优先继续验证"
    if entry["converged"]:
        return "可作为备选"
    return "先降低优先级"


def _run_single_candidate(
    candidate: Dict[str, object],
    *,
    task_dir: Path,
    question: str,
    device: str,
    model_path: Optional[str],
    steps: int,
    fmax: float,
    optimizer_name: str,
    filter_name: str,
    constrain_symmetry: bool,
    calculator_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    materials_dir = get_materials_dir(task_dir)
    structure_path = Path(str(candidate.get("structure_path", ""))).expanduser()
    if not structure_path.exists():
        raise FileNotFoundError(f"候选结构文件不存在: {structure_path}")

    atoms = read(structure_path)
    metadata = dict(candidate.get("metadata", {}) or {})
    if "formula" not in metadata:
        metadata["formula"] = atoms.get_chemical_formula()
    display_formula = (
        metadata.get("original_formula")
        or metadata.get("materials_project_formula")
        or metadata.get("formula")
        or atoms.get_chemical_formula()
    )

    relaxed_atoms, converged = run_relaxation(
        atoms,
        device=device,
        model_path=model_path,
        steps=steps,
        fmax=fmax,
        optimizer_name=optimizer_name,
        filter_name=filter_name,
        constrain_symmetry=constrain_symmetry,
        calculator_kwargs=calculator_kwargs,
    )
    summary_rows = make_summary_table(relaxed_atoms, converged, metadata)
    summary_map = {row["字段"]: row["数值"] for row in summary_rows}
    slug = _slugify(str(candidate.get("name", metadata.get("formula", "candidate"))))
    relaxed_path = materials_dir / f"relaxed_{slug}.cif"
    write(relaxed_path, relaxed_atoms)

    result = {
        "name": str(candidate.get("name", metadata.get("formula", slug))),
        "formula": str(metadata.get("formula", relaxed_atoms.get_chemical_formula())),
        "display_formula": str(display_formula),
        "source": str(metadata.get("source", "unknown")),
        "converged": bool(converged),
        "energy_per_atom": float(summary_map["每原子能量 (eV/atom)"]),
        "max_force": float(summary_map["最大受力 (eV/Å)"]),
        "relaxed_structure": relaxed_path.name,
        "metadata": metadata,
        "details": summary_rows,
    }
    result["priority_score"] = _priority_score(result, question)
    result["recommendation"] = _recommendation(result, question)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--structure", default="")
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--candidate-manifest", default="")
    parser.add_argument("--params-json", default="")
    args = parser.parse_args()

    task_dir = Path(args.output_dir) / args.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    materials_dir = get_materials_dir(task_dir)
    runtime_dir = get_runtime_dir(task_dir)
    progress_path = runtime_dir / "progress.json"
    result_path = runtime_dir / "result.json"
    image_path = runtime_dir / "summary.svg"
    structure_path = materials_dir / "relaxed.cif"
    source_structure_path = Path(args.structure).expanduser() if args.structure else None
    metadata_from_file = load_metadata(args.metadata_json)
    manifest_payload = load_candidate_manifest(args.candidate_manifest)
    candidates = list(manifest_payload.get("candidates", []))
    query_stats = list(manifest_payload.get("query_stats", []))
    requested_queries = list(manifest_payload.get("requested_queries", []))
    params = load_params(args.params_json)

    device = str(params.get("device") or os.environ.get("MATTERSIM_DEVICE", "cpu"))
    model_path = str(params.get("custom_params", {}).get("model_path") or os.environ.get("MATTERSIM_MODEL_PATH", "")).strip() or None
    calculator_kwargs = {
        key: value
        for key, value in dict(params.get("custom_params", {}) or {}).items()
        if key not in {"model_path"}
    }
    relax_steps = int(params.get("relax_steps") or os.environ.get("MATTERSIM_RELAX_STEPS", "200"))
    relax_fmax = float(params.get("relax_fmax") or os.environ.get("MATTERSIM_RELAX_FMAX", "0.05"))
    optimizer_name = str(params.get("optimizer") or os.environ.get("MATTERSIM_OPTIMIZER", "FIRE"))
    filter_name = str(params.get("filter") if params.get("filter") is not None else os.environ.get("MATTERSIM_FILTER", "ExpCellFilter"))
    constrain_symmetry = bool(params.get("constrain_symmetry", os.environ.get("MATTERSIM_CONSTRAIN_SYMMETRY", "0") == "1"))

    try:
        update_progress(progress_path, 0, "已接收材料设计指令")

        update_progress(progress_path, 1, "正在解析结构输入")
        if not candidates:
            if source_structure_path and source_structure_path.exists():
                atoms = read(source_structure_path)
                metadata = {
                    "source": "file",
                    "structure_path": str(source_structure_path),
                    "formula": atoms.get_chemical_formula(),
                }
            else:
                atoms, metadata = build_atoms(args.question)
            if metadata_from_file:
                metadata.update(metadata_from_file)
            candidates = [
                {
                    "name": metadata.get("formula", "candidate-1"),
                    "structure_path": str(source_structure_path) if source_structure_path else "",
                    "metadata": metadata,
                }
            ]
            if not source_structure_path:
                temp_input_path = materials_dir / "input_single.cif"
                write(temp_input_path, atoms)
                candidates[0]["structure_path"] = str(temp_input_path)

        update_progress(progress_path, 2, "正在加载计算模型")
        # 先尝试构造计算器，尽早暴露模型文件或下载问题
        _ = build_calculator(device=device, model_path=model_path, extra_kwargs=calculator_kwargs)

        candidate_results: List[Dict[str, object]] = []
        for index, candidate in enumerate(candidates, start=1):
            update_progress(progress_path, 3, f"正在执行 MatterSim 结构弛豫 ({index}/{len(candidates)})")
            candidate_results.append(
                _run_single_candidate(
                    candidate,
                    task_dir=task_dir,
                    question=args.question,
                    device=device,
                    model_path=model_path,
                    steps=relax_steps,
                    fmax=relax_fmax,
                    optimizer_name=optimizer_name,
                    filter_name=filter_name,
                    constrain_symmetry=constrain_symmetry,
                    calculator_kwargs=calculator_kwargs,
                )
            )

        candidate_results.sort(key=lambda item: item["priority_score"], reverse=True)
        for rank, item in enumerate(candidate_results, start=1):
            item["rank"] = rank

        update_progress(progress_path, 4, "正在汇总 MatterSim 结果")
        top = candidate_results[0]
        rows = [
            {"字段": "最佳候选", "数值": top["name"]},
            {"字段": "化学式", "数值": top["formula"]},
            {"字段": "每原子能量 (eV/atom)", "数值": round(float(top["energy_per_atom"]), 6)},
            {"字段": "最大受力 (eV/Å)", "数值": round(float(top["max_force"]), 6)},
            {"字段": "优先级分数", "数值": round(float(top["priority_score"]), 4)},
            {"字段": "推荐", "数值": top["recommendation"]},
        ]
        if requested_queries:
            rows = [
                {"字段": "请求检索体系数", "数值": len(requested_queries)},
                {"字段": "实际进入计算候选数", "数值": len(candidate_results)},
            ] + rows
    except Exception as exc:
        write_failure(progress_path, result_path, min(4, len(SIMULATION_STEPS) - 1), str(exc))
        raise

    try:
        if len(candidate_results) == 1:
            single_path = task_dir / candidate_results[0]["relaxed_structure"]
            if single_path.exists():
                structure_path.write_bytes(single_path.read_bytes())
        write_svg(image_path, rows, top["formula"])

        result = {
            "question": args.question,
            "formula": top["formula"],
            "summary": f"Agent 已完成 {len(candidate_results)} 个候选的快排，当前优先推荐 {top['name']}。",
            "table": rows,
            "candidates": candidate_results,
            "query_stats": query_stats,
            "artifacts": {
                "image": image_path.name,
                "result_json": result_path.name,
            },
            "metadata": {"candidate_count": len(candidate_results)},
        }
        result["metadata"]["simulation_params"] = {
            "relax_steps": relax_steps,
            "relax_fmax": relax_fmax,
            "optimizer": optimizer_name,
            "filter": filter_name,
            "constrain_symmetry": constrain_symmetry,
            "device": device,
            "custom_params": params.get("custom_params", {}),
        }
        result["metadata"]["requested_queries"] = requested_queries
        result["metadata"]["query_stats"] = query_stats
        if requested_queries:
            missing_queries = [str(item.get("query", "")) for item in query_stats if int(item.get("returned_candidates", 0) or 0) == 0]
            if missing_queries:
                result["summary"] += f" 其中未返回候选的体系有: {'、'.join(missing_queries)}。"
        if len(candidate_results) == 1 and structure_path.exists():
            result["artifacts"]["relaxed_structure"] = structure_path.name
        result["run_id"] = hashlib.md5(f"{args.task_id}:{result['formula']}".encode("utf-8")).hexdigest()[:12]
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        update_progress(progress_path, len(SIMULATION_STEPS) - 1, "MatterSim 任务完成", status="completed")
    except Exception as exc:
        write_failure(progress_path, result_path, len(SIMULATION_STEPS) - 1, f"结果写出失败: {exc}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"matesim_dft failed: {exc}", file=sys.stderr)
        raise
