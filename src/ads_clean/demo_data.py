from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class DemoRun:
    run_dir: Path
    label: str
    metrics: Dict[str, object]


def available_runs(roots: Optional[Iterable[Path]] = None) -> List[DemoRun]:
    roots = list(roots or [Path("outputs/demo_trace_runs"), Path("outputs/experiments"), Path("outputs")])
    runs: List[DemoRun] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in root.rglob("metrics.json"):
            run_dir = metrics_path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            metrics = _read_json(metrics_path)
            runs.append(DemoRun(run_dir, _run_label(run_dir, metrics), metrics))
    return sorted(runs, key=lambda run: run.run_dir.stat().st_mtime if run.run_dir.exists() else 0, reverse=True)


def load_run(run_dir: Path) -> Dict[str, object]:
    run_dir = Path(run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    workflow = _read_json(run_dir / "workflow_trace.json")
    action_trace = _read_csv(run_dir / "action_trace.csv")
    if action_trace.empty:
        action_trace = _read_csv(run_dir / "decision_log.csv")
    bundle = {
        "run_dir": run_dir,
        "metrics": metrics,
        "workflow": workflow,
        "action_trace": action_trace,
        "operation_trace": _read_csv(run_dir / "operation_trace.csv"),
        "operator_trace": _read_csv(run_dir / "operator_trace.csv"),
        "operation_rule_trace": _read_csv(run_dir / "operation_rule_trace.csv"),
        "operator_weight_trace": _read_csv(run_dir / "operator_weight_trace.csv"),
        "model_trace": _read_csv(run_dir / "model_trace.csv"),
    }
    bundle["capabilities"] = trace_capabilities(bundle)
    return bundle


def trace_capabilities(bundle: Dict[str, object]) -> Dict[str, bool]:
    return {
        "action_trace": not bundle["action_trace"].empty,
        "operation_trace": not bundle["operation_trace"].empty,
        "operator_trace": not bundle["operator_trace"].empty,
        "operation_rule_trace": not bundle["operation_rule_trace"].empty,
        "operator_weight_trace": not bundle["operator_weight_trace"].empty,
        "model_trace": not bundle["model_trace"].empty,
    }


def build_workflow_graph(bundle: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    metrics = bundle["metrics"]
    action_trace: pd.DataFrame = bundle["action_trace"]
    operation_trace: pd.DataFrame = bundle["operation_trace"]
    operator_trace: pd.DataFrame = bundle["operator_trace"]

    nodes = [
        {
            "id": "task",
            "label": f"{metrics.get('dataset', 'dataset')}\n{metrics.get('task_type', '')}, {metrics.get('model_type', '')}",
            "kind": "task",
            "count": int(metrics.get("rows", 0) or 0),
        },
        {
            "id": "controller",
            "label": "Action allocation\nzero-cost policy",
            "kind": "controller",
            "count": int(len(action_trace)),
        },
    ]
    edges = [{"source": "task", "target": "controller", "label": "task and budget"}]

    action_name = {0: "no_op", 1: "repair_value", 2: "delete", 3: "replace_nearby"}
    if not action_trace.empty and "action" in action_trace.columns:
        action_counts = action_trace["action"].astype(int).map(action_name).value_counts().to_dict()
    else:
        action_counts = {}

    for action in ("no_op", "repair_value", "delete", "replace_nearby"):
        count = int(action_counts.get(action, 0))
        if count == 0:
            continue
        node_id = f"action:{action}"
        nodes.append({"id": node_id, "label": f"{action}\n{count} cells", "kind": "action", "count": count})
        edges.append({"source": "controller", "target": node_id, "label": str(count)})

    if not operator_trace.empty:
        op_count = int(len(operator_trace))
        nodes.append({
            "id": "operators",
            "label": f"Available operator orchestration\n{op_count} operators",
            "kind": "operator_stage",
            "count": op_count,
        })
        edges.append({"source": "task", "target": "operators", "label": "runtime value source"})
        for action, label in (("repair_value", "repair action"), ("replace_nearby", "value source")):
            source_id = f"action:{action}"
            if any(node["id"] == source_id for node in nodes):
                edges.append({"source": source_id, "target": "operators", "label": label})

    op_rows = int(len(operation_trace))
    accepted_rows = int(operation_trace["accepted_by_verifier"].fillna(False).astype(bool).sum()) if "accepted_by_verifier" in operation_trace.columns else op_rows
    nodes.append({
        "id": "operations",
        "label": f"Data operation records\n{accepted_rows}/{op_rows} accepted",
        "kind": "operation",
        "count": op_rows,
    })
    uses_operator_values = any(node["id"] in {"action:repair_value", "action:replace_nearby"} for node in nodes)
    if not operator_trace.empty and uses_operator_values:
        edges.append({"source": "operators", "target": "operations", "label": "compiled values"})
    for action in ("repair_value", "delete", "replace_nearby"):
        source_id = f"action:{action}"
        if any(node["id"] == source_id for node in nodes):
            edges.append({"source": source_id, "target": "operations", "label": "execute"})

    nodes.append({
        "id": "verifier",
        "label": f"Verifier\n{metrics.get('verifier_selected', 'candidate')}",
        "kind": "verifier",
        "count": 1,
    })
    edges.append({"source": "operations", "target": "verifier", "label": "evaluate"})
    edges.append({"source": "verifier", "target": "controller", "label": "feedback"})
    return {"nodes": nodes, "edges": edges}


def action_summary(action_trace: pd.DataFrame) -> pd.DataFrame:
    if action_trace.empty or "action" not in action_trace.columns:
        return pd.DataFrame()
    action_names = {0: "no_op", 1: "repair_value", 2: "delete", 3: "replace_nearby"}
    df = action_trace.copy()
    df["action_name"] = df["action"].astype(int).map(action_names)
    for col in ("state_feature_importance", "state_remaining_budget_ratio", "state_remaining_errors_ratio"):
        if col not in df.columns:
            df[col] = pd.NA
    grouped = df.groupby("action_name", dropna=False).agg(
        count=("action", "size"),
        mean_feature_importance=("state_feature_importance", "mean"),
        mean_remaining_budget=("state_remaining_budget_ratio", "mean"),
        mean_remaining_errors=("state_remaining_errors_ratio", "mean"),
    )
    return grouped.reset_index().sort_values("count", ascending=False)


def _run_label(run_dir: Path, metrics: Dict[str, object]) -> str:
    dataset = metrics.get("dataset", run_dir.name)
    scenario = metrics.get("scenario", "")
    error_rate = metrics.get("error_rate") or "native"
    model = metrics.get("model_type", "")
    return f"{dataset} | {scenario}/{error_rate} | {model} | {run_dir.name}"


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
