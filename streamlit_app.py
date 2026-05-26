from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ads_clean.demo_data import action_summary, available_runs, build_workflow_graph, load_run


ROOT = Path(__file__).resolve().parent


TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "toggle": "中文",
        "title": "DDPAgent Workflow Explorer",
        "caption": "Default view uses real cached artifacts. The run button starts a real DDPAgent execution with runtime operator tracing.",
        "no_runs": "No experiment outputs were found. Run `bash scripts/run_demo_trace.sh` or use the real-run panel below.",
        "run": "Run",
        "run_dir": "Run directory",
        "dataset": "Dataset",
        "scenario": "Scenario",
        "task": "Task",
        "model": "Model",
        "delta": "Fixed split delta",
        "missing_operator": "This cached run has no runtime operator trace. The viewer shows real action and data-operation traces only. Run with operator tracing to inspect operator orchestration.",
        "real_panel": "Run Real Pipeline",
        "real_note": "This button invokes `python -m ads_clean.cli run` with `--force-uniclean-run --trace-operators`. It does not simulate results.",
        "error_rate": "Injected error rate",
        "episodes": "Episodes",
        "max_errors": "Max detected errors, 0 means full detector output",
        "single_max": "UniClean block threshold",
        "start_real": "Start real run",
        "running": "Running DDPAgent. This may take minutes because UniClean is executed instead of reading cached cleaned tables.",
        "finished": "Real run completed.",
        "failed": "Real run failed.",
        "inspect": "Inspect workflow node",
        "task_model": "Task And Model",
        "action_alloc": "Action Allocation",
        "no_action_trace": "No action trace is available for this run.",
        "action_records": "Action Records",
        "operators": "Operator Orchestration",
        "no_operator": "No runtime operator trace was collected for this run.",
        "rules": "Generated repair rules",
        "weights": "Execution feedback weights",
        "operations": "Data Operation Records",
        "no_operations": "No data operation records are available. This can happen when the policy chose no-op.",
        "changed": "Changed records",
        "accepted": "Accepted by verifier",
        "verifier": "Verifier And Downstream Utility",
        "operator": "Operator",
    },
    "zh": {
        "toggle": "English",
        "title": "DDPAgent 工作流可视化",
        "caption": "默认展示真实缓存结果。点击真实运行按钮会启动带算子 trace 的真实 DDPAgent 流程。",
        "no_runs": "没有找到实验输出。可以先运行 `bash scripts/run_demo_trace.sh`，或使用下方真实运行面板。",
        "run": "运行结果",
        "run_dir": "运行目录",
        "dataset": "数据集",
        "scenario": "场景",
        "task": "任务",
        "model": "模型",
        "delta": "固定划分提升",
        "missing_operator": "该缓存结果没有运行时算子 trace。界面只展示真实动作分配和数据操作 trace。若要查看算子编排，请使用带算子 trace 的真实运行。",
        "real_panel": "真实运行流程",
        "real_note": "该按钮会调用 `python -m ads_clean.cli run`，并启用 `--force-uniclean-run --trace-operators`。不会模拟结果。",
        "error_rate": "人工注入错误率",
        "episodes": "训练轮数",
        "max_errors": "最大检测错误数，0 表示使用完整检测结果",
        "single_max": "UniClean 分块阈值",
        "start_real": "开始真实运行",
        "running": "正在运行 DDPAgent。由于会真实执行 UniClean 而不是读取缓存，可能需要数分钟。",
        "finished": "真实运行完成。",
        "failed": "真实运行失败。",
        "inspect": "查看工作流节点",
        "task_model": "任务与模型",
        "action_alloc": "动作分配",
        "no_action_trace": "该运行没有动作 trace。",
        "action_records": "动作记录",
        "operators": "算子编排",
        "no_operator": "该运行没有收集运行时算子 trace。",
        "rules": "生成的修复规则",
        "weights": "执行反馈权重",
        "operations": "数据操作记录",
        "no_operations": "该运行没有数据操作记录。这可能表示策略选择了 no-op。",
        "changed": "发生变化的记录",
        "accepted": "验证器接受",
        "verifier": "验证器与下游收益",
        "operator": "算子",
    },
}


def main() -> None:
    st.set_page_config(page_title="DDPAgent Workflow Explorer", layout="wide")
    lang = _language()
    t = lambda key: TEXT[lang][key]

    st.sidebar.button(t("toggle"), on_click=_toggle_language)
    st.title(t("title"))
    st.caption(t("caption"))

    _render_real_run_panel(lang)

    runs = available_runs()
    if not runs:
        st.info(t("no_runs"))
        st.stop()

    selected_label = st.sidebar.selectbox(t("run"), [run.label for run in runs])
    selected = next(run for run in runs if run.label == selected_label)
    bundle = load_run(selected.run_dir)
    metrics = bundle["metrics"]

    st.sidebar.write(t("run_dir"))
    st.sidebar.code(str(selected.run_dir))

    top = st.columns(5)
    top[0].metric(t("dataset"), str(metrics.get("dataset", "")))
    top[1].metric(t("scenario"), f"{metrics.get('scenario', '')}/{metrics.get('error_rate') or 'native'}")
    top[2].metric(t("task"), str(metrics.get("task_type", "")))
    top[3].metric(t("model"), str(metrics.get("model_type", "")))
    delta = metrics.get("downstream_fixed_delta", metrics.get("downstream_delta", ""))
    top[4].metric(t("delta"), _fmt_metric(delta))

    caps = bundle["capabilities"]
    if not caps["operator_trace"]:
        st.warning(t("missing_operator"))

    graph = _localize_graph(build_workflow_graph(bundle), lang)
    st.plotly_chart(_workflow_figure(graph), use_container_width=True)

    node_options = [f"{node['id']} | {node['label'].replace(chr(10), ' ')}" for node in graph["nodes"]]
    chosen_node = st.selectbox(t("inspect"), node_options)
    chosen_id = chosen_node.split(" | ", 1)[0]
    _render_node(chosen_id, bundle, metrics, lang)


def _language() -> str:
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    return st.session_state.lang


def _toggle_language() -> None:
    st.session_state.lang = "zh" if st.session_state.get("lang", "en") == "en" else "en"


def _render_real_run_panel(lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    with st.expander(t("real_panel"), expanded=False):
        st.write(t("real_note"))
        with st.form("real_run_form"):
            cols = st.columns(4)
            dataset = cols[0].selectbox(t("dataset"), ["beers", "flights", "hospitals", "rayyan", "tax"])
            scenario = cols[1].selectbox(t("scenario"), ["original", "artificial"])
            model_options = ["ridge", "linear", "random_forest"] if dataset == "tax" else ["random_forest", "svm"]
            model = cols[2].selectbox(t("model"), model_options)
            error_rate = cols[3].selectbox(t("error_rate"), ["1", "025", "05", "075", "125", "15", "175", "2"])

            cols2 = st.columns(3)
            episodes = int(cols2[0].number_input(t("episodes"), min_value=1, max_value=1000, value=50, step=10))
            max_errors = int(cols2[1].number_input(t("max_errors"), min_value=0, max_value=100000, value=50, step=10))
            single_max = int(cols2[2].number_input(t("single_max"), min_value=100, max_value=100000, value=10000, step=1000))
            submitted = st.form_submit_button(t("start_real"))

        if submitted:
            with st.spinner(t("running")):
                result = _run_real_pipeline(dataset, scenario, error_rate, model, episodes, max_errors, single_max)
            if result.returncode == 0:
                st.success(t("finished"))
                st.code(result.stdout[-4000:])
                st.rerun()
            else:
                st.error(t("failed"))
                st.code((result.stdout + "\n" + result.stderr)[-8000:])


def _run_real_pipeline(dataset: str, scenario: str, error_rate: str, model: str, episodes: int, max_errors: int, single_max: int):
    cmd = [
        sys.executable,
        "-m",
        "ads_clean.cli",
        "run",
        "--dataset",
        dataset,
        "--scenario",
        scenario,
        "--output-root",
        "outputs/demo_trace_runs",
        "--profile",
        "default",
        "--episodes",
        str(episodes),
        "--rf-estimators",
        "10",
        "--base-cv-folds",
        "3",
        "--max-detected-errors",
        str(max_errors),
        "--single-max",
        str(single_max),
        "--model-type-override",
        model,
        "--force-uniclean-run",
        "--trace-operators",
        "--quiet",
    ]
    if scenario == "artificial":
        cmd.extend(["--error-rate", error_rate, "--result-assets"])
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True, timeout=None)


def _render_node(chosen_id: str, bundle, metrics, lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    if chosen_id == "task":
        st.subheader(t("task_model"))
        st.json({
            "dataset": metrics.get("dataset"),
            "task_type": metrics.get("task_type"),
            "target": metrics.get("target"),
            "model_type": metrics.get("model_type"),
            "rows": metrics.get("rows"),
            "feature_count": metrics.get("feature_count"),
            "verifier_selected": metrics.get("verifier_selected"),
        })
    elif chosen_id == "controller":
        st.subheader(t("action_alloc"))
        summary = action_summary(bundle["action_trace"])
        if summary.empty:
            st.info(t("no_action_trace"))
        else:
            st.dataframe(summary, use_container_width=True)
            st.bar_chart(summary.set_index("action_name")["count"])
    elif chosen_id.startswith("action:"):
        action = chosen_id.split(":", 1)[1]
        st.subheader(f"{t('action_records')} {action}")
        st.dataframe(_display_sample(_filter_action(bundle["action_trace"], action)), use_container_width=True)
    elif chosen_id == "operators":
        st.subheader(t("operators"))
        if bundle["operator_trace"].empty:
            st.info(t("no_operator"))
        else:
            st.dataframe(bundle["operator_trace"], use_container_width=True)
            _operator_details(bundle, lang)
    elif chosen_id == "operations":
        st.subheader(t("operations"))
        op_df = bundle["operation_trace"]
        if op_df.empty:
            st.info(t("no_operations"))
        else:
            st.dataframe(_display_sample(op_df), use_container_width=True)
            changed = int(op_df["changed"].fillna(False).astype(bool).sum()) if "changed" in op_df.columns else len(op_df)
            accepted = int(op_df["accepted_by_verifier"].fillna(False).astype(bool).sum()) if "accepted_by_verifier" in op_df.columns else len(op_df)
            cols = st.columns(2)
            cols[0].metric(t("changed"), changed)
            cols[1].metric(t("accepted"), accepted)
    elif chosen_id == "verifier":
        st.subheader(t("verifier"))
        model_trace = bundle["model_trace"]
        if not model_trace.empty:
            st.dataframe(model_trace, use_container_width=True)
        else:
            st.json({
                "downstream_before": metrics.get("downstream_before"),
                "downstream_after": metrics.get("downstream_after"),
                "downstream_fixed_before": metrics.get("downstream_fixed_before"),
                "downstream_fixed_after": metrics.get("downstream_fixed_after"),
                "verifier_selected": metrics.get("verifier_selected"),
            })


def _operator_details(bundle, lang: str) -> None:
    t = lambda key: TEXT[lang][key]
    operators = bundle["operator_trace"]
    if "operator_id" not in operators.columns:
        return
    operator_id = st.selectbox(t("operator"), operators["operator_id"].astype(str).drop_duplicates().tolist())
    selected = operators[operators["operator_id"].astype(str) == operator_id]
    st.dataframe(selected, use_container_width=True)

    rules = bundle["operation_rule_trace"]
    if not rules.empty and "operator_id" in rules.columns:
        block_id = f"block:{selected.iloc[0].get('node', '')}" if not selected.empty else ""
        matched_rules = rules[rules["operator_id"].astype(str).isin({operator_id, block_id})]
        st.write(t("rules"))
        st.dataframe(_display_sample(matched_rules), use_container_width=True)

    weights = bundle["operator_weight_trace"]
    if not weights.empty and "operator_id" in weights.columns:
        matched_weights = weights[weights["operator_id"].astype(str) == operator_id]
        st.write(t("weights"))
        st.dataframe(matched_weights, use_container_width=True)


def _localize_graph(graph, lang: str):
    if lang == "en":
        return graph
    localized = {"nodes": [], "edges": []}
    for node in graph["nodes"]:
        new_node = dict(node)
        count = node.get("count", 0)
        if node["id"] == "task":
            new_node["label"] = node["label"]
        elif node["id"] == "controller":
            new_node["label"] = f"动作分配\n{count} 个错误"
        elif node["id"].startswith("action:"):
            action = node["id"].split(":", 1)[1]
            new_node["label"] = f"{_action_zh(action)}\n{count} 个单元"
        elif node["id"] == "operators":
            new_node["label"] = f"可用算子编排\n{count} 个算子"
        elif node["id"] == "operations":
            new_node["label"] = node["label"].replace("Data operation records", "数据操作记录").replace("accepted", "已接受")
        elif node["id"] == "verifier":
            new_node["label"] = node["label"].replace("Verifier", "验证器")
        localized["nodes"].append(new_node)
    edge_map = {
        "task and budget": "任务与预算",
        "runtime value source": "运行时值来源",
        "execute": "执行",
        "evaluate": "评估",
        "feedback": "反馈",
        "repair action": "修复动作",
        "value source": "值来源",
        "compiled values": "编译后的值",
    }
    for edge in graph["edges"]:
        new_edge = dict(edge)
        new_edge["label"] = edge_map.get(edge.get("label"), edge.get("label"))
        localized["edges"].append(new_edge)
    return localized


def _action_zh(action: str) -> str:
    return {
        "no_op": "不操作",
        "repair_value": "真值修复",
        "delete": "删除",
        "replace_nearby": "近邻替换",
    }.get(action, action)


def _fmt_metric(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _filter_action(df: pd.DataFrame, action_name: str) -> pd.DataFrame:
    if df.empty or "action" not in df.columns:
        return pd.DataFrame()
    action_map = {"no_op": 0, "repair_value": 1, "delete": 2, "replace_nearby": 3}
    return df[df["action"].astype(int) == action_map[action_name]]


def _display_sample(df: pd.DataFrame, n: int = 500) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(n)


def _workflow_figure(graph) -> go.Figure:
    nodes = graph["nodes"]
    edges = graph["edges"]
    layers = {
        "task": 0,
        "controller": 1,
        "action": 2,
        "operator_stage": 3,
        "operation": 4,
        "verifier": 5,
    }
    by_layer = {}
    for node in nodes:
        by_layer.setdefault(layers.get(node["kind"], 0), []).append(node)

    positions = {}
    for layer, layer_nodes in by_layer.items():
        total = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            positions[node["id"]] = (layer, (total - 1) / 2 - i)

    edge_x = []
    edge_y = []
    annotations = []
    for edge in edges:
        if edge["source"] not in positions or edge["target"] not in positions:
            continue
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor="#667085",
                opacity=0.75,
            )
        )

    color_map = {
        "task": "#3B82F6",
        "controller": "#7C3AED",
        "action": "#059669",
        "operator_stage": "#D97706",
        "operation": "#DC2626",
        "verifier": "#475569",
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.3, color="#98A2B3"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=[positions[node["id"]][0] for node in nodes],
            y=[positions[node["id"]][1] for node in nodes],
            mode="markers+text",
            marker=dict(size=44, color=[color_map.get(node["kind"], "#475569") for node in nodes], line=dict(width=1.5, color="white")),
            text=[node["label"] for node in nodes],
            textposition="bottom center",
            textfont=dict(size=13, color="#111827"),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=30, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(visible=False, range=(-0.5, 5.5)),
        yaxis=dict(visible=False),
        annotations=annotations,
    )
    return fig


if __name__ == "__main__":
    main()
