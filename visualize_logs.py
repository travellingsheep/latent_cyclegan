import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh


# 训练日志中主要使用的指标字段。
LOSS_METRICS = [
    "loss_G",
    "loss_D",
    "loss_gan_G",
    "loss_gan_F",
    "loss_cyc",
    "loss_id",
    "loss_D_A",
    "loss_D_B",
]
LR_METRICS = ["lr_G", "lr_D"]
ALL_METRICS = LOSS_METRICS + LR_METRICS
DEFAULT_CONFIG_PATH = "configs/example.yaml"


def resolve_config_path_value(config_path: str, path_value: str) -> str:
    """将配置中的相对路径解析为绝对路径。"""
    if os.path.isabs(path_value):
        return path_value
    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    return os.path.abspath(os.path.join(config_dir, path_value))


def parse_args() -> argparse.Namespace:
    """解析命令行参数，同时忽略 Streamlit 注入的额外参数。"""
    parser = argparse.ArgumentParser(description="CycleGAN 训练日志交互式看板")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="默认 YAML 配置文件路径",
    )
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件，并返回顶层字典。"""
    with open(config_path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML 配置顶层必须是字典映射")
    return config


def resolve_log_path(config_path: str, config: Dict[str, Any]) -> str:
    """根据配置文件中的 logging 段动态解析 JSONL 日志路径。"""
    logging_cfg = config.get("logging", {})
    if not isinstance(logging_cfg, dict):
        raise ValueError("配置项 logging 必须是字典")

    log_dir = logging_cfg.get("log_dir")
    log_file = logging_cfg.get("log_file")
    if not isinstance(log_dir, str) or not log_dir:
        raise ValueError("配置项 logging.log_dir 缺失或不是有效字符串")
    if not isinstance(log_file, str) or not log_file:
        raise ValueError("配置项 logging.log_file 缺失或不是有效字符串")

    resolved_log_dir = resolve_config_path_value(config_path, log_dir)
    return os.path.join(resolved_log_dir, log_file)


def discover_sweep_log_paths(config_path: str, config: Dict[str, Any]) -> List[str]:
    """扫描 exp_root 下的 run_*/logs 目录，发现可用的 sweep 日志文件。"""
    exp_root = config.get("exp_root")
    logging_cfg = config.get("logging", {})
    if not isinstance(exp_root, str) or not exp_root.strip():
        return []
    if not isinstance(logging_cfg, dict):
        return []

    log_file = logging_cfg.get("log_file")
    if not isinstance(log_file, str) or not log_file:
        return []

    exp_root_path = resolve_config_path_value(config_path, exp_root)
    if not os.path.isdir(exp_root_path):
        return []

    discovered_paths: List[str] = []
    for child_name in sorted(os.listdir(exp_root_path)):
        if not child_name.startswith("run_"):
            continue
        candidate = os.path.join(exp_root_path, child_name, "logs", log_file)
        if os.path.exists(candidate):
            discovered_paths.append(candidate)

    discovered_paths.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return discovered_paths


def render_log_selector(primary_log_path: str, discovered_log_paths: List[str]) -> str:
    """渲染日志选择器，支持在基础训练日志和 sweep 日志之间切换。"""
    options: List[str] = []
    if primary_log_path not in options:
        options.append(primary_log_path)
    for path in discovered_log_paths:
        if path not in options:
            options.append(path)

    if "selected_log_path" not in st.session_state:
        default_path = primary_log_path if os.path.exists(primary_log_path) else (discovered_log_paths[0] if discovered_log_paths else primary_log_path)
        st.session_state["selected_log_path"] = default_path

    if options:
        if st.session_state["selected_log_path"] not in options:
            st.session_state["selected_log_path"] = options[0]
        selected = st.sidebar.selectbox(
            "日志文件",
            options=options,
            index=options.index(st.session_state["selected_log_path"]),
            help="优先读取基础训练日志；若不存在，可切换到 sweep 输出目录下的 run_*/logs/train_log.jsonl。",
        )
        st.session_state["selected_log_path"] = selected
        return selected

    return primary_log_path


@st.cache_data(show_spinner=False)
def load_data(log_path: str) -> Tuple[pd.DataFrame, int]:
    """读取 JSONL 日志并转换为 DataFrame，同时统计被跳过的坏行数量。"""
    records: List[Dict[str, Any]] = []
    skipped_lines = 0

    with open(log_path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_lines += 1
                continue

            if not isinstance(item, dict):
                skipped_lines += 1
                continue

            records.append(item)

    df = pd.DataFrame(records)
    if df.empty:
        return df, skipped_lines

    if "kimg" not in df.columns:
        raise ValueError("日志文件缺少必要字段 kimg")

    numeric_columns = ["kimg"] + [metric for metric in ALL_METRICS if metric in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["kimg"]).sort_values("kimg").reset_index(drop=True)
    return df, skipped_lines


def build_metric_figure(df: pd.DataFrame, metrics: List[str], title: str, log_y: bool = False) -> go.Figure:
    """基于指标列表构建 Plotly 折线图。"""
    figure = go.Figure()
    available_metrics = [metric for metric in metrics if metric in df.columns]

    for metric in available_metrics:
        metric_df = df[["kimg", metric]].dropna()
        figure.add_trace(
            go.Scatter(
                x=metric_df["kimg"],
                y=metric_df[metric],
                mode="lines",
                name=metric,
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title="kimg",
        yaxis_title="value",
        template="plotly_white",
        hovermode="x unified",
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=40, r=20, t=80, b=40),
    )

    if log_y:
        figure.update_yaxes(type="log")

    return figure


def render_metric_tab(df: pd.DataFrame, metrics: List[str], title: str, log_y: bool = False) -> None:
    """渲染单个标签页的图表，并对空数据或缺字段做友好提示。"""
    available_metrics = [metric for metric in metrics if metric in df.columns]
    if df.empty:
        st.warning("当前日志文件为空，暂无可展示的数据。")
        return
    if not available_metrics:
        st.warning("当前日志文件中没有找到该视图所需的指标字段。")
        return

    figure = build_metric_figure(df, available_metrics, title=title, log_y=log_y)
    st.plotly_chart(figure, use_container_width=True)


def render_overview_tab(df: pd.DataFrame) -> None:
    """渲染综合主控台视图，支持指标多选和对数坐标。"""
    selected_metrics = st.multiselect(
        "选择要展示的损失指标",
        options=LOSS_METRICS,
        default=["loss_G", "loss_D"],
    )
    use_log_scale = st.checkbox("启用对数 Y 轴 (Log Scale)", value=False)

    if not selected_metrics:
        st.info("请至少选择一个损失指标后再绘图。")
        return

    render_metric_tab(df, selected_metrics, title="Overview: Loss Metrics vs kimg", log_y=use_log_scale)


def render_sidebar(default_config_path: str) -> str:
    """渲染侧边栏，并允许用户通过按钮切换当前使用的配置文件。"""
    st.sidebar.header("配置面板")

    if "active_config_path" not in st.session_state:
        st.session_state["active_config_path"] = default_config_path

    input_config_path = st.sidebar.text_input(
        "配置文件路径",
        value=st.session_state["active_config_path"],
        help="默认读取 configs/example.yaml，可手动修改后点击加载配置。",
    )

    if st.sidebar.button("加载配置", use_container_width=True):
        st.session_state["active_config_path"] = input_config_path.strip() or default_config_path
        st.cache_data.clear()

    st.sidebar.divider()
    st.sidebar.subheader("刷新设置")

    if "auto_refresh_enabled" not in st.session_state:
        st.session_state["auto_refresh_enabled"] = True
    if "auto_refresh_seconds" not in st.session_state:
        st.session_state["auto_refresh_seconds"] = 10

    auto_refresh_enabled = st.sidebar.checkbox(
        "启用自动刷新",
        value=bool(st.session_state["auto_refresh_enabled"]),
        help="启用后会按固定间隔自动重新加载配置和日志。",
    )
    auto_refresh_seconds = st.sidebar.number_input(
        "自动刷新间隔（秒）",
        min_value=1,
        max_value=3600,
        value=int(st.session_state["auto_refresh_seconds"]),
        step=1,
    )

    st.session_state["auto_refresh_enabled"] = auto_refresh_enabled
    st.session_state["auto_refresh_seconds"] = int(auto_refresh_seconds)

    st.sidebar.caption(f"当前配置: {st.session_state['active_config_path']}")
    if auto_refresh_enabled:
        st.sidebar.caption(f"自动刷新间隔: {int(auto_refresh_seconds)} 秒")
    return st.session_state["active_config_path"]


def main() -> None:
    """Streamlit 入口函数。"""
    args = parse_args()

    st.set_page_config(page_title="CycleGAN 训练日志看板", layout="wide")
    st.title("CycleGAN 潜空间训练交互式数据看板")
    st.caption("使用 Streamlit + Plotly 对训练 JSONL 日志进行本地交互式可视化")

    active_config_path = render_sidebar(args.config)

    # 自动刷新模式下，周期性触发整页重跑，确保能看到最新训练日志。
    if st.session_state.get("auto_refresh_enabled", True):
        st_autorefresh(
            interval=int(st.session_state.get("auto_refresh_seconds", 10)) * 1000,
            key="visualize-logs-autorefresh",
        )

    # 顶部刷新按钮用于主动清理缓存并重新读取最新日志内容。
    if st.button("刷新日志", type="primary"):
        st.cache_data.clear()
        st.rerun()

    config_path = os.path.abspath(active_config_path)
    st.write(f"当前配置文件: {config_path}")

    if not os.path.exists(config_path):
        st.error(f"配置文件不存在: {config_path}")
        return

    try:
        config = load_yaml_config(config_path)
        primary_log_path = resolve_log_path(config_path, config)
        discovered_log_paths = discover_sweep_log_paths(config_path, config)
    except Exception as exc:
        st.error(f"加载配置失败: {exc}")
        return

    log_path = render_log_selector(primary_log_path, discovered_log_paths)

    st.write(f"目标日志文件: {log_path}")
    if not os.path.exists(log_path):
        if discovered_log_paths:
            st.warning("基础日志文件不存在，但已经发现 sweep 运行目录。请在侧边栏切换到已有的 run_*/logs/train_log.jsonl。")
        else:
            st.warning(
                "日志文件暂时不存在。"
                "如果你刚启动训练，这通常表示还没到第一次写日志的步数；"
                "如果你在跑 sweep，请先确认对应 run_*/logs 目录下是否已经生成 train_log.jsonl。"
            )
        return

    try:
        df, skipped_lines = load_data(log_path)
    except Exception as exc:
        st.error(f"读取日志失败: {exc}")
        return

    if skipped_lines > 0:
        st.warning(f"检测到 {skipped_lines} 行损坏或不合法 JSON，已自动跳过。")

    if not df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("日志条数", f"{len(df)}")
        col2.metric("最新 kimg", f"{df['kimg'].iloc[-1]:.3f}")
        col3.metric("可用指标数", f"{sum(metric in df.columns for metric in ALL_METRICS)}")

    overview_tab, g_tab, d_tab, penalty_tab, lr_tab = st.tabs(
        ["综合主控台 (Overview)", "生成器视图 (G-View)", "判别器视图 (D-View)", "约束惩罚视图 (Penalty-View)", "学习率监控 (LR-View)"]
    )

    with overview_tab:
        render_overview_tab(df)

    with g_tab:
        render_metric_tab(df, ["loss_G", "loss_gan_G", "loss_gan_F"], title="G-View: Generator Metrics vs kimg")

    with d_tab:
        render_metric_tab(df, ["loss_D", "loss_D_A", "loss_D_B"], title="D-View: Discriminator Metrics vs kimg")

    with penalty_tab:
        render_metric_tab(df, ["loss_cyc", "loss_id"], title="Penalty-View: Constraint Metrics vs kimg")

    with lr_tab:
        render_metric_tab(df, ["lr_G", "lr_D"], title="LR-View: Learning Rate vs kimg")


if __name__ == "__main__":
    main()