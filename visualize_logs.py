import argparse
import json
import os
import re
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
D_VIEW_Y_TICK_STEP = 0.05


def resolve_config_path_value(config_path: str, path_value: str) -> str:
    """将配置中的相对路径解析为绝对路径。"""
    if os.path.isabs(path_value):
        return path_value
    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    return os.path.abspath(os.path.join(config_dir, path_value))


def natural_sort_key(text: str) -> List[Any]:
    """按自然顺序排序字符串，使 batch_size_2 排在 batch_size_16 前。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


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
    """根据配置文件动态解析单次训练的 JSONL 日志路径。"""
    logging_cfg = config.get("logging", {})
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}

    exp_root = str(config.get("exp_root", "") or "").strip()
    log_file = str(logging_cfg.get("log_file", "train_log.jsonl") or "").strip()
    if not log_file:
        raise ValueError("配置项 logging.log_file 缺失或不是有效字符串")

    log_dir_value = logging_cfg.get("log_dir")
    if isinstance(log_dir_value, str) and log_dir_value.strip():
        resolved_log_dir = resolve_config_path_value(config_path, log_dir_value)
    elif exp_root:
        resolved_exp_root = resolve_config_path_value(config_path, exp_root)
        resolved_log_dir = os.path.join(resolved_exp_root, "logs")
    else:
        resolved_log_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs", "logs"))
    return os.path.join(resolved_log_dir, log_file)


def resolve_experiment_dir(config_path: str, config: Dict[str, Any], log_path: str) -> str:
    """根据配置与日志路径推断当前实验目录。"""
    logging_cfg = config.get("logging", {})
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}

    log_dir_value = logging_cfg.get("log_dir")
    if isinstance(log_dir_value, str) and log_dir_value.strip():
        log_dir = resolve_config_path_value(config_path, log_dir_value)
    else:
        log_dir = os.path.dirname(log_path)

    if os.path.basename(log_dir) == "logs":
        return os.path.dirname(log_dir)

    exp_root = str(config.get("exp_root", "") or "").strip()
    if exp_root:
        return resolve_config_path_value(config_path, exp_root)
    return os.path.dirname(log_dir)


def resolve_visualize_scan_root(config_path: str, config: Dict[str, Any]) -> str:
    """解析 visualize 用的实验扫描根目录，优先使用 visualization.scan_root。"""
    visualization_cfg = config.get("visualization", {})
    if not isinstance(visualization_cfg, dict):
        visualization_cfg = {}

    scan_root = str(visualization_cfg.get("scan_root", "") or "").strip()
    if scan_root:
        return resolve_config_path_value(config_path, scan_root)

    exp_root = str(config.get("exp_root", "") or "").strip()
    if exp_root:
        return resolve_config_path_value(config_path, exp_root)
    return ""


def build_experiment_record(
    *,
    exp_dir: str,
    log_path: str,
    config_path: str,
    source: str,
) -> Dict[str, Any]:
    """构造统一的实验目录记录。"""
    exp_dir = os.path.abspath(exp_dir)
    log_path = os.path.abspath(log_path)
    config_path = os.path.abspath(config_path)
    return {
        "name": os.path.basename(exp_dir) or exp_dir,
        "exp_dir": exp_dir,
        "log_path": log_path,
        "config_path": config_path,
        "source": source,
        "has_log": os.path.exists(log_path),
        "has_config": os.path.exists(config_path),
    }


def list_directory_entries(directory: str) -> Tuple[List[str], List[str]]:
    """列出当前目录的直接子文件夹与文件。"""
    child_dirs: List[str] = []
    child_files: List[str] = []
    for child_name in sorted(os.listdir(directory), key=natural_sort_key):
        child_path = os.path.join(directory, child_name)
        if os.path.isdir(child_path):
            child_dirs.append(child_path)
        else:
            child_files.append(child_path)
    return child_dirs, child_files


def build_experiment_from_directory(directory: str, config_path: str, root_config: Dict[str, Any]) -> Dict[str, Any]:
    """当当前目录已经是叶子节点时，将其视为实验目录。"""
    directory = os.path.abspath(directory)
    logging_cfg = root_config.get("logging", {})
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}
    log_file = str(logging_cfg.get("log_file", "train_log.jsonl"))

    if os.path.basename(directory) == "logs":
        experiment_dir = os.path.dirname(directory)
        experiment_config_path = os.path.join(experiment_dir, "config.yaml")
        if os.path.exists(experiment_config_path):
            selected_config_path = experiment_config_path
            selected_config = load_yaml_config(selected_config_path)
            log_path = resolve_log_path(selected_config_path, selected_config)
            source = "selected_logs_dir_config"
        else:
            selected_config_path = os.path.abspath(config_path)
            log_path = os.path.join(directory, log_file)
            source = "selected_logs_dir_fallback"
        return build_experiment_record(
            exp_dir=experiment_dir,
            log_path=log_path,
            config_path=selected_config_path,
            source=source,
        )

    experiment_config_path = os.path.join(directory, "config.yaml")
    if os.path.exists(experiment_config_path):
        selected_config_path = experiment_config_path
        selected_config = load_yaml_config(selected_config_path)
        log_path = resolve_log_path(selected_config_path, selected_config)
        source = "selected_leaf_config"
    else:
        selected_config_path = os.path.abspath(config_path)
        log_path = os.path.join(directory, "logs", log_file)
        source = "selected_leaf_fallback"

    return build_experiment_record(
        exp_dir=directory,
        log_path=log_path,
        config_path=selected_config_path,
        source=source,
    )


def render_directory_navigator(scan_root_path: str, config_path: str, config: Dict[str, Any]) -> Dict[str, Any] | None:
    """逐层展示目录内容，直到选到叶子目录后再解析实验。"""
    if not scan_root_path:
        st.warning("当前配置没有可用的 visualization.scan_root。")
        return None
    if not os.path.isdir(scan_root_path):
        st.warning(f"扫描根目录不存在: {scan_root_path}")
        return None

    if "visualize_nav_root" not in st.session_state or st.session_state["visualize_nav_root"] != scan_root_path:
        st.session_state["visualize_nav_root"] = scan_root_path
        st.session_state["visualize_nav_current_dir"] = scan_root_path

    current_dir = st.session_state.get("visualize_nav_current_dir", scan_root_path)
    if not os.path.isdir(current_dir) or not os.path.abspath(current_dir).startswith(os.path.abspath(scan_root_path)):
        current_dir = scan_root_path
        st.session_state["visualize_nav_current_dir"] = current_dir

    child_dirs, child_files = list_directory_entries(current_dir)

    st.subheader("目录导航")
    st.write(f"扫描根目录: {scan_root_path}")
    st.write(f"当前目录: {current_dir}")

    parent_dir = os.path.dirname(current_dir.rstrip(os.sep))
    can_go_up = os.path.abspath(current_dir) != os.path.abspath(scan_root_path)
    if can_go_up and st.button("返回上一级", use_container_width=True):
        st.session_state["visualize_nav_current_dir"] = parent_dir
        st.rerun()

    if child_dirs:
        st.caption("当前层可进入的子文件夹")
        next_dir = st.selectbox(
            "选择一个子文件夹继续",
            options=child_dirs,
            format_func=lambda value: os.path.basename(value) or value,
            key=f"dir-select::{current_dir}",
            help="这里只展示当前层；选中后点击进入下一层。",
        )
        if st.button("进入选中文件夹", type="primary", use_container_width=True):
            st.session_state["visualize_nav_current_dir"] = next_dir
            st.rerun()

    if child_files:
        st.caption("当前层文件")
        for file_path in child_files:
            st.code(file_path, language=None)

    if child_dirs:
        return None

    st.info("当前目录下没有子文件夹，已将其视为候选实验目录。")
    return build_experiment_from_directory(current_dir, config_path, config)


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


def build_metric_figure(
    df: pd.DataFrame,
    metrics: List[str],
    title: str,
    log_y: bool = False,
    y_tick_step: float | None = None,
) -> go.Figure:
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
    elif y_tick_step is not None:
        figure.update_yaxes(dtick=y_tick_step)

    return figure


def render_metric_tab(
    df: pd.DataFrame,
    metrics: List[str],
    title: str,
    log_y: bool = False,
    y_tick_step: float | None = None,
) -> None:
    """渲染单个标签页的图表，并对空数据或缺字段做友好提示。"""
    available_metrics = [metric for metric in metrics if metric in df.columns]
    if df.empty:
        st.warning("当前日志文件为空，暂无可展示的数据。")
        return
    if not available_metrics:
        st.warning("当前日志文件中没有找到该视图所需的指标字段。")
        return

    figure = build_metric_figure(df, available_metrics, title=title, log_y=log_y, y_tick_step=y_tick_step)
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
        st.session_state.pop("visualize_nav_root", None)
        st.session_state.pop("visualize_nav_current_dir", None)
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


def render_summary_metrics(df: pd.DataFrame) -> None:
    """渲染日志摘要指标。"""
    if df.empty:
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("日志条数", f"{len(df)}")
    col2.metric("最新 kimg", f"{df['kimg'].iloc[-1]:.3f}")
    col3.metric("可用指标数", f"{sum(metric in df.columns for metric in ALL_METRICS)}")


def render_single_log_dashboard(df: pd.DataFrame) -> None:
    """渲染单个日志的 tab 视图。"""
    overview_tab, g_tab, d_tab, penalty_tab, lr_tab = st.tabs(
        ["综合主控台 (Overview)", "生成器视图 (G-View)", "判别器视图 (D-View)", "约束惩罚视图 (Penalty-View)", "学习率监控 (LR-View)"]
    )

    with overview_tab:
        render_overview_tab(df)

    with g_tab:
        render_metric_tab(df, ["loss_G", "loss_gan_G", "loss_gan_F"], title="G-View: Generator Metrics vs kimg")

    with d_tab:
        render_metric_tab(
            df,
            ["loss_D", "loss_D_A", "loss_D_B"],
            title="D-View: Discriminator Metrics vs kimg",
            y_tick_step=D_VIEW_Y_TICK_STEP,
        )

    with penalty_tab:
        render_metric_tab(df, ["loss_cyc", "loss_id"], title="Penalty-View: Constraint Metrics vs kimg")

    with lr_tab:
        render_metric_tab(df, ["lr_G", "lr_D"], title="LR-View: Learning Rate vs kimg")


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
        scan_root_path = resolve_visualize_scan_root(config_path, config)
    except Exception as exc:
        st.error(f"加载配置失败: {exc}")
        return

    selected_experiment = render_directory_navigator(scan_root_path, config_path, config)
    if selected_experiment is None:
        return

    st.divider()
    st.subheader("当前选中的实验")
    st.write(f"实验目录: {selected_experiment['exp_dir']}")
    st.write(f"实验配置: {selected_experiment['config_path']}")
    log_path = selected_experiment["log_path"]
    st.write(f"目标日志文件: {log_path}")

    if not os.path.exists(log_path):
        st.warning(
            "当前实验目录已检测到，但日志文件暂时不存在。"
            "如果你刚启动训练，这通常表示还没到第一次写日志的步数；"
            "如果这是 sweep 子实验，请确认该目录下的 logs 目录已经生成 train_log.jsonl。"
        )
        return

    try:
        df, skipped_lines = load_data(log_path)
    except Exception as exc:
        st.error(f"读取日志失败: {exc}")
        return

    if skipped_lines > 0:
        st.warning(f"检测到 {skipped_lines} 行损坏或不合法 JSON，已自动跳过。")

    render_summary_metrics(df)
    render_single_log_dashboard(df)


if __name__ == "__main__":
    main()
