import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import torch


StateDict = Dict[str, torch.Tensor]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _is_state_dict(obj: Any) -> bool:
    return isinstance(obj, (dict, OrderedDict)) and len(obj) > 0 and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    )


def _extract_core_model(obj: Any) -> Dict[str, StateDict]:
    if _is_state_dict(obj):
        return {"model": dict(obj)}

    if not isinstance(obj, (dict, OrderedDict)):
        raise ValueError(f"不支持的 .pt 顶层类型: {type(obj)}")

    extracted: Dict[str, StateDict] = {}
    for key, value in obj.items():
        if isinstance(key, str) and _is_state_dict(value):
            extracted[key] = dict(value)

    if not extracted:
        raise ValueError(
            "未找到可提取的网络权重。期望输入是单个 state_dict，或包含 G/F/D_A/D_B 等权重字段的 checkpoint。"
        )

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从训练保存的 .pt 文件中提取网络权重，去掉优化器等非核心内容"
    )
    parser.add_argument("pt", type=str, help="输入 .pt / .pth 文件路径")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出文件路径；默认保存在输入文件同目录下，命名为 extracted_model.pt",
    )
    args = parser.parse_args()

    in_path = Path(args.pt).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    if args.out.strip():
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_path = in_path.with_name("extracted_model.pt")

    obj = _torch_load(in_path)
    extracted = _extract_core_model(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, str(out_path))

    print(f"输入文件: {in_path}")
    print(f"输出文件: {out_path}")
    print(f"提取到 {len(extracted)} 个网络权重部分: {', '.join(extracted.keys())}")


if __name__ == "__main__":
    main()
