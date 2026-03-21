import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _find_state_dicts(obj: Any) -> List[Tuple[str, StateDict]]:
    results: List[Tuple[str, StateDict]] = []

    if _is_state_dict(obj):
        results.append(("root", obj))
        return results

    if isinstance(obj, (dict, OrderedDict)):
        for key, value in obj.items():
            if isinstance(key, str) and _is_state_dict(value):
                results.append((key, value))

    return results


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _count_params(state_dict: StateDict) -> Tuple[int, int]:
    total_params = 0
    trainable_params = 0
    for tensor in state_dict.values():
        n = tensor.numel()
        total_params += n
        if bool(getattr(tensor, "requires_grad", False)):
            trainable_params += n
    return total_params, trainable_params


def _count_bytes(state_dict: StateDict) -> int:
    return sum(_tensor_bytes(tensor) for tensor in state_dict.values())


def _format_params(num_params: int) -> str:
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.3f}K"
    return str(num_params)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def _top_tensors(state_dict: StateDict, limit: int) -> Iterable[Tuple[str, torch.Tensor]]:
    return sorted(
        state_dict.items(),
        key=lambda item: item[1].numel(),
        reverse=True,
    )[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 .pt / .pth 文件中的模型参数量和参数占用大小"
    )
    parser.add_argument("pt", type=str, help=".pt / .pth 文件路径")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="每个 state_dict 额外展示参数量最大的前 K 个张量，设为 0 可关闭",
    )
    args = parser.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"文件不存在: {pt_path}")

    obj = _torch_load(pt_path)
    state_dicts = _find_state_dicts(obj)
    if not state_dicts:
        raise ValueError(
            "未在该文件中找到可识别的 state_dict。支持单个 state_dict，或包含 G/F/D_A/D_B 等键的 checkpoint。"
        )

    print(f"文件: {pt_path}")
    print(f"顶层对象类型: {type(obj).__name__}")
    print(f"检测到 {len(state_dicts)} 个 state_dict\n")

    grand_total_params = 0
    grand_total_trainable = 0
    grand_total_bytes = 0

    for name, state_dict in state_dicts:
        total_params, trainable_params = _count_params(state_dict)
        total_bytes = _count_bytes(state_dict)

        grand_total_params += total_params
        grand_total_trainable += trainable_params
        grand_total_bytes += total_bytes

        print(f"[{name}]")
        print(f"参数张量数: {len(state_dict)}")
        print(f"总参数量: {total_params} ({_format_params(total_params)})")
        print(f"可训练参数量: {trainable_params} ({_format_params(trainable_params)})")
        print(f"参数占用: {_format_bytes(total_bytes)}")

        if args.topk > 0:
            print(f"Top {args.topk} 参数张量:")
            for tensor_name, tensor in _top_tensors(state_dict, args.topk):
                print(
                    "  "
                    f"{tensor_name}: shape={tuple(tensor.shape)}, "
                    f"params={tensor.numel()} ({_format_params(tensor.numel())}), "
                    f"dtype={tensor.dtype}, size={_format_bytes(_tensor_bytes(tensor))}"
                )
        print()

    if len(state_dicts) > 1:
        print("[汇总]")
        print(f"总参数量: {grand_total_params} ({_format_params(grand_total_params)})")
        print(
            f"总可训练参数量: {grand_total_trainable} "
            f"({_format_params(grand_total_trainable)})"
        )
        print(f"总参数占用: {_format_bytes(grand_total_bytes)}")


if __name__ == "__main__":
    main()
