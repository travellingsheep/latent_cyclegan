import argparse
from collections import OrderedDict

import torch
def tensor_meta(tensor):
    return (
        f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, requires_grad={tensor.requires_grad})"
    )


def summarize_obj(obj, indent=0, max_depth=3, max_items=30):
    pad = "  " * indent
    if indent > max_depth:
        print(f"{pad}... (超过最大展开层数)")
        return

    if isinstance(obj, torch.Tensor):
        print(f"{pad}{tensor_meta(obj)}")
        return

    if isinstance(obj, torch.nn.Module):
        print(f"{pad}Module: {obj.__class__.__name__}")
        return

    if isinstance(obj, (dict, OrderedDict)):
        print(f"{pad}{type(obj).__name__}(len={len(obj)})")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{pad}  ... (其余 {len(obj) - max_items} 项省略)")
                break
            print(f"{pad}  key={k!r} -> ", end="")
            if isinstance(v, torch.Tensor):
                print(tensor_meta(v))
            elif isinstance(v, (dict, OrderedDict, list, tuple)):
                print()
                summarize_obj(v, indent + 2, max_depth, max_items)
            else:
                if isinstance(v, str):
                    text = v if len(v) <= 200 else (v[:200] + "...(截断)")
                    print(f"str(len={len(v)}): {text!r}")
                elif isinstance(v, (int, float, bool)) or v is None:
                    print(f"{type(v).__name__}: {v}")
                else:
                    print(f"{type(v).__name__}")
        return

    if isinstance(obj, (list, tuple)):
        print(f"{pad}{type(obj).__name__}(len={len(obj)})")
        for i, v in enumerate(obj[:max_items]):
            print(f"{pad}  [{i}] -> ", end="")
            if isinstance(v, torch.Tensor):
                print(tensor_meta(v))
            elif isinstance(v, (dict, OrderedDict, list, tuple)):
                print()
                summarize_obj(v, indent + 2, max_depth, max_items)
            elif isinstance(v, str):
                text = v if len(v) <= 200 else (v[:200] + "...(截断)")
                print(f"str(len={len(v)}): {text!r}")
            elif isinstance(v, (int, float, bool)) or v is None:
                print(f"{type(v).__name__}: {v}")
            else:
                print(f"{type(v).__name__}")
        if len(obj) > max_items:
            print(f"{pad}  ... (其余 {len(obj) - max_items} 项省略)")
        return

    if isinstance(obj, str):
        text = obj if len(obj) <= 500 else (obj[:500] + "...(截断)")
        print(f"{pad}str(len={len(obj)}): {text!r}")
        return

    if isinstance(obj, (int, float, bool)) or obj is None:
        print(f"{pad}{type(obj).__name__}: {obj}")
        return

    print(f"{pad}{type(obj).__name__}")


def main():
    parser = argparse.ArgumentParser(description="查看 .pt 文件结构（隐藏长数值内容）")
    parser.add_argument(
        "--pt",
        default="./outputs/small_id_02/new_baseline_medium_w_r1/ckpt/epoch_0001.pt",
        help=".pt 文件路径",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="最大展开层数")
    parser.add_argument("--max-items", type=int, default=30, help="每层最多展示条目数")
    args = parser.parse_args()

    obj = torch.load(args.pt, map_location="cpu")
    print(f"✅ 已加载: {args.pt}")
    print(f"顶层对象类型: {type(obj).__name__}\n")

    if isinstance(obj, (dict, OrderedDict)) and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    ):
        print("检测到 state_dict，以下仅展示参数名和张量元信息：")
        total_params = 0
        for name, param in obj.items():
            print(
                f"参数名: {name:50s} | 形状: {tuple(param.shape)} | dtype: {param.dtype}"
            )
            total_params += param.numel()
        print(f"\n模型总参数量: {total_params / 1e6:.2f}M")
    else:
        print("检测到通用对象，以下仅展示结构与文字信息（不展开长数值）：")
        summarize_obj(obj, max_depth=args.max_depth, max_items=args.max_items)


if __name__ == "__main__":
    main()