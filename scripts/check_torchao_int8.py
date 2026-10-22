"""torchao INT8 weight-only 量子化の動作確認（CPU・小型 Linear のみ。GPU・モデルは使わない）。

固定環境（torch 2.7.1+cu128 / torchao 0.16.0）で `Int8WeightOnlyConfig` が使えるか、
量子化後の出力誤差、CPU<->デバイス移動の挙動、保持バイト数の削減率を確認する。
実機の DiT で同じ結果になる保証はない（あくまで前提条件の切り分け）。

使い方: python -I scripts/check_torchao_int8.py
"""
import json
import sys
import warnings

import torch
import torch.nn as nn


def tensor_bytes(m):
    total = 0
    for p in m.parameters():
        t = p.data
        # torchao の量子化テンソルは内部テンソル群を持つ。実保持量の概算には属性を辿る
        if hasattr(t, "tensor_impl") or hasattr(t, "_data") or hasattr(t, "qdata"):
            for name in ("qdata", "_data", "int_data"):
                inner = getattr(t, name, None)
                if inner is None and hasattr(t, "tensor_impl"):
                    inner = getattr(t.tensor_impl, name, None)
                if isinstance(inner, torch.Tensor):
                    total += inner.numel() * inner.element_size()
                    break
            else:
                total += t.numel() * 1  # int8 とみなす
        else:
            total += t.numel() * t.element_size()
    return total


def main():
    res = {"torch": torch.__version__}
    warnings.simplefilter("ignore")
    try:
        import torchao
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        res["torchao"] = torchao.__version__
    except Exception as e:
        res["import_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(1)

    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 2048)).to(torch.bfloat16)
    x = torch.randn(4, 2048, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = base(x)
    res["bf16_param_mb"] = round(sum(p.numel() * p.element_size() for p in base.parameters()) / 1048576, 2)

    try:
        quantize_(base, Int8WeightOnlyConfig())
        res["quantize_ok"] = True
    except Exception as e:
        res["quantize_ok"] = False
        res["quantize_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(1)

    with torch.no_grad():
        out = base(x)
    err = (out.float() - ref.float()).abs()
    res["forward_ok"] = bool(torch.isfinite(out).all())
    res["max_abs_err"] = round(float(err.max()), 5)
    res["rel_err"] = round(float(err.mean() / ref.float().abs().mean()), 5)
    res["int8_param_mb_estimate"] = round(tensor_bytes(base) / 1048576, 2)
    res["weight_type"] = type(base[0].weight.data).__name__

    try:
        base.to("cpu")
        base.to(dtype=torch.bfloat16)  # 実装は offload 時に dtype 付き .to() を使う
        with torch.no_grad():
            out2 = base(x)
        res["to_cpu_with_dtype_ok"] = bool(torch.isfinite(out2).all())
        res["weight_type_after_to"] = type(base[0].weight.data).__name__
    except Exception as e:
        res["to_cpu_with_dtype_ok"] = False
        res["to_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0 if res.get("forward_ok") and res.get("to_cpu_with_dtype_ok") else 1)


if __name__ == "__main__":
    main()
