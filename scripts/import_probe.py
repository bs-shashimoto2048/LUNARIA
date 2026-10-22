"""D0b の子プロセス: import（と CUDA 初期化）を、トークンの順に 1 つずつ実行し、各トークンのメモリ増分を測る。

    python -I scripts/import_probe.py --spec "torch,cuda,transformers" --out <json>

トークン:
  python               何もしない（基準）
  torch / numpy / soundfile / loguru / tqdm / safetensors / transformers / accelerate / diffusers /
  torchaudio / torchao / triton / fastapi / flash_attn   そのモジュールを import
  transformers_auto    from transformers import AutoTokenizer, AutoModelForCausalLM（acestep.llm_inference と同じ形）
  acestep.gpu_config / acestep.handler / acestep.llm_inference / acestep.api_server   ACE-Step のモジュールを import
  cuda                 torch.cuda.init() + 1 要素のテンソル作成（CUDA コンテキストの作成。モデルは使わない）
  cuda_work            小さなテンソルで cuBLAS（matmul）・cuDNN（conv1d）・SDPA を 1 回ずつ実行（最初の実 GPU 処理。モデルは使わない）
  peft / lycoris / lightning / acestep.training / acestep.core.generation.handler   原因の切り分け用の import

各トークンの前後で、private / RSS / システム commit 余裕 / 空き物理 RAM / VRAM / 所要時間 / 新しく読み込まれたモジュールと DLL を記録する。
各構成は新しいプロセスで実行する（import キャッシュや前の確保が結果に影響しないように）。実モデルのロード・量子化・生成はしない。
"""
import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import psutil  # noqa: E402

import dit_probe as dp  # noqa: E402  (mem / avail_gb / commit_headroom_gb / SelfMonitor / JsonlStream / vram_used_mb)

P = psutil.Process()
MB = 1048576
STDLIB = set(sys.stdlib_module_names)


def top_level_modules():
    return {m.split(".")[0] for m in sys.modules if m and not m.startswith("_")} - STDLIB


def mapped_dlls():
    """読み込まれている DLL / pyd の {ファイル名: マップサイズ MB}（ユーザー名を含むパスは載せない）。"""
    out = {}
    try:
        for e in P.memory_maps(grouped=True):
            n = os.path.basename(e.path)
            if n.lower().endswith((".dll", ".pyd")):
                out[n] = round(e.rss / MB)
    except Exception:
        pass
    return out


def torch_groups():
    """torch 内の読み込み済みサブモジュールを、2 階層（例: torch._dynamo）で集約した集合。"""
    out = set()
    for m in list(sys.modules):
        if m.startswith("torch."):
            p = m.split(".")
            out.add(".".join(p[:2]))
    return out


def snapshot():
    m = dp.mem()
    return {"private_mb": m["private_mb"], "rss_mb": m["rss_mb"], "commit_headroom_gb": round(dp.commit_headroom_gb(), 3),
            "avail_gb": round(dp.avail_gb(), 3), "vram_used_mb": dp.vram_used_mb()}


def do_token(tok):
    """トークンを実行する。実行した内容の説明を返す。"""
    if tok == "python":
        return "no-op"
    if tok == "cuda":
        import torch
        torch.cuda.init()
        x = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del x
        return "torch.cuda.init + zeros(1)"
    if tok == "cuda_work":
        # 最初の実 GPU 処理（cuBLAS / cuDNN / SDPA の初期化）。CUDA 初期化時の削減が「処理の初回に戻る」だけでないかを確認する。
        # 小さなテンソルのみ（モデルは使わない）
        import torch
        import torch.nn.functional as F
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        b = a @ a                                                              # cuBLAS
        c = F.conv1d(torch.randn(1, 16, 4096, device="cuda"), torch.randn(16, 16, 3, device="cuda"))  # cuDNN
        q = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.bfloat16)
        o = F.scaled_dot_product_attention(q, q, q)                            # SDPA
        torch.cuda.synchronize()
        del a, b, c, q, o
        torch.cuda.empty_cache()
        return "matmul(bf16) + conv1d + sdpa（小さなテンソル）"
    if tok == "transformers_auto":
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        return "from transformers import AutoTokenizer, AutoModelForCausalLM"
    importlib.import_module(tok)
    return f"import {tok}"


def run(spec, out):
    tokens = [t for t in spec.split(",") if t]
    res = {"spec": spec, "tokens": [], "status": "running", "pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "generation": "NOT RUN", "models": "NOT LOADED"}
    stream = dp.JsonlStream(Path(out).with_suffix(".stream.jsonl"))
    dp.STREAM = stream

    def save():
        dp.atomic_write(out, res)

    def on_abort(msg):
        res["status"] = "aborted_memory_floor"
        res["abort_reason"] = msg
        save()
        os._exit(3)

    mon = dp.SelfMonitor(on_abort=on_abort, interval=0.1, stream=stream, vram_fn=dp.vram_used_mb)
    mon.start()
    res["process_start"] = {**snapshot(), "modules": len(sys.modules)}
    save()
    try:
        for tok in tokens:
            mon.step = tok
            mon.reset_window()
            before = snapshot()
            mods_before, dlls_before = top_level_modules(), mapped_dlls()
            tg_before = torch_groups()
            n_before = len(sys.modules)
            dp.event(f"{tok}:pre")
            t0 = time.perf_counter()
            what = do_token(tok)
            seconds = round(time.perf_counter() - t0, 3)
            time.sleep(0.3)  # サンプラーに直後の状態を捉えさせる
            after = snapshot()
            dp.event(f"{tok}:post")
            dlls_after = mapped_dlls()
            new_dlls = {k: v for k, v in dlls_after.items() if k not in dlls_before}
            win = mon.window_summary()
            torch = sys.modules.get("torch")
            rec = {
                "token": tok, "what": what, "seconds": seconds,
                "before": before, "after": after,
                "private_delta_mb": after["private_mb"] - before["private_mb"],
                "private_window_max_delta_mb": win["max_private_mb"] - before["private_mb"],
                "rss_delta_mb": after["rss_mb"] - before["rss_mb"],
                "system_commit_delta_gb": round(before["commit_headroom_gb"] - after["commit_headroom_gb"], 3),
                "avail_drop_gb": round(before["avail_gb"] - after["avail_gb"], 3),
                "vram_delta_mb": (None if before["vram_used_mb"] is None or after["vram_used_mb"] is None
                                  else after["vram_used_mb"] - before["vram_used_mb"]),
                "cuda_initialized_after": bool(torch is not None and torch.cuda.is_initialized()),
                "new_modules": len(sys.modules) - n_before,
                "new_top_level_packages": sorted(top_level_modules() - mods_before),
                "new_torch_groups": sorted(torch_groups() - tg_before),
                "new_dlls": dict(sorted(new_dlls.items(), key=lambda kv: -kv[1])[:12]),
                "new_dll_count": len(new_dlls), "new_dll_total_mb": sum(new_dlls.values()),
            }
            res["tokens"].append(rec)
            save()
        res["status"] = "ok"
    except BaseException as e:
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        mon.stop_flag = True
        res["sampler"] = mon.summary()
        res["totals"] = {"private_mb_from_start": snapshot()["private_mb"] - res["process_start"]["private_mb"],
                         "system_commit_gb_from_start": round(res["process_start"]["commit_headroom_gb"]
                                                              - snapshot()["commit_headroom_gb"], 3),
                         "avail_drop_gb_from_start": round(res["process_start"]["avail_gb"] - snapshot()["avail_gb"], 3),
                         "cuda_initialized_final": bool(sys.modules.get("torch") is not None
                                                        and sys.modules["torch"].cuda.is_initialized()),
                         "seconds_total": round(sum(t["seconds"] for t in res["tokens"]), 3)}
        res["events"] = list(dp.EVENTS)
        save()
        stream.close()
    return 0 if res["status"] == "ok" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.exit(run(a.spec, a.out))


if __name__ == "__main__":
    main()
