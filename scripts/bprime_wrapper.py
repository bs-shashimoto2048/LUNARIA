"""プロファイル B′（0.6B LM + INT8 DiT 常駐 + GPU VAE 優先）の起動ラッパー。

上流 ACE-Step のソースは変更しない。`AceStepHandler.initialize_service` を実行時に差し替え、
次の順序を明示する:

    1. DiT を GPU へロード（offload_to_cpu=Auto のため Text Encoder / VAE / LM は CPU 側）
    2. 上流が DiT 読み込み後に `quantization="int8_weight_only"` を適用（torchao）
    3. ラッパーが「本当に量子化されたか」を GPU 上の重みで検証
    4. 検証に失敗したら (status, False) を返す → 上流は起動を失敗させる。**黙って bf16 に戻さない**

上流は torchao の import 失敗時に警告だけで量子化なしに進む実装（init_service_orchestrator.py）なので、
その黙示フォールバックをここで検知して異常終了にする。

使い方（プロジェクト専用 venv）:
    python -I scripts/bprime_wrapper.py --serve
    python -I scripts/bprime_wrapper.py --diagnose-init --report <path>   # モデル初期化と量子化までの診断のみ

--diagnose-init はモデルの初期化・量子化・配置・VRAM 余裕の確認だけを行い、音楽生成はしない。
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import lunaria_profiles as lp  # noqa: E402

QUANT = "int8_weight_only"
VERIFY_RESULTS = []   # パッチした initialize_service が検証結果を追記する
HANDLERS = []
VAE_CPU_FALLBACK_THRESHOLD_GB = 0.5   # 上流 generate_music_decode.py の閾値（audit で固定 commit を確認済み）


# ---------------------------------------------------------------------------------------------
# 検証ロジック（モデルに依存しない純粋関数。単体テスト可能）
# ---------------------------------------------------------------------------------------------
def aqt_class():
    """torchao の AffineQuantizedTensor を返す（無ければ None）。上流と同じ 2 経路を試す。"""
    for mod, attr in (("torchao.dtypes.affine_quantized_tensor", "AffineQuantizedTensor"),
                      ("torchao.quantization.affine_quantized", "AffineQuantizedTensor")):
        try:
            m = __import__(mod, fromlist=[attr])
            return getattr(m, attr)
        except Exception:
            continue
    return None


def _quantized_bytes(t):
    """量子化テンソルが保持する整数データのバイト数。取得できなければ int8 とみなす（exact=False）。"""
    for name in ("qdata", "int_data", "_data"):
        inner = getattr(t, name, None)
        if inner is None and hasattr(t, "tensor_impl"):
            inner = getattr(t.tensor_impl, name, None)
        if hasattr(inner, "numel") and hasattr(inner, "element_size"):
            return inner.numel() * inner.element_size(), True
    return t.numel(), False


def verify_int8(handler, require_device="cuda", cls=None):
    """handler.model が実際に INT8 量子化されているかを重みから検証する。

    ok の条件: handler.quantization == QUANT、量子化テンソルが 1 つ以上、量子化テンソルがすべて require_device 上。
    """
    reasons = []
    quant_attr = getattr(handler, "quantization", None)
    if quant_attr != QUANT:
        reasons.append(f"handler.quantization={quant_attr!r}（想定 {QUANT!r}）。上流が量子化を黙って無効化した疑い")
    cls = cls or aqt_class()
    if cls is None:
        reasons.append("AffineQuantizedTensor クラスを取得できない（torchao の互換性）")
    model = getattr(handler, "model", None)
    if model is None:
        reasons.append("handler.model が無い")
    res = {"quantization_attr": quant_attr, "n_params": 0, "n_quantized": 0, "numel_total": 0, "numel_quantized": 0,
           "quantized_devices": {}, "other_devices": {}, "quantized_bytes": 0, "quantized_bytes_exact": True,
           "other_bytes": 0}
    if model is not None and cls is not None:
        for _, p in model.named_parameters():
            res["n_params"] += 1
            res["numel_total"] += p.numel()
            is_q = isinstance(p.data, cls) or isinstance(p, cls)
            dev = p.device.type
            if is_q:
                res["n_quantized"] += 1
                res["numel_quantized"] += p.numel()
                res["quantized_devices"][dev] = res["quantized_devices"].get(dev, 0) + 1
                b, exact = _quantized_bytes(p.data)
                res["quantized_bytes"] += b
                res["quantized_bytes_exact"] = res["quantized_bytes_exact"] and exact
            else:
                res["other_devices"][dev] = res["other_devices"].get(dev, 0) + 1
                res["other_bytes"] += p.numel() * p.element_size()
        if res["n_quantized"] == 0:
            reasons.append("量子化された重みが 1 つも無い")
        elif set(res["quantized_devices"]) != {require_device}:
            reasons.append(f"量子化された重みの配置が {require_device} のみではない: {res['quantized_devices']}")
    res["quantized_numel_fraction"] = (round(res["numel_quantized"] / res["numel_total"], 4)
                                       if res["numel_total"] else 0.0)
    res["reasons"] = reasons
    res["ok"] = not reasons
    return res


def module_device(m):
    try:
        return next(m.parameters()).device.type
    except (StopIteration, AttributeError, TypeError):
        return None


def placement_report(handler, llm_handler):
    llm = getattr(llm_handler, "llm", None)
    return {
        "dit(model)": module_device(getattr(handler, "model", None)),
        "text_encoder": module_device(getattr(handler, "text_encoder", None)),
        "vae": module_device(getattr(handler, "vae", None)),
        "lm": module_device(llm),
        "lm_backend": getattr(llm_handler, "llm_backend", None),
        "offload_to_cpu": getattr(handler, "offload_to_cpu", None),
        "offload_dit_to_cpu": getattr(handler, "offload_dit_to_cpu", None),
        "compiled": getattr(handler, "compiled", None),
        "dtype": str(getattr(handler, "dtype", None)),
        "quantization": getattr(handler, "quantization", None),
    }


def assert_profile_env(profile_id=lp.PROFILE_B_PRIME):
    """想定外の環境でラッパーが起動されたら止める（プロファイル外の設定の混入を防ぐ）。"""
    problems = []
    for k, v in lp.PROFILES[profile_id]["env"].items():
        if os.environ.get(k) != v:
            problems.append(f"{k}={os.environ.get(k)!r}（想定 {v!r}）")
    for k in lp.UNSET_ENV:
        if k in os.environ:
            problems.append(f"{k} が設定されている（プロファイルでは未設定が前提）")
    if problems:
        raise RuntimeError("プロファイル環境の不一致: " + "; ".join(problems))


# ---------------------------------------------------------------------------------------------
# 実行時計測
# ---------------------------------------------------------------------------------------------
def proc_mem():
    import psutil
    mi = psutil.Process(os.getpid()).memory_info()
    return {"rss_mb": round(mi.rss / 1048576), "private_mb": round(getattr(mi, "private", 0) / 1048576)}


def cuda_mem():
    import torch
    if not torch.cuda.is_available():
        return {"cuda": False}
    free_b, total_b = torch.cuda.mem_get_info(0)
    out = {"cuda": True,
           "allocated_mb": round(torch.cuda.memory_allocated(0) / 1048576),
           "reserved_mb": round(torch.cuda.memory_reserved(0) / 1048576),
           "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / 1048576),
           "device_free_mb": round(free_b / 1048576), "device_total_mb": round(total_b / 1048576)}
    try:
        from acestep.gpu_config import get_effective_free_vram_gb
        out["effective_free_gb"] = round(get_effective_free_vram_gb(0), 3)
    except Exception as e:
        out["effective_free_gb"] = f"ERR {type(e).__name__}"
    return out


def snapshot():
    return {"t": round(time.time(), 1), **proc_mem(), "gpu": cuda_mem()}


# ---------------------------------------------------------------------------------------------
# initialize_service の差し替え
# ---------------------------------------------------------------------------------------------
def apply_patch(handler_cls=None):
    """AceStepHandler.initialize_service を、INT8 適用と検証つきのものに差し替える。上流ソースは無変更。

    handler_cls はテスト用の差し替え口（通常は None = acestep.handler.AceStepHandler）。
    """
    if handler_cls is None:
        from acestep.handler import AceStepHandler as handler_cls

    AceStepHandler = handler_cls
    orig = AceStepHandler.initialize_service
    if getattr(orig, "_lunaria_bprime", False):
        return

    def patched(self, *args, **kwargs):
        if kwargs.get("offload_dit_to_cpu"):
            return "B′ は DiT の GPU 常駐が前提。offload_dit_to_cpu=True は拒否", False
        if kwargs.get("quantization") not in (None, QUANT):
            return f"B′ は quantization={QUANT!r} のみ許可。指定値: {kwargs.get('quantization')!r}", False
        kwargs["quantization"] = QUANT
        before = snapshot()
        t0 = time.time()
        status, ok = orig(self, *args, **kwargs)
        info = {"init_service_seconds": round(time.time() - t0, 1), "before": before, "after": snapshot(),
                "init_service_ok": bool(ok)}
        if not ok:
            VERIFY_RESULTS.append({"ok": False, "reasons": [f"initialize_service 失敗: {str(status)[:300]}"], **info})
            return status, False
        res = verify_int8(self, require_device="cuda")
        res.update(info)
        VERIFY_RESULTS.append(res)
        HANDLERS.append(self)
        if not res["ok"]:
            return "B′: INT8 の適用を確認できないため中止（bf16 へ黙ってフォールバックしない）: " + \
                   "; ".join(res["reasons"]), False
        return status, True

    patched._lunaria_bprime = True
    AceStepHandler.initialize_service = patched


# ---------------------------------------------------------------------------------------------
# 診断（モデル初期化と量子化まで。音楽生成はしない）
# ---------------------------------------------------------------------------------------------
def _probe_context(label, ctx_factory):
    """モデルを一時的に GPU に載せ、その間の VRAM を記録する（生成はしない）。"""
    import torch
    out = {"label": label}
    try:
        torch.cuda.reset_peak_memory_stats(0)
        out["before"] = cuda_mem()
        with ctx_factory():
            out["inside"] = cuda_mem()
        out["peak_allocated_mb_inside"] = round(torch.cuda.max_memory_allocated(0) / 1048576)
        out["after"] = cuda_mem()
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def run_probes(app):
    handler = HANDLERS[-1]
    llm_handler = app.state.llm_handler
    report = {"placement": placement_report(handler, llm_handler), "after_init": snapshot()}
    if hasattr(handler, "_empty_cache"):
        handler._empty_cache()
    # VAE 用 VRAM の余裕: 実デコードと同じく VAE を GPU に載せ、上流の閾値（実効空き 0.5GB）と比較
    vae = _probe_context("vae_to_gpu", lambda: handler._load_model_context("vae"))
    if vae.get("ok"):
        eff = vae["inside"].get("effective_free_gb")
        vae["effective_free_gb_inside"] = eff
        vae["threshold_gb"] = VAE_CPU_FALLBACK_THRESHOLD_GB
        vae["gpu_decode_expected"] = bool(isinstance(eff, (int, float)) and eff >= VAE_CPU_FALLBACK_THRESHOLD_GB)
        try:
            vae["auto_tile_chunk_size"] = handler._get_auto_decode_chunk_size()
        except Exception as e:
            vae["auto_tile_chunk_size"] = f"ERR {type(e).__name__}"
    report["vae_headroom"] = vae
    # DiT 常駐 + LM / Text Encoder を GPU に載せた時の VRAM（offload コンテキスト内のみ）
    report["lm_on_gpu"] = _probe_context("lm_to_gpu", lambda: llm_handler._load_model_context())
    report["text_encoder_on_gpu"] = _probe_context("text_encoder_to_gpu",
                                                   lambda: handler._load_model_context("text_encoder"))
    report["final"] = snapshot()
    report["final"]["placement"] = placement_report(handler, llm_handler)
    return report


def atomic_write(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def diag_thread(app, server, report_path, timeout_s):
    t0 = time.time()
    report = {"mode": "diagnose-init", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "generation": "NOT RUN"}
    try:
        while time.time() - t0 < timeout_s and not getattr(server, "started", False):
            if server.should_exit:
                break
            time.sleep(1)
        report["startup_seconds"] = round(time.time() - t0, 1)
        report["server_started"] = bool(getattr(server, "started", False))
        report["verify_int8"] = VERIFY_RESULTS[-1] if VERIFY_RESULTS else None
        report["llm_initialized"] = bool(getattr(app.state, "_llm_initialized", False))
        report["llm_init_error"] = getattr(app.state, "_llm_init_error", None)
        if report["server_started"] and VERIFY_RESULTS and VERIFY_RESULTS[-1].get("ok"):
            report.update(run_probes(app))
            report["status"] = "ok"
        else:
            report["status"] = "failed_before_probes"
    except Exception as e:
        report["status"] = "error"
        report["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        atomic_write(report_path, report)
        server.should_exit = True


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", action="store_true")
    g.add_argument("--diagnose-init", action="store_true")
    ap.add_argument("--report", default=str(ROOT / "data" / "logs" / "bprime_diag_report.json"))
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    assert_profile_env()
    apply_patch()

    import uvicorn
    import acestep.api_server as srv

    host = os.environ["ACESTEP_API_HOST"]
    port = int(os.environ["ACESTEP_API_PORT"])
    server = uvicorn.Server(uvicorn.Config(srv.app, host=host, port=port, workers=1, reload=False))
    if args.diagnose_init:
        threading.Thread(target=diag_thread, args=(srv.app, server, args.report, args.timeout), daemon=True).start()
    code = 0
    try:
        server.run()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    if args.diagnose_init and not Path(args.report).exists():
        atomic_write(args.report, {"mode": "diagnose-init", "status": "failed_startup", "exit_code": code,
                                   "generation": "NOT RUN",
                                   "verify_int8": VERIFY_RESULTS[-1] if VERIFY_RESULTS else None})
    sys.exit(code)


if __name__ == "__main__":
    main()
