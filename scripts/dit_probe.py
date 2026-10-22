"""DiT 単独の段階的 GPU 診断の子プロセス。LM / Text Encoder / VAE は初期化せず、楽曲生成もしない。

    python -I scripts/dit_probe.py --step b1 --out <json>   # import のみ（torch / transformers / acestep.api_server まで）
    python -I scripts/dit_probe.py --step b2 --out <json>   # import（acestep.handler まで）+ CUDA コンテキストの作成
    python -I scripts/dit_probe.py --step cd --out <json>   # C: DiT の GPU 直接ロード / D: INT8 適用（C が安全な場合のみ）

通常は scripts/dit_diag.py（事前確認・所有プロセス管理つき）から起動される。単独で直接実行しないこと。

安全:
- 自己監視スレッド（0.25 秒）が、空き RAM が下限（3GB）を 2 回連続で下回ったら**自プロセスを終了**する
  （他プロセスには触れない）。終了前に途中までの結果を保存する
- 結果は各マイルストーンで原子的に保存（途中停止しても残る）
- モデルは未量子化のまま成功扱いにしない。INT8 の検証に失敗したら status は success にならない
"""
import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import psutil  # noqa: E402

import lunaria_profiles as lp  # noqa: E402

P = psutil.Process()
MB = 1048576
GB = 1024**3
FLOOR_GB = lp.SAFETY_FLOOR_GB
MODELS = ROOT / "models"
DIT_DIR = "acestep-v15-turbo"
QUANT = "int8_weight_only"


def mem():
    mi = P.memory_info()
    return {"rss_mb": round(mi.rss / MB), "private_mb": round(getattr(mi, "private", 0) / MB)}


def avail_gb():
    return psutil.virtual_memory().available / GB


def commit_headroom_gb():
    """システムの commit 余裕（GlobalMemoryStatusEx: 上限 − 使用）。物理 RAM に余裕があっても commit は尽きうる。"""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / GB


def settle(sec=1.0):
    gc.collect()
    time.sleep(sec)


BASE0 = {}  # プロセス開始時点の空き RAM / RSS / private（物理消費を「開始時からの変化」で測るため）


def phys_since_start():
    """プロセス開始からの物理 RAM 消費。available の低下量と RSS の増加の大きい方を採る（保守的）。

    private（commit 量）は numpy の OpenBLAS 等が予約するだけで触れていない領域を含み、物理消費を過大評価しうる。
    他プロセスへの影響は物理の空き（available）で決まるため、判定にはこちらを使い、private は commit の判定に使う。
    """
    m = mem()
    drop = BASE0["avail_gb"] - avail_gb()
    rss = (m["rss_mb"] - BASE0["rss_mb"]) / 1024
    return {"avail_drop_gb": round(drop, 3), "rss_delta_gb": round(rss, 3), "phys_gb": round(max(drop, rss), 3),
            "private_delta_gb": round((m["private_mb"] - BASE0["private_mb"]) / 1024, 3)}


def atomic_write(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


class SelfMonitor(threading.Thread):
    """プロセス private と空き RAM を 0.25 秒間隔で採取する軽量スレッド（系列は 1 秒に 1 点だけ保持）。

    空き RAM が floor_gb を consecutive 回連続で下回ったら on_abort を呼ぶ（既定: 結果を保存して os._exit(3)）。
    """

    def __init__(self, floor_gb=FLOOR_GB, consecutive=2, avail_fn=avail_gb, private_fn=None, rss_fn=None,
                 on_abort=None, interval=0.25, commit_fn=commit_headroom_gb, commit_floor_gb=FLOOR_GB):
        super().__init__(daemon=True)
        self.floor_gb, self.consecutive, self.interval = floor_gb, consecutive, interval
        self.commit_fn, self.commit_floor_gb = commit_fn, commit_floor_gb
        self.min_commit = 1e9
        self._clow = 0
        self.avail_fn = avail_fn
        self.private_fn = private_fn or (lambda: getattr(P.memory_info(), "private", 0))
        self.rss_fn = rss_fn or (lambda: P.memory_info().rss)
        self.on_abort = on_abort
        self.stop_flag = False
        self.max_private = 0
        self.max_rss = 0
        self.min_avail = 1e9
        self.series = []
        self._low = 0
        self._ticks = 0
        self.aborted = False
        self.step = "init"

    def tick(self):
        a = self.avail_fn()
        c = self.commit_fn()
        pv = self.private_fn()
        rss = self.rss_fn()
        self.max_private = max(self.max_private, pv)
        self.max_rss = max(self.max_rss, rss)
        self.min_avail = min(self.min_avail, a)
        self.min_commit = min(self.min_commit, c)
        self._ticks += 1
        if self._ticks % max(1, round(1.0 / self.interval)) == 0:
            self.series.append({"t": round(time.time(), 1), "step": self.step, "private_mb": round(pv / MB),
                                "rss_mb": round(rss / MB), "avail_gb": round(a, 2), "commit_headroom_gb": round(c, 2)})
        self._low = self._low + 1 if a < self.floor_gb else 0
        self._clow = self._clow + 1 if c < self.commit_floor_gb else 0
        if not self.aborted:
            reason = None
            if self._low >= self.consecutive:
                reason = f"空き RAM {a:.2f}GB が下限 {self.floor_gb}GB を下回った（{self.step}）"
            elif self._clow >= self.consecutive:
                reason = f"commit 余裕 {c:.2f}GB が下限 {self.commit_floor_gb}GB を下回った（{self.step}）"
            if reason:
                self.aborted = True
                if self.on_abort:
                    self.on_abort(reason)
        return a

    def run(self):
        while not self.stop_flag:
            self.tick()
            time.sleep(self.interval)

    def summary(self):
        return {"max_private_mb": round(self.max_private / MB), "max_rss_mb": round(self.max_rss / MB),
                "min_avail_gb": None if self.min_avail == 1e9 else round(self.min_avail, 2),
                "min_commit_headroom_gb": None if self.min_commit == 1e9 else round(self.min_commit, 2),
                "aborted": self.aborted, "series_points": len(self.series),
                "series": self.series[-300:]}


def cuda_mem(torch):
    free_b, total_b = torch.cuda.mem_get_info(0)
    return {"allocated_mb": round(torch.cuda.memory_allocated(0) / MB),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / MB),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / MB),
            "device_free_mb": round(free_b / MB), "device_total_mb": round(total_b / MB),
            "device_used_mb": round((total_b - free_b) / MB)}


def param_summary(model, bw):
    """重みの dtype / デバイス別の個数とバイト数（量子化テンソルは保持している整数データのバイト数）。"""
    cls = bw.aqt_class()
    out = {"n_params": 0, "numel": 0, "by_kind": {}, "devices": {}, "bytes_total": 0, "bytes_exact": True}
    for _, p in model.named_parameters():
        is_q = cls is not None and (isinstance(p.data, cls) or isinstance(p, cls))
        if is_q:
            b, exact = bw._quantized_bytes(p.data)
            kind = "int8(AffineQuantizedTensor)"
            out["bytes_exact"] = out["bytes_exact"] and exact
        else:
            b, kind = p.numel() * p.element_size(), str(p.dtype).replace("torch.", "")
        out["n_params"] += 1
        out["numel"] += p.numel()
        k = out["by_kind"].setdefault(kind, {"count": 0, "bytes": 0})
        k["count"] += 1
        k["bytes"] += b
        out["bytes_total"] += b
        out["devices"][p.device.type] = out["devices"].get(p.device.type, 0) + 1
    out["bytes_total_mb"] = round(out["bytes_total"] / MB)
    for k in out["by_kind"].values():
        k["mb"] = round(k["bytes"] / MB)
    return out


def import_stage(level):
    """level: full = api_server まで（B1） / handler = acestep.handler まで（B2・C）。"""
    import warnings
    warnings.simplefilter("ignore")
    res = {"start": mem()}
    import torch  # noqa: F401
    res["after_torch"] = mem()
    import accelerate  # noqa: F401
    import diffusers  # noqa: F401
    import transformers  # noqa: F401
    res["after_transformers"] = mem()
    if level == "full":
        import acestep.api_server  # noqa: F401
        import acestep.handler  # noqa: F401
        import acestep.llm_inference  # noqa: F401
    else:
        import acestep.handler  # noqa: F401
    settle()
    res["after_import"] = mem()
    res["import_private_mb"] = res["after_import"]["private_mb"] - res["start"]["private_mb"]
    res["avail_after_import_gb"] = round(avail_gb(), 2)
    return res


def cuda_init_stage():
    import torch
    t0 = time.time()
    before = mem()
    torch.cuda.init()
    x = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    del x
    settle()
    after = mem()
    return {"before": before, "after": after, "ctx_host_private_mb": after["private_mb"] - before["private_mb"],
            "seconds": round(time.time() - t0, 1), "cuda": cuda_mem(torch),
            "device_name": torch.cuda.get_device_name(0)}


def run(step, out):
    BASE0.update({"avail_gb": avail_gb(), **{k: v for k, v in mem().items()}})
    res = {"step": step, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
           "generation": "NOT RUN", "lm_te_vae": "NOT INITIALIZED",
           "process_start": {"avail_gb": round(BASE0["avail_gb"], 2), "rss_mb": BASE0["rss_mb"],
                             "private_mb": BASE0["private_mb"]}}
    holder = {"res": res}

    def save():
        atomic_write(out, holder["res"])

    def on_abort(msg):
        holder["res"]["status"] = "aborted_memory_floor"
        holder["res"]["abort_reason"] = msg
        holder["res"]["sampler"] = mon.summary()
        save()
        os._exit(3)  # 自プロセスのみ終了（他プロセスには触れない）

    mon = SelfMonitor(on_abort=on_abort)
    mon.start()
    save()
    try:
        if step == "b1":
            mon.step = "import_full"
            res["import"] = import_stage("full")
            res["phys_after_import"] = phys_since_start()
            res["status"] = "ok"
        elif step == "b2":
            mon.step = "import_handler"
            res["import"] = import_stage("handler")
            res["phys_after_import"] = phys_since_start()
            mon.step = "cuda_init"
            res["cuda_init"] = cuda_init_stage()
            res["private_after_cuda_gb"] = round(res["cuda_init"]["after"]["private_mb"] / 1024, 3)
            res["phys_after_cuda"] = phys_since_start()
            res["status"] = "ok"
        elif step == "cd":
            step_cd(res, mon, save)
        res["sampler"] = mon.summary()
        res["final"] = mem()
    except BaseException as e:  # 例外でも成功扱いにしない
        res["status"] = "error"
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["sampler"] = mon.summary()
    finally:
        mon.stop_flag = True
        save()
    return 0 if res.get("status") == "ok" else 1


def step_cd(res, mon, save):
    mon.step = "import_handler"
    res["import"] = import_stage("handler")
    import torch
    mon.step = "cuda_init"
    res["cuda_init"] = cuda_init_stage()
    res["private_after_cuda_gb"] = round(res["cuda_init"]["after"]["private_mb"] / 1024, 3)
    res["phys_after_cuda"] = phys_since_start()
    save()

    import bprime_wrapper as bw
    from acestep.handler import AceStepHandler
    disable_warmup = os.environ.get("LUNARIA_DISABLE_WARMUP") == "1"  # 既定オフ。dit_diag.py --disable-warmup で明示した時のみ
    bw.apply_low_peak_patches(components=("dit",), disable_warmup=disable_warmup)  # DiT の GPU 直接ロード + RAM ガード
    res["patches"] = list(bw.LOWPEAK_APPLIED)
    res["disable_warmup"] = disable_warmup
    h = AceStepHandler()
    h.device = "cuda"
    h.offload_to_cpu = True       # tier3 Auto と同じ（<16GB）。DiT は offload_dit_to_cpu=False のため GPU 常駐
    h.offload_dit_to_cpu = False
    h.dtype = torch.bfloat16
    h.compiled = False
    h.quantization = None

    # ---- Step C: DiT の GPU 直接ロード（未量子化。量子化は Step D で別に適用して前後を計測する）----
    mon.step = "C_dit_load"
    torch.cuda.reset_peak_memory_stats(0)
    pre = {"mem": mem(), "avail_gb": round(avail_gb(), 2), "cuda": cuda_mem(torch)}
    res["C_pre"] = pre
    save()
    t0 = time.time()
    h._load_main_model_from_checkpoint(model_checkpoint_path=str(MODELS / DIT_DIR), device="cuda",
                                       use_flash_attention=True, compile_model=False, quantization=None)
    torch.cuda.synchronize()
    c_seconds = round(time.time() - t0, 1)
    settle()
    post = {"mem": mem(), "avail_gb": round(avail_gb(), 2), "cuda": cuda_mem(torch),
            "seconds": c_seconds, "params": param_summary(h.model, bw),
            "private_delta_mb": mem()["private_mb"] - pre["mem"]["private_mb"],
            "rss_delta_mb": mem()["rss_mb"] - pre["mem"]["rss_mb"],
            "avail_drop_gb": round(pre["avail_gb"] - avail_gb(), 3),
            "phys_since_start": phys_since_start(),
            "sampler_max_private_mb": round(mon.max_private / MB), "sampler_max_rss_mb": round(mon.max_rss / MB),
            "sampler_min_avail_gb": round(mon.min_avail, 2),
            "load_events": list(bw.LOAD_EVENTS)}
    res["C_post"] = post
    res["C_ok"] = bool(post["params"]["devices"] == {"cuda": post["params"]["n_params"]})
    res["status"] = "C_done"
    save()

    # ---- Step D の実行条件: RAM・VRAM に余裕がある場合のみ ----
    cuda_free_mb = post["cuda"]["device_free_mb"]
    ram_ok = avail_gb() >= FLOOR_GB + 0.5
    vram_ok = cuda_free_mb >= 1024  # 量子化は層ごとに INT8 を作って bf16 を解放する。一時領域として 1GiB 以上の空きを要求
    res["D_gate"] = {"ram_ok": ram_ok, "vram_ok": vram_ok, "avail_gb": round(avail_gb(), 2),
                     "device_free_mb": cuda_free_mb}
    if not (res["C_ok"] and ram_ok and vram_ok):
        res["status"] = "C_done_D_skipped"
        return

    # ---- Step D: INT8 適用（失敗しても未量子化のまま成功扱いにしない）----
    mon.step = "D_int8"
    torch.cuda.reset_peak_memory_stats(0)
    h.quantization = QUANT
    d_pre = {"mem": mem(), "cuda": cuda_mem(torch), "avail_gb": round(avail_gb(), 2)}
    t0 = time.time()
    try:
        h._apply_dit_quantization(QUANT)
        torch.cuda.synchronize()
        res["D_error"] = None
    except BaseException as e:
        res["D_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    d_seconds = round(time.time() - t0, 1)
    after_q = cuda_mem(torch)
    torch.cuda.empty_cache()
    settle()
    res["D"] = {"pre": d_pre, "seconds": d_seconds, "after_quantize_before_empty_cache": after_q,
                "after_empty_cache": cuda_mem(torch), "mem": mem(), "avail_gb": round(avail_gb(), 2),
                "peak_allocated_mb_during_quantize": after_q["max_allocated_mb"],
                "params": param_summary(h.model, bw)}
    ver = bw.verify_int8(h, require_device="cuda")
    res["D_verify"] = ver
    before_b = post["params"]["bytes_total"]
    after_b = res["D"]["params"]["bytes_total"]
    res["D_weight_bytes_reduction_mb"] = round((before_b - after_b) / MB)
    res["D_vram_allocated_reduction_mb"] = d_pre["cuda"]["allocated_mb"] - res["D"]["after_empty_cache"]["allocated_mb"]
    res["status"] = "ok" if (ver["ok"] and res["D_error"] is None) else "D_failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["b1", "b2", "cd"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(args.step, args.out))


if __name__ == "__main__":
    main()
