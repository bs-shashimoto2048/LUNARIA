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


import ctypes  # noqa: E402


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


_GMS_LOCK = threading.Lock()
_GMS = _MEMORYSTATUSEX()
_GMS.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)


def _global_memory_status():
    """(空き物理 RAM, commit 余裕) のバイト数。GlobalMemoryStatusEx 1 回（約 2µs）。

    psutil.virtual_memory() はこの環境で約 20ms かかり、0.1 秒間隔のサンプラーには重すぎるため使わない
    （psutil の available も Windows では ullAvailPhys と同じ値）。
    """
    with _GMS_LOCK:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(_GMS))
        return _GMS.ullAvailPhys, _GMS.ullAvailPageFile


def avail_gb():
    return _global_memory_status()[0] / GB


def commit_headroom_gb():
    """システムの commit 余裕（GlobalMemoryStatusEx: 上限 − 使用）。物理 RAM に余裕があっても commit は尽きうる。"""
    return _global_memory_status()[1] / GB


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


class JsonlStream:
    """1 行ごとに flush する JSONL 書き出し。プロセスがクラッシュしても、そこまでの系列とイベントが残る。"""

    def __init__(self, path):
        self.path = Path(path)
        self._f = None
        self._lock = threading.Lock()

    def write(self, row):
        with self._lock:
            if self._f is None:
                self._f = open(self.path, "a", encoding="utf-8")
            self._f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._f.flush()

    def close(self):
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


STREAM = None  # 実行時に JsonlStream を設定する（event() とサンプラーが同じストリームへ書く）
EVENTS = []


def event(name, **kw):
    """工程の区切りを、その瞬間の private / commit 余裕とともに記録する（高頻度サンプルの補助）。"""
    row = {"event": name, "t": round(time.time(), 3), "private_mb": mem()["private_mb"],
           "commit_headroom_gb": round(commit_headroom_gb(), 3), **kw}
    EVENTS.append(row)
    if STREAM is not None:
        STREAM.write(row)
    return row


class _SafeOpenProxy:
    """safetensors.safe_open の戻り値の薄いプロキシ。開閉の瞬間を記録する（挙動は変えない）。"""

    def __init__(self, real, device):
        self._real, self._device = real, device
        event("safe_open", device=str(device))

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *a):
        r = self._real.__exit__(*a)
        event("safe_open_close", device=str(self._device))
        return r

    def __getattr__(self, name):
        return getattr(self._real, name)


def install_load_instrumentation(mu=None):
    """transformers.modeling_utils の safe_open と caching_allocator_warmup を、挙動を変えずに計装する。

    どの工程で commit / private が急増するかを時刻つきで特定するため。戻り値は元に戻す関数（テスト用）。
    """
    if mu is None:
        import transformers.modeling_utils as mu
    orig_open, orig_warm = mu.safe_open, mu.caching_allocator_warmup
    if getattr(orig_open, "_lunaria_instr", False):
        return lambda: None

    def open_wrapper(*args, **kwargs):
        dev = kwargs.get("device", "cpu")
        real = orig_open(*args, **kwargs)
        return _SafeOpenProxy(real, dev)

    def warm_wrapper(*args, **kwargs):
        event("warmup_start")
        try:
            return orig_warm(*args, **kwargs)
        finally:
            event("warmup_end")

    open_wrapper._lunaria_instr = True
    mu.safe_open, mu.caching_allocator_warmup = open_wrapper, warm_wrapper

    def uninstall():
        mu.safe_open, mu.caching_allocator_warmup = orig_open, orig_warm
    return uninstall


class SelfMonitor(threading.Thread):
    """プロセス private・RSS・空き RAM・commit 余裕（と VRAM）を高頻度で採取する軽量スレッド。

    - 全サンプルを JSONL ストリームへ即時書き出し（クラッシュしても残る）。最終 JSON には 1 秒に 1 点だけ保持
    - 空き RAM / commit 余裕が下限を consecutive 回連続で下回ったら on_abort を呼ぶ（既定: 結果を保存して os._exit(3)）
    - 精度と負荷の評価のため、実際のサンプル間隔（最大・平均）と 1 回のサンプルの処理時間を記録する
    """

    def __init__(self, floor_gb=FLOOR_GB, consecutive=2, avail_fn=avail_gb, private_fn=None, rss_fn=None,
                 on_abort=None, interval=0.25, commit_fn=commit_headroom_gb, commit_floor_gb=FLOOR_GB,
                 stream=None, vram_fn=None):
        super().__init__(daemon=True)
        self.floor_gb, self.consecutive, self.interval = floor_gb, consecutive, interval
        self.commit_fn, self.commit_floor_gb = commit_fn, commit_floor_gb
        self.stream, self.vram_fn = stream, vram_fn
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
        self._last_t = None
        self.max_gap = 0.0
        self.sum_gap = 0.0
        self.tick_cost_total = 0.0
        self.reset_window()

    def reset_window(self):
        """工程ごとの「処理中の最大値」を取るための区間統計をリセットする。"""
        self._w = {"max_private": 0, "min_commit": 1e9, "min_avail": 1e9, "max_vram": None, "samples": 0}

    def window_summary(self):
        w = self._w
        return {"max_private_mb": round(w["max_private"] / MB),
                "min_commit_headroom_gb": None if w["min_commit"] == 1e9 else round(w["min_commit"], 3),
                "min_avail_gb": None if w["min_avail"] == 1e9 else round(w["min_avail"], 3),
                "max_vram_used_mb": w["max_vram"], "samples": w["samples"]}

    def tick(self):
        t0 = time.perf_counter()
        if self._last_t is not None:
            gap = t0 - self._last_t
            self.max_gap = max(self.max_gap, gap)
            self.sum_gap += gap
        self._last_t = t0
        a = self.avail_fn()
        c = self.commit_fn()
        pv = self.private_fn()
        rss = self.rss_fn()
        vram = self.vram_fn() if self.vram_fn else None
        self.max_private = max(self.max_private, pv)
        self.max_rss = max(self.max_rss, rss)
        self.min_avail = min(self.min_avail, a)
        self.min_commit = min(self.min_commit, c)
        w = self._w
        w["max_private"] = max(w["max_private"], pv)
        w["min_commit"] = min(w["min_commit"], c)
        w["min_avail"] = min(w["min_avail"], a)
        if vram is not None:
            w["max_vram"] = vram if w["max_vram"] is None else max(w["max_vram"], vram)
        w["samples"] += 1
        self._ticks += 1
        row = {"t": round(time.time(), 3), "step": self.step, "private_mb": round(pv / MB), "rss_mb": round(rss / MB),
               "avail_gb": round(a, 2), "commit_headroom_gb": round(c, 2)}
        if vram is not None:
            row["vram_used_mb"] = vram
        if self.stream is not None:
            self.stream.write(row)
        if self._ticks % max(1, round(1.0 / self.interval)) == 0:
            self.series.append(row)
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
        self.tick_cost_total += time.perf_counter() - t0
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
                "samples": self._ticks, "interval_target_s": self.interval,
                "interval_max_s": round(self.max_gap, 3),
                "interval_mean_s": round(self.sum_gap / max(1, self._ticks - 1), 4),
                "tick_cost_mean_ms": round(1000 * self.tick_cost_total / max(1, self._ticks), 3),
                "series": self.series[-300:]}


def vram_used_mb():
    """GPU 全体の VRAM 使用量（MiB）。CUDA 未初期化のときは None（サンプラーが CUDA を初期化しないため）。"""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            free_b, total_b = torch.cuda.mem_get_info(0)
            return round((total_b - free_b) / MB)
    except Exception:
        pass
    return None


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

    global STREAM
    STREAM = JsonlStream(Path(out).with_suffix(".stream.jsonl"))  # 全サンプルとイベントをクラッシュ耐性つきで保存
    # 0.1 秒間隔: 1 秒間隔（ランナー側）では見えない瞬間的なピークを捉える。負荷は summary の tick_cost_mean_ms に記録
    mon = SelfMonitor(on_abort=on_abort, interval=0.1, stream=STREAM, vram_fn=vram_used_mb)
    mon.start()
    event("process_start", step=step)
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
        elif step == "d0":
            step_d0(res, mon, save)
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
        res["events"] = list(EVENTS)  # 工程の区切り（時刻・private・commit 余裕）。高頻度サンプルは .stream.jsonl
        save()
        STREAM.close()
    return 0 if res.get("status") == "ok" else 1


def sync_point(label, **kw):
    """重要な確保・解放の直前直後に行う同期的な計測。0.1 秒間隔のサンプラーが取りこぼす短時間の変化も、この時点の値は確実に残る。"""
    return event(label, vram_used_mb=vram_used_mb(), avail_gb=round(avail_gb(), 3), **kw)


def d0_budget(need_ram_gb, need_commit_gb):
    """工程の直前にメモリ予算を再評価する。物理の空き・commit 余裕の両方で、確保後も安全下限（3GB）を保てるか。"""
    a, c = avail_gb(), commit_headroom_gb()
    ok = (a - need_ram_gb >= FLOOR_GB) and (c - need_commit_gb >= FLOOR_GB)
    return ok, {"avail_gb": round(a, 3), "commit_headroom_gb": round(c, 3), "need_ram_gb": round(need_ram_gb, 3),
                "need_commit_gb": round(need_commit_gb, 3), "floor_gb": FLOOR_GB}


def d0_deltas(rec):
    """pre を基準にした、処理後・処理中の最大・解放後の増分（private / system commit / VRAM）。"""
    pre, post, win = rec["pre"], rec["post"], rec["window"]
    d = {"private_post_mb": post["private_mb"] - pre["private_mb"],
         "private_window_max_mb": win["max_private_mb"] - pre["private_mb"],
         "commit_post_gb": round(pre["commit_headroom_gb"] - post["commit_headroom_gb"], 3),
         "commit_window_max_gb": (None if win["min_commit_headroom_gb"] is None
                                  else round(pre["commit_headroom_gb"] - win["min_commit_headroom_gb"], 3)),
         "avail_post_drop_gb": round(pre["avail_gb"] - post["avail_gb"], 3)}
    if pre.get("vram_used_mb") is not None and post.get("vram_used_mb") is not None:
        d["vram_post_mb"] = post["vram_used_mb"] - pre["vram_used_mb"]
        d["vram_window_max_mb"] = (None if win["max_vram_used_mb"] is None
                                   else win["max_vram_used_mb"] - pre["vram_used_mb"])
    rel = rec.get("released")
    if rel:
        d["private_released_mb"] = rel["private_mb"] - pre["private_mb"]
        d["commit_released_gb"] = round(pre["commit_headroom_gb"] - rel["commit_headroom_gb"], 3)
        if pre.get("vram_used_mb") is not None and rel.get("vram_used_mb") is not None:
            d["vram_released_mb"] = rel["vram_used_mb"] - pre["vram_used_mb"]
    return d


def d0_stage(name, mon, fn, release=None, settle_s=0.4):
    """1 工程を、前・処理中の最大・後・解放後・所要時間・エラーつきで記録する（実行は独立した 1 工程）。"""
    rec = {"name": name, "pre": sync_point(f"{name}:pre")}
    mon.step = name
    mon.reset_window()
    t0 = time.perf_counter()
    try:
        rec["result"], rec["error"] = fn(), None
    except BaseException as e:  # CUDA エラー・確保失敗も記録して、成功扱いにしない
        rec["result"], rec["error"] = None, f"{type(e).__name__}: {str(e)[:300]}"
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    time.sleep(settle_s)  # サンプラーに処理直後の状態を捉えさせる
    rec["post"] = sync_point(f"{name}:post")
    rec["window"] = mon.window_summary()
    if release is not None:
        try:
            release()
            rec["release_error"] = None
        except BaseException as e:
            rec["release_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        time.sleep(settle_s)
        rec["released"] = sync_point(f"{name}:released")
    rec["deltas"] = d0_deltas(rec)
    return rec


def step_d0(res, mon, save):
    """D0: モデルなしの合成 GPU 診断（実 DiT・LM・Text Encoder・VAE・INT8・楽曲生成は行わない）。

    DiT 診断で観測した commit の急増が、(M1) safetensors のファイルマッピング、(M2) CUDA の大きな確保、のどちらに
    由来するかを切り分け、`safe_open(device="cuda")` が単独でアクセス違反を起こすかを確認する。
    工程は独立し、各工程の前に予算を再評価する。安全な順（CUDA の小さな確保 → CPU の safetensors → CUDA の safetensors）で進める。
    """
    mon.step = "import"
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    res["import_torch"] = mem()
    mon.step = "cuda_init"
    sync_point("cuda_init:pre")
    res["cuda_init"] = cuda_init_stage()
    sync_point("cuda_init:post")
    res["private_after_cuda_gb"] = round(res["cuda_init"]["after"]["private_mb"] / 1024, 3)
    res["phys_after_cuda"] = phys_since_start()
    stages, holder = [], {}
    res["d0_stages"] = stages
    res["budget_stops"] = []
    save()

    def run_stage(name, fn, release=None, need_ram_gb=0.3, need_commit_gb=1.0):
        ok, b = d0_budget(need_ram_gb, need_commit_gb)
        if not ok:
            res["budget_stops"].append({"stage": name, **b})
            return None
        rec = d0_stage(name, mon, fn, release)
        rec["budget"] = b
        stages.append(rec)
        save()
        return rec

    # ---- A. CUDA の合成確保（小さい容量から段階的に。各段階の前に予算を再評価。warmup と同じ torch.empty(fp16) の形式）----
    for size_mib in (64, 128, 256, 512, 1024):
        gib = size_mib / 1024

        def alloc(size_mib=size_mib):
            holder["x"] = torch.empty(size_mib * MB // 2, dtype=torch.float16, device="cuda")
            torch.cuda.synchronize()
            return {"size_mib": size_mib, "reserved_mb": round(torch.cuda.memory_reserved(0) / MB)}

        def free():
            holder.pop("x", None)
            torch.cuda.empty_cache()

        rec = run_stage(f"cuda_alloc_{size_mib}MiB", alloc, free, need_ram_gb=0.3, need_commit_gb=3.0 * gib)
        if rec is None or rec["error"]:
            break

    # ---- B. 小さな確保の積み重ね（transformers の warmup 無効時のパラメータごとの確保に相当）: 32MiB × 16 = 512MiB ----
    def alloc_many():
        holder["many"] = [torch.empty(32 * MB // 2, dtype=torch.float16, device="cuda") for _ in range(16)]
        torch.cuda.synchronize()
        return {"count": 16, "each_mib": 32}

    def free_many():
        holder.pop("many", None)
        torch.cuda.empty_cache()

    run_stage("cuda_alloc_many_32MiBx16", alloc_many, free_many, need_commit_gb=1.5)

    # ---- C. 合成 safetensors（256MiB）の作成 ----
    work = ROOT / "data" / "logs" / "d0_tmp"
    work.mkdir(parents=True, exist_ok=True)
    path = work / "d0_synthetic.safetensors"
    n_t, elems = 8, 16 * MB  # bf16 で 32MiB × 8 = 256MiB

    def make_file():
        tensors = {f"w{i}": torch.randn(elems, dtype=torch.bfloat16) for i in range(n_t)}
        save_file(tensors, str(path))
        holder["ref0"] = tensors["w0"].clone()  # CUDA 読み込みの正しさの照合用（32MiB）
        return {"file_mb": round(path.stat().st_size / MB)}

    try:
        run_stage("make_synthetic_file", make_file, lambda: gc.collect(), need_ram_gb=0.8, need_commit_gb=1.0)
        res["synthetic_file_mb"] = round(path.stat().st_size / MB) if path.exists() else None
        if path.exists():
            # ---- D. safetensors CPU: 開く / get_slice ----
            def cpu_open():
                holder["f_cpu"] = safe_open(str(path), framework="pt", device="cpu")
                holder["slices"] = [holder["f_cpu"].get_slice(k) for k in holder["f_cpu"].keys()]
                return {"keys": len(holder["slices"])}

            def cpu_close():
                holder.pop("slices", None)
                holder.pop("f_cpu").__exit__(None, None, None)

            run_stage("st_cpu_open_get_slice", cpu_open, cpu_close, need_commit_gb=1.0)

            # ---- E. safetensors CPU: 実体化（[...]）----
            def cpu_materialize():
                f = safe_open(str(path), framework="pt", device="cpu")
                holder["f_cpu"] = f
                holder["t_cpu"] = [f.get_slice(k)[...] for k in f.keys()]
                return {"tensors": len(holder["t_cpu"]), "device": str(holder["t_cpu"][0].device)}

            def cpu_materialize_release():
                holder.pop("t_cpu", None)
                holder.pop("f_cpu").__exit__(None, None, None)
                gc.collect()

            run_stage("st_cpu_materialize", cpu_materialize, cpu_materialize_release, need_commit_gb=1.5)

            # ---- F. 同じファイルの 3 重オープン（マッピングの多重度と commit の関係）----
            def multi_open():
                holder["multi"] = [safe_open(str(path), framework="pt", device="cpu") for _ in range(3)]
                return {"opens": 3}

            def multi_close():
                for h in holder.pop("multi"):
                    h.__exit__(None, None, None)

            run_stage("st_cpu_open_x3", multi_open, multi_close, need_commit_gb=2.5)

            # ---- G. safetensors CUDA: safe_open(device="cuda:0") で 1 テンソルずつ GPU へ（transformers と同じ手順）----
            def cuda_open_load():
                f = safe_open(str(path), framework="pt", device="cuda:0")
                holder["f_cuda"] = f
                holder["t_cuda"] = {}
                for i, k in enumerate(f.keys()):
                    sync_point(f"st_cuda_tensor_{i}:pre")
                    holder["t_cuda"][k] = f.get_slice(k)[...]
                    sync_point(f"st_cuda_tensor_{i}:post")
                torch.cuda.synchronize()
                t0c = holder["t_cuda"]["w0"]
                return {"tensors": len(holder["t_cuda"]), "device": str(t0c.device),
                        "values_equal_to_cpu_reference": bool(torch.equal(t0c.cpu(), holder["ref0"])),
                        "all_finite": all(bool(torch.isfinite(t.float()).all()) for t in holder["t_cuda"].values())}

            def cuda_open_release():
                holder.pop("t_cuda", None)
                holder.pop("f_cuda").__exit__(None, None, None)
                torch.cuda.empty_cache()

            rec_g = run_stage("st_cuda_open_load", cuda_open_load, cuda_open_release, need_commit_gb=2.0)

            # ---- H. transformers と同じ順序: warmup 相当の確保 → 解放 → safe_open(cuda) で読み込み ----
            if rec_g is not None and not rec_g["error"]:
                def warm_then_load():
                    w = torch.empty(256 * MB // 2, dtype=torch.float16, device="cuda")  # warmup 相当（モデルのバイト数 1 回）
                    torch.cuda.synchronize()
                    sync_point("warmup_like_alloc:post")
                    del w
                    sync_point("warmup_like_alloc:freed_ref")
                    f = safe_open(str(path), framework="pt", device="cuda:0")
                    holder["f_cuda"] = f
                    holder["t_cuda"] = {k: f.get_slice(k)[...] for k in f.keys()}
                    torch.cuda.synchronize()
                    return {"tensors": len(holder["t_cuda"])}

                run_stage("warmup_like_then_cuda_open_load", warm_then_load, cuda_open_release, need_commit_gb=3.0)
    finally:
        holder.clear()
        try:
            path.unlink()
            work.rmdir()
        except OSError:
            pass
    errors = [s["name"] for s in stages if s["error"] or s.get("release_error")]
    res["d0_errors"] = errors
    # M1 / M2 の指標（commit 増分 ÷ 対象サイズ）。1 に近ければ「そのサイズ分が commit に計上された」
    ind = {}
    for s in stages:
        d = s["deltas"]
        n = s["name"]
        if n.startswith("cuda_alloc_") and "MiB" in n and "many" not in n:
            ind[n] = {"commit_post_over_size": round(d["commit_post_gb"] / (int(n.split("_")[2][:-3]) / 1024), 3),
                      "commit_window_max_over_size": None if d["commit_window_max_gb"] is None else round(
                          d["commit_window_max_gb"] / (int(n.split("_")[2][:-3]) / 1024), 3)}
        elif n.startswith("st_") or n == "cuda_alloc_many_32MiBx16" or n.startswith("warmup_like"):
            base_gib = 0.5 if n == "cuda_alloc_many_32MiBx16" else 0.25
            ind[n] = {"commit_post_over_size": round(d["commit_post_gb"] / base_gib, 3),
                      "commit_window_max_over_size": None if d["commit_window_max_gb"] is None else round(
                          d["commit_window_max_gb"] / base_gib, 3)}
    res["d0_indicators"] = ind
    g = next((s for s in stages if s["name"] == "st_cuda_open_load"), None)
    values_ok = bool(g and g["result"] and g["result"]["values_equal_to_cpu_reference"] and g["result"]["all_finite"])
    res["d0_cuda_values_ok"] = values_ok
    if errors:
        res["status"] = "d0_stage_error"
    elif res["budget_stops"]:
        res["status"] = "ok_budget_stopped"  # 予算が足りず一部の工程を実施しなかった（安全側の停止）。失敗ではない
    else:
        res["status"] = "ok" if values_ok else "d0_values_mismatch"


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
    # ロード方式: gpu-direct = device_map で GPU へ直接（D1 / warmup 無効なら D2）、cpu-via = 従来の CPU 実体化 → GPU 転送（D3。
    # 上流のまま。device_map の注入も RAM ガードも付けない）
    dit_path = os.environ.get("LUNARIA_DIT_PATH", "gpu-direct")
    res["dit_path"] = dit_path
    if dit_path == "gpu-direct":
        bw.apply_low_peak_patches(components=("dit",), disable_warmup=disable_warmup)  # DiT の GPU 直接ロード + RAM ガード
    install_load_instrumentation()  # safe_open の開閉と warmup の開始・終了を、時刻と commit つきで記録（挙動は変えない）
    res["patches"] = list(bw.LOWPEAK_APPLIED)
    res["instrumentation"] = ["safe_open (open/close)", "caching_allocator_warmup (start/end)"]
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
    event("C_start")
    h._load_main_model_from_checkpoint(model_checkpoint_path=str(MODELS / DIT_DIR), device="cuda",
                                       use_flash_attention=True, compile_model=False, quantization=None)
    torch.cuda.synchronize()
    event("C_end")
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
    event("D_start")
    try:
        h._apply_dit_quantization(QUANT)
        torch.cuda.synchronize()
        event("D_end")
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
    ap.add_argument("--step", required=True, choices=["b1", "b2", "cd", "d0"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(args.step, args.out))


if __name__ == "__main__":
    main()
