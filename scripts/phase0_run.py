"""LUNARIA Phase 0 実機検証ランナー（Stage 単位で独立実行）。

使い方（プロジェクトルートの専用 venv で実行）:
    .venv-engine\\Scripts\\python.exe -I scripts\\phase0_run.py --stage A [--force] [--dry-preflight]

設計方針:
- 自分が起動した ACE-Step プロセス（とその子孫）だけを停止する。python.exe の一括終了はしない
- 状態は data/logs/phase0_state.json に原子的に逐次保存（tmp → os.replace）。失敗を成功と記録しない
- RAM/VRAM テレメトリは data/logs/phase0_telemetry.jsonl に逐次追記（異常終了後も残る）
- メモリ逼迫時は生成を始めない／始めた後でも自分のエンジンのみ正常停止して中断する
- エンジン stdout/stderr は直接ファイルへ（パイプ詰まりなし）。ログ本文は読み込まず、必要行のみ抽出する
- 歌詞本文は結果・ログに残さない（文字数のみ）。テスト歌詞は全て架空
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine" / "ACE-Step-1.5"
MODELS = ROOT / "models"
LOGS = ROOT / "data" / "logs"
OUT = ROOT / "data" / "outputs" / "phase0"
DOCS = ROOT / "docs"
STATE = LOGS / "phase0_state.json"
TELEMETRY = LOGS / "phase0_telemetry.jsonl"
ENGINE_LOG = LOGS / "acestep_api.log"  # run_stage で実行ごとのファイル名に差し替え
RUN_ID = "none"
EXPECTED_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
REQUIRED_MODELS = ["acestep-v15-turbo", "acestep-5Hz-lm-0.6B", "vae", "Qwen3-Embedding-0.6B"]

HOST, PORT = "127.0.0.1", 8001
BASE = f"http://{HOST}:{PORT}"
INIT_TIMEOUT = 30 * 60
GEN_TIMEOUT = 20 * 60

# メモリ余裕（実機の状態に応じて調整）
MIN_FREE_RAM_GB = 14.0      # 開始前に必要な空き物理 RAM（実測: エンジン最大 RSS 約 11.7GB + 余裕）
MIN_COMMIT_HEADROOM_GB = 8.0
MIN_FREE_VRAM_MIB = 3500
ABORT_FREE_RAM_GB = 1.5    # 実行中にこれを下回ったら自エンジンを正常停止して中断
ABORT_CONSECUTIVE = 3

sys.path.insert(0, str(ROOT / "scripts"))
from validate_wav import validate  # noqa: E402

STAGES = {
    "A": [dict(name="inst30", sec=30, instrumental=True, bpm=104, key="C Major",
               prompt="Bright educational synth-pop instrumental, clean minimal arrangement, moderate tempo",
               lyrics="[Instrumental]")],
    "B": [dict(name="ja30", sec=30, instrumental=False, bpm=104, key="C Major",
               prompt="Japanese educational mnemonic synth-pop song with a clear female vocal, precise Japanese "
                      "pronunciation, moderate tempo, simple memorable melody, syllabic singing.",
               lyrics="[Verse 1]\nこれはテスト用の架空の歌詞です\nあおいほしがそらにひかる\n"
                      "[Chorus]\nルナリア ルナリア つきのうた\nゆっくりうたう ことばのうた")],
    "C": [dict(name="ja60", sec=60, instrumental=False, bpm=104, key="C Major",
               prompt="Japanese educational mnemonic synth-pop song with a clear female vocal, precise Japanese "
                      "pronunciation, moderate tempo, simple memorable melody, syllabic singing.",
               lyrics="[Verse 1]\nこれはテスト用の架空の歌詞です\nあおいほしがそらにひかる\nしずかなよるにかぜがふく\n"
                      "[Chorus]\nルナリア ルナリア つきのうた\nゆっくりうたう ことばのうた\n"
                      "[Verse 2]\nみっつかぞえてひとやすみ\nあしたのゆめをおもいだす\n"
                      "[Chorus]\nルナリア ルナリア つきのうた\nゆっくりうたう ことばのうた")],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atomic_write(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class State:
    """途中停止しても壊れない状態ファイル。各段階の前後で save() する。"""

    """実行ごとに一意の run_id でエントリを追加する。過去の実行記録は上書きしない。"""

    def __init__(self):
        self.data = {}
        if STATE.exists():
            try:
                self.data = json.loads(STATE.read_text(encoding="utf-8"))
            except Exception:
                # 壊れた状態ファイルは退避して新規開始（破棄しない）
                STATE.replace(STATE.with_name(f"phase0_state.corrupt.{int(time.time())}.json"))
                self.data = {}
        if "stages" in self.data:  # run_id 導入前の形式は退避して保持
            self.data.setdefault("legacy_stages_before_run_id", self.data.pop("stages"))
        self.data.setdefault("runs", {})

    def save(self):
        atomic_write(STATE, self.data)

    def succeeded(self, stage_key):
        return [rid for rid, r in self.data["runs"].items()
                if r.get("stage") == stage_key and r.get("status") == "ok"]


def port_in_use(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def gpu_mem_mib():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        used, total = [int(x) for x in o.strip().split(",")]
        return used, total
    except Exception:
        return None, None


def mem_snapshot():
    vm = psutil.virtual_memory()
    # Windows: swap_memory().total は ページファイル、commit は ctypes 経由が正確だが軽量版として
    # virtual_memory + swap_memory を記録する
    sw = psutil.swap_memory()
    return {"free_ram_gb": round(vm.available / 1024**3, 2), "ram_used_pct": vm.percent,
            "pagefile_used_gb": round(sw.used / 1024**3, 2), "pagefile_total_gb": round(sw.total / 1024**3, 2)}


def commit_info():
    """コミット済みメモリとコミット上限（GlobalMemoryStatusEx: ullTotalPageFile = 上限）。"""
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
    gb = 1024**3
    return {"commit_limit_gb": round(m.ullTotalPageFile / gb, 1),
            "commit_used_gb": round((m.ullTotalPageFile - m.ullAvailPageFile) / gb, 1),
            "commit_headroom_gb": round(m.ullAvailPageFile / gb, 1)}


def top_consumers(n=8):
    """メモリ消費上位をプロセス名で集約（情報提供のみ。停止はしない）。"""
    agg = {}
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            agg[p.info["name"]] = agg.get(p.info["name"], 0) + p.info["memory_info"].rss
        except (psutil.Error, TypeError):
            pass
    return [{"name": k, "rss_gb": round(v / 1024**3, 2)}
            for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:n]]


def gpu_processes():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                     "--format=csv,noheader"], text=True)
        return [x.strip() for x in o.strip().splitlines() if x.strip()]
    except Exception:
        return []


def residual_engine_procs():
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if p.info["pid"] != me and re.search(r"acestep-api|ACE-Step-1\.5", cl, re.I) \
                    and "phase0_run.py" not in cl:
                out.append({"pid": p.info["pid"], "name": p.info["name"]})
        except psutil.Error:
            pass
    return out


class PhaseTracker:
    """エンジンログを追記分だけ読み、工程の区切りと VAE 実行デバイスを記録する。
    ログ時刻（loguru 形式）を使い、本文は保存しない（マーカー名のみ）。"""
    MARKERS = [
        ("lm_start", re.compile(r"LLM usage decision|Loading LLM to cuda")),
        ("dit_start", re.compile(r"\[service_generate\] Generating audio")),
        ("vae_start", re.compile(r"Model generation completed\. Decoding latents")),
        ("vae_device_cpu", re.compile(r"auto-enabling CPU VAE decode|Moving VAE to CPU|_decode_on_cpu\] Moving")),
        ("vae_end", re.compile(r"VAE decode completed")),
        ("generate_done", re.compile(r"\[generate_music\] Done! Generated")),
    ]
    TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
    ANSI = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, monitor):
        self.mon = monitor
        self.offset = ENGINE_LOG.stat().st_size if ENGINE_LOG.exists() else 0
        self.events = {}
        self.diffusion_time_cost = None

    def poll(self):
        try:
            with open(ENGINE_LOG, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
                self.offset += len(chunk)
        except FileNotFoundError:
            return
        for raw in chunk.decode("utf-8", "replace").splitlines():
            line = self.ANSI.sub("", raw)
            for name, pat in self.MARKERS:
                if name not in self.events and pat.search(line):
                    m = self.TS.search(line)
                    self.events[name] = m.group(1) if m else time.strftime("%H:%M:%S")
                    self.mon.phase = {"lm_start": "lm", "dit_start": "dit", "vae_start": "vae",
                                      "vae_end": "save", "generate_done": "save"}.get(name, self.mon.phase)
            m = re.search(r"'diffusion_time_cost': ([0-9.]+)", line)
            if m and self.diffusion_time_cost is None:
                self.diffusion_time_cost = round(float(m.group(1)), 2)

    def summary(self):
        from datetime import datetime

        def t(k):
            v = self.events.get(k)
            try:
                return datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
            except Exception:
                return None

        def d(a, b):
            ta, tb = t(a), t(b)
            return round((tb - ta).total_seconds(), 1) if ta and tb else None

        return {"events": self.events,
                "lm_seconds": d("lm_start", "dit_start"),
                "dit_seconds": d("dit_start", "vae_start"),
                "dit_diffusion_time_cost": self.diffusion_time_cost,
                "vae_seconds": d("vae_start", "vae_end"),
                "vae_device": "cpu" if "vae_device_cpu" in self.events else "gpu(未確認:CPUマーカー無し)"}


def http_json(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def preflight():
    """生成前の安全確認。1つでも NG なら実行しない。"""
    r = {"checks": {}, "ok": True}
    snap = mem_snapshot()
    used, total = gpu_mem_mib()
    r["memory"] = snap
    r["gpu_used_mib"], r["gpu_total_mib"] = used, total

    def chk(name, ok, detail=""):
        r["checks"][name] = {"ok": bool(ok), "detail": detail}
        r["ok"] = r["ok"] and bool(ok)

    chk("project_root", (ROOT / ".git").exists() and ENGINE.exists(), str(ROOT))
    try:
        sha = subprocess.check_output(["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    chk("acestep_commit", sha == EXPECTED_COMMIT, sha)
    for m in REQUIRED_MODELS:
        d = MODELS / m
        chk(f"model:{m}", d.is_dir() and any(d.glob("*.safetensors")), str(d))
    chk("no_17B_xl_requested", True, "acestep-v15-turbo + 0.6B only")
    ck = ENGINE / "checkpoints"
    try:
        junction_ok = ck.is_dir() and ck.resolve() == MODELS.resolve()
    except OSError:
        junction_ok = False
    chk("checkpoints_junction", junction_ok, f"{ck} -> {MODELS}")
    resid = residual_engine_procs()
    r["residual_engine_procs"] = resid
    chk("no_residual_engine_procs", not resid, str(resid))
    r["commit"] = commit_info()
    chk("commit_headroom", r["commit"]["commit_headroom_gb"] >= MIN_COMMIT_HEADROOM_GB,
        f"{r['commit']['commit_headroom_gb']}GB >= {MIN_COMMIT_HEADROOM_GB}GB "
        f"(used {r['commit']['commit_used_gb']} / limit {r['commit']['commit_limit_gb']})")
    r["gpu_processes"] = gpu_processes()
    r["top_memory_consumers"] = top_consumers()
    chk("port_free", not port_in_use(PORT), f"{HOST}:{PORT}")
    chk("free_ram", snap["free_ram_gb"] >= MIN_FREE_RAM_GB, f"{snap['free_ram_gb']}GB >= {MIN_FREE_RAM_GB}GB")
    chk("free_vram", used is not None and (total - used) >= MIN_FREE_VRAM_MIB,
        f"{None if used is None else total - used}MiB >= {MIN_FREE_VRAM_MIB}MiB")
    return r


class Monitor(threading.Thread):
    """1秒毎に RAM/VRAM/自プロセスツリー RSS を採取し JSONL に逐次追記。逼迫検知フラグを持つ。"""

    def __init__(self, root_pid, tag):
        super().__init__(daemon=True)
        self.root_pid, self.tag = root_pid, tag
        self.stop_flag = False
        self.phase = "init"
        self.max_vram = 0
        self.max_rss = 0
        self.min_free_ram = 1e9
        self.pressure = False
        self._low = 0
        self.lock = threading.Lock()

    def run(self):
        n = 0
        with open(TELEMETRY, "a", encoding="utf-8") as f:
            while not self.stop_flag:
                used, _ = gpu_mem_mib()
                rss = 0
                try:
                    p = psutil.Process(self.root_pid)
                    for q in [p] + p.children(recursive=True):
                        try:
                            rss += q.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                snap = mem_snapshot()
                with self.lock:
                    if used:
                        self.max_vram = max(self.max_vram, used)
                    self.max_rss = max(self.max_rss, rss)
                    self.min_free_ram = min(self.min_free_ram, snap["free_ram_gb"])
                    self._low = self._low + 1 if snap["free_ram_gb"] < ABORT_FREE_RAM_GB else 0
                    if self._low >= ABORT_CONSECUTIVE:
                        self.pressure = True
                n += 1
                if True:  # 1秒間隔で書き出し
                    f.write(json.dumps({"run_id": RUN_ID, "t": time.strftime("%H:%M:%S"), "tag": self.tag,
                                        "phase": self.phase,
                                        "vram_mib": used,
                                        "tree_rss_mb": round(rss / 1048576), **snap}) + "\n")
                    f.flush()
                time.sleep(1)

    def snapshot_and_reset(self):
        with self.lock:
            r = {"max_vram_mib": self.max_vram, "max_rss_mb": round(self.max_rss / 1048576),
                 "min_free_ram_gb": None if self.min_free_ram == 1e9 else self.min_free_ram}
            self.max_vram = self.max_rss = 0
            self.min_free_ram = 1e9
        return r


def owned_tree(proc):
    """自分が起動した PID の子孫のみを列挙。実行ファイルが当プロジェクトの venv 配下であることも確認。"""
    try:
        parent = psutil.Process(proc.pid)
        exe = parent.exe().lower()
    except psutil.Error:
        return []
    if str(ROOT / ".venv-engine").lower() not in exe:
        log(f"所有権確認失敗: PID {proc.pid} の exe={exe} は想定外。停止しない。")
        return []
    return [parent] + parent.children(recursive=True)


def stop_owned(proc):
    """正常停止要求 → 待機 → 所有 PID のみ terminate → 残存のみ kill。"""
    out = {"pids": [], "graceful": False, "terminated": [], "killed": []}
    tree = owned_tree(proc)
    out["pids"] = [p.pid for p in tree]
    if not tree:
        return out
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)  # 新プロセスグループ宛の正常停止要求
    except Exception as e:
        out["graceful_error"] = type(e).__name__
    _, alive = psutil.wait_procs(tree, timeout=40)
    out["graceful"] = not alive
    for p in alive:
        try:
            p.terminate()
            out["terminated"].append(p.pid)
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(alive, timeout=20)
    for p in alive:
        try:
            p.kill()
            out["killed"].append(p.pid)
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=10)
    return out


def extract_engine_facts():
    """エンジンログから構成情報の行のみ抽出（歌詞・プロンプト行は対象外のパターンのみ）。"""
    facts = []
    pat = re.compile(r"Configuration Tier|offload|backend|quantiz|int8|Keeping main model|attention implementation|"
                     r"GPU auto-detection|vLLM|initialized successfully", re.I)
    try:
        with open(ENGINE_LOG, "rb") as f:
            for raw in f:
                line = re.sub(r"\x1b\[[0-9;]*m", "", raw.decode("utf-8", "replace")).strip()
                if "query_result" in line or "Xet" in line or "Fetching" in line:
                    continue
                if pat.search(line):
                    facts.append(line[:220])
                if len(facts) >= 40:
                    break
    except FileNotFoundError:
        pass
    return facts


def run_stage(stage_key, force):
    global ENGINE_LOG, RUN_ID
    RUN_ID = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    LOGS.mkdir(parents=True, exist_ok=True)

    # --dry-preflight は読み取り専用: phase0_state.json に触れず、別の診断ファイルへ記録する
    if "--dry-preflight" in sys.argv:
        pf = preflight()
        diag = LOGS / f"preflight_{RUN_ID}.json"
        atomic_write(diag, {"run_id": RUN_ID, "stage": stage_key, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "preflight": pf})
        bad = [k for k, v in pf["checks"].items() if not v["ok"]]
        log(f"事前確認 {'OK' if pf['ok'] else 'NG ' + str(bad)}（--dry-preflight: 状態ファイル不変）→ {diag.name}")
        return 0 if pf["ok"] else 3

    st = State()
    done = st.succeeded(stage_key)
    if done and not force:
        log(f"Stage {stage_key} は成功済み（run_id={done[-1]}）。重複生成を避けて終了（再実行は --force）。")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    cases = STAGES[stage_key]
    for c in cases:
        w = OUT / f"{c['name']}.wav"
        if w.exists() and not force:
            log(f"{w.name} が既に存在。上書きしない（--force で退避せず再生成）。")
            return 4

    ENGINE_LOG = LOGS / f"acestep_api_{RUN_ID}.log"  # 実行ごとに分離（過去ログを上書きしない）
    s = {"run_id": RUN_ID, "stage": stage_key, "status": "preflight",
         "started": time.strftime("%Y-%m-%d %H:%M:%S"), "engine_log": ENGINE_LOG.name, "cases": []}
    st.data["runs"][RUN_ID] = s
    s["preflight"] = preflight()
    st.save()
    if not s["preflight"]["ok"]:
        s["status"] = "blocked_preflight"
        st.save()
        bad = [k for k, v in s["preflight"]["checks"].items() if not v["ok"]]
        log(f"事前確認 NG: {bad} → 実行しない")
        return 3

    env = os.environ.copy()
    cache = ROOT / ".cache"
    env.update({
        "ACESTEP_API_HOST": HOST, "ACESTEP_API_PORT": str(PORT),
        "ACESTEP_INIT_LLM": "true",
        "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
        "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B",
        "ACESTEP_NO_INIT": "false",
        "ACESTEP_CHECKPOINTS_DIR": str(MODELS),
        "ACESTEP_TMPDIR": str(cache / "acestep" / "tmp"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "HF_HOME": str(cache / "hf"),
        "HF_HUB_OFFLINE": "1",  # 再ダウンロード禁止（モデルは取得済み）
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
    })
    for k in ("ACESTEP_LM_BACKEND", "ACESTEP_OFFLOAD_TO_CPU", "ACESTEP_OFFLOAD_DIT_TO_CPU", "MAX_CUDA_VRAM",
              "ACESTEP_PROJECT_ROOT"):
        env.pop(k, None)

    exe = ROOT / ".venv-engine" / "Scripts" / "acestep-api.exe"
    logf = open(ENGINE_LOG, "wb")
    t0 = time.time()
    proc = subprocess.Popen([str(exe), "--host", HOST, "--port", str(PORT)], cwd=str(ENGINE), env=env,
                            stdout=logf, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    s.update({"status": "engine_starting", "owned_pid": proc.pid})
    st.save()
    log(f"ACE-Step API 起動 PID={proc.pid}")
    mon = Monitor(proc.pid, f"stage{stage_key}")
    mon.start()
    rc = 1
    try:
        # エンジン起動（ポート待受開始）
        while not port_in_use(PORT):
            if proc.poll() is not None:
                raise RuntimeError(f"EngineCrashed exit={proc.returncode}")
            if mon.pressure or time.time() - t0 > INIT_TIMEOUT:
                raise RuntimeError("MemoryPressure/Timeout before listen")
            time.sleep(1)
        s["engine_start_seconds"] = round(time.time() - t0, 1)
        t_listen = time.time()
        s["status"] = "loading_models"
        st.save()
        # モデルロード
        health = None
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"EngineCrashed exit={proc.returncode}")
            if mon.pressure:
                raise RuntimeError("MemoryPressure")
            if time.time() - t0 > INIT_TIMEOUT:
                raise RuntimeError("InitTimeout")
            try:
                health = http_json("/health", timeout=5)
                d = health["data"]
                if d.get("models_initialized") and d.get("llm_initialized"):
                    break
            except Exception:
                pass
            time.sleep(2)
        s["model_load_seconds"] = round(time.time() - t_listen, 1)
        s["total_init_seconds"] = round(time.time() - t0, 1)
        s["init_perf"] = mon.snapshot_and_reset()
        s["health"] = health["data"]
        s["engine_facts"] = extract_engine_facts()
        s["status"] = "ready"
        st.save()
        atomic_write(DOCS / "ace-step-health.json", health)
        with urllib.request.urlopen(BASE + "/openapi.json", timeout=30) as rr:
            (DOCS / "ace-step-openapi.json").write_bytes(rr.read())
        log(f"Ready: engine起動 {s['engine_start_seconds']}s / モデルロード {s['model_load_seconds']}s")

        for c in cases:
            payload = {
                "prompt": c["prompt"], "lyrics": c["lyrics"], "thinking": True,
                "vocal_language": "ja" if not c["instrumental"] else "en",
                "audio_format": "wav", "audio_duration": c["sec"], "bpm": c["bpm"], "key_scale": c["key"],
                "batch_size": 1, "use_random_seed": False, "seed": 42, "inference_steps": 8,
            }
            case = {"name": c["name"], "expected_sec": c["sec"], "lyrics_chars": len(c["lyrics"])}
            s["cases"].append(case)
            s["status"] = f"generating:{c['name']}"
            st.save()
            mon.snapshot_and_reset()
            tracker = PhaseTracker(mon)
            mon.phase = "lm"
            tg = time.time()
            sub = http_json("/release_task", payload, timeout=60)  # 1ジョブにつき要求は1回のみ
            tid = sub["data"]["task_id"]
            case["task_id"] = tid
            st.save()
            item = None
            while time.time() - tg < GEN_TIMEOUT:
                if proc.poll() is not None:
                    raise RuntimeError("EngineCrashed during generation")
                tracker.poll()
                case["phases"] = tracker.summary()  # 中断しても最新の工程情報を残す
                if mon.pressure:
                    case["phases"]["aborted_in_phase"] = mon.phase
                    st.save()
                    raise RuntimeError(f"MemoryPressure during generation (phase={mon.phase})")
                q = http_json("/query_result", {"task_id_list": [tid]}, timeout=30)
                item = q["data"][0]
                if item["status"] in (1, 2):
                    break
                time.sleep(2)
            tracker.poll()
            case["phases"] = tracker.summary()
            case["gen_seconds"] = round(time.time() - tg, 1)
            case.update(mon.snapshot_and_reset())
            case["status_code"] = item["status"] if item else None
            if item and item["status"] == 1:
                r0 = json.loads(item["result"])[0]
                case.update({"dit_model": r0.get("dit_model"), "lm_model": r0.get("lm_model"),
                             "seed_value": r0.get("seed_value"), "metas": r0.get("metas")})
                wav = OUT / f"{c['name']}.wav"
                mon.phase = "save"
                ts = time.time()
                with urllib.request.urlopen(BASE + r0["file"], timeout=120) as rr:
                    wav.write_bytes(rr.read())
                case["wav_save_seconds"] = round(time.time() - ts, 2)
                mon.phase = "validate"
                tv = time.time()
                case["validation"] = validate(str(wav), c["sec"])
                case["wav_validate_seconds"] = round(time.time() - tv, 2)
                case["ok"] = bool(case["validation"].get("ok"))
            else:
                case["ok"] = False
                case["error"] = "generation failed or timed out"
            st.save()
            log(f"[{c['name']}] {case['gen_seconds']}s ok={case['ok']}")
        s["status"] = "generated_ok" if all(x.get("ok") for x in s["cases"]) else "generated_with_failures"
        rc = 0 if s["status"] == "generated_ok" else 1
    except Exception as e:
        s["status"] = "failed"
        s["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR {s['error']}")
        rc = 1
    finally:
        mon.stop_flag = True
        t_stop = time.time()
        s["stop"] = stop_owned(proc)
        s["stop_seconds"] = round(time.time() - t_stop, 1)
        logf.close()
        time.sleep(5)
        s["port_released"] = not port_in_use(PORT)
        used, _ = gpu_mem_mib()
        s["gpu_after_stop_used_mib"] = used
        s["orphan_pids"] = [p for p in s["stop"]["pids"] if psutil.pid_exists(p)]
        s["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        clean = s["port_released"] and not s["orphan_pids"]
        s["shutdown_clean"] = clean
        if s["status"] == "generated_ok" and clean:
            s["status"] = "ok"
        elif s["status"] == "generated_ok":
            s["status"] = "generated_but_shutdown_unclean"
            rc = 1
        st.save()
        log(f"停止: graceful={s['stop']['graceful']} port_released={s['port_released']} "
            f"gpu_used={used}MiB orphans={s['orphan_pids']} status={s['status']}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-preflight", action="store_true")
    args = ap.parse_args()
    sys.exit(run_stage(args.stage, args.force))


if __name__ == "__main__":
    main()
