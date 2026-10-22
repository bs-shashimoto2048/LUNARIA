"""DiT 単独の段階的 GPU 診断ランナー（LM / Text Encoder / VAE は初期化しない。楽曲生成もしない）。

    .venv-engine\\Scripts\\python.exe -I scripts\\dit_diag.py --dry-run   # Step A と各ステップの許可判定のみ。子プロセスなし
    .venv-engine\\Scripts\\python.exe -I scripts\\dit_diag.py             # 許可条件を満たすステップだけを順に 1 回ずつ実行

ステップ:
  A   軽量診断（モデルロードなし。環境・依存関係・メモリ状況。このランナー自身）
  B1  import のみ（torch / transformers / acestep.api_server まで）→ 前回の約 2.55GB と比較
  B2  import（acestep.handler まで）+ CUDA コンテキスト → CUDA のホスト側 RAM / VRAM を実測
  C+D DiT の GPU 直接ロード（C）→ 安全な場合のみ INT8 適用（D）

許可条件（各ステップの直前に、その時点の空き資源で判定）: 空き RAM >= 見込み消費量 + 安全余裕 3GB。
見込み消費量は「実測済みは実測値、未測定は上限見積り」。安全下限（3GB）は引き下げない。
満たさなければそのステップで停止して報告する。自動再試行はしない。

- 子プロセスは自分が起動した PID のみ停止する（他プロセスには触れない）
- 結果: data/logs/dit_diag_<run_id>.json（Git 管理外）。実行ごとに一意の run_id。--dry-run は状態ファイルを変更しない
- このランナーは軽量（psutil と nvidia-smi のみ。モデルは読まない）
"""
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lunaria_profiles as lp  # noqa: E402
import phase0_run as p0  # noqa: E402  (State / Monitor / stop_owned / commit_info などを再利用)
import psutil  # noqa: E402

ROOT = p0.ROOT
LOGS = p0.LOGS
STAGE = "DIT_DIAG"
PROBE = ROOT / "scripts" / "dit_probe.py"
TIMEOUTS = {"b1": 300, "b2": 300, "cd": 900, "d0": 300}
VALID_STEPS = ("b1", "b2", "cd", "d0")


def parse_steps(argv):
    """--steps b1,b2,cd（既定）または d0 など。無効な指定は拒否する。"""
    for a in argv:
        if a.startswith("--steps="):
            steps = tuple(x for x in a.split("=", 1)[1].split(",") if x)
            bad = [s for s in steps if s not in VALID_STEPS]
            if bad or not steps:
                raise SystemExit(f"無効な --steps: {bad or '空'}（有効: {','.join(VALID_STEPS)}）")
            return steps
    return ("b1", "b2", "cd")


def parse_path(argv):
    """--path=gpu-direct（既定。D1 / --disable-warmup で D2）または --path=cpu-via（D3: 従来の CPU 経由ロード）。"""
    for a in argv:
        if a.startswith("--path="):
            p = a.split("=", 1)[1]
            if p not in ("gpu-direct", "cpu-via"):
                raise SystemExit(f"無効な --path: {p}")
            return p
    return "gpu-direct"
MB = 1048576


def vram_state():
    used, total = p0.gpu_mem_mib()
    return {"used_mib": used, "total_mib": total, "free_mib": None if used is None else total - used}


def current_resources():
    snap = p0.mem_snapshot()
    commit = p0.commit_info()
    v = vram_state()
    return {"avail_gb": snap["free_ram_gb"], "commit_headroom_gb": commit["commit_headroom_gb"],
            "free_vram_mib": v["free_mib"], "vram_used_mib": v["used_mib"], "pagefile_used_gb": snap["pagefile_used_gb"]}


def step_a():
    """Step A: モデルロードなしの軽量診断。"""
    def ver(n):
        try:
            return importlib.metadata.version(n)
        except Exception:
            return None
    r = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
         "versions": {n: ver(n) for n in ("torch", "torchao", "transformers", "accelerate", "diffusers",
                                          "safetensors", "triton-windows", "psutil")},
         "resources": current_resources(), "memory": p0.mem_snapshot(), "commit": p0.commit_info(),
         "gpu_processes": p0.gpu_processes(), "top_memory_consumers": p0.top_consumers(),
         "port_8001_free": not p0.port_in_use(p0.PORT), "residual_engine_procs": p0.residual_engine_procs(),
         "runner_private_mb": round(psutil.Process().memory_info().private / MB)}
    try:
        r["acestep_commit"] = subprocess.check_output(["git", "-C", str(p0.ENGINE), "rev-parse", "HEAD"],
                                                      text=True).strip()
    except Exception:
        r["acestep_commit"] = None
    r["acestep_commit_ok"] = r["acestep_commit"] == p0.EXPECTED_COMMIT
    dit = p0.MODELS / "acestep-v15-turbo"
    r["dit_files_ok"] = dit.is_dir() and (dit / "model.safetensors").exists()
    ck = p0.ENGINE / "checkpoints"
    try:
        r["junction_ok"] = ck.is_dir() and ck.resolve() == p0.MODELS.resolve()
    except OSError:
        r["junction_ok"] = False
    r["static_ok"] = bool(r["acestep_commit_ok"] and r["dit_files_ok"] and r["junction_ok"]
                          and r["port_8001_free"] and not r["residual_engine_procs"])
    return r


class SysMonitor(threading.Thread):
    """ランナー側の高頻度システム監視（0.1 秒、GlobalMemoryStatusEx のみ）。

    子プロセスの Python サンプラーは、GIL を保持するネイティブ呼び出し（例: safetensors の save_file）の間は止まる
    （D0 で 4.39 秒の空白を実測）。ランナーは別プロセスなので影響を受けず、システム全体の空き RAM と commit 余裕を
    途切れずに記録できる。nvidia-smi のような重い処理は行わない（VRAM は 1 秒周期の Monitor と子プロセス内で記録）。
    """

    def __init__(self, path, status_fn=None, interval=0.1):
        super().__init__(daemon=True)
        self.path = Path(path)
        self.status_fn = status_fn or p0.fast_status
        self.interval = interval
        self.stop_flag = False
        self.samples = 0
        self.max_gap = 0.0
        self.min_avail = 1e9
        self.min_commit = 1e9
        self.cost_total = 0.0

    def tick(self, f=None):
        t0 = time.perf_counter()
        avail, load, commit = self.status_fn()
        self.min_avail = min(self.min_avail, avail)
        self.min_commit = min(self.min_commit, commit)
        self.samples += 1
        if f is not None:
            f.write(json.dumps({"t": round(time.time(), 3), "avail_gb": round(avail, 3), "commit_headroom_gb": commit},
                               separators=(",", ":")) + "\n")
            f.flush()
        self.cost_total += time.perf_counter() - t0

    def run(self):
        last = None
        with open(self.path, "a", encoding="utf-8") as f:
            while not self.stop_flag:
                t = time.perf_counter()
                if last is not None:
                    self.max_gap = max(self.max_gap, t - last)
                last = t
                self.tick(f)
                time.sleep(max(0.0, self.interval - (time.perf_counter() - t)))

    def summary(self):
        return {"samples": self.samples, "interval_target_s": self.interval, "interval_max_s": round(self.max_gap, 3),
                "tick_cost_mean_ms": round(1000 * self.cost_total / max(1, self.samples), 4),
                "min_avail_gb": None if self.min_avail == 1e9 else round(self.min_avail, 3),
                "min_commit_headroom_gb": None if self.min_commit == 1e9 else round(self.min_commit, 2)}


def _sanitize(text):
    """ログに含まれる個人情報（ユーザー名つきのパス）を伏せる。"""
    import re
    s = str(text).replace(str(ROOT), "<ROOT>")
    return re.sub(r"([A-Za-z]:\\Users\\)[^\\\s]+", r"\1<USER>", s)


def collect_crash_info(t_start, t_end, log_path):
    """子プロセスが異常終了したときの、Windows イベントログ（Application Error / WER）と faulthandler の記録を保存する。

    読み取りのみ。自動再試行はしない。結果はローカル（Git 管理外）の結果ファイルに入り、公開文書には要点だけを書く。
    """
    import datetime
    import re
    out = {"events": [], "faulthandler_tail": []}
    fmt = "%Y-%m-%d %H:%M:%S"
    s = datetime.datetime.fromtimestamp(t_start - 5).strftime(fmt)
    e = datetime.datetime.fromtimestamp(t_end + 90).strftime(fmt)
    ps = ("$ev = Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=(Get-Date 'START'); EndTime=(Get-Date 'END')} "
          "-ErrorAction SilentlyContinue | Where-Object { $_.ProviderName -match 'Application Error|Windows Error Reporting' } | "
          "ForEach-Object { [pscustomobject]@{Time=$_.TimeCreated.ToString('HH:mm:ss.fff'); Provider=$_.ProviderName; "
          "Id=$_.Id; Message=$_.Message} }; if ($ev) { $ev | ConvertTo-Json -Compress -Depth 3 } else { '[]' }"
          ).replace("START", s).replace("END", e)
    try:
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps], text=True, timeout=60,
                                      encoding="utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else []
        if isinstance(data, dict):
            data = [data]
        keep = re.compile(r"例外コード|Exception code|フォールト オフセット|Fault offset|障害が発生したモジュール|Faulting module|"
                          r"障害が発生しているアプリケーション|Faulting application|P4:|P7:|P8:|イベント名|Event Name")
        for ev in data:
            lines = [_sanitize(x.strip()) for x in str(ev.get("Message", "")).splitlines() if keep.search(x)]
            out["events"].append({"time": ev.get("Time"), "provider": ev.get("Provider"), "id": ev.get("Id"),
                                  "summary": lines[:12]})
    except Exception as ex:
        out["events_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    try:
        txt = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        idx = [i for i, ln in enumerate(txt) if "fatal exception" in ln.lower() or "Fatal Python" in ln]
        tail = txt[idx[0]:idx[0] + 40] if idx else []
        out["faulthandler_tail"] = [_sanitize(x)[:240] for x in tail]
        out["faulthandler_found"] = bool(idx)
    except Exception as ex:
        out["faulthandler_error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    return out


def run_child(step, run_id, results):
    """子プロセスを所有プロセスとして起動し、外側からも RAM を監視する。自動再試行はしない。"""
    out = LOGS / f"dit_probe_{run_id}_{step}.json"
    log = LOGS / f"dit_probe_{run_id}_{step}.log"
    cache = ROOT / ".cache"
    env = lp.build_env(lp.PROFILE_B_PRIME, os.environ)
    env.update({
        "ACESTEP_CHECKPOINTS_DIR": str(p0.MODELS),
        "ACESTEP_TMPDIR": str(cache / "acestep" / "tmp"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "HF_HOME": str(cache / "hf"),
    })
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONFAULTHANDLER"] = "1"  # ネイティブクラッシュ（アクセス違反）の Python 側スタックをログへ出す
    if "--disable-warmup" in sys.argv:  # 既定オフ。caching_allocator_warmup を無効化して commit の急増を避けられるか確認する用
        env["LUNARIA_DISABLE_WARMUP"] = "1"
    env["LUNARIA_DIT_PATH"] = parse_path(sys.argv)  # gpu-direct（D1/D2）または cpu-via（D3）
    py = ROOT / ".venv-engine" / "Scripts" / "python.exe"
    before = {"resources": current_resources(), "vram": vram_state()}
    logf = open(log, "wb")
    t0 = time.time()
    proc = subprocess.Popen([str(py), "-I", "-X", "faulthandler", str(PROBE), "--step", step, "--out", str(out)],
                            cwd=str(p0.ENGINE),
                            env=env, stdout=logf, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    p0.log(f"[{step}] 子プロセス開始 PID={proc.pid}")
    mon = p0.Monitor(proc.pid, f"dit_{step}")
    mon.phase = step
    mon.start()
    sysmon = SysMonitor(LOGS / f"dit_sys_{run_id}_{step}.jsonl")  # 子の GIL に左右されない 0.1 秒のシステム監視
    sysmon.start()
    info = {"step": step, "owned_pid": proc.pid, "before": before, "result_file": out.name, "log_file": log.name}
    try:
        while proc.poll() is None:
            if mon.pressure:
                info["abort"] = "外側の監視: 空き RAM が下限未満（所有プロセスのみ停止）"
                break
            if time.time() - t0 > TIMEOUTS[step]:
                info["abort"] = "タイムアウト"
                break
            time.sleep(1)
    finally:
        mon.stop_flag = True
        sysmon.stop_flag = True
        info["sys_monitor"] = sysmon.summary()
        info["perf"] = mon.snapshot_and_reset()
        info["wall_seconds"] = round(time.time() - t0, 1)
        info["stop"] = p0.stop_owned(proc)  # 既に終了していれば何もしない。生きていれば所有 PID のみ停止
        logf.close()
        time.sleep(3)
        info["exit_code"] = proc.poll()
        info["after"] = {"resources": current_resources(), "vram": vram_state()}
        info["ram_recovery_gb"] = round(info["after"]["resources"]["avail_gb"] - before["resources"]["avail_gb"], 2)
        info["vram_after_minus_before_mib"] = info["after"]["vram"]["used_mib"] - before["vram"]["used_mib"]
        info["orphans"] = [x for x in info["stop"]["pids"] if psutil.pid_exists(x)]
        info["residual_after"] = p0.residual_engine_procs()
        if info["exit_code"] not in (0, 1, 3, None):  # 0xC0000005 などのネイティブクラッシュ: 記録を保存（再試行はしない）
            info["access_violation"] = info["exit_code"] in (3221225477, -1073741819)
            info["crash_info"] = collect_crash_info(t0, time.time(), log)
    try:
        info["result"] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    except Exception as e:
        info["result"] = {"read_error": type(e).__name__}
    return info


def main():
    dry = "--dry-run" in sys.argv
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    p0.RUN_ID = run_id
    p0.ABORT_FREE_RAM_GB = lp.SAFETY_FLOOR_GB  # 外側の監視: 空き RAM がこれ未満で所有プロセスのみ停止
    LOGS.mkdir(parents=True, exist_ok=True)
    out_file = LOGS / (f"dit_diag_dry_{run_id}.json" if dry else f"dit_diag_{run_id}.json")
    steps = parse_steps(sys.argv)
    R = {"run_id": run_id, "mode": "dry-run" if dry else "run", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "generation": "NOT RUN", "lm_te_vae": "NOT INITIALIZED", "steps": {}, "admission": {},
         "options": {"steps": list(steps), "path": parse_path(sys.argv),
                     "disable_warmup": "--disable-warmup" in sys.argv}}
    st = None
    if not dry:
        st = p0.State()
        st.data["runs"][run_id] = {"run_id": run_id, "stage": STAGE, "status": "running",
                                   "started": R["started"], "result_file": out_file.name,
                                   "generation": "NOT RUN"}
        st.save()

    def save():
        p0.atomic_write(out_file, R)

    A = step_a()
    R["steps"]["A"] = A
    save()
    res_now = A["resources"]
    if not A["static_ok"]:
        R["status"] = "blocked_step_a"
        R["blocked_reason"] = "静的条件（commit / モデル / ジャンクション / ポート / 残存プロセス）を満たさない"
        save()
        p0.log(f"Step A NG → 停止（{out_file.name}）")
        return finish(R, st, save, 3)

    measured = {}

    def admit(step):
        r = current_resources()
        a = lp.dit_diag_admission(step, measured, r["avail_gb"], r["commit_headroom_gb"], r["free_vram_mib"])
        R["admission"][step] = a
        save()
        return a

    # 判定のみ（--dry-run）: 現在の資源で各ステップの許可判定を記録して終了
    if dry:
        for s in steps:
            admit(s)
        R["status"] = "dry_run_done"
        save()
        for s, a in R["admission"].items():
            bad = [k for k, c in a["checks"].items() if not c["ok"]]
            p0.log(f"[{s}] {'許可' if a['ok'] else '不許可 ' + str(bad)}: 消費 {a['consumption_gb']}GB + 余裕 "
                   f"{a['safety_floor_gb']}GB = 必要 {a['required_free_ram_gb']}GB / 現在 {res_now['avail_gb']}GB")
        return finish(R, st, save, 0)

    status, rc = "ok", 0
    for step in steps:
        a = admit(step)
        if not a["ok"]:
            bad = [f"{k}: {c['detail']}" for k, c in a["checks"].items() if not c["ok"]]
            R["status"] = f"blocked_at_{step}"
            R["blocked_reason"] = bad
            p0.log(f"[{step}] 事前条件を満たさないため実行しない: {bad}（自動再試行なし）")
            status, rc = R["status"], 3
            break
        info = run_child(step, run_id, R)
        R["steps"][step] = info
        save()
        child = info.get("result") or {}
        p0.log(f"[{step}] 子プロセス終了: status={child.get('status')} exit={info['exit_code']} "
               f"RAM 回復 {info['ram_recovery_gb']:+.2f}GB / VRAM 差 {info['vram_after_minus_before_mib']:+d}MiB")
        if step == "b1" and child.get("status") == "ok":
            measured["b1_phys_gb"] = child["phys_after_import"]["phys_gb"]
            R["measured"] = dict(measured)
        if step == "b2" and child.get("status") == "ok":
            measured["b2_private_after_cuda_gb"] = child["private_after_cuda_gb"]
            measured["b2_phys_after_cuda_gb"] = child["phys_after_cuda"]["phys_gb"]
            base_vram = info["before"]["vram"]["used_mib"]
            measured["ctx_vram_mib"] = max(0, (info["perf"]["max_vram_mib"] or base_vram) - base_vram)
            R["measured"] = dict(measured)
        if child.get("status") not in ("ok", "C_done", "C_done_D_skipped") or info.get("abort") \
                or info["orphans"] or info["residual_after"]:
            R["status"] = f"failed_at_{step}"
            status, rc = R["status"], 1
            break
    else:
        last = steps[-1]
        lr = (R["steps"].get(last) or {}).get("result") or {}
        R["status"] = "ok" if lr.get("status") == "ok" else f"completed_with_{lr.get('status')}"
        status = R["status"]
        rc = 0 if lr.get("status") == "ok" else 1
    R["runner_private_mb_end"] = round(psutil.Process().memory_info().private / MB)
    R["final_resources"] = current_resources()
    save()
    return finish(R, st, save, rc)


def finish(R, st, save, rc):
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    R["runner_private_mb_end"] = round(psutil.Process().memory_info().private / MB)
    save()
    if st is not None:
        st.data["runs"][R["run_id"]].update({"status": R.get("status"), "finished": R["finished"]})
        st.save()
    p0.log(f"完了: status={R.get('status')} runner_private={R['runner_private_mb_end']}MB → {R['run_id']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
