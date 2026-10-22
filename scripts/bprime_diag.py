"""B′ プロファイルの診断ランナー（モデル初期化と INT8 量子化までの診断 1 回。音楽生成はしない）。

    .venv-engine\\Scripts\\python.exe -I scripts\\bprime_diag.py --dry-preflight   # 読み取り専用。状態ファイル不変
    .venv-engine\\Scripts\\python.exe -I scripts\\bprime_diag.py                    # 事前確認に合格した場合のみ診断を 1 回実行

- 事前確認の必要量は lunaria_profiles.b_prime_admission()（A の初期化時の実測値が根拠。推定値だけで確定しない）
- 診断は自分が起動したラッパー（bprime_wrapper.py）の子プロセスとして実行し、所有 PID のみ停止する
- テレメトリ（1 秒間隔）は data/logs/phase0_telemetry.jsonl に run_id 付きで追記
- 自動再試行はしない。成功済みの診断は --force なしでは再実行しない
- 実行ごとに一意の run_id。過去の記録は上書きしない
"""
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lunaria_profiles as lp  # noqa: E402
import phase0_run as p0  # noqa: E402  (State / Monitor / stop_owned などを再利用)
import psutil  # noqa: E402

ROOT = p0.ROOT
LOGS = p0.LOGS
STAGE = "B_PRIME_DIAG"
WRAPPER = ROOT / "scripts" / "bprime_wrapper.py"
RUN_TIMEOUT_S = 30 * 60
PINNED_TORCHAO = "0.16.0"


def residual_procs():
    """ACE-Step / B′ ラッパーの残存プロセス（自分自身は除外）。"""
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
            if p.info["pid"] != me and ("acestep-api" in cl or "bprime_wrapper.py" in cl):
                out.append({"pid": p.info["pid"], "name": p.info["name"]})
        except psutil.Error:
            pass
    return out


def bprime_preflight():
    adm = lp.b_prime_admission()
    r = {"profile": lp.PROFILE_B_PRIME, "admission": adm, "checks": {}, "ok": True}
    snap = p0.mem_snapshot()
    used, total = p0.gpu_mem_mib()
    r["memory"], r["commit"] = snap, p0.commit_info()
    r["gpu_used_mib"], r["gpu_total_mib"] = used, total
    r["gpu_processes"] = p0.gpu_processes()
    r["top_memory_consumers"] = p0.top_consumers()

    def chk(name, ok, detail=""):
        r["checks"][name] = {"ok": bool(ok), "detail": detail}
        r["ok"] = r["ok"] and bool(ok)

    chk("project_root", (ROOT / ".git").exists() and p0.ENGINE.exists(), "<ROOT>")
    try:
        sha = subprocess.check_output(["git", "-C", str(p0.ENGINE), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    chk("acestep_commit", sha == p0.EXPECTED_COMMIT, sha)
    for m in p0.REQUIRED_MODELS:
        d = p0.MODELS / m
        chk(f"model:{m}", d.is_dir() and any(d.glob("*.safetensors")), m)
    ck = p0.ENGINE / "checkpoints"
    try:
        junction_ok = ck.is_dir() and ck.resolve() == p0.MODELS.resolve()
    except OSError:
        junction_ok = False
    chk("checkpoints_junction", junction_ok, "engine/ACE-Step-1.5/checkpoints -> models")
    try:
        tv = importlib.metadata.version("torchao")
    except Exception:
        tv = None
    chk("torchao_pinned", importlib.util.find_spec("torchao") is not None and tv == PINNED_TORCHAO,
        f"torchao={tv}（固定 {PINNED_TORCHAO}。torch 2.7.1 との互換性警告あり）")
    resid = residual_procs()
    chk("no_residual_engine_procs", not resid, str(resid))
    chk("port_free", not p0.port_in_use(p0.PORT), f"{p0.HOST}:{p0.PORT}")
    chk("free_ram", snap["free_ram_gb"] >= adm["required_free_ram_gb"],
        f"{snap['free_ram_gb']}GB >= {adm['required_free_ram_gb']}GB（根拠: {adm['basis']}）")
    chk("commit_headroom", r["commit"]["commit_headroom_gb"] >= adm["required_commit_headroom_gb"],
        f"{r['commit']['commit_headroom_gb']}GB >= {adm['required_commit_headroom_gb']}GB")
    free_vram = None if used is None else total - used
    chk("free_vram", free_vram is not None and free_vram >= adm["required_free_vram_mib"],
        f"{free_vram}MiB >= {adm['required_free_vram_mib']}MiB")
    return r


def main():
    dry = "--dry-preflight" in sys.argv
    force = "--force" in sys.argv
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    p0.RUN_ID = run_id
    LOGS.mkdir(parents=True, exist_ok=True)

    pf = bprime_preflight()
    pf_file = LOGS / f"preflight_bprime_{run_id}.json"
    p0.atomic_write(pf_file, {"run_id": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "preflight": pf})
    bad = [k for k, v in pf["checks"].items() if not v["ok"]]

    if dry:  # 読み取り専用: 状態ファイルには触れない
        p0.log(f"B′ 事前確認 {'OK' if pf['ok'] else 'NG ' + str(bad)}（--dry-preflight: 状態ファイル不変）→ {pf_file.name}")
        return 0 if pf["ok"] else 3

    st = p0.State()
    done = st.succeeded(STAGE)
    if done and not force:
        p0.log(f"B′ 診断は成功済み（run_id={done[-1]}）。再実行は --force（承認が必要）")
        return 0
    s = {"run_id": run_id, "stage": STAGE, "profile": lp.PROFILE_B_PRIME, "status": "preflight",
         "started": time.strftime("%Y-%m-%d %H:%M:%S"), "generation": "NOT RUN", "preflight_file": pf_file.name,
         "preflight_ok": pf["ok"]}
    st.data["runs"][run_id] = s
    st.save()
    if not pf["ok"]:
        s["status"] = "blocked_preflight"
        s["blocked_by"] = bad
        st.save()
        p0.log(f"B′ 事前確認 NG: {bad} → 実行しない（自動再試行なし）")
        return 3

    # ---- 実行（診断 1 回）----
    report_path = LOGS / f"bprime_report_{run_id}.json"
    engine_log = LOGS / f"bprime_engine_{run_id}.log"
    cache = ROOT / ".cache"
    env = lp.build_env(lp.PROFILE_B_PRIME, os.environ)
    env.update({
        "ACESTEP_CHECKPOINTS_DIR": str(p0.MODELS),
        "ACESTEP_TMPDIR": str(cache / "acestep" / "tmp"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "HF_HOME": str(cache / "hf"),
    })
    p0.ABORT_FREE_RAM_GB = lp.HARD_ABORT_FLOOR_GB  # 測定モード: 空き RAM がこれ未満で所有プロセスのみ停止
    py = ROOT / ".venv-engine" / "Scripts" / "python.exe"
    logf = open(engine_log, "wb")
    t0 = time.time()
    proc = subprocess.Popen([str(py), "-I", str(WRAPPER), "--diagnose-init", "--report", str(report_path)],
                            cwd=str(p0.ENGINE), env=env, stdout=logf, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    s.update({"status": "running", "owned_pid": proc.pid, "engine_log": engine_log.name,
              "report_file": report_path.name})
    st.save()
    p0.log(f"B′ 診断を開始 PID={proc.pid}（音楽生成は行わない）")
    mon = p0.Monitor(proc.pid, "bprime_diag")
    mon.start()
    rc = 1
    try:
        while True:
            if report_path.exists():
                break
            if proc.poll() is not None:
                break
            if mon.pressure:
                raise RuntimeError("MemoryPressure（空き RAM が下限未満）")
            if time.time() - t0 > RUN_TIMEOUT_S:
                raise RuntimeError("Timeout")
            time.sleep(2)
        # ラッパーは診断後に自分で終了する。最大 60 秒待つ
        for _ in range(30):
            if proc.poll() is not None:
                break
            time.sleep(2)
        s["wall_seconds"] = round(time.time() - t0, 1)
        if report_path.exists():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            s["report_status"] = rep.get("status")
            s["verify_int8_ok"] = bool((rep.get("verify_int8") or {}).get("ok"))
            s["report_summary"] = {k: rep.get(k) for k in ("startup_seconds", "placement")}
            rc = 0 if rep.get("status") == "ok" and s["verify_int8_ok"] else 1
        else:
            s["report_status"] = "missing"
            s["child_exit_code"] = proc.poll()
    except Exception as e:
        s["error"] = f"{type(e).__name__}: {e}"
        p0.log(f"ERROR {s['error']}")
    finally:
        mon.stop_flag = True
        s["perf"] = mon.snapshot_and_reset()
        s["pressure_abort"] = bool(mon.pressure)
        t_stop = time.time()
        s["stop"] = p0.stop_owned(proc)
        s["stop_seconds"] = round(time.time() - t_stop, 1)
        logf.close()
        time.sleep(5)
        s["port_released"] = not p0.port_in_use(p0.PORT)
        used, _ = p0.gpu_mem_mib()
        s["gpu_after_stop_used_mib"] = used
        s["orphan_pids"] = [x for x in s["stop"]["pids"] if psutil.pid_exists(x)]
        s["residual_after"] = residual_procs()
        s["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        clean = s["port_released"] and not s["orphan_pids"] and not s["residual_after"]
        s["shutdown_clean"] = clean
        s["status"] = "ok" if (rc == 0 and clean and "error" not in s) else "failed"
        st.save()
        p0.log(f"停止: port_released={s['port_released']} gpu_used={used}MiB orphans={s['orphan_pids']} "
               f"→ status={s['status']}")
    return 0 if s["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
