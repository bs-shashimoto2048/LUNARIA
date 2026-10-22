"""D0b: import（と CUDA 初期化）のメモリ消費を、構成ごとに新しいプロセスで切り分ける診断ランナー。

    .venv-engine\\Scripts\\python.exe -I scripts\\d0b_diag.py --dry-run            # 各構成の許可判定のみ。子プロセスなし
    .venv-engine\\Scripts\\python.exe -I scripts\\d0b_diag.py                      # 許可される構成を、1 つずつ逐次実行
    .venv-engine\\Scripts\\python.exe -I scripts\\d0b_diag.py --only=C_torch_cuda  # 指定した構成だけ

- 実 DiT・LM・Text Encoder・VAE のロード、INT8 量子化、楽曲生成はしない（import と CUDA コンテキストの作成のみ）
- 各構成の直前に、その時点の空き RAM / commit 余裕 / VRAM で許可判定（見込み消費量 + 安全余裕 3GB、commit は + 4GB）。
  不許可の構成は実行せず記録する。子プロセスが異常終了・中断したら、そこで全体を止める（自動再試行なし）
- 子プロセスは自分が起動した PID のみ停止する。他プロジェクト・Chrome・WSL には触れない
- 結果は data/logs/d0b_<run_id>.json（Git 管理外）。実行ごとに一意の run_id
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dit_diag as dd  # noqa: E402  (SysMonitor / collect_crash_info / current_resources / step_a を再利用)
import lunaria_profiles as lp  # noqa: E402
import phase0_run as p0  # noqa: E402
import psutil  # noqa: E402

ROOT = p0.ROOT
LOGS = p0.LOGS
PROBE = ROOT / "scripts" / "import_probe.py"
TIMEOUT_S = 240

# (名前, トークン列)。各構成は新しいプロセスで実行する。順序の影響を見るため、同じライブラリを CUDA 初期化の前後で import する構成も含める。
VARIANTS = [
    ("A_python", "python"),
    ("B_torch", "torch"),
    ("C_torch_cuda", "torch,cuda"),
    ("E_torch_safetensors", "torch,safetensors"),
    ("E2_torch_safetensors_cuda", "torch,safetensors,cuda"),
    ("D_torch_transformers", "torch,transformers"),
    ("D2_transformers_then_cuda", "torch,transformers,cuda"),
    ("D3_cuda_then_transformers", "torch,cuda,transformers"),
    ("D4_transformers_auto_then_cuda", "torch,transformers_auto,cuda"),
    ("H_accelerate_then_cuda", "torch,accelerate,cuda"),
    ("I_diffusers_then_cuda", "torch,diffusers,cuda"),
    ("J_torchaudio_then_cuda", "torch,torchaudio,cuda"),
    ("K_torchao_then_cuda", "torch,torchao,cuda"),
    ("L_triton_then_cuda", "torch,triton,cuda"),
    ("M_np_sf_loguru_tqdm_then_cuda", "torch,numpy,soundfile,loguru,tqdm,cuda"),
    ("F1_acestep_gpu_config_then_cuda", "torch,acestep.gpu_config,cuda"),
    ("G_acestep_handler_then_cuda", "torch,acestep.handler,cuda"),          # handler 自体が eager に読む範囲（本番の下限）
    ("G2_cuda_then_acestep_handler", "torch,cuda,acestep.handler"),
    ("F2_llm_inference_then_cuda", "torch,acestep.llm_inference,cuda"),
    ("P_api_server_then_cuda", "torch,acestep.api_server,cuda"),            # 本番の入口（api_server）
    ("R_b2_replica_then_cuda", "torch,accelerate,diffusers,transformers,acestep.handler,cuda"),  # 前回 B2 と同じ構成
    ("X_cuda_then_flash_attn", "torch,cuda,flash_attn"),                    # flash_attn は未測定。最後に実行
    # --- 第 2 段: CUDA 初期化の増分が import 順序に依存する原因の切り分けと、「最初の実 GPU 処理」での再確認 ---
    ("W3_cuda_then_work", "torch,cuda,cuda_work"),                          # 基準: import なしの CUDA 初期化と最初の GPU 処理
    ("W1_handler_then_cuda_work", "torch,acestep.handler,cuda,cuda_work"),   # 本番の順序（import → CUDA）
    ("W2_cuda_then_handler_work", "torch,cuda,acestep.handler,cuda_work"),   # 逆順（CUDA → import）。初回の GPU 処理で戻らないか
    ("N1_peft_then_cuda", "torch,peft,cuda"),
    ("N2_lycoris_then_cuda", "torch,lycoris,cuda"),
    ("N3_lightning_then_cuda", "torch,lightning,cuda"),
    ("N4_training_libs_then_cuda", "torch,peft,lycoris,lightning,torchaudio,cuda"),
    ("N5_acestep_training_then_cuda", "torch,acestep.training,cuda"),
    ("N6_acestep_core_handler_then_cuda", "torch,acestep.core.generation.handler,cuda"),
    # --- 第 3 段: 因果の直接確認。peft を経由せず、flash_attn だけを CUDA 初期化の前に import する ---
    ("Y_flash_attn_then_cuda", "torch,flash_attn,cuda"),
]


def custom_variants(argv):
    """--custom=名前:トークン列 で、その場の構成を追加する（原因の切り分けで、解析結果に応じた構成を試すため）。複数可。"""
    out = []
    for a in argv:
        if a.startswith("--custom="):
            name, _, spec = a.split("=", 1)[1].partition(":")
            if not name or not spec:
                raise SystemExit(f"無効な --custom: {a}（形式: --custom=名前:トークン,トークン）")
            out.append((name, spec))
    return out


def pick_variants(argv):
    """--only=名前1,名前2,... を指定した順に実行する（指定なしは全構成）。--custom= の構成は末尾に追加する。"""
    cust = custom_variants(argv)
    if cust and not any(a.startswith("--only=") for a in argv):
        return cust
    only = []
    for a in argv:
        if a.startswith("--only="):
            only += [x for x in a.split("=", 1)[1].split(",") if x]
    if not only:
        return VARIANTS
    table = dict(VARIANTS)
    bad = [o for o in only if o not in table]
    if bad:
        raise SystemExit(f"無効な --only: {bad}（有効: {sorted(table)}）")
    return [(n, table[n]) for n in only] + cust


def run_variant(name, spec, run_id):
    out = LOGS / f"d0b_{run_id}_{name}.json"
    log = LOGS / f"d0b_{run_id}_{name}.log"
    env = dict(os.environ)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONFAULTHANDLER": "1", "HF_HUB_OFFLINE": "1"})
    env.pop("CUDA_VISIBLE_DEVICES", None)
    py = ROOT / ".venv-engine" / "Scripts" / "python.exe"
    before = {"resources": dd.current_resources(), "vram": dd.vram_state()}
    logf = open(log, "wb")
    t0 = time.time()
    proc = subprocess.Popen([str(py), "-I", "-X", "faulthandler", str(PROBE), "--spec", spec, "--out", str(out)],
                            cwd=str(p0.ENGINE), env=env, stdout=logf, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    mon = p0.Monitor(proc.pid, f"d0b_{name}")
    mon.phase = name
    mon.start()
    sysmon = dd.SysMonitor(LOGS / f"d0b_sys_{run_id}_{name}.jsonl")
    sysmon.start()
    info = {"name": name, "spec": spec, "owned_pid": proc.pid, "before": before, "result_file": out.name}
    try:
        while proc.poll() is None:
            if mon.pressure:
                info["abort"] = "外側の監視: 空き RAM または commit 余裕が下限未満（所有プロセスのみ停止）"
                break
            if time.time() - t0 > TIMEOUT_S:
                info["abort"] = "タイムアウト"
                break
            time.sleep(0.5)
    finally:
        sysmon.stop_flag = True
        info["sys_monitor"] = sysmon.summary()
        info["perf"] = mon.snapshot_and_reset()
        mon.stop_flag = True
        info["wall_seconds"] = round(time.time() - t0, 1)
        info["stop"] = p0.stop_owned(proc)
        logf.close()
        time.sleep(2)
        info["exit_code"] = proc.poll()
        info["after"] = {"resources": dd.current_resources(), "vram": dd.vram_state()}
        info["orphans"] = [x for x in info["stop"]["pids"] if psutil.pid_exists(x)]
        info["residual_after"] = [p for p in p0.residual_engine_procs()]
        if info["exit_code"] not in (0, 1, 3, None):
            info["access_violation"] = info["exit_code"] in (3221225477, -1073741819)
            info["crash_info"] = dd.collect_crash_info(t0, time.time(), log)
    try:
        info["result"] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    except Exception as e:
        info["result"] = {"read_error": type(e).__name__}
    return info


def main():
    dry = "--dry-run" in sys.argv
    variants = pick_variants(sys.argv)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    p0.RUN_ID = run_id
    p0.ABORT_FREE_RAM_GB = lp.SAFETY_FLOOR_GB
    LOGS.mkdir(parents=True, exist_ok=True)
    out_file = LOGS / (f"d0b_dry_{run_id}.json" if dry else f"d0b_{run_id}.json")
    R = {"run_id": run_id, "mode": "dry-run" if dry else "run", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "generation": "NOT RUN", "models": "NOT LOADED", "variants": {}, "admission": {}, "skipped": []}
    st = None
    if not dry:
        st = p0.State()
        st.data["runs"][run_id] = {"run_id": run_id, "stage": "D0B", "status": "running", "started": R["started"],
                                   "result_file": out_file.name, "generation": "NOT RUN"}
        st.save()

    def save():
        p0.atomic_write(out_file, R)

    A = dd.step_a()
    R["step_a"] = A
    save()
    if not A["static_ok"]:
        R["status"] = "blocked_step_a"
        save()
        p0.log("Step A NG → 停止")
        return finish(R, st, save, 3)

    rc = 0
    for name, spec in variants:
        r = dd.current_resources()
        a = lp.d0b_admission(spec, r["avail_gb"], r["commit_headroom_gb"], r["free_vram_mib"])
        R["admission"][name] = a
        save()
        if not a["ok"]:
            bad = [f"{k}: {c['detail']}" for k, c in a["checks"].items() if not c["ok"]]
            R["skipped"].append({"name": name, "reason": bad})
            p0.log(f"[{name}] 不許可のため実行しない: {bad}")
            continue
        if dry:
            p0.log(f"[{name}] 許可（必要 空き RAM {a['required_free_ram_gb']}GB / commit {a['required_commit_headroom_gb']}GB）")
            continue
        info = run_variant(name, spec, run_id)
        R["variants"][name] = info
        save()
        child = info.get("result") or {}
        p0.log(f"[{name}] status={child.get('status')} exit={info['exit_code']} {info['wall_seconds']}s "
               f"cuda={child.get('totals', {}).get('cuda_initialized_final')}")
        if child.get("status") != "ok" or info.get("abort") or info["orphans"] or info["residual_after"]:
            R["status"] = f"failed_at_{name}"
            p0.log(f"[{name}] 異常のため全体を停止（自動再試行なし）")
            rc = 1
            break
        time.sleep(1.0)
    else:
        R["status"] = "dry_run_done" if dry else ("ok" if not R["skipped"] else "ok_with_skipped")
    R.setdefault("status", "stopped")
    R["final_resources"] = dd.current_resources()
    return finish(R, st, save, rc)


def finish(R, st, save, rc):
    R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    R["runner_private_mb_end"] = round(psutil.Process().memory_info().private / 1048576)
    save()
    if st is not None:
        st.data["runs"][R["run_id"]].update({"status": R.get("status"), "finished": R["finished"]})
        st.save()
    p0.log(f"完了: status={R.get('status')} runner_private={R['runner_private_mb_end']}MB → {R['run_id']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
