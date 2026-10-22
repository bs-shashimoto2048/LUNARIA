"""D0b の結果ファイルから、構成ごと・トークンごとの増分を表にして出力する（読み取りのみ）。

使い方: python -I scripts/summarize_d0b.py [d0b_<run_id>.json]   （省略時は最新）
"""
import json
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "data" / "logs"


def main():
    cands = sorted(p for p in LOGS.glob("d0b_2*.json") if p.name.count("_") == 1)  # d0b_<run_id>.json（子の結果は d0b_<run_id>_<name>.json）
    if "--all" in sys.argv:
        # 複数回の実行をまとめる（同じ構成名は新しい方を採用）
        R = {"run_id": "+".join(json.loads(p.read_text(encoding="utf-8"))["run_id"] for p in cands), "status": "merged",
             "variants": {}, "skipped": []}
        for p in cands:
            r = json.loads(p.read_text(encoding="utf-8"))
            R["variants"].update(r["variants"])
            R["skipped"] += [s for s in r["skipped"]]
        R["skipped"] = [s for s in R["skipped"] if s["name"] not in R["variants"]]
    else:
        path = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else cands[-1]
        R = json.loads(path.read_text(encoding="utf-8"))
    print(f"run_id={R['run_id']} status={R.get('status')} variants_run={len(R['variants'])} skipped={len(R['skipped'])}")
    print("\n--- 構成ごとの合計（プロセス開始から）")
    print(f"{'variant':34s} {'sec':>6} {'priv_MB':>8} {'commit_GB':>9} {'avail_drop':>10} {'cuda':>5} {'exit':>5} {'min_free':>8} {'min_commit':>10}")
    for n, v in R["variants"].items():
        r = v.get("result") or {}
        t = r.get("totals", {})
        sm = v.get("sys_monitor", {})
        print(f"{n:34s} {t.get('seconds_total', 0):6.1f} {t.get('private_mb_from_start', 0):8d} "
              f"{t.get('system_commit_gb_from_start', 0):9.2f} {t.get('avail_drop_gb_from_start', 0):10.2f} "
              f"{str(t.get('cuda_initialized_final')):>5} {str(v.get('exit_code')):>5} "
              f"{str(sm.get('min_avail_gb')):>8} {str(sm.get('min_commit_headroom_gb')):>10}")
    print("\n--- トークンごとの増分（その前の状態からの差）")
    print(f"{'variant':32s} {'token':20s} {'sec':>6} {'priv_MB':>8} {'win_max':>8} {'rss_MB':>7} {'commit_GB':>9} {'avail_dr':>8} {'vram_MB':>7} {'cuda':>5} {'newmod':>6} {'dll#':>5} {'dllMB':>6}")
    for n, v in R["variants"].items():
        for tk in (v.get("result") or {}).get("tokens", []):
            print(f"{n:32s} {tk['token']:20s} {tk['seconds']:6.2f} {tk['private_delta_mb']:8d} {tk['private_window_max_delta_mb']:8d} "
                  f"{tk['rss_delta_mb']:7d} {tk['system_commit_delta_gb']:9.2f} {tk['avail_drop_gb']:8.2f} "
                  f"{str(tk['vram_delta_mb']):>7} {str(tk['cuda_initialized_after']):>5} {tk['new_modules']:6d} "
                  f"{tk['new_dll_count']:5d} {tk['new_dll_total_mb']:6d}")
    print("\n--- 不許可で実行しなかった構成")
    for s in R["skipped"]:
        print(f"  {s['name']}: {s['reason']}")
    print("\n--- 異常終了・クラッシュ")
    bad = [(n, v) for n, v in R["variants"].items() if v.get("exit_code") not in (0, None) or v.get("abort") or v.get("orphans")]
    print("  なし" if not bad else "\n".join(f"  {n}: exit={v.get('exit_code')} abort={v.get('abort')}" for n, v in bad))


if __name__ == "__main__":
    main()
