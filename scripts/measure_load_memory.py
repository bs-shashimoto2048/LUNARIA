"""モデルロード経路のメモリ挙動を、合成データ（CPU のみ・約 0.5GB）で実測する。GPU・実モデルは使わない。

確認すること（docs/PHASE0_INIT_MEMORY_OPTIMIZATION.md の根拠）:
  E1  ライブラリの import だけで private がどれだけ増えるか（初期化前のベースライン L。CUDA コンテキストは含まない）
  E2  safetensors を safe_open(device="cpu") + get_slice()[...] で実体化すると private が増えるか、
      del + gc 後に Windows で OS へ返却されるか（上流 transformers のロード方法の再現）
  E3  fp32 で読んだモジュールを .to(bfloat16) するときのピークと常駐（上流の LM / Text Encoder 経路の再現）

各実験は**別プロセス**で測定する（前の実験の解放遅延が混入しないようにするため）。

使い方: python -I scripts/measure_load_memory.py <作業ディレクトリ>
結果は標準出力に JSON。private = psutil の memory_info().private（コミット量。ファイルマップ分を含まない）。
"""
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU を一切使わない（torch の import 前に設定）

import psutil  # noqa: E402

P = psutil.Process()
MB = 1048576
N_TENSORS, TENSOR_ELEMS = 8, 32 * MB  # bf16 で 64MB × 8 = 512MB


def mem():
    mi = P.memory_info()
    return {"rss_mb": round(mi.rss / MB), "private_mb": round(mi.private / MB)}


class PeakSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak_private = 0
        self.stop = False

    def run(self):
        while not self.stop:
            self.peak_private = max(self.peak_private, P.memory_info().private)
            time.sleep(0.003)


def settle(sec=2.0):
    gc.collect()
    time.sleep(sec)


def e1():
    import warnings
    warnings.simplefilter("ignore")
    res = {"start": mem()}
    import torch  # noqa: F401
    res["after_import_torch"] = mem()
    import accelerate  # noqa: F401
    import diffusers  # noqa: F401
    import transformers  # noqa: F401
    res["after_import_transformers_diffusers"] = mem()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine" / "ACE-Step-1.5"))
    import acestep.api_server  # noqa: F401
    import acestep.handler  # noqa: F401
    import acestep.llm_inference  # noqa: F401
    settle()
    res["after_import_acestep_api_server"] = mem()
    return res


def e2gen(path):
    import torch
    from safetensors.torch import save_file
    save_file({f"w{i}": torch.randn(TENSOR_ELEMS, dtype=torch.bfloat16) for i in range(N_TENSORS)}, path)
    return {"file_mb": os.path.getsize(path) // MB}


def e2(path):
    import torch
    from safetensors import safe_open
    settle()
    b0 = mem()
    res = {"baseline_after_import_torch": b0}
    held = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for n in list(f.keys()):
            held.append(f.get_slice(n)[...])  # 上流（transformers 4.57 _load_state_dict_into_meta_model）と同じ実体化
    p1 = mem()
    res["after_materialize"] = p1
    res["materialize_private_delta_mb"] = p1["private_mb"] - b0["private_mb"]
    res["materialize_rss_delta_mb"] = p1["rss_mb"] - b0["rss_mb"]
    # 実体化したテンソルを触る（実際のロードでは .to(device) / dtype 変換で全要素を読む）
    s = 0.0
    for t in held:
        s += float(t[::4096].float().sum())
    res["after_touch"] = mem()
    del held
    settle(3.0)
    p2 = mem()
    res["after_del_gc"] = p2
    res["private_returned_to_os_mb"] = res["after_touch"]["private_mb"] - p2["private_mb"]
    return res


def e3():
    import torch
    import torch.nn as nn
    settle()
    b0 = mem()
    layers = nn.Sequential(*[nn.Linear(2048, 2048, bias=False) for _ in range(32)])  # fp32 で 512MB
    settle()
    p1 = mem()
    sampler = PeakSampler()
    sampler.start()
    layers.to(torch.bfloat16)  # 上流の LM: AutoModelForCausalLM(fp32) → .to("cpu").to(self.dtype=bf16) の再現
    sampler.stop = True
    sampler.join()
    settle(3.0)
    p2 = mem()
    res = {"baseline": b0, "fp32_resident": p1, "after_to_bf16": p2,
           "fp32_private_mb": p1["private_mb"] - b0["private_mb"],
           "bf16_private_mb": p2["private_mb"] - b0["private_mb"],
           "peak_private_during_convert_vs_baseline_mb": round(sampler.peak_private / MB) - b0["private_mb"]}
    del layers
    settle(3.0)
    res["after_del"] = mem()
    return res


def e4(path):
    """E4: 解放後に Windows 側で OS へ返す手段（ヒープ縮小・ワーキングセット縮小）が private を減らせるか。"""
    import ctypes
    import torch
    from safetensors import safe_open
    settle()
    b0 = mem()
    res = {"baseline": b0}
    held = []
    with safe_open(path, framework="pt", device="cpu") as f:
        for n in list(f.keys()):
            held.append(f.get_slice(n)[...])
    res["after_materialize"] = mem()
    del held
    settle(3.0)
    res["after_del_gc"] = mem()
    k32 = ctypes.windll.kernel32
    try:
        res["msvcrt__heapmin_result"] = int(ctypes.cdll.msvcrt._heapmin())
    except Exception as e:
        res["msvcrt__heapmin_result"] = f"ERR {type(e).__name__}"
    time.sleep(1.0)
    res["after__heapmin"] = mem()
    try:
        k32.GetProcessHeap.restype = ctypes.c_void_p
        k32.HeapCompact.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k32.HeapCompact.restype = ctypes.c_size_t
        res["HeapCompact_largest_free_block"] = int(k32.HeapCompact(k32.GetProcessHeap(), 0))
    except Exception as e:
        res["HeapCompact_largest_free_block"] = f"ERR {type(e).__name__}"
    time.sleep(1.0)
    res["after_HeapCompact"] = mem()
    try:
        ctypes.windll.psapi.EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception:
        pass
    time.sleep(1.0)
    res["after_EmptyWorkingSet"] = mem()
    res["private_returned_by_heapmin_mb"] = res["after_del_gc"]["private_mb"] - res["after__heapmin"]["private_mb"]
    res["private_returned_total_mb"] = res["after_del_gc"]["private_mb"] - res["after_EmptyWorkingSet"]["private_mb"]
    return res


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        mode = sys.argv[2]
        out = {"e1": e1, "e3": e3}.get(mode)
        if mode == "e4":
            out = lambda: e4(sys.argv[3])  # noqa: E731
        if mode == "e2gen":
            out = lambda: e2gen(sys.argv[3])  # noqa: E731
        elif mode == "e2":
            out = lambda: e2(sys.argv[3])  # noqa: E731
        print(json.dumps(out()))
        return
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    path = str(work / "synthetic.safetensors")
    result = {}
    for label, args in (("E1_import_baseline", ["e1"]), ("E2_generate_file", ["e2gen", path]),
                        ("E2_safetensors_materialize", ["e2", path]), ("E3_fp32_to_bf16", ["e3"]),
                        ("E4_windows_heap_trim", ["e4", path])):
        r = subprocess.run([sys.executable, "-I", __file__, "--child", *args], capture_output=True, text=True,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
        line = [x for x in r.stdout.splitlines() if x.startswith("{")]
        result[label] = json.loads(line[-1]) if line else {"error": r.stderr[-300:]}
    try:
        os.unlink(path)
    except OSError:
        pass
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
