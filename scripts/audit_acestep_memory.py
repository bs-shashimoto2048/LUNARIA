"""ACE-Step 固定 commit のメモリ管理に関する前提を、ソースの静的検査だけで確認する診断スクリプト。

GPU・モデル・ネットワークは使わない（読み取りのみ）。上流を更新したときに、
docs/PHASE0_LOW_MEMORY_DESIGN.md の前提が崩れていないかを検知する用途。

使い方: python -I scripts/audit_acestep_memory.py   （終了コード 0=全項目が想定どおり / 1=想定と異なる項目あり）
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UP = ROOT / "engine" / "ACE-Step-1.5"
PKG = UP / "acestep"
EXPECTED_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"


def read(rel):
    return (PKG / rel).read_text(encoding="utf-8", errors="replace")


def find(rel, pattern, flags=0):
    """最初に一致した行番号を返す（無ければ None）。"""
    for i, line in enumerate(read(rel).splitlines(), 1):
        if re.search(pattern, line, flags):
            return i
    return None


def tier_block(name):
    """gpu_config.py の GPU_TIER_CONFIGS から tier ブロックのテキストを取り出す。"""
    text = read("gpu_config.py")
    m = re.search(rf'"{name}":\s*\{{(.*?)\n    \}},', text, re.S)
    return m.group(1) if m else ""


def tree_has(pattern, globs=("**/*.py",)):
    rx = re.compile(pattern)
    for g in globs:
        for p in PKG.glob(g):
            if p.name.endswith("_test.py"):
                continue
            if rx.search(p.read_text(encoding="utf-8", errors="replace")):
                return str(p.relative_to(PKG))
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    checks = []

    def add(item, ok, detail):
        checks.append({"item": item, "ok": bool(ok), "detail": detail})

    try:
        sha = subprocess.check_output(["git", "-C", str(UP), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    add("上流 commit が固定値と一致", sha == EXPECTED_COMMIT, sha)

    # 1. Tier 判定（<= 8GB が tier3）
    ln = find("gpu_config.py", r"elif gpu_memory_gb <= 8:")
    add("Tier 判定: VRAM <= 8GB は tier3（8.0GB の実機は tier3）", ln is not None, f"gpu_config.py:{ln}")
    t3 = tier_block("tier3")
    add("tier3 既定: offload_to_cpu / offload_dit_to_cpu / quantization がすべて True",
        all(re.search(rf"{k}\":\s*True", t3) for k in ("offload_to_cpu_default", "offload_dit_to_cpu_default",
                                                       "quantization_default")),
        "GPU_TIER_CONFIGS['tier3']（ドキュメント上の既定値）")
    add("tier3 既定: LM は 0.6B のみ・LM 初期化 True",
        'init_lm_default": True' in t3 and 'available_lm_models": ["acestep-5Hz-lm-0.6B"]' in t3, "")

    # 2/4. API 経路は tier の offload_dit / quantization 既定値を使わない
    for rel in ("api/startup_model_init.py", "api/http/model_init_service.py"):
        text = read(rel)
        call = re.search(r"initialize_service\((.*?)\n\s*\)", text, re.S)
        has_q = bool(call and "quantization" in call.group(1))
        ln = find(rel, r'env_bool\("ACESTEP_OFFLOAD_DIT_TO_CPU", False\)')
        add(f"{rel}: DiT offload は環境変数のみ（既定 False。tier 既定を使わない）", ln is not None, f"{rel}:{ln}")
        add(f"{rel}: initialize_service に quantization を渡さない（= INT8 は API 経路で無効）", not has_q, rel)
    where = tree_has(r"ACESTEP_QUANT")
    add("INT8 を指定する環境変数は存在しない（ACESTEP_QUANT* 無し）", where is None, str(where))
    add("DiT 量子化の実装は存在（initialize_service の quantization 引数経由のみ）",
        find("core/generation/handler/init_service_orchestrator.py", r"quantization: Optional\[str\] = None") is not None,
        "init_service_orchestrator.py")
    add("INT8 の検証は torchao の import 可否のみ（compile 必須の検証なし）",
        "compile_model" in read("core/generation/handler/init_service_setup.py")
        and "import torchao" in read("core/generation/handler/init_service_setup.py"),
        "init_service_setup.py::_validate_quantization_setup")

    # 3/4. offload の対象と DiT 常駐条件
    ln = find("core/generation/handler/init_service_loader.py", r"Keeping main model on .* \(persistent\)")
    add("offload_to_cpu=True かつ DiT offload=False のとき DiT は GPU 常駐", ln is not None,
        f"init_service_loader.py:{ln}")
    ln = find("core/generation/handler/init_service_offload_context.py", r'model_name == "model" and not self\.offload_dit_to_cpu')
    add("DiT offload=True のときのみ DiT は生成中だけ GPU に載る（他モデルは常にコンテキスト内のみ）", ln is not None,
        f"init_service_offload_context.py:{ln}")

    # 5. LM のロード・解放
    ln = find("llm_inference.py", r"Loading LLM to \{self\.device\}")
    add("LM(pt) は生成コンテキスト内のみ GPU、終了後 CPU へ戻す（offload_to_cpu 時）", ln is not None,
        f"llm_inference.py:{ln}")
    routes = tree_has(r'@app\.(post|get|delete)\("/v1/(unload|release)', ("api/**/*.py",))
    add("LM / モデルを解放する REST ルートは存在しない", routes is None, str(routes))
    ln = find("api/llm_generation_inputs.py", r"llm_handler\.unload\(\)")
    add("LM の unload は MPS の cover 経路でのみ呼ばれる（CUDA では呼ばれない）", ln is not None,
        f"api/llm_generation_inputs.py:{ln}")

    # 6. LM なし経路
    ln = find("api/llm_readiness.py", r'ACESTEP_INIT_LLM=false')
    add("ACESTEP_INIT_LLM=false で LM を読み込まず、use_cot_* は自動で無効化される", ln is not None,
        f"api/llm_readiness.py:{ln}")

    # 7/8. VAE デコード
    add("VAE タイルデコードは API 既定で有効（use_tiled_decode=True）",
        find("api/http/release_task_models.py", r"use_tiled_decode: bool = True") is not None, "release_task_models.py")
    ln = find("core/generation/handler/generate_music_decode.py", r"if effective_free < 0\.5:")
    add("VAE の CPU フォールバック条件: 実効空き VRAM < 0.5GB", ln is not None,
        f"generate_music_decode.py:{ln}")
    ln = find("gpu_config.py", r"effective_free_bytes = device_free_bytes \+ pytorch_cache_free_bytes")
    add("実効空き VRAM = デバイス空き + (reserved - allocated)。他アプリの使用分は差し引かれる", ln is not None,
        f"gpu_config.py:{ln}")
    ln = find("core/generation/handler/memory_utils.py", r"return min\(128, max_chunk\)")
    add("GPU タイルのチャンクは空き 12GB 未満で 128 固定（env ACESTEP_VAE_DECODE_CHUNK_SIZE で上書き可）",
        ln is not None, f"memory_utils.py:{ln}")
    add("VAE を強制 CPU にする env: ACESTEP_VAE_ON_CPU",
        find("core/generation/handler/generate_music_decode.py", r'ACESTEP_VAE_ON_CPU') is not None, "")

    # 9. 解放処理（Windows では libc/malloc_trim は無効）
    ln = find("core/generation/handler/init_service_memory_basic.py", r'platform\.system\(\) == "Linux"')
    add("malloc_trim / mmap 閾値調整は Linux 限定（Windows では OS ヒープ縮小は行われない）", ln is not None,
        f"init_service_memory_basic.py:{ln}")

    # 10. 必要時のみ起動
    add("ACESTEP_NO_INIT=true でモデルを遅延ロード（初回リクエスト時）",
        find("api/startup_model_init.py", r'env_bool\("ACESTEP_NO_INIT", True\)') is not None, "startup_model_init.py")
    add("アイドル時の自動解放（idle unload）機構は存在しない", tree_has(r"idle_unload|ACESTEP_IDLE") is None, "")

    # ---- 初期化ピーク RAM の経路（docs/PHASE0_INIT_MEMORY_OPTIMIZATION.md の根拠）----
    loader = "core/generation/handler/init_service_loader.py"
    comp = "core/generation/handler/init_service_loader_components.py"
    orch = "core/generation/handler/init_service_orchestrator.py"
    add("[初期化] DiT の from_pretrained に device_map が無い（全重みが CPU 上に実体化される）",
        find(loader, r"AutoModel\.from_pretrained\(") is not None and "device_map" not in read(loader),
        f"{loader}:{find(loader, r'AutoModel.from_pretrained')}")
    ln_to = find(loader, r"self\.model = self\.model\.to\(device\)\.to\(self\.dtype\)")
    ln_q = find(loader, r"self\._apply_dit_quantization\(quantization\)")
    add("[初期化] INT8 は DiT を GPU へ移した後に適用される（CPU 上では量子化しない）",
        ln_to is not None and ln_q is not None and ln_to < ln_q, f"{loader}:{ln_to} -> :{ln_q}")
    o_dit = find(orch, r"self\._load_main_model_from_checkpoint\(")
    o_vae = find(orch, r"self\._load_vae_model\(")
    o_te = find(orch, r"self\._load_text_encoder_and_tokenizer\(")
    add("[初期化] ロード順序: DiT → VAE → Text Encoder",
        None not in (o_dit, o_vae, o_te) and o_dit < o_vae < o_te, f"{orch}:{o_dit},{o_vae},{o_te}")
    add("[初期化] Text Encoder は dtype 指定なしで読み込む（bf16 保存でも fp32 に展開）",
        "AutoModel.from_pretrained(text_encoder_path)" in read(comp), f"{comp}:{find(comp, 'AutoModel.from_pretrained')}")
    add("[初期化] offload 時、VAE / Text Encoder は CPU で _get_vae_dtype('cpu') = float32 へ変換される",
        '_get_vae_dtype("cpu")' in read(comp)
        and re.search(r'target_device == "cpu":\s*\n\s*return torch\.float32',
                      read("core/generation/handler/memory_utils.py")) is not None, "")
    add("[初期化] LM(pt) は dtype 指定なしで読み込み、その後 .to('cpu').to(dtype) で変換する",
        "AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)" in read("llm_inference.py")
        and 'self.llm = self.llm.to("cpu").to(self.dtype)' in read("llm_inference.py"),
        f"llm_inference.py:{find('llm_inference.py', 'AutoModelForCausalLM.from_pretrained')}")
    o_llm = find("api/startup_model_init.py", r"initialize_llm_at_startup\(")
    o_svc = find("api/startup_model_init.py", r"handler\.initialize_service\(")
    add("[初期化] LM は DiT / VAE / Text Encoder の初期化後に初期化される",
        None not in (o_llm, o_svc) and o_svc < o_llm, f"startup_model_init.py:{o_svc} -> :{o_llm}")
    add("[初期化] DiT のロード処理内で _release_system_memory が呼ばれない",
        "_release_system_memory" not in read(loader), loader)

    # transformers（固定版）のロード実装（site-packages を読むだけ）
    mu = ROOT / ".venv-engine" / "Lib" / "site-packages" / "transformers" / "modeling_utils.py"
    mu_text = mu.read_text(encoding="utf-8", errors="replace") if mu.exists() else ""
    add("[transformers] safetensors は safe_open(device=tensor_device) + get_slice で 1 パラメータずつ実体化",
        'safe_open(shard_file, framework="pt", device=tensor_device)' in mu_text and "file_pointer.get_slice(" in mu_text,
        "transformers/modeling_utils.py::_load_state_dict_into_meta_model")
    add("[transformers] device_map の int / str / torch.device は {'': device} に正規化され、dict は _no_split_modules なしで受理",
        "device_map = {\"\": device_map}" in mu_text and "elif device_map is not None:" in mu_text
        and "check_tied_parameters_on_same_device" in mu_text, "transformers/modeling_utils.py::from_pretrained / _get_device_map")
    add("[transformers] device_map が None なら param_device='cpu'（CPU 上に実体化して保持）",
        'if device_map is None:\n                param_device = "cpu"' in mu_text, "transformers/modeling_utils.py")
    add("[transformers] low_cpu_mem_usage は無視される（常に meta 初期化）",
        'kwargs.pop("low_cpu_mem_usage", None)' in mu_text, "transformers/modeling_utils.py")

    ok_all = all(c["ok"] for c in checks)
    for c in checks:
        print(f"[{'OK' if c['ok'] else 'NG'}] {c['item']}  ({c['detail']})")
    print(json.dumps({"all_ok": ok_all, "total": len(checks), "ng": [c['item'] for c in checks if not c['ok']]},
                     ensure_ascii=False))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
