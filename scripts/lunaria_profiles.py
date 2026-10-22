"""LUNARIA 実行プロファイル定義と、プロファイル別の実行許可条件。

- A        : 現行構成（0.6B LM + bf16 DiT 常駐）。保護条件は従来どおり空き RAM 14GB 以上
- B_PRIME  : 0.6B LM + INT8 DiT（GPU 常駐）+ GPU VAE 優先。LUNARIA 側の起動ラッパーで INT8 を適用
- C        : LM なし。**補助的な診断手段として定義だけ保持**（現時点では実行しない）

許可条件の数値は「実測の根拠」を明記する。推定値だけで安全条件を確定しない。
"""

PROFILE_A = "A"
PROFILE_B_PRIME = "B_PRIME"
PROFILE_C = "C"

# --- B′ の根拠となる実測値（docs/PHASE0_GPU_BASELINE.md / data/logs の実測）---------------
# B′ 固有の実測は未取得。DiT の bf16 重みを先に読み込んでから INT8 化する（量子化は GPU 上で
# 読み込み後に適用される）ため、初期化経路は A と共通。A の初期化時の実測を上限として用いる。
MEASURED_A_INIT_PEAK_RSS_GB = 9.99       # A の初期化時プロセスツリー RSS ピーク（実測）
MEASURED_A_INIT_MIN_FREE_RAM_GB = 0.81   # A の初期化時に空き RAM が落ちた最小値（開始時は約 8GB）
MEASURED_A_INIT_PEAK_VRAM_OURS_MIB = 7186 - 1756  # 初期化時 GPU 使用 7186MiB − 他アプリ 1756MiB
MEASURED_A_MAX_RSS_GB = 11.74            # 生成中を含む A の RSS 最大

A_MIN_FREE_RAM_GB = 14.0                 # 従来構成 A の保護条件（維持）

PROTECT_FLOOR_GB = 2.0                   # OS・他プロセス用に常に残す空き RAM
HARD_ABORT_FLOOR_GB = 3.0                # 測定モード中、これを下回ったら所有プロセスのみ正常停止
MIN_COMMIT_MARGIN_GB = 4.0
VRAM_MARGIN_MIB = 512


# --- 段階ごとの RAM ガード（低ピーク・ロード中に、各段階の確保前に空き RAM を確認する）----------
# 必要量 = bf16 チェックポイントのサイズ × 1.14（実測: 512MB の safetensors 実体化で private +585MB）+ 余裕。
# これは「その段階の確保前に安全に失敗させる」ための保護で、preflight の閾値（上の b_prime_admission）とは別物。
# preflight の閾値は、初期化時の RAM 消費の改善が実機で確認されるまで引き下げない。
STAGE_NEED_GB = {
    "dit_gpu_direct": 0.6,   # GPU 直接ロードでは CPU 側はパラメータ 1 つ分の一時領域のみ（最大テンソル未測定のため余裕を見る）
    "text_encoder": 1.6,     # Qwen3-Embedding-0.6B bf16 1.12GB × 1.14 ≒ 1.28GB + 余裕
    "lm": 1.8,               # acestep-5Hz-lm-0.6B bf16 1.28GB × 1.14 ≒ 1.46GB + 余裕
}
STAGE_GUARD_FLOOR_GB = HARD_ABORT_FLOOR_GB  # 段階の確保後も、空き RAM をこの値以上残す


def b_prime_admission():
    """B′ の初期化診断（測定モード）に必要な空き資源。根拠は A の初期化時の実測。"""
    peak = MEASURED_A_INIT_PEAK_RSS_GB
    margin = max(1.0, peak * 0.15)
    return {
        "mode": "measurement",
        "basis": "A の初期化ピーク RSS 実測値を上限として使用（B′ 固有の実測は未取得）",
        "peak_ram_gb": peak,
        "required_free_ram_gb": round(peak + margin + PROTECT_FLOOR_GB, 2),
        "required_commit_headroom_gb": round(peak + MIN_COMMIT_MARGIN_GB, 2),
        "required_free_vram_mib": MEASURED_A_INIT_PEAK_VRAM_OURS_MIB + VRAM_MARGIN_MIB,
        "hard_abort_free_ram_gb": HARD_ABORT_FLOOR_GB,
    }


# --- 環境変数（プロファイルごとに明示。ここに無いものは Auto = 上流の自動判定）---------------
COMMON_ENV = {
    "ACESTEP_API_HOST": "127.0.0.1",
    "ACESTEP_API_PORT": "8001",
    "ACESTEP_CONFIG_PATH": "acestep-v15-turbo",
    "ACESTEP_NO_INIT": "false",
    "HF_HUB_OFFLINE": "1",               # 再ダウンロード禁止
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}

# 上流が参照する、プロファイルで明示的に未設定にするキー（外部環境の混入を防ぐ）
UNSET_ENV = [
    "ACESTEP_OFFLOAD_DIT_TO_CPU",  # B′: DiT は GPU 常駐（既定 False）
    "ACESTEP_OFFLOAD_TO_CPU",      # Auto（<16GB で有効）。無条件に無効化しない
    "ACESTEP_LM_OFFLOAD_TO_CPU",   # Auto（offload_to_cpu に従う）
    "ACESTEP_COMPILE_MODEL",       # compile は使わない（INT8 は compile 不要。併用は未確認）
    "ACESTEP_VAE_ON_CPU",          # VAE は GPU 優先
    "ACESTEP_PROJECT_ROOT",
    "MAX_CUDA_VRAM",
]

PROFILES = {
    PROFILE_A: {
        "description": "現行構成: 0.6B LM + bf16 DiT 常駐（tier3 Auto）",
        "env": {**COMMON_ENV, "ACESTEP_INIT_LLM": "true", "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B"},
        "int8": False,
        "min_free_ram_gb": A_MIN_FREE_RAM_GB,
        "status": "measured",
    },
    PROFILE_B_PRIME: {
        "description": "0.6B LM + INT8 DiT（GPU 常駐）+ GPU VAE 優先。offload は Auto（Text Encoder / VAE / LM）",
        "env": {
            **COMMON_ENV,
            "ACESTEP_INIT_LLM": "true",
            "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B",
            # 判断: backend が Auto だと、INT8 で VRAM が空いた時点で vLLM が選ばれ KV キャッシュが VRAM を
            # 消費しうる（llm_inference.py: 空き VRAM >= 2.0GB で vLLM）。VAE 用 VRAM を確保するため pt に固定。
            # 上流 README も 6-8GB 帯は pt を推奨。
            "ACESTEP_LM_BACKEND": "pt",
        },
        "int8": True,
        "quantization": "int8_weight_only",
        "status": "estimated",  # 実測前。初回は測定モードでのみ実行
    },
    PROFILE_C: {
        "description": "LM なし + bf16 DiT（補助的な診断手段。現時点では実行しない）",
        "env": {**COMMON_ENV, "ACESTEP_INIT_LLM": "false"},
        "request_overrides": {"thinking": False, "use_cot_caption": False, "use_cot_language": False},
        "int8": False,
        "status": "defined_not_run",
    },
}


def build_env(profile_id, base_env):
    """プロファイルの環境変数を base_env に適用して返す（UNSET_ENV は削除）。"""
    env = dict(base_env)
    for k in UNSET_ENV:
        env.pop(k, None)
    env.update(PROFILES[profile_id]["env"])
    return env
