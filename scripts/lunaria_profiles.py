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
    # 当初は 0.6（パラメータ 1 つ分の一時領域の見込み）。DiT 診断（run 20261022-164908）の観測で、クラッシュ前までに
    # 物理 RAM が約 0.86GB 低下したため 1.0 に引き上げた（観測値は途中までのため、実際はさらに大きい可能性がある）
    "dit_gpu_direct": 1.0,
    "text_encoder": 1.6,     # Qwen3-Embedding-0.6B bf16 1.12GB × 1.14 ≒ 1.28GB + 余裕
    "lm": 1.8,               # acestep-5Hz-lm-0.6B bf16 1.28GB × 1.14 ≒ 1.46GB + 余裕
}
STAGE_GUARD_FLOOR_GB = HARD_ABORT_FLOOR_GB  # 段階の確保後も、空き RAM をこの値以上残す


# --- DiT 単独の段階的 GPU 診断の許可条件（docs/PHASE0_DIT_DIAGNOSTIC.md）-------------------------
# 条件: 実行開始前の空き RAM >= 見込み消費量 + 安全余裕 3GB（安全下限は引き下げない）。
# 見込み消費量は「実測済みの成分は実測値、未測定の成分は上限見積り」で積む（推定だけで許可しない）。
IMPORT_FULL_PRIVATE_GB = 2.553   # E1 実測: torch + transformers + diffusers + acestep.api_server の import（private）
CUDA_HOST_UPPER_GB = 1.0         # CUDA コンテキストのホスト側 private の上限見積り（未測定。B2 の実測値で置換する）
DIT_FILE_VRAM_MIB = 4570         # DiT bf16 の重み 4.46GiB（実測のファイルサイズ）
CUDA_CTX_VRAM_UPPER_MIB = 1024   # CUDA コンテキストの VRAM 上限見積り（未測定。B2 の実測値で置換する）
SAFETY_FLOOR_GB = HARD_ABORT_FLOOR_GB  # 3.0
# DiT 診断（run 20261022-164908、未量子化 DiT の GPU 直接ロードで torch_cpu.dll がアクセス違反）の観測:
#   ロード開始と同時にプロセスの commit が CUDA コンテキスト後の 6.5GB から 20.3GB へ急増し、その後 11.2GB（+4.7GB）へ落ちた。
#   急増分（+13.8GB）は DiT ファイル（約 4.6GB）の約 3 倍、低下分は約 2 倍、残りは約 1 倍。
#   CPU のみの実測（E5）で、safetensors の 1 マッピングごとに commit がファイルサイズ分計上されることを確認済み（機構は PHASE0_COMMIT_SPIKE）。
#   原因（mmap の重複 / CUDA 確保 / 両方）は未確定のため、どの DiT ロード方式の許可条件にも、この観測値を使う。
DIT_COMMIT_PEAK_OBSERVED_GB = 13.8
# 同診断の B2（import + CUDA コンテキスト）の実測。以降の許可条件の既定値に使う（同じ構成の再測定で置き換わる）。
CUDA_BASE_PHYS_MEASURED_GB = 3.40     # 物理消費（import 0.81 + CUDA コンテキスト 約 2.6）
CUDA_BASE_PRIVATE_MEASURED_GB = 6.38  # private（commit 量）


def dit_diag_admission(step, measured=None, avail_gb=None, commit_headroom_gb=None, free_vram_mib=None):
    """DiT 単独診断の各ステップ（b1 / b2 / cd）を実行してよいかを判定する。

    step: "b1" = import のみ / "b2" = import（handler のみ）+ CUDA コンテキスト / "cd" = DiT の GPU 直接ロード + INT8
    measured: 直前ステップの実測値 {"b2_private_after_cuda_gb": float, "ctx_vram_mib": int} があれば置換に使う
    """
    # 空き RAM の判定には「物理消費」（available の低下量と RSS 増加の大きい方）を使う。他プロセスへの影響は物理の空きで決まるため。
    # commit 余裕の判定には private（commit 量）を使う。private は numpy 等の予約分を含み物理消費を過大評価しうる。
    # 実測が無い成分は、private ベースの E1 実測（2.553GB）や上限見積りを使う（保守側）。
    measured = measured or {}
    if step == "b1":
        ram, commit = IMPORT_FULL_PRIVATE_GB, IMPORT_FULL_PRIVATE_GB
        basis = "import は E1 実測（private 2.553GB。物理消費は未測定のため保守側の値を使用）"
        vram_need = 0
    elif step == "b2":
        base_ram, base_commit = CUDA_BASE_PHYS_MEASURED_GB, CUDA_BASE_PRIVATE_MEASURED_GB
        ram, commit = base_ram, base_commit
        basis = f"import + CUDA コンテキストは前回診断の実測値（物理 {base_ram}GB / private {base_commit}GB）"
        vram_need = CUDA_CTX_VRAM_UPPER_MIB + 256
    elif step == "cd":
        dit_upper = STAGE_NEED_GB["dit_gpu_direct"]
        base_ram = measured.get("b2_phys_after_cuda_gb", CUDA_BASE_PHYS_MEASURED_GB)
        base_commit = measured.get("b2_private_after_cuda_gb", CUDA_BASE_PRIVATE_MEASURED_GB)
        tag = "今回の B2 の実測値" if "b2_phys_after_cuda_gb" in measured else "前回診断の B2 の実測値"
        # commit は観測した急増（+13.8GB）を下回らないようにする（推定の上限ではなく実際に観測した値）
        ram, commit = base_ram + dit_upper, base_commit + max(dit_upper, DIT_COMMIT_PEAK_OBSERVED_GB)
        basis = (f"import + CUDA は{tag}（物理 {base_ram:.2f}GB / private {base_commit:.2f}GB）+ DiT の物理上限 {dit_upper}GB / "
                 f"commit は観測した急増 +{DIT_COMMIT_PEAK_OBSERVED_GB}GB を使用")
        ctx = measured.get("ctx_vram_mib", 861)  # B2 の実測（MiB）。未取得なら前回の実測値
        vram_need = DIT_FILE_VRAM_MIB + ctx + VRAM_MARGIN_MIB
    elif step == "d0":
        # D0: モデルなしの合成 CUDA 確保（最大 1GiB）+ 合成 safetensors（256MB）の GPU 直接読み込み。GPU の commit 計上の切り分け用
        base_ram = measured.get("b2_phys_after_cuda_gb", CUDA_BASE_PHYS_MEASURED_GB)
        base_commit = measured.get("b2_private_after_cuda_gb", CUDA_BASE_PRIVATE_MEASURED_GB)
        ram = base_ram + 0.5
        commit = base_commit + 3.0  # 確保 1GiB + マッピング 256MB が最大 3 倍に計上されうるとして、保守側に 3GB
        basis = f"import + CUDA は実測 + 合成テストの物理 0.5GB / commit は最大 3 倍計上を想定して +3.0GB"
        vram_need = measured.get("ctx_vram_mib", 861) + 1024 + 256 + VRAM_MARGIN_MIB
    else:
        raise ValueError(step)
    cons = ram
    out = {"step": step, "consumption_gb": round(cons, 2), "commit_consumption_gb": round(commit, 2), "basis": basis,
           "safety_floor_gb": SAFETY_FLOOR_GB,
           "required_free_ram_gb": round(cons + SAFETY_FLOOR_GB, 2),
           "required_commit_headroom_gb": round(commit + MIN_COMMIT_MARGIN_GB, 2),
           "required_free_vram_mib": int(vram_need)}
    if avail_gb is not None:
        out["checks"] = {
            "free_ram": {"ok": avail_gb >= out["required_free_ram_gb"],
                         "detail": f"{avail_gb:.2f}GB >= {out['required_free_ram_gb']}GB"
                                   f"（消費 {out['consumption_gb']} + 余裕 {SAFETY_FLOOR_GB}）"},
            "commit_headroom": {"ok": commit_headroom_gb is not None
                                and commit_headroom_gb >= out["required_commit_headroom_gb"],
                                "detail": f"{commit_headroom_gb}GB >= {out['required_commit_headroom_gb']}GB"},
            "free_vram": {"ok": free_vram_mib is not None and free_vram_mib >= vram_need,
                          "detail": f"{free_vram_mib}MiB >= {int(vram_need)}MiB"},
        }
        out["ok"] = all(c["ok"] for c in out["checks"].values())
    return out


# --- D0b（import のメモリ消費の切り分け。モデルなし）の許可条件 ------------------------------------
# 診断対象ごとのメモリ予算が不明なので、過去の最大実測値を基準に保守的に判定する:
#   import のみ（acestep.api_server まで）: 物理 0.66〜0.81GB / commit 2.55GB（B1・B2）
#   torch のみ + CUDA 初期化（D0）: 物理 0.63GB / private 約 1.9GB
#   重い構成 + CUDA 初期化（前回 B2）: 物理 3.40GB / private 6.38GB / VRAM 861MiB
D0B_HEAVY_TOKENS = ("transformers", "transformers_auto", "accelerate", "diffusers", "torchaudio", "torchao", "triton",
                    "acestep.handler", "acestep.llm_inference", "acestep.api_server", "acestep.gpu_config",
                    "peft", "lycoris", "lightning", "acestep.training", "acestep.core.generation.handler")


# 前回 B2 が実際に読み込んだ範囲（acestep.handler の連鎖に含まれる torchaudio・torchao 等も B2 で読み込まれた）
D0B_B2_TOKENS = frozenset({"torch", "accelerate", "diffusers", "transformers", "transformers_auto", "torchaudio", "torchao",
                           "acestep.gpu_config", "acestep.handler", "cuda", "safetensors", "numpy", "soundfile", "loguru",
                           "tqdm", "peft", "lycoris", "lightning", "acestep.training", "acestep.core.generation.handler",
                           "cuda_work", "flash_attn"})  # 末尾は、B2 の連鎖（acestep.handler の eager な学習用モジュール、peft 経由の flash_attn）に含まれるもの + 小さな GPU 処理


def d0b_admission(spec, avail_gb=None, commit_headroom_gb=None, free_vram_mib=None):
    """D0b の 1 構成（トークン列）を実行してよいか。見込み消費量 + 安全余裕 3GB（commit は + 4GB）。"""
    tokens = [t for t in spec.split(",") if t]
    cuda = "cuda" in tokens
    # torch の内部モジュール（torch._dynamo など。原因の切り分け用）は、B2 の連鎖の中で読み込まれていたものとして重い構成に含める
    heavy = any(t in D0B_HEAVY_TOKENS or t.startswith("torch.") for t in tokens)
    # D0b の実測: flash_attn（flash_attn_2_cuda）を CUDA 初期化より前にロードすると、CUDA 初期化が +4.0GB（物理 +2.7GB）になる。
    # CUDA 初期化より前の flash_attn は重い構成として扱う（B2 では peft 経由でロードされていた = B2 の範囲内）
    flash_before_cuda = cuda and "flash_attn" in tokens and tokens.index("flash_attn") < tokens.index("cuda")
    heavy = heavy or flash_before_cuda
    flash = "flash_attn" in tokens and not flash_before_cuda  # CUDA 初期化の後の flash_attn は、ロード自体の追加（未測定）のみ
    if cuda and heavy and all(t in D0B_B2_TOKENS or t.startswith("torch.") for t in tokens):
        # 前回 B2（torch + accelerate + diffusers + transformers + acestep.handler + CUDA）の部分集合。
        # 読み込むものが B2 の範囲内なので、B2 の実測値（物理 3.40GB / private 6.38GB / ピーク private 6.54GB）が上限になる見込み
        ram, commit, basis = 3.40 + 0.3, 6.54 + 0.6, "前回 B2 の実測値が上限になる部分集合（物理 3.40GB / ピーク private 6.54GB）+ 余裕"
    elif cuda and heavy:
        ram, commit, basis = 3.40 + 0.5, 6.38 + 1.0, "前回 B2 の最大実測（物理 3.40GB / private 6.38GB）+ 余裕（B2 の範囲外の読み込みを含む）"
    elif cuda:
        ram, commit, basis = 1.0, 2.5, "D0 の実測（物理 0.63GB / private 約 1.9GB）+ 余裕"
    elif heavy:
        ram, commit, basis = 1.5, 3.5, "import のみの最大実測（物理 0.81GB / commit 2.55GB）+ 余裕"
    else:
        ram, commit, basis = 1.0, 2.5, "軽い import（torch 以下）+ 余裕"
    if "cuda_work" in tokens and heavy:
        # 実測（D0b 第 2 段 W1）: acestep.handler を import した後の最初の GPU 処理（cuBLAS・cuDNN・SDPA）で、空き RAM が
        # 開始時 8.33GB → 2.94GB（-5.4GB）まで低下し、なお増加中に子プロセスが下限で自己停止した。同じ処理は import なしでは物理 0.18GB。
        # 最終的な消費は不明なので、観測値に未観測分の上乗せ（+1.5GB）をして判定する。
        ram, commit = 5.4 + 1.5, 9.5 + 1.0
        basis = "W1 の実測（重い連鎖の後の最初の GPU 処理で物理 -5.4GB 以上、増加中に停止）+ 未観測分の上乗せ"
    if flash:  # flash_attn の拡張モジュールの読み込みは未測定のため、さらに上乗せ
        ram, commit = ram + 1.0, commit + 1.5
        basis += " + flash_attn（未測定のため +1.0GB / +1.5GB）"
    vram = 0 if not cuda else 861 + 256 + 512
    out = {"spec": spec, "consumption_gb": round(ram, 2), "commit_consumption_gb": round(commit, 2), "basis": basis,
           "safety_floor_gb": SAFETY_FLOOR_GB, "required_free_ram_gb": round(ram + SAFETY_FLOOR_GB, 2),
           "required_commit_headroom_gb": round(commit + MIN_COMMIT_MARGIN_GB, 2), "required_free_vram_mib": vram}
    if avail_gb is not None:
        out["checks"] = {
            "free_ram": {"ok": avail_gb >= out["required_free_ram_gb"],
                         "detail": f"{avail_gb:.2f}GB >= {out['required_free_ram_gb']}GB（消費 {out['consumption_gb']} + 余裕 {SAFETY_FLOOR_GB}）"},
            "commit_headroom": {"ok": commit_headroom_gb is not None and commit_headroom_gb >= out["required_commit_headroom_gb"],
                                "detail": f"{commit_headroom_gb}GB >= {out['required_commit_headroom_gb']}GB"},
            "free_vram": {"ok": vram == 0 or (free_vram_mib is not None and free_vram_mib >= vram),
                          "detail": f"{free_vram_mib}MiB >= {vram}MiB"},
        }
        out["ok"] = all(c["ok"] for c in out["checks"].values())
    return out


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
