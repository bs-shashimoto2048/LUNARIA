# PHASE0_SETUP — 環境構築・再現手順

対象: Windows 11 / PowerShell。ルートは `<LUNARIA>` と表記（Claude Code 起動ディレクトリ直下の `LUNARIA`）。

## 1. 前提

- Git、NVIDIA ドライバー（実測: 537.53）、C: に 約 25GB 以上の空き
- グローバル Python は変更しない。uv と Python 3.11 は `<LUNARIA>\.tools` に閉じる

## 2. 手順

```powershell
# 1) 上流 ACE-Step は submodule（commit ca1e85fe9430179831e6bc6be790c332190a3866 に固定。改変・更新しない）
git clone --recurse-submodules https://github.com/bs-shashimoto2048/LUNARIA.git
#   既に clone 済みなら: git submodule update --init

# 2) uv をプロジェクト内に導入（PATH を変更しない）
$env:UV_INSTALL_DIR = '<LUNARIA>\.tools\uv'; $env:UV_NO_MODIFY_PATH = '1'
irm https://astral.sh/uv/install.ps1 | iex

# 3) 専用仮想環境（Python 3.11 / PyTorch 2.7.1+cu128 は uv.lock で固定）
$d = '<LUNARIA>'
$env:UV_CACHE_DIR = "$d\.tools\uv-cache"; $env:UV_PYTHON_INSTALL_DIR = "$d\.tools\python"
$env:UV_PROJECT_ENVIRONMENT = "$d\.venv-engine"; $env:UV_PYTHON_PREFERENCE = 'only-managed'
cd "$d\engine\ACE-Step-1.5"
& "$d\.tools\uv\uv.exe" sync --python 3.11 --frozen

# 4) モデル（models/ に 4 種）。初回のみ取得（約 7GB）。1.7B / XL は使用しない
#    acestep-v15-turbo, acestep-5Hz-lm-0.6B, vae, Qwen3-Embedding-0.6B

# 5) 上流の固定参照先 checkpoints を models へ向ける（ジャンクション）
cmd /c mklink /J "$d\engine\ACE-Step-1.5\checkpoints" "$d\models"
```

### checkpoints をジャンクションにする理由

上流 `api/startup_model_init.py` は `<engine>/checkpoints` を固定で参照し、`ACESTEP_CHECKPOINTS_DIR` を無視する。
上流を改変せず、実体を `models/` の 1 か所に統一するための措置（詳細は `PHASE0_MIGRATION.md`）。
外すときは `rmdir`（再帰削除しない）。

## 3. 動作確認

```powershell
& "$d\.venv-engine\Scripts\python.exe" -I -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
# 事前確認のみ（エンジンは起動しない）
& "$d\.venv-engine\Scripts\python.exe" -I "$d\scripts\phase0_run.py" --stage A --dry-preflight
```

## 4. Stage の実行

```powershell
& "$d\.venv-engine\Scripts\python.exe" -I "$d\scripts\phase0_run.py" --stage A    # 30 秒インスト
# --stage B: 30 秒日本語歌詞 / --stage C: 60 秒日本語歌詞（承認後のみ実行）
# --force: 成功済み Stage／既存 WAV を無視して再生成
```

`phase0_run.py` の性質:

- 事前確認（commit、モデル、ポート、空き RAM ≥ 6GB、空き VRAM ≥ 4000MiB）を満たさないと起動しない
- エンジンは自プロセスの子としてのみ起動・停止（CTRL_BREAK → 所有 PID のみ terminate → 残存のみ kill）
- 実行中に空き RAM < 1.5GB が 3 秒続くと、自エンジンを正常停止して中断
- 実行ごとに一意の `run_id`（例: `20261022-143042-105372`）を付与。過去の実行記録は上書きしない
- 状態: `data/logs/phase0_state.json` の `runs.<run_id>` に原子的に逐次保存。run_id 導入前の記録は `legacy_stages_before_run_id` に退避
- `--dry-preflight` は読み取り専用。状態ファイルを変更せず、結果は `data/logs/preflight_<run_id>.json` に保存（終了コード 0=OK / 3=NG）
- テレメトリ: `data/logs/phase0_telemetry.jsonl`（1 秒間隔、`run_id` と工程名 `phase` を含む）
- エンジンログ: `data/logs/acestep_api_<run_id>.log`（実行ごとに分離。歌詞本文はステータス JSON に含めない）
- 成功済み Stage の再実行、既存 WAV の上書きはしない
- `HF_HUB_OFFLINE=1` で再ダウンロードを禁止

## 5. 推奨する実行環境（実測に基づく）

- 実行前の空き物理 RAM を **約 14GB 以上**（エンジンの最大 RSS 約 11.7GB ＋ 余裕）
- 実行前の空き VRAM を **約 3.5GB 以上**（他アプリの GPU 使用を減らす）
- Chrome 等の多プロセスアプリ、WSL を終了（ユーザー操作。自動終了はしない）

## 6. 環境変数（Auto 設定を維持）

`ACESTEP_INIT_LLM=true`, `ACESTEP_CONFIG_PATH=acestep-v15-turbo`, `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B`,
`ACESTEP_CHECKPOINTS_DIR=<LUNARIA>\models`, `ACESTEP_NO_INIT=false`, `HF_HUB_OFFLINE=1`。
backend / offload / quantization は未指定（上流の GPU tier 判定に委ねる）。
`start_api_server.bat` は起動時に自動更新確認を行うため使用しない。
