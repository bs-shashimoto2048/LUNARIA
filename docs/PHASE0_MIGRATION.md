# PHASE0_MIGRATION — 開発環境の移行記録

作成日: 2026-10-22

## 概要

| 項目 | 内容 |
|---|---|
| 移行元 | 旧開発先 `C:\dev\LUNARIA`（**削除せず保全**） |
| 移行先 | `<PROJECT_ROOT>`（Claude Code 起動ディレクトリ直下の `LUNARIA`。実パスは公開しない） |
| 方式 | robocopy によるコピー（移動・削除なし）。ログ: `data/logs/migration_robocopy.log` |
| robocopy 終了コード | 本体=3（コピー＋移行先のみのファイル＝ログ自身）、その他=1（正常コピー）。8 以上なし |

## コピー対象と除外

| 対象 | 扱い |
|---|---|
| 本体（scripts / docs / data / .git / .gitignore / 上流 ACE-Step ツリー） | コピー（1251 ファイル、件数・容量一致） |
| `.tools`（uv 本体、uv 管理 Python 3.11.17、uv-cache） | コピー（35326 ファイル、7.096 GB、件数・容量一致）。wheel の再取得を避けるため |
| `.venv-engine` | **コピーしない**。絶対パス依存のため移行先で再作成 |
| モデル | 必要な 4 種のみコピー（下記） |
| `engine/ACE-Step-1.5/checkpoints`（旧） | コピーしない。移行先では `models` へのジャンクションに置換 |
| 旧 `models`（9.4GB、0.6B LM なし・1.7B あり） | コピーしない（重複・不完全） |
| `acestep-5Hz-lm-1.7B`（3.5GB） | コピーしない（使用禁止方針。旧環境に残存） |

### モデルの取捨選択

旧環境は 2 か所にモデルがあった（`models` と上流 `checkpoints`）。
`checkpoints` 側が 4 種すべてを含む完全版だったため、そこから以下のみ `models/` へコピーした。

- `acestep-v15-turbo`（DiT）
- `acestep-5Hz-lm-0.6B`（LM）
- `vae`
- `Qwen3-Embedding-0.6B`

## 整合性確認

- 本体・`.tools`・各モデルのファイル数と総容量が移行元と一致
- 重みファイルの SHA256 が移行元と一致（turbo / 0.6B LM / vae / Qwen3-Embedding）
- 上流 ACE-Step の HEAD が `ca1e85fe9430179831e6bc6be790c332190a3866` で一致、上流ツリーは無変更（`git status` 空）
- LUNARIA 本体のリポジトリは移行先で正常（remote 未設定、commit なし）

## 設計上の判断

### checkpoints をジャンクションにした理由

上流 `acestep/api/startup_model_init.py:64` は `os.path.join(project_root, "checkpoints")`
（`project_root` は `api_server.py` の `__file__` 基準）を固定で使い、`ACESTEP_CHECKPOINTS_DIR` を参照しない。
一方 `model_downloader.py` は `ACESTEP_CHECKPOINTS_DIR` を参照する。
このため旧環境では取得先（`models`）と読込先（`engine/.../checkpoints`）が分かれて二重取得になっていた。

上流を改変せずに 1 か所へ統一するため、

```text
engine\ACE-Step-1.5\checkpoints  ==(junction)==>  models\
```

とした（`mklink /J`、管理者権限不要）。実体は `models/` のみ。
ジャンクションは `.gitignore`（`engine/ACE-Step-1.5/`）で追跡対象外。
**注意:** ジャンクションを `rmdir` ではなく再帰削除すると実体が消える。外すときは `rmdir` のみ。

### 仮想環境の再作成

```powershell
$d = '<LUNARIA ルート>'
$env:UV_CACHE_DIR = "$d\.tools\uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$d\.tools\python"
$env:UV_PROJECT_ENVIRONMENT = "$d\.venv-engine"
$env:UV_PYTHON_PREFERENCE = 'only-managed'
cd "$d\engine\ACE-Step-1.5"
& "$d\.tools\uv\uv.exe" sync --python 3.11 --frozen --offline
```

`uv.lock` を固定（`--frozen`）し、オフラインで再現。PyTorch 2.7.1+cu128 / Python 3.11.17 は不変。

### パス修正

- `scripts/phase0_run.py`: ルートを `Path(__file__)` から導出（絶対パス定数を廃止）
- `scripts/validate_wav.py`: パス依存なし
- 旧 `C:\dev\LUNARIA` を参照するファイルは `scripts/` 内に残っていない

## 移行元の保全

`C:\dev\LUNARIA` は変更・削除していない。不要になった時点でユーザー判断で整理する。
（旧環境の `models` と `checkpoints` の重複 約 20GB、旧 `.venv-engine` 7.3GB が残っている。）
