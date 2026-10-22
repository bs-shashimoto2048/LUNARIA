# LUNARIA（ルナリア）

ACE-Step 1.5 を使い、音楽生成をローカル環境で完結させる Windows デスクトップアプリ。
将来は日本語の問題・回答から暗記用楽曲を生成する機能も実装する。

- 開発の正本: 開発引き継ぎ仕様書（非公開・開発者のローカルで保管。製品名は **LUNARIA**。旧仮称は使用しない）
- アプリアイコン: 三日月と星のモチーフ案を採用予定（正式素材の提供後に適用）

## 現在の状態

**Phase 0（GPU・ACE-Step 基礎検証）進行中。Stage A は未達**（メモリ不足により中断。詳細は `docs/PHASE0_GPU_BASELINE.md`）。
Electron / React / FastAPI の製品機能は未実装（Phase 1 以降）。

## 構成

```text
LUNARIA/
├─ apps/desktop/            （Phase 2）
├─ services/orchestrator/   （Phase 1）
├─ engine/ACE-Step-1.5/     上流 ACE-Step（git submodule・commit 固定・無改変）
├─ packages/contracts/
├─ models/                  モデル実体（Git 管理外）。engine/.../checkpoints はここへのジャンクション
├─ data/{database,outputs,projects,logs}/   Git 管理外（.gitkeep のみ追跡）
├─ scripts/                 phase0_run.py / validate_wav.py
├─ docs/                    PHASE0_*.md、ACE-Step の health / openapi 保存
├─ .tools/                  uv・uv 管理 Python・キャッシュ（Git 管理外）
└─ .venv-engine/            専用仮想環境（Git 管理外）
```

## ローカル開発環境

Windows 11 / RTX 4070 Laptop GPU（8GB）/ Python 3.11.17（uv）/ PyTorch 2.7.1+cu128 / ACE-Step `ca1e85f`。
構築・再現手順: `docs/PHASE0_SETUP.md`。移行記録: `docs/PHASE0_MIGRATION.md`。実測結果: `docs/PHASE0_GPU_BASELINE.md`。
低メモリ運用の設計と ACE-Step のメモリ管理の監査: `docs/PHASE0_LOW_MEMORY_DESIGN.md`。
プロファイル B′（INT8 + DiT GPU 常駐）の実装と診断準備: `docs/PHASE0_BPRIME_PROFILE.md`。
互換性・ライセンス（`PHASE0_COMPATIBILITY.md`）は未作成。ACE-Step 本体は MIT、モデルカードも MIT。

## 取得とセットアップ（概要）

```powershell
git clone --recurse-submodules https://github.com/bs-shashimoto2048/LUNARIA.git
# 既に clone 済みなら: git submodule update --init
```

- 上流 ACE-Step は submodule で **commit `ca1e85fe9430179831e6bc6be790c332190a3866` に固定**。上流を直接改変・更新しない
- モデル実体（約 7GB）・仮想環境・生成音源・ログは Git 管理外。モデルは `models/` に配置する
- `engine/ACE-Step-1.5/checkpoints` は `models/` へのジャンクション（`mklink /J`）。ローカルで作成する
- 仮想環境（Python 3.11 / PyTorch 2.7.1+cu128）の再現とジャンクション作成は `docs/PHASE0_SETUP.md`

## Git 運用

- `main` は安定版。通常の変更は `feature/*` `fix/*` `chore/*` ブランチから PR を作成し、原則 squash merge
- force push 禁止。モデル・生成音源・ログ・認証情報は登録しない

## 方針

- すべて `127.0.0.1` に限定。外部公開しない
- 歌詞・問題・回答の本文をログへ残さない。実際の社内試験データはテストに使わない
- 自分が起動したプロセスだけを停止する
- commit / push はユーザー承認後。リモートは未設定
