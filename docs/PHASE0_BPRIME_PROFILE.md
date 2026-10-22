# PHASE0_BPRIME_PROFILE — プロファイル B′（INT8 + DiT GPU 常駐）の実装と診断準備

作成日: 2026-10-22 / Issue: #3 / 状態: **実装・事前確認まで完了。実 GPU 診断は未実施（事前確認が基準未達）**

> Phase 0 Stage A（30 秒インスト生成）は未達のまま。本書は B′ の GPU 動作や Stage A の成功を主張しない。

## 1. 位置づけ

低メモリ運用（`PHASE0_LOW_MEMORY_DESIGN.md`）で選んだ検証対象。RTX 4070 Laptop GPU の 8GB VRAM を活かしつつ、
CPU RAM への過剰な offload を抑える。プロファイル C（LM なし）は定義のみ保持し、現時点では実行しない。

| 項目 | B′ の設定 |
|---|---|
| DiT / LM | `acestep-v15-turbo` / `acestep-5Hz-lm-0.6B` |
| INT8 | 有効（`int8_weight_only`、LUNARIA 側ラッパーで適用） |
| DiT | GPU 常駐（`ACESTEP_OFFLOAD_DIT_TO_CPU` 未設定 = False） |
| CPU offload | Auto のまま。Text Encoder / VAE / LM は生成コンテキスト内のみ GPU（無条件には無効化しない） |
| LM backend | **`pt` に固定**（下記の判断） |
| VAE | GPU 優先（`ACESTEP_VAE_ON_CPU` 未設定。タイルデコードは API 既定） |
| batch size | 1 |
| 再ダウンロード | 禁止（`HF_HUB_OFFLINE=1`） |

### LM backend を pt に固定した判断

上流は backend が Auto だと vLLM を優先し、実機の空き VRAM が 2.0GB 以上あれば vLLM を選ぶ（`llm_inference.py`）。
前回は DiT 常駐でほぼ空きが無く pt にフォールバックしていたが、**INT8 で DiT が小さくなると vLLM が選ばれ、KV キャッシュが VRAM を消費して
VAE 用の余裕を削る可能性がある**。B′ の目的（VAE を GPU で動かす）に反するため、上流 README が 6〜8GB 帯で推奨する `pt` に固定した。
この固定が不要かどうかは実測で確認する。

## 2. 実装方式

上流ソースは変更しない。LUNARIA 側のファイルのみ:

| ファイル | 役割 |
|---|---|
| `scripts/lunaria_profiles.py` | プロファイル A / B′ / C の環境変数定義、B′ の実行許可条件（実測値が根拠） |
| `scripts/bprime_wrapper.py` | 起動ラッパー。`AceStepHandler.initialize_service` を実行時に差し替え、INT8 を適用して検証 |
| `scripts/bprime_diag.py` | 診断ランナー。事前確認・所有プロセス管理・テレメトリ・停止検証 |
| `scripts/test_bprime_wrapper.py` | 単体テスト 14 件（GPU・モデル不使用） |

### モデル初期化と量子化の順序（明示）

1. ラッパーが環境（プロファイルの環境変数）を検証し、不一致なら起動しない
2. `initialize_service` を差し替え（`quantization="int8_weight_only"` を注入。`offload_dit_to_cpu=True` や他の量子化指定は**拒否**）
3. 上流が DiT を GPU へロード（`init_service_loader.py`: `model.to(device).to(dtype)`）
4. 上流が DiT のロード後に `quantize_()` を適用（同ファイル。**GPU 上の bf16 重みを INT8 に置換**）
5. ラッパーが GPU 上の重みを走査して量子化を検証（失敗なら `(status, False)` → 上流は起動を失敗させる）
6. 以降は上流の通常の起動処理（VAE / Text Encoder は CPU、LM は pt で初期化）

### 量子化の対象

上流の `_apply_dit_quantization` は、DiT の `decoder.*` の **Linear 層のみ**を INT8 重み量子化する（tokenizer / detokenizer は除外）。
それ以外（embedding・norm・decoder 以外の層）は bf16 のまま。実機 DiT での量子化率は未測定（診断で記録する）。

### 黙示フォールバックの防止

上流は torchao の import に失敗すると、警告だけで量子化なしに初期化を続ける（`init_service_orchestrator.py`）。
ラッパーは次を満たさない場合に失敗として扱い、**bf16 へ黙って戻さない**:

- `handler.quantization == "int8_weight_only"`
- `AffineQuantizedTensor` の重みが 1 つ以上ある
- 量子化された重みがすべて `cuda` 上にある

単体テストで、上流の黙示フォールバック、属性だけ立っていて実体が bf16 のケース、デバイス不一致を検知できることを確認した。

## 3. torchao 互換性の評価

| 項目 | 結果 |
|---|---|
| 固定バージョン | torchao 0.16.0 / torch 2.7.1+cu128（上流 `pyproject.toml` の pin） |
| 警告 1 | 「torch 2.7.1 との非互換により **C++ 拡張の import をスキップ**」。torchao のカスタム CUDA/C++ カーネルが使われず、PyTorch 実装にフォールバックする |
| 警告 2 | `Int8WeightOnlyConfig` の version 1 は非推奨（上流は既定の v1 を使用）。将来の torchao で削除される |
| CPU・小型 Linear | 量子化・順伝播・`.to(dtype)` 後の量子化維持を確認（重み約 50% 縮小、相対誤差約 0.8%） |
| **GPU・実機 DiT** | **未確認**。速度（C++ 拡張なし）、数値の安定性、生成音声への影響は実機診断・生成試験で確認が必要 |

評価: 警告は致命的ではなく、小型層では動作する。ただし実機 DiT の互換性は未確認で、**速度低下と品質影響が最大のリスク**。
診断（初期化と量子化まで）で INT8 の適用自体は確認できるが、速度と品質は 30 秒生成の試験まで分からない。

## 4. 診断の内容（音楽生成は行わない）

`bprime_wrapper.py --diagnose-init` は、サーバー起動（モデル初期化・量子化・LM 初期化）が完了した時点で次を記録して終了する。

- INT8 の適用結果: 量子化テンソル数・割合・配置デバイス・保持バイト数・初期化前後の RAM（RSS / private）・VRAM
- モデル配置: DiT / Text Encoder / VAE / LM のデバイス、LM backend、offload・compile・dtype の実値
- VAE 用 VRAM の余裕: VAE を一時的に GPU へ載せた状態の実効空き VRAM、上流の CPU フォールバック閾値（0.5GB）との比較、タイルのチャンクサイズ
- LM / Text Encoder を一時的に GPU へ載せた時の VRAM（DiT 常駐との併存量）
- 起動・停止の安全性（所有プロセスの終了、ポート解放、GPU メモリ回復、残存プロセス）

## 5. 事前確認（B′ 専用）

推定値だけで安全条件を確定しない。B′ は DiT の bf16 重みを先に読み込んでから INT8 化するため、初期化の経路は A と共通である。
**A の初期化時の実測値を上限として用いる**（B′ 固有の実測は未取得）。

| 項目 | 必要量 | 根拠 |
|---|---|---|
| 空き物理 RAM | 13.49GB 以上 | A の初期化ピーク RSS 9.99GB（実測）+ 余裕 max(1GB, 15%) + 保護下限 2GB |
| コミット余裕 | 13.99GB 以上 | 同 9.99GB + 4GB |
| 空き VRAM | 5,942MiB 以上 | A の初期化時 VRAM（他アプリを除く）5,430MiB（実測）+ 512MiB |
| 中断の下限 | 空き RAM 3GB 未満が 3 秒連続 | 所有プロセスのみ正常停止 |
| その他 | commit 固定、モデル 4 種、ジャンクション、torchao 0.16.0、残存プロセスなし、ポート 8001 空き | |

従来構成 A の保護条件（空き RAM 14GB 以上）は維持している。

### 今回の事前確認の結果（診断は実行せず）

| 項目 | 実測 | 判定 |
|---|---|---|
| 空き物理 RAM | 6.79GB | **NG**（必要 13.49GB） |
| コミット余裕 | 16.0GB | OK |
| 空き VRAM | 6,427MiB | OK |
| ポート 8001 / 残存プロセス / モデル / ジャンクション / torchao | 問題なし | OK |

主な RAM 使用: Chrome 約 5.7GB、メモリ圧縮 約 1.8GB、WSL 約 1.0GB、msedgewebview2 約 0.9GB。他プロセスは停止していない。

## 6. 実行手順（承認後）

```powershell
# 事前確認のみ（読み取り専用。状態ファイルを変更しない）
& "<ROOT>\.venv-engine\Scripts\python.exe" -I "<ROOT>\scripts\bprime_diag.py" --dry-preflight
# 診断 1 回（事前確認に合格した場合のみ実行。自動再試行なし）
& "<ROOT>\.venv-engine\Scripts\python.exe" -I "<ROOT>\scripts\bprime_diag.py"
```

結果は `data/logs/`（Git 管理外）の `preflight_bprime_<run_id>.json`、`bprime_report_<run_id>.json`、`bprime_engine_<run_id>.log`、
`phase0_state.json`（`runs.<run_id>`）、`phase0_telemetry.jsonl`（1 秒間隔）に run_id 別で保存される。

## 7. 代替方式の整理（実装が複雑化する場合）

| 方式 | 内容 | 評価 |
|---|---|---|
| **採用: 実行時ラップ** | `initialize_service` を差し替えて INT8 を注入・検証（約 300 行） | 上流無改変で最小。上流の内部シグネチャ変更に弱い（`audit_acestep_memory.py` と単体テストで検知） |
| 自前サーバー | 上流のハンドラーを import して LUNARIA 専用の API を実装 | 制御は最大だが実装量が大きい。Phase 1 のオーケストレーター設計と重複 |
| 上流への機能追加 | `ACESTEP_QUANTIZATION` 環境変数を上流に追加（PR） | 最も素直だが、上流の改変は現時点で禁止 |
| Gradio / CLI 経路 | 既に `quantization` 引数を持つ | UI・CLI 前提で LUNARIA の構成に合わない |

実装は複雑化しておらず、採用方式のままでよい。上流を更新する場合は `audit_acestep_memory.py` と `test_bprime_wrapper.py` を再実行する。

## 8. 未実施・未確認

- 実 GPU でのモデルロード・INT8 適用・VRAM / RAM の実測（事前確認が基準未達のため未実施）
- 実機 DiT での torchao 互換性（速度・数値・生成音声）
- LM backend を pt に固定することの要否
- 30 秒インストの生成と WAV 検証（Stage A）
