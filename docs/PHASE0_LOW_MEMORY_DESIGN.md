# PHASE0_LOW_MEMORY_DESIGN — 低メモリ運用の設計と ACE-Step メモリ管理の監査

作成日: 2026-10-22 / Issue: #1 / 状態: **調査・設計のみ（実 GPU 検証は未実施）**
対象: ACE-Step 固定 commit `ca1e85fe9430179831e6bc6be790c332190a3866`、RTX 4070 Laptop GPU（VRAM 8GB）

> **Phase 0 Stage A（30 秒インスト生成）は依然として未達。** 本書は Stage A の成功を主張しない。

## 0. 方針変更

ユーザーは複数プロジェクトを同時進行しており、Chrome・WSL・他プロジェクトを止めて空き RAM 14GB 以上を確保する運用は、
LUNARIA の標準運用条件として採用しない。目標は次のとおり（**設計目標であり、動作保証ではない**）。

- 他の開発環境を維持したまま、空き RAM 6〜8GB 程度で利用できる構成を目指す
- LUNARIA の追加物理 RAM 消費 4〜6GB 程度を暫定目標とする
- メモリ不足を無視して実行するのではなく、**実際の消費量を削減する**

### 根拠の区分

| 記号 | 意味 |
|---|---|
| **[実測]** | 本プロジェクトの実機ログ・テレメトリで確認した値 |
| **[コード]** | 固定 commit のソースを読んで確認した挙動（`scripts/audit_acestep_memory.py` で 25 項目を静的に検証済み） |
| **[推定]** | 上記からの見積り。未測定。実機検証が必要 |
| **[未確認]** | 現時点で根拠がない |

## 1. 現行 ACE-Step のメモリ管理（監査結果）

| # | 調査項目 | 結果 |
|---|---|---|
| 1 | Tier 3 判定 | `get_gpu_tier()` は VRAM ≤ 8GB を tier3 とする（`gpu_config.py:760`）。実機の 8.0GB は tier3 **[コード][実測]**。tier3 の既定値は offload_to_cpu / **offload_dit_to_cpu** / **quantization がすべて True**、LM は 0.6B のみ、LM backend 制限は "all" |
| 2 | INT8 の有効条件と適用 | **API サーバー経路では INT8 は適用されない。** `initialize_service()` の `quantization` は既定 `None` で、API の起動処理（`api/startup_model_init.py`）と HTTP 初期化（`api/http/model_init_service.py`）はこの引数を渡さない。INT8 を指定する環境変数・リクエスト項目も存在しない（`ACESTEP_QUANT*` 無し）**[コード]**。tier3 の `quantization_default=True` は Gradio UI 用の既定値で、REST では効かない。ドキュメント（GPU_COMPATIBILITY.md）の「tier3 は INT8 が既定」は **API 経路には当てはまらない** |
| 3 | CPU offload の対象 | `offload_to_cpu=True`（<16GB で自動）の場合、Text Encoder・VAE・LM(pt) は生成コンテキスト内だけ GPU に載り、終了後 CPU へ戻る。DiT は次項 **[コード]** |
| 4 | DiT offload の条件 | DiT の offload は **環境変数 `ACESTEP_OFFLOAD_DIT_TO_CPU` のみ**で、既定 `False`。tier3 の `offload_dit_to_cpu_default=True` は API では使われない **[コード]**。そのため API 経路では DiT が **GPU 常駐**（`Keeping main model on cuda (persistent)` **[実測]**、`init_service_loader.py:200`）。`True` にすると、DiT は拡散中だけ GPU に載り、終了後 CPU へ戻る（`init_service_offload_context.py`） |
| 5 | LM のロード・解放 | pt backend + offload 時、LM は生成コンテキスト内だけ GPU、終了後 CPU へ戻す（`llm_inference.py:4131` 付近）。完全解放の `llm_handler.unload()` は存在するが、**CUDA の通常経路では呼ばれず**（MPS の cover 経路のみ）、REST の解放ルートも無い **[コード]** |
| 6 | LM なしの生成経路 | `ACESTEP_INIT_LLM=false` で LM を読み込まず、`use_cot_caption` / `use_cot_language` は自動で無効化される（`api/llm_readiness.py`）。`thinking=false` なら DiT の text2music 単独で動く **[コード]**。**歌声品質への影響は [未確認]** |
| 7 | GPU VAE タイルデコード | API 既定で `use_tiled_decode=True`。チャンクは実効空き VRAM が 12GB 未満で **128 固定**（`ACESTEP_VAE_DECODE_CHUNK_SIZE` で上書き可）**[コード]** |
| 8 | VAE の CPU フォールバック | **実効空き VRAM < 0.5GB** で自動的に CPU デコード（`generate_music_decode.py:158`）。強制は `ACESTEP_VAE_ON_CPU=1` **[コード]** |
| 9 | 推論後のメモリ解放 | `_release_system_memory()` は gc・`empty_cache`・`malloc_trim` を行うが、**`malloc_trim` と mmap 閾値調整は Linux 限定**（`_get_libc()` は Linux のみ）。Windows では OS ヒープ縮小は行われない **[コード]**。実際にどれだけ RAM が戻るかは **[未確認]** |
| 10 | 必要時のみ起動 | `ACESTEP_NO_INIT=true` でモデルを初回リクエスト時に遅延ロードできる。アイドル時の自動解放機構は無い **[コード]**。プロセス停止は完全な解放手段（前回の停止後、RAM・VRAM は起動前の水準に回復 **[実測]**）。起動〜モデルロード完了は **175 秒**（取得済み・ディスクキャッシュあり）**[実測]** |

## 2. 前回 VAE が CPU にフォールバックした原因

前回（run: Stage A 中断）の時系列 **[実測]**:

1. 拡散（DiT）は **約 6 秒**で完了（`diffusion_time_cost` 5.82 秒、8 ステップ、30 秒音声）
2. VAE 直前: torch の allocated = **4.80GB**（≒ DiT 常駐分）、max = 5.93GB
3. ログ: `Effective free VRAM before VAE decode: 0.00 GB` → `Only 0.00 GB free VRAM; auto-enabling CPU VAE decode`
4. タイルデコード（chunk 128、CPU）が **6 分以上完了せず**、空き RAM 1.5GB でウォッチドッグが中断

原因の構造:

- **直接原因**: 実効空き VRAM（= デバイス空き + (reserved − allocated)）が閾値 0.5GB を下回った。
  デバイス空きは「他アプリの使用分（約 1.74GB）」「CUDA コンテキスト等」「DiT 常駐 4.8GB」を引いた残りで、ほぼ 0 だった
- **根本原因**: API 経路では DiT が GPU 常駐（項目 4）で、tier3 の想定（DiT offload + INT8）が適用されていない（項目 2・4）。
  8GB カードで他アプリが約 1.7GB を使っていると、VAE に回せる VRAM が残らない
- **増幅要因**: CPU デコードは大量の RAM を要し（RSS 9.1→11.7GB 付近）、空き RAM が 1.5〜2.5GB の状況でページングし、極端に遅くなった **[推定]**
  （CPU デコード中の private メモリは未計測）

つまり「VRAM 不足 → CPU デコード → RAM 逼迫」の連鎖で、VRAM と RAM の両方が足りなかった。
VRAM を空ければ CPU デコード自体を避けられ、RAM の追加消費も抑えられる。

## 3. 低メモリプロファイルの比較

### 前提となる実測値

- モデルのディスクサイズ **[実測]**: DiT 4.46GB（bf16）、LM 0.6B 1.28GB、Text Encoder 1.12GB、VAE 0.31GB
- 現行構成（プロファイル A）の実測 **[実測]**: GPU 全体のピーク VRAM 7.89GB（他アプリ約 1.76GB を含む → LUNARIA 分は約 6.1GB）、
  プロセスツリー RSS ピーク 11.74GB（初期化時 9.99GB）
- RSS には**ファイルマップされたモデルファイルの共有ページを含む可能性**があり、RAM 圧迫の実態は private（コミット）量で見るべき。
  今回 `phase0_run.py` に `tree_private_mb` を追加した。**private の実測値はまだ無い**
- INT8（torchao 0.16.0 / torch 2.7.1）は、CPU 上の小型 Linear で動作し、重みが約 50% に縮小、相対誤差約 0.8%、`.to(dtype)` 後も量子化が維持されることを確認 **[実測：`scripts/check_torchao_int8.py`]**。
  ただし torchao は「torch 2.7.1 との非互換により C++ 拡張をスキップ」と警告する。**実機 DiT での速度・品質は [未確認]**

### 比較表（数値は特記なき限り [推定]）

| 項目 | A: 現行<br>0.6B LM + bf16 DiT 常駐 | B: 0.6B LM + INT8 + DiT offload<br>+ GPU VAE タイル | **B′: 0.6B LM + INT8（DiT は GPU 常駐）<br>+ GPU VAE タイル（推奨候補）** | C: LM なし + DiT<br>（切り分け用） |
|---|---|---|---|---|
| 設定 | tier3 Auto（API 既定） | INT8 + `ACESTEP_OFFLOAD_DIT_TO_CPU=true` | INT8 + DiT offload なし（offload_to_cpu は Auto） | `ACESTEP_INIT_LLM=false` + リクエストで `thinking/use_cot_*=false` |
| DiT の重み | bf16 4.5GB（GPU） | INT8 約 2.4〜3.0GB（拡散中のみ GPU、普段は CPU RAM） | INT8 約 2.4〜3.0GB（GPU 常駐） | bf16 4.5GB（GPU 常駐） |
| LUNARIA の VRAM | 約 6.1GB **[実測]** | 約 3.5〜4.5GB（VAE 時は約 1GB まで低下） | 約 4〜5GB | 約 5.5〜6GB |
| VAE の実行先 | **CPU に落ちた** **[実測]** | GPU の見込み（DiT 退避で空きが出る） | GPU の見込み（空き約 1.5〜3GB > 0.5GB 閾値） | **CPU に落ちる恐れ**（DiT 常駐は同じ。LM は VAE 時すでに CPU） |
| LUNARIA の RAM（RSS） | 最大 11.7GB **[実測]** | 約 7〜9GB（CPU に DiT・LM・TE・VAE 計 5.2〜5.8GB が常駐） | **約 5〜6.5GB**（LM・TE・VAE 計 約 2.7GB ＋ランタイム 約 2〜3GB） | 約 4.5〜6GB（A から LM 約 1.3GB と CoT 一時領域を除く） |
| 生成速度 | DiT 5.8 秒 **[実測]**（LM 工程・VAE は未計測） | 拡散ごとに CPU↔GPU 転送が発生し遅くなる。INT8 の速度は [未確認] | INT8 の速度は [未確認]（C++ 拡張なしの経路） | LM 工程が無くなり短縮。DiT のみ |
| 日本語歌詞への影響 | 基準（Stage B 未実施で歌声は未確認） | INT8 の量子化誤差の影響は [未確認] | 同左 | LM の計画（コード・メタデータ補完）が無い。歌詞は Text Encoder 経由で DiT に入るため経路は残る **[コード]** が、歌唱の聞き取りやすさは [未確認] |
| 既存 API で制御 | 可（env のみ） | **不可**（INT8 は REST から指定できない。DiT offload は env で可） | **不可**（INT8 のみ。同上） | 可（env＋リクエスト項目） |
| 上流の改変 | 不要 | **ソース改変は不要**。LUNARIA 側の起動スクリプトで、`initialize_service` の呼び出し時に `quantization="int8_weight_only"` を渡す実行時ラップが必要 | 同左 | 不要 |
| 実機検証 | 14GB 未満では禁止（前回失敗） | 必要 | 必要 | 必要（最初の測定候補） |

補足:

- **参考 A+: A に `ACESTEP_OFFLOAD_DIT_TO_CPU=true` だけを足した構成**（API で制御可能）は、VAE を GPU に戻せる一方、DiT が CPU RAM に常駐して
  RSS が約 14〜16GB へ増える見込み **[推定]**。VRAM と RAM はトレードオフで、RAM を減らす目的には逆効果。
- 「DiT を GPU に置き、他を CPU に置く」現行方針が RAM を 約 2.7GB の常駐に抑えている。B′ はそれを保ったまま DiT を INT8 で小さくし、VAE に回す VRAM を確保する案。

## 4. RAM 6〜8GB 環境での実行可能性

結論: **現時点の見積りでは、空き RAM 6GB での実行は困難。空き 8〜9GB 程度なら B′ / C で可能性がある（いずれも未測定）。**

- 目標の「追加 4〜6GB」に対し、B′ は約 5〜6.5GB、C は約 4.5〜6GB **[推定]**。RSS の中にはファイルマップ分が含まれる可能性があり、private で見れば下がるかもしれない
  が、**未計測**
- 他プロセスを守るには、起動後も空き RAM を一定以上残す必要がある。例えば保護下限 2GB・余裕 1GB を取ると、B′ の必要空き RAM は 約 8〜9.5GB
- 6GB まで下げるには、上流レベルの追加削減（工程ごとのモデルの逐次ロード、mmap ロード、Text Encoder の解放など）が必要になる可能性が高い。今回は設計せず、測定後に検討する
- 「必要時のみ起動し、アイドル時に停止」（項目 10）は**常駐時のピークは下げない**が、他プロジェクトとの共存には有効
  （停止後は RAM・VRAM が完全に戻る **[実測]**）。起動に約 3 分かかる点がトレードオフ

## 5. 事前確認（実行許可条件）の再設計

現行の固定条件（空き RAM ≥ 14GB）を全モードへ一律に適用する方式をやめ、**プロファイルごとの実測値**から許可条件を導く。
現行の 14GB 閾値は変更しない（プロファイル A の実測 RSS 11.7GB からも、14GB は妥当）。

### プロファイル定義（案）

```json
{
  "id": "B-prime-int8-gpu-dit",
  "status": "estimated",                  // estimated | measured（測定 3 回以上）
  "env": { "ACESTEP_INIT_LLM": "true", "...": "..." },
  "measured": { "ram_private_peak_mb": null, "ram_rss_peak_mb": null,
                "commit_peak_mb": null, "vram_ours_peak_mib": null, "runs": 0 },
  "estimated": { "ram_peak_gb": [5.0, 6.5], "vram_ours_gib": [4.0, 5.0] }
}
```

### 許可条件（案）

- `peak` = 測定済みなら実測の最大 private（無ければ RSS）。推定のみなら推定の上限値
- 必要な空き物理 RAM = `peak + max(1.0GB, peak × 0.15) + 保護下限 2.0GB`
- 必要なコミット余裕 = `peak + 4GB`
- 必要な空き VRAM = `プロファイルの LUNARIA 分ピーク + 0.5GB`（他アプリ分は実行前の `nvidia-smi` で差し引く）
- 例: A（実測 RSS 11.7GB）→ 約 14.7〜15GB。現行の 14GB より厳しい
- 例: B′（推定上限 6.5GB）→ 約 9.5GB

### 実行中の保護

- テレメトリ（1 秒間隔）で空き RAM・コミット余裕・VRAM を監視
- **ソフト下限**（空き < 3GB）: 新規ジョブを開始せず待機
- **ハード下限**（空き < 2GB が 3 秒連続、または コミット余裕 < 3GB）: 所有プロセスのみ正常停止して中断
- 他プロセスは**一切停止しない**。不足時は不足量とユーザー側の操作案を報告するだけ
- `estimated` のプロファイルは**測定モード**でのみ実行する: 明示承認、30 秒・1 件のみ、ハード下限を 3GB に引き上げ、
  結果を `measured` に追記（3 回の実測後に `status: measured` へ昇格）

## 6. 推奨する検証順序

1. **V0（完了）**: 静的監査 25 項目（`scripts/audit_acestep_memory.py`）、INT8 の CPU 動作確認（`scripts/check_torchao_int8.py`）、
   テレメトリへ private メモリを追加（`scripts/phase0_run.py`）。
2. **V1: プロファイル C の測定**（env とリクエスト項目のみ。上流改変・追加コード不要）。最も RAM が小さく、
   LM 起因の消費と private/RSS の差を切り分けられる。VAE が CPU に落ちるかも記録する。
   ※ C は LM なしのため「Stage A の定義（0.6B LM 初期化）」と異なる。Stage A の達成条件の扱いは承認時に確認する
3. **V2: プロファイル B′ の測定**。LUNARIA 側の起動スクリプト（INT8 を渡す実行時ラップ）の実装と承認が前提。
   実機 DiT での INT8 の速度・品質（WAV 検証）を確認する
4. **V3: プロファイル B（DiT offload）** は、B′ で VRAM が不足した場合のみ
5. **A は空き RAM 15GB 未満では再実行しない**

## 7. 次回の実 GPU 検証の安全条件

- ユーザーの明示承認（プロファイルと件数を指定）
- 自動停止の権限は所有プロセスのみ。他プロジェクト・Chrome・WSL は停止しない
- 事前確認（`--dry-preflight`）に合格すること。測定モードでは必要空き RAM を推定上限で計算し、ハード下限を 3GB にする
- 生成は 30 秒・1 件・batch size 1。モデル・依存関係・Python・CUDA・ドライバーを変更しない。再ダウンロードしない（`HF_HUB_OFFLINE=1`）
- 1 秒間隔のテレメトリ（RSS・private・空き RAM・コミット余裕・VRAM・工程名）と VAE 実行デバイスを記録
- 失敗時は自動再試行しない。結果は run_id 別に保存し、過去の記録を上書きしない
- 停止後に、所有プロセス・ポート 8001・GPU メモリの回復を確認

## 8. リスク・未確認事項

- torchao 0.16.0 は torch 2.7.1 との非互換警告を出す（C++ 拡張なし）。小型層では動作したが、実機 DiT の速度・数値安定性は未確認
- INT8 は Windows の `torch.compile`（triton-windows 3.3.1 導入済み）を使わない前提（`compile_model` 既定 False、検証は torchao の import のみ）。compile との併用は未確認
- Windows では `malloc_trim` が無いため、`.to("cpu")` 後に RAM が OS へ戻る量は未確認（private の実測で確認する）
- 他アプリの VRAM 使用（約 1.7GB）は変動する。VAE の CPU フォールバック閾値（0.5GB）に近い運用になる
- 上流の更新でこれらの前提が変わり得る。固定 commit を維持し、更新時は `audit_acestep_memory.py` を再実行する

## 9. 関連ファイル

- `scripts/audit_acestep_memory.py` — 固定 commit のメモリ管理の静的検証（25 項目、GPU 不使用）
- `scripts/check_torchao_int8.py` — INT8 の CPU 動作確認（小型層のみ）
- `scripts/phase0_run.py` — テレメトリに `tree_private_mb` を追加
- `docs/PHASE0_GPU_BASELINE.md` — 前回の実測結果
