# PHASE0_IMPORT_MEMORY_INVESTIGATION — D0b（import による RAM・commit 増加の原因特定）

状態: 調査完了（実モデルのロード・INT8・楽曲生成は未実施。Phase 0 Stage A は未達のまま）。
詳細ログは `data/logs`（Git 管理外）。本書は要約値のみを記載する。

## 1. 要点

- ACE-Step の `acestep.handler` を import すると、プロセスの private bytes が約 **+1.0GB**、所要 12〜46 秒。うち学習用ライブラリ（peft / lycoris / lightning / torchaudio）が eager に読み込まれる。
- **CUDA 初期化の増分は、先に何を import したかで 13 倍変わる**。torch のみ、または CUDA を先に初期化した場合は **+306MB**（物理 約 −0.1GB、VRAM +145MiB）。`acestep.handler` / peft / lycoris などを import した後では **+3,986〜4,003MB**（物理 約 −2.4〜2.8GB、VRAM +861MiB）。
- 同じ import を CUDA 初期化の前後どちらで行うかで、合計の private は 6,516MB（handler → CUDA）と 2,779MB（CUDA → handler）。**CUDA 先行で private −3.7GB、commit −4.4GB、物理 −3.4GB、VRAM −0.7GiB**。
- 直接機構（peft / diffusers / torchao に共通して CUDA 初期化を重くする要因）は **未特定**。flash_attn・sympy・torch._dynamo/_inductor は反証した（第6章）。
- 最初の GPU 処理（cuBLAS / cuDNN / SDPA）はさらに増える。handler → CUDA → GPU 処理（W1）は、空き RAM が 5.4GB 以上減る途中で、設計どおり下限 3GB で子プロセスが自己停止した。**CUDA 先行が最初の GPU 処理・実生成ピークでも有効かは未検証**（W2 は安全条件により未実行）。このため、CUDA 先行をメモリ削減の成功とは評価しない。

## 2. 実施内容

| 項目 | 実績 |
|---|---|
| 子プロセス数 | 38（5 パス） |
| 安全な自己停止 | 1（W1、下限 3GB、exit 3） |
| クラッシュ | 0（0xC0000005 なし）、孤児プロセス 0 |
| 禁止事項 | 実 DiT/LM/TE/VAE のロード、INT8、楽曲生成は未実施。上流ソース無改変 |

各構成は 1 つずつ逐次実行し、直前に `lunaria_profiles.d0b_admission`（見込み消費量 + 安全余裕 3GB）を判定した。自動再試行はしない。

## 3. import の静的解析（`scripts/analyze_import_chain.py`）

- `acestep.handler` の eager 連鎖は親パッケージの `__init__` を含め **105 モジュール**。`acestep.api_server` は 177 モジュール、加えて transformers・torchaudio・fastapi・uvicorn 等。
- eager な外部ライブラリ（handler）: torch, loguru, numpy, soundfile, tqdm, **peft, lycoris, torchaudio, lightning**、bitsandbytes（try）。`acestep.training` 経由で、生成に不要な学習用ライブラリまで読み込まれる。
- 関数内（lazy）: diffusers, flash_attn, safetensors, torchao, transformers 等。
- import 時に CUDA・モデルロード・大きな確保を行う呼び出しはない。

## 4. import 別の増分（子プロセス、CUDA 未初期化）

| import | private 増分 | 所要 | 備考 |
|---|---|---|---|
| torch | +1,533MB（commit 約 1.5GB、物理 約 0.4GB） | 3.5〜8 秒（初回 42 秒） | 948 モジュール、DLL 62 個 4,762MB をマップ |
| safetensors / numpy 等 / gpu_config | ≈ 0 | | |
| transformers | +23MB（auto で +106MB） | 10〜16 秒 | |
| accelerate / diffusers / torchao / torchaudio / triton | +9 / +100 / +103 / +7 / +4MB | | |
| lightning | 約 +950MB（peft 後は +61MB） | | |
| peft | +923MB | | flash_attn_2_cuda（941MB のマップ）を含む |
| lycoris | +924MB | | |
| acestep.llm_inference | +107MB | 21 秒 | |
| **acestep.handler** | **+992MB** | 31〜46 秒 | |
| acestep.api_server | +1,009MB | 33 秒 | |
| acestep.training | +993MB | | |

private とシステム commit の増分は 1 対 1（±1%）。

## 5. CUDA 初期化の増分

| 先に import したもの | private | RSS | commit | 物理 | VRAM |
|---|---|---|---|---|---|
| 正常群（torch のみ、safetensors、numpy 等、gpu_config、llm_inference、transformers、accelerate、torchaudio、triton、lightning、CUDA 先行） | +306MB | +118MB | +0.3GB | −0.1GB | +145MiB |
| 中間（diffusers、torchao） | +3,319〜3,324MB | +2,224MB | | −2.0〜2.1GB | |
| 重い（peft、lycoris、acestep.training / handler / api_server、B2 再現構成） | +3,986〜4,003MB | +2,827MB | +3.6〜4.0GB | −2.4〜2.8GB | +861MiB |

新規 DLL は重い場合もなし。CUDA 初期化の時間は正常 0.2 秒、重い構成 2.3〜3.1 秒。

### 5.1 順序の効果

| 順序 | private 合計 | commit | 物理 | VRAM |
|---|---|---|---|---|
| handler → CUDA（G） | 6,516MB | 6.83GB | −3.75GB | +861MiB |
| CUDA → handler（G2） | 2,779MB | 2.46GB | −0.39GB | +145MiB |

G2 でも handler の import 自体は +938MB で変わらない。減るのは CUDA 初期化側の約 3.7GB。

### 5.2 最初の GPU 処理（W 系）

- W3（import なし、CUDA → GPU 処理）: private +191MB、物理 −0.18GB、VRAM +34MiB。
- W1（handler → CUDA → GPU 処理）: 約 1 秒で private 8,875 → 9,459MB、空き RAM 8.33 → 2.94GB（−5.4GB 以上、増加中）。子プロセスの自己停止が作動（下限 3GB、他プロセスへの影響なし）。
- W2（CUDA → handler → GPU 処理）: 必要量 9.9GB（空き + commit 判定）を満たさないため **未実行**。

## 6. 反証した仮説

| 仮説 | 結果 |
|---|---|
| flash_attn の事前ロードが原因 | flash_attn → CUDA（Y）は +306MB で正常。9/9 の重い構成が事前に flash_attn_2_cuda を読んでいた相関は交絡 |
| sympy | 正常構成 3 件にも存在 |
| torch._dynamo / _inductor | 正常構成でも読み込まれている |

peft / diffusers / torchao に共通する直接機構は未特定。以前の記録（B2）にあった「CUDA コンテキストが物理 2.6GB・VRAM 861MiB を使う」は **訂正**する: これは CUDA 自体ではなく、先行 import に由来する増分だった（CUDA 単独は物理 約 0.1GB 減・VRAM 145MiB）。

## 7. 起動時間への影響

handler import は 12〜46 秒（G 54.5 秒 vs G2 42.7 秒、順序で約 12 秒短縮）。CUDA 初期化は 0.2 秒（正常）対 2.3〜3.1 秒（重い）。

## 8. 計測の精度

子プロセスのサンプラーは 1 tick 0.032〜0.4ms。ランナー側 SysMonitor は 0.1 秒間隔（0.004ms/tick）。ランナー側の最大間隔は 1.0 秒。

## 9. 改善方式の比較

「単に import を遅延させただけで、実生成時のピークが変わらない方式」はメモリ削減成功と評価しない。

| 観点 | A: lazy import | B: プロセス分離 | C: import 連鎖整理 | D: 必要時起動 |
|---|---|---|---|---|
| 初期化ピーク | 初期化時のみ下がる。生成時は同じ（遅延のみなら不成立） | 変わらない（モデル分は減らない） | 学習用を「ロードしない」場合は下がる（約 −1GB、CUDA 初期化の重さへの効果は未検証） | 変わらない |
| 常駐 | 使用後は同じ | 軽量オーケストレーターで削減可 | 恒常的に下がる | 停止後は RAM/VRAM が完全回復（実測済み） |
| GPU（VRAM） | 同じ | 同じ | 同じ（CUDA 初期化が正常なら −0.7GiB の可能性） | 同じ |
| 品質 | 影響なし | 影響なし | LoRA 使用時に学習ライブラリが必要 | 影響なし |
| 起動時間 | 初回生成時に移る | 増える | 短縮（peft 等 約 1〜3 秒以上） | 約 3 分 |
| 上流無改変 | 困難（sys.modules スタブ等が必要） | 可 | 同左 | 可 |
| Electron との整合 | 可 | 良（IPC 設計と一致） | 可 | 良 |

- **CUDA 先行初期化**（ラッパーで実現、上流無改変）: 初期化時の削減は実測済み（物理 −3.4GB 等）。最初の GPU 処理・実生成ピークでの効果は未検証。実装は `LUNARIA_CUDA_FIRST=1`（`bprime_wrapper.py`）と `dit_diag.py --cuda-first` で**オプトイン（既定オフ）**。
- 推奨: 次回は先に **W2** を実施し、効果が確認できた場合に限り CUDA 先行を B′ に採用する。学習用ライブラリの除外（C）は LoRA 機能との整合を確認してから検討する。

## 10. 次回の実 GPU モデル診断の安全条件

- 空き RAM は W2 で 9.9GB 以上（または承認済みの測定モード: 子プロセス自己停止下限 3GB）、commit 余裕は観測値で判定。
- DiT のロード診断（D1〜D3）は、DiT ファイルの commit が 3 倍になる現象のうち **1 倍分（約 4.6GB）が未特定**のため、観測値（過去の急増 13.8GB）を使って判定する。現状（commit 余裕約 19GB）では不許可（必要約 24.2GB）。
- 安全下限 3GB は引き下げない。自動再試行はしない。

## 11. 関連ファイル

- `scripts/analyze_import_chain.py`: import 連鎖の静的解析
- `scripts/import_probe.py`: 子プロセスでの import 単位測定
- `scripts/d0b_diag.py`: ランナー（`--dry-run` / `--only` / `--custom`）
- `scripts/summarize_d0b.py`: 結果の集計（`--all`）
- `scripts/test_import_diag.py`: テスト 20 件
- `scripts/lunaria_profiles.py`: `d0b_admission`
- 関連文書: `PHASE0_D0_RESULTS.md`、`PHASE0_DIT_DIAGNOSTIC.md`、`PHASE0_INIT_MEMORY_OPTIMIZATION.md`、`PHASE0_BPRIME_PROFILE.md`
