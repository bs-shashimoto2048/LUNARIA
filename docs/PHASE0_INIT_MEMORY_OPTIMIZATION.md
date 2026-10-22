# PHASE0_INIT_MEMORY_OPTIMIZATION — B′ 初期化時のピーク RAM 削減

作成日: 2026-10-22 / Issue: #3 / PR: #4 / 状態: **静的解析・CPU 実測・実装・単体テストまで完了。実 GPU 検証は未実施**

> Phase 0 Stage A（30 秒インスト生成）は未達のまま。本書は B′ の GPU 動作や Stage A の成功を主張しない。

> **訂正（DiT 診断後。詳細は `PHASE0_DIT_DIAGNOSTIC.md`）**
> - 本書の `L = 2.55GB` は **commit 量（private）であり、物理消費ではない**。import の物理消費は約 0.7〜0.8GB（E1 の private に numpy の予約約 780MB が含まれていた）
> - 一方、本書が見落としていた **CUDA コンテキストが物理 RAM を約 2.6GB（commit は約 4.0GB）消費する**。import + CUDA の物理ベースラインは約 3.4GB [実測]
> - したがって §4・§7 の絶対値は置き換えが必要（8GB 環境の見通しは本書より厳しい）。主要な発見（DiT の CPU 実体化、LM・Text Encoder の fp32 展開と居座り、`device_map` / `dtype` の注入が transformers で受理されること）は変わらない
> - **DiT の `device_map` による GPU 直接ロードは、実機で `torch_cpu.dll` のアクセス違反により失敗した**（commit の急増と同時。commit 上限の超過が有力な原因）。§5 の候補 A の「採用（実装済み）」は、実機で動作が確認できていない状態である

## 0. 方針と目標

- 複数プロジェクト並行・Chrome / WSL / 遠方監視システムの稼働を前提に、**空き物理 RAM 8GB 程度**で安全に初期化できる方式を探す（6GB は追加目標）
- 目的は量子化後の削減ではなく、**モデル初期化時のピーク削減**
- 制約: DiT turbo・LM 0.6B・日本語歌詞機能・INT8 を維持。上流は改変しない。Python / PyTorch / CUDA / ドライバー更新なし。再ダウンロードなし。他プロジェクトを止めない
- **preflight の閾値（B′: 空き RAM 13.49GB、A: 14GB）は引き下げていない**（初期化時の RAM 消費の改善が実機で確認できていないため）

根拠の区分: **[実測]**（本プロジェクトで測定）／**[コード]**（固定 commit と固定版 transformers のソースを読んで確認。`scripts/audit_acestep_memory.py` で 37 項目を静的検証）／**[推定]**／**[未確認]**

## 1. 結論

初期化時の RAM ピークの主因は、**モデルの重みが CPU 上に private メモリとして実体化され、しかも fp32 に展開される箇所がある**こと。

1. **DiT（bf16 4.46GB）**: `device_map` なしで読むため、全重みが CPU 上に実体化される。実測では、safetensors の CPU 実体化は**ファイルサイズの約 1.14 倍**の private を消費する
2. **Text Encoder（bf16 1.12GB）**: `dtype` 指定なしで読むため fp32（約 2.2〜2.6GB）に展開され、offload 時は CPU 上で fp32 のまま**常駐**する
3. **LM 0.6B（bf16 1.28GB）**: 同様に fp32（約 2.6〜2.9GB）で読み、その後 bf16 へ変換する。**実測では、fp32 → bf16 変換後も private は減らなかった**（fp32 の領域がプロセス内に居座る）
4. 上流の「解放」処理（`malloc_trim`）は Linux 専用で、Windows では効かない

これに対し、**上流を改変せず実行時ラップだけで**、DiT を GPU へ直接ロード、LM と Text Encoder を最初から bf16 でロードする方式を実装した（単体テスト 38 件で検証）。
削減見込みは**ピーク −1.7〜2.4GB・常駐 −1.0〜1.3GB [推定]**。ただし、これだけでは空き RAM 8GB での安全な初期化は**まだ保証できない**（§7）。

## 2. 初期化経路の実ソース解析（固定 commit `ca1e85f`）

| # | 処理 | ソース・関数 | CPU RAM の挙動 | GPU |
|---|---|---|---|---|
| 1 | 起動・import | `acestep/api_server.py` ほか | import だけで private **2,553MB** [実測] | CUDA コンテキストは未作成 |
| 2 | DiT のロード | `core/generation/handler/init_service_loader.py:175` `_load_main_model_from_checkpoint` → `AutoModel.from_pretrained(path, trust_remote_code=True, attn_implementation=…, dtype=bf16)` | **`device_map` なし → 全パラメータが CPU 上に実体化**（transformers `_load_state_dict_into_meta_model`: `device_map is None` → `param_device="cpu"`）。private ≒ 4.46GB × 1.14 [実測係数] | — |
| 3 | DiT の GPU 転送 | 同 `:198` `self.model = self.model.to(device).to(self.dtype)` | `.to()` 後は CPU 側の参照が外れ、**参照がすべて外れれば OS へ返る**（E4: 513MB 返却）[実測]。ただし参照が残ると返らない（E2）[実測] | DiT 常駐 |
| 4 | INT8 量子化 | 同 `:209` `_apply_dit_quantization` → `quantize_(self.model, config, filter_fn)` | **GPU 上で適用**（`:198` の転送後）。CPU に量子化前の複製を作らない [コード] | bf16 重み → INT8 を GPU 上で置換 |
| 5 | VAE | `init_service_loader_components.py` `_load_vae_model` → `AutoencoderOobleck.from_pretrained(path)` | fp32（0.34GB）。offload 時は CPU 常駐（`_get_vae_dtype("cpu")` = fp32） | — |
| 6 | Text Encoder | 同 `:95` `_load_text_encoder_and_tokenizer` → `AutoModel.from_pretrained(text_encoder_path)`（**dtype 指定なし**） | **bf16 保存でも fp32 に展開**。offload 時は `.to("cpu").to(float32)` で **fp32 のまま常駐**（約 2.2〜2.6GB）[コード] | — |
| 7 | LM(pt) | `llm_inference.py:404` `_load_pytorch_model` → `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`（**dtype 指定なし**）→ `.to("cpu").to(bf16)` | **fp32 で読み込み（約 2.6〜2.9GB）→ bf16 へ変換**。変換後も private が減らない [実測] | — |
| 8 | ロード順序 | `init_service_orchestrator.py:156,163,169` / `api/startup_model_init.py:80,165` | DiT → VAE → Text Encoder → **LM（最後）** [コード] | — |
| 9 | 解放処理 | `init_service_memory_basic.py:221` `_release_system_memory`（gc + `empty_cache` + `malloc_trim`） | DiT のロード処理内では**呼ばれない**。`malloc_trim` は Linux 限定 [コード] | — |

### transformers 4.57.6（固定版）のロード実装 [コード]

- safetensors は `safe_open(shard_file, device=tensor_device)` + `get_slice()[...]` で **1 パラメータずつ**実体化する。`device_map` が GPU を指す場合は、その GPU 上へ直接実体化される（CPU に全体を持たない）
- `device_map` に int / str / `torch.device` を渡すと `{"": device}` に正規化される。dict は `_no_split_modules` を要求されず（要求されるのは `"auto"` 等の文字列のみ）、tied パラメータの整合チェックだけが行われる
- `low_cpu_mem_usage` は無視され、常に meta デバイスで初期化する
- `device_map` が GPU のとき、`caching_allocator_warmup` が空き VRAM − 1.2GiB を上限に GPU メモリを先に確保する（GPU 側の挙動。CPU RAM には影響しない）

## 3. CPU・合成データでの実測（`scripts/measure_load_memory.py`、各実験は別プロセス、3 回再現）

private = `psutil.memory_info().private`（コミット量）。GPU・実モデルは使っていない。

| 実験 | 結果 |
|---|---|
| E1 import ベースライン L | **2,553MB**（torch 1,551 → transformers/diffusers 1,656 → acestep 2,553）。CUDA コンテキストは含まない |
| E2 safetensors の CPU 実体化 | 512MB のファイルで private **+585MB**（×1.14）。実体化したテンソルへの参照が残ると、解放しても **0MB 返却** |
| E3 fp32 → bf16 変換 | fp32 で 641MB → bf16 変換後も **664MB**（減らない）。変換のピークは fp32 と同水準 |
| E4 参照をすべて落とした場合 | 解放後に 513MB が OS へ返った。`_heapmin` / `HeapCompact` / `EmptyWorkingSet` は追加で返さない（不要） |

解釈（一部は推定）:

- CPU 上でモデルを実体化すると、ファイルサイズ以上の private を消費する **[実測]**
- **一度 CPU で fp32 に展開すると、bf16 へ変換しても private は下がらない** **[実測]**。LM と Text Encoder を fp32 で読む上流の経路は、そのまま常駐量を押し上げる
- E2 と E4 の差（参照の有無）は、実体化テンソルが同一ストレージを共有し、1 つ参照が残ると全体が保持されたためと考えられる **[推定]**。実際のロードでは各パラメータが `.to(device)` で個別に移るため、DiT は E4 型（返却される）になる見込み **[推定]**

## 4. ピークと常駐の見積り（プロセス private、GB）

L = 2.55 [実測]。各モデルは「ファイルサイズ × 1.14（実測係数）」、fp32 は bf16 の 2 倍で見積もり **[推定]**。CUDA コンテキストのホスト側消費は**未測定**で、両方式に共通。

| 段階 | 現状（上流のまま） | 低ピーク方式（実装済み） |
|---|---|---|
| import | L = 2.55 | L = 2.55 |
| DiT ロード | + 5.1（CPU 実体化。転送後に返却される見込み） | + 約 0.1〜0.3（パラメータ 1 つ分の一時領域） |
| VAE | + 0.34（fp32） | + 0.34（変更なし） |
| Text Encoder | + 2.2〜2.6（fp32 常駐） | + 1.28（bf16） |
| LM 読み込み中 | + 2.6〜2.9（fp32 一時） | + 1.46（bf16 直接） |
| LM 読み込み後（常駐） | + 1.46（bf16。fp32 の残骸が居座る恐れあり） | + 1.46 |
| **ピーク（private）** | **約 7.7〜8.3**（DiT 段階または LM 段階） | **約 5.9**（常駐と一致） |
| **常駐（private）** | 約 6.9 以上 | 約 5.9 |

- 実測の裏付け: A の初期化で、実機の空き RAM は開始時約 8GB から最小 0.81GB へ落ちた（約 7.2GB の消費）。プロセスツリー RSS ピークは 9.99GB（ファイルマップ分を含む）。現状列の見積り（7.7〜8.3）と整合する
- 削減見込み: **ピーク −1.7〜2.4GB、常駐 −1.0〜1.3GB [推定]**
- 実測が必要: 低ピーク方式の実際の private ピーク、CUDA コンテキストのホスト側消費、DiT の最大テンソルサイズ

## 5. 改善候補の比較

| 候補 | 内容 | 固定 ACE-Step での実現性 | 効果 [推定] | リスク | 判断 |
|---|---|---|---|---|---|
| **A. GPU 直接ロード** | DiT の `from_pretrained` に `device_map={"":"cuda:0"}` を足す | **可能**。transformers が dict を受理し、1 パラメータずつ GPU へ実体化する **[コード]**。CPU で `device_map` の dict 受理と値の一致を確認 **[実測]**。上流ソース改変不要（実行時ラップ） | DiT 段階のピーク −約 5GB | 実 GPU・カスタムモデル（trust_remote_code）での動作は**未確認**。`caching_allocator_warmup` が VRAM を先に確保する | **採用（実装済み）** |
| **B. ストリーミングロード** | 重みを分割読み込みしてピークを抑える | A と同一。transformers が既に 1 パラメータずつ読む | A に含まれる | 自前実装は不要 | A に包含 |
| **C. モジュール単位量子化** | 層ごとに順次量子化 | 上流の `quantize_()` が GPU 上でモジュール単位に置換する **[コード]**。CPU 側の複製は無い。別案として HF の `TorchAoConfig` をロード時に使う方法があるが、**量子化の対象範囲が上流の `decoder.*` の Linear のみと変わる**（品質影響が未確認）| 追加効果なし | 範囲変更による品質リスク | 不採用（現行の `quantize_` を維持） |
| **D. 初期化順序の変更** | LM / DiT / VAE / TE の順序を変える | ピークの主因は順序ではなく**fp32 展開と CPU 実体化**。dtype 修正後は常駐が単調増加するため、順序を変えても RAM ピークは変わらない | ほぼ無し | — | 不採用 |
| **E. 事前量子化モデルの再利用** | 量子化済み重みを保存して次回から読む | 上流はロード後に `quantize_` する構造のため、再利用には自前でモデルを組み立てて `load_state_dict(assign=True)` する必要がある。torchao 0.16.0 の量子化テンソルの保存・再読込の互換性は**未確認**（torch 2.7.1 との非互換警告あり）。初回変換のピークは A と同じ | 初期化の高速化・VRAM ピーク低減。RAM ピークへの追加効果は小 | 互換性が未確認で複雑 | **保留**（A で足りなければ再検討） |
| **F. Text Encoder の bf16 常駐** | offload 時の CPU 常駐を fp32 → bf16 | 実装可能。`_get_vae_dtype("cpu")` をロード中だけ bf16 にする。**数値は同一**（bf16 → fp32 → bf16 は可逆。上流も初回生成後は bf16 で CPU に戻す）| 常駐 −1.27GB | 無し | **採用（実装済み）** |
| **G. LM / TE の bf16 直接ロード** | `dtype` を注入して fp32 を経由しない | 可能（`dtype` 引数は固定版 transformers が受理 **[実測]**）。**値は fp32 経由と完全一致**（極小モデルで `torch.equal`）**[実測]** | LM 段階のピーク −約 1.4GB、E3 の居座りを回避 | 無し | **採用（実装済み）** |
| **H. LM を GPU に常駐** | `ACESTEP_LM_OFFLOAD_TO_CPU=false` | 環境変数で可能（既存 API）。RAM −1.46GB、VRAM +1.3GB | 常駐 −1.5GB | VAE 用 VRAM の余裕が減る。TE の GPU 転送時の VRAM が逼迫しうる（推定 6.1GB vs 空き 6.4GB）| **次の候補**（VRAM の実測後に判断） |

いずれも「実装できる」と決めつけず、実 GPU で確認が必要な項目を §8 に列挙した。

## 6. 実装（`scripts/bprime_wrapper.py`、上流無改変）

| 機能 | 内容 |
|---|---|
| DiT の GPU 直接ロード | `transformers.AutoModel.from_pretrained` を差し替え、パスが `acestep-v15-turbo` で `device_map` 未指定かつ CUDA 利用可なら `device_map={"":"cuda:0"}` を注入。指定済みなら尊重 |
| Text Encoder の bf16 ロード | パスが `Qwen3-Embedding-0.6B` なら `dtype=bf16` を注入。`_load_text_encoder_and_tokenizer` をラップし、ロード中だけ `_get_vae_dtype("cpu")` を bf16 に（例外時も必ず元へ戻す） |
| LM の bf16 直接ロード | `LLMHandler._load_pytorch_model` をラップし、ロード中だけ `AutoModelForCausalLM` を「LM 自身の dtype を注入するプロキシ」に差し替え |
| 上流の前提チェック | 起動時に、パッチが前提とする上流ソースの形（`device_map` が無い、dtype 指定が無い、`_get_vae_dtype("cpu")` が fp32 など）を静的に確認。崩れていれば**パッチせず起動を失敗**させる |
| **段階ごとの RAM ガード** | DiT / Text Encoder / LM の各ロードの**確保前**に「空き RAM − 必要量 ≥ 3GB」を確認し、足りなければその段階で安全に失敗する（必要量は bf16 サイズ × 1.14 + 余裕: DiT 0.6 / TE 1.6 / LM 1.8GB）。preflight の閾値とは別の追加保護 |
| LM 未初期化の検知 | RAM ガード等で LM が初期化されなかった場合、診断は成功扱いにしない |
| 段階別の計測 | 各ロード段階の private メモリと注入内容を `load_events` として診断レポートに記録（A の初期化との比較用） |

## 7. RAM 8GB 環境での実行可能性（見積り・未検証）

プロセスの起動前に測る「空き RAM」には、これから import で消費される約 2.55GB が含まれる点に注意。起動前の空きが 8GB の場合の推移（**[推定]**、CUDA ホスト側消費は含まない）:

| 時点 | 空き RAM（概算） |
|---|---|
| preflight 時点 | 8.0GB |
| import 後（−2.55） | 5.5GB |
| DiT（GPU 直接）+ VAE（−0.4〜0.6） | 4.9〜5.1GB |
| Text Encoder bf16（−1.28） | 3.6〜3.8GB |
| LM ガード判定: 必要 1.8GB + 下限 3GB | 3.6〜3.8 − 1.8 = 1.8〜2.0GB < 3GB → **LM 段階でガードが中止する見込み** |
| （仮に LM まで読み込めた場合）LM 後（−1.46） | 2.2〜2.4GB |

評価:

- **低ピーク方式の実装だけでは、起動前の空き 8GB で、3GB の安全下限を保ったまま初期化を完了するのは難しい**（残りは 2GB 台）。前回のウォッチドッグ中断（空き 1.5GB）に近い水準で、現時点では勧められない
- 空き 6GB は不可能に近い（LM 段階の前にガードが止める）
- 8GB で下限を保つには、さらに約 1.5GB の削減が要る。最有力は候補 H（LM を GPU に常駐。RAM −1.46GB、VRAM +1.3GB）で、その場合の常駐は約 4.5GB 前後 **[推定]**。VAE 用 VRAM の余裕と、Text Encoder の GPU 転送時の VRAM を**実機で測ってから**判断する
- 以上はすべて推定であり、CUDA コンテキストのホスト側消費（未測定）で 0.3〜1GB ずれうる

## 8. 次回の GPU 診断で確認する項目（承認後）

- 低ピーク方式の**実際の private ピークと常駐**（`load_events` と 1 秒間隔テレメトリ）。現状 A の初期化（約 7.2GB 消費）との比較
- DiT の GPU 直接ロードの成否（カスタムモデル、flash attention、`caching_allocator_warmup` の VRAM 先確保）と、CPU 側の private 増加がほぼ 0 であること
- INT8 の適用位置と量子化率（GPU 上）、torchao 0.16.0 × torch 2.7.1 の実機動作
- Text Encoder と LM が bf16 でロードされ、LM が初期化されていること（`llm_initialized`）
- VAE 用 VRAM の余裕（実効空き VRAM と閾値 0.5GB）、LM / TE を GPU へ載せた時の VRAM（候補 H の判断材料）
- CUDA コンテキストのホスト側消費
- 停止後のプロセス・ポート・GPU メモリの回復

### 安全条件（変更なし）

- ユーザーの明示承認が必要。**preflight の閾値は、改善が実測で確認されるまで引き下げない**（B′: 空き RAM 13.49GB）
- 空き RAM が閾値に届かない現在の環境では、診断は実行しない
- 閾値の見直し案（承認が必要）: 段階ごとの RAM ガードを前提に、**段階的な診断**（まず DiT のみ初期化して private を実測 → 結果を見て次の段階へ）を、現在の空き（約 7GB）でも安全に行えるか検討する。DiT 段階の必要量は L 2.55 + 約 0.3 + 余裕 + 下限 約 5.5〜6GB と見積もられる **[推定]**。実装には、初期化を段階で止めるモード（`--stop-after dit` 相当）の追加と承認が要る
- 所有プロセスのみ停止。他プロジェクト・Chrome・WSL は停止しない。自動再試行なし

## 9. 未検証・リスク

- 低ピーク方式の実機での効果（本書の削減値はすべて推定）
- DiT の `device_map` 直接ロードが、カスタムモデル（`trust_remote_code`）と flash attention の組み合わせで実機動作するか
- 数値同一性: LM は bf16 直接ロードと fp32 経由で**値が一致**することを極小モデルで確認済み。DiT の GPU 直接ロードは、CPU 実体化 + 転送と値が一致することを極小モデルの CPU（`device_map={"":"cpu"}`）で確認したのみ
- Windows でのメモリ返却の挙動は、参照の残り方に依存する（E2 と E4）。実際のロードでの返却量は未測定
- torchao 0.16.0 の互換性警告（C++ 拡張なし）、`Int8WeightOnlyConfig` v1 の非推奨
- 上流の更新で前提が崩れうる。固定 commit を維持し、更新時は `audit_acestep_memory.py`（37 項目）と `test_bprime_wrapper.py`（38 件）を再実行する

## 10. 関連ファイル

- `scripts/bprime_wrapper.py` — 低ピーク・ロード、RAM ガード、前提チェック、診断
- `scripts/lunaria_profiles.py` — 段階ごとの必要量（`STAGE_NEED_GB`）、preflight 閾値（変更なし）
- `scripts/measure_load_memory.py` — CPU・合成データによるメモリ挙動の実測（E1〜E4）
- `scripts/audit_acestep_memory.py` — 固定 commit・固定版 transformers の静的検証（37 項目）
- `scripts/test_bprime_wrapper.py` — 単体テスト 38 件（GPU・実モデル不使用）
