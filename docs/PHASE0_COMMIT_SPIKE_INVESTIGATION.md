# PHASE0_COMMIT_SPIKE_INVESTIGATION — DiT の GPU 直接ロード時の commit 急増とアクセス違反の調査

作成日: 2026-10-22 / Issue: #3 / PR: #4
状態: **静的解析・CPU のみの実測・小型モデルの検証まで。実 DiT のロード・GPU 量子化・楽曲生成は行っていない**

> Phase 0 Stage A（30 秒インスト生成）は未達のまま。本書は DiT のロード成功や INT8 の適用を主張しない。
> 原因は**確定していない**。複数の仮説を、根拠の強さとともに整理する。

根拠の区分: **[実測]**／**[コード]**（固定 commit・固定版 transformers のソースを読んで確認）／**[推定]**／**[未確認]**

## 1. 要点

1. **commit の急増には、safetensors のファイルマッピングが関与している可能性が高い**。CPU のみの実測で、`safe_open` 1 回ごとに commit（private とシステム commit の両方）が**ファイルサイズ分**増え、同じファイルを 3 重に開くと 3 倍になることを確認した [実測 E5]。前回の急増（+13.8GB ≒ DiT ファイル 約 4.6GB × 3）、低下（−9.15GB ≒ × 2）、残り（+4.7GB ≒ × 1）の比と一致する
2. ただし、**3 重マッピングの出どころは transformers のコードから特定できていない**（読み込み中に開いたままなのは 1 か所）。CUDA の大きな確保（VRAM の先確保）が commit を増やす機構（M2）も排除できない。**どちらか、または両方かは未確定** [未確認]
3. **前回の評価を訂正する**: 「commit 上限の超過が有力」とした点は弱まった。クラッシュ（16:52:58）は commit 急増（16:52:53〜54）が収まった**約 3〜4 秒後**で、急増の最中ではない。commit 余裕の最小値は測っておらず、上限超過は**仮説の一つ**に格下げする
4. **アクセス違反の原因は未特定**。`torch_cpu.dll`（オフセット 0x6013f94）の例外で、物理 RAM は十分だった。可能性を §6 に列挙した
5. **`--disable-warmup` は commit 急増を止めない可能性が高い**（急増の主因がマッピングなら無関係）。無効化の仕組み自体は安全に検証できた（§5）が、実モデルでの効果と安全性は未確認。**正式な既定値にはしない**

## 2. 観測した事実（前回の診断、run `20261022-164908`）

| 項目 | 値 | 区分 |
|---|---|---|
| 障害 | `torch_cpu.dll`、例外 0xC0000005、フォルト オフセット 0x6013f94、タイムスタンプ 0x6837cbfa。プロセス ID は診断の子プロセスと一致 | [実測] Windows イベントログ |
| 障害の時刻 | 16:52:58.754（Application Error）。WER の記録は 16:53:28。クラッシュ済みプロセスは 16:53:45 まで残った | [実測] |
| commit（private）の推移 | 6.5GB（CUDA 後）→ **20.3GB**（16:52:53〜54）→ **11.2GB**（16:52:55 以降） | [実測] 1 秒間隔（実際の間隔は 1〜3 秒） |
| VRAM の推移 | 2,630MiB → **7,194MiB**（16:52:53。+4,564MiB ≒ DiT 重み 4.46GiB） | [実測] |
| 物理 RAM | 空きは 4.3GB 以上を維持。RSS 最大 4.0GB | [実測] |
| 急増の比 | +13.8GB : −9.15GB : +4.7GB ≒ 3 : 2 : 1（DiT ファイル 約 4.6GB 単位） | [実測] |
| クラッシュ時の commit 余裕 | **未測定**（開始前 16.3GB、終了後 19.6GB） | [未確認] |

時系列の注意: 急増（20.3GB）→ 低下（11.2GB）→ クラッシュの順で、クラッシュは低下の約 3〜4 秒後。1 秒間隔のサンプル間が最大 3 秒空いたため、クラッシュ直前の数秒は観測できていない。

## 3. A. CUDA allocator warmup（`caching_allocator_warmup`）

固定版 transformers 4.57.6（`modeling_utils.py`）[コード]。`audit_acestep_memory.py` で 7 項目を静的に検証済み。

### 呼び出し関係と条件

```
AutoModel.from_pretrained(path, device_map=...)              # DiT: acestep の _load_main_model_from_checkpoint が呼ぶ
 └─ PreTrainedModel.from_pretrained
     ├─ （meta デバイスでモデルを構築）
     └─ _load_pretrained_model                                  (:5285)
         ├─ _move_missing_keys_from_meta_to_cpu                 (:5359)  欠損キーを CPU に実体化
         ├─ _initialize_missing_keys → initialize_weights()     (:5362)  欠損キーを CPU で初期化
         ├─ if device_map is not None and not is_hqq_or_quark:  (:5430)
         │    └─ caching_allocator_warmup(model, expanded_device_map, hf_quantizer)   (:5432)
         └─ for shard in checkpoint_files: load_shard_file(...)  (直列。並列は HF_ENABLE_PARALLEL_LOADING が真のときだけ)
              └─ _load_state_dict_into_meta_model               (:671)   1 パラメータずつ読み込み
```

- 呼び出し条件: **`device_map` が指定されたときだけ**。従来の経路（`device_map` なし）では呼ばれない
- `warmup` はモジュール変数として呼ばれるため、`transformers.modeling_utils.caching_allocator_warmup` を差し替えれば無効化できる
- `device_map` が CPU / disk / meta のみなら、アクセラレータがないため**何も確保せず早期 return**

### 確保量の計算

1. デバイスごとに、そのデバイスへ置くパラメータのバイト数を合計（`byte_count`）
2. 量子化器がなければ `factor = 2`
3. 空き VRAM が分かる場合、`byte_count = min(byte_count, max(0, 空き VRAM − 1.2GiB))`
4. すでに予約済みで未使用の VRAM（`memory_reserved − memory_allocated`）があれば、その分を差し引く
5. **`torch.empty(byte_count // factor, dtype=float16, device=device)` を 1 回**（= `byte_count` バイト）。戻り値はすぐ参照が外れるが、PyTorch の caching allocator に予約として残る

DiT（4.46GiB）では、前回の空き VRAM は約 6.19GiB で、`min(4.46, 6.19 − 1.2 = 4.99)` = 4.46GiB を 1 回で確保した。観測した VRAM の +4,564MiB と一致 [実測]。

小型モデルで、CUDA をモックして次を確認した [実測・テスト済み]: 確保は 1 回、要素数 × 2 バイト = モデルのバイト数、空き VRAM が小さいときは 0、未使用の予約分だけ減る、CPU の `device_map` では確保なし。

### CPU 側の commit が増える理由（warmup について）

warmup は GPU 上の確保（`device="cuda"`）で、CPU のコードパスではない。**warmup 自体が CPU の private / commit を増やす機構は、ソースからは見つからない** [コード]。ただし、Windows（WDDM）で VRAM の割り当てがプロセスの commit に計上されるかは未測定（M2）。CUDA コンテキストの作成では、VRAM 861MiB に対して private が +3,987MB 増えた [実測]。これは GPU 関連の確保が commit を大きく動かしうることを示すが、warmup の確保が同様かは分からない。

### `--disable-warmup` の実装と副作用

- 実装: `bprime_wrapper.apply_low_peak_patches(disable_warmup=True)` が `transformers.modeling_utils.caching_allocator_warmup` を何もしない関数に置き換える。**既定オフ**。`dit_diag.py --disable-warmup` で明示した時だけ有効
- 副作用 [コード・推定]:
  - VRAM の先確保がなくなり、重みの読み込み時に GPU への確保が**パラメータごとに行われる**。最終的な VRAM 使用量は同じ（4.46GiB + コンテキスト）。ロード時間は伸びる（transformers の docstring は warmup で読み込みが大幅に速くなると説明）。VRAM が逼迫している環境では、パラメータごとの確保でキャッシュが断片化しうる
  - 数値は変わらない（確保のヒントにすぎない）。小型モデルで重み・dtype の一致を確認済み
  - PyTorch 2.7.1 との互換性: 関数の呼び出しを差し替えるだけで、PyTorch の API には依存しない
- **未確認**: 実 DiT で commit の急増が消えるか、アクセス違反が回避されるか。マッピングが主因なら効果はない

## 4. B. DiT の GPU 直接ロード

| 項目 | 内容 | 区分 |
|---|---|---|
| `device_map` の適用 | `from_pretrained` が `{"": "cuda:0"}` を受け取り、`_load_pretrained_model` へ渡す。`from_pretrained` は int / str / `torch.device` を `{"": device}` に正規化するが、**dict はそのまま**通す（値は文字列 `"cuda:0"` のまま） | [コード] |
| 重みの読み込み | `safe_open(shard_file, device="cuda:0")` を開いたまま、パラメータごとに `get_slice(name)[...]` → `.to(param_device)`。ループの終了後に閉じる | [コード] |
| 一時複製 | CPU 側の一時複製はパラメータ単位。ただし safetensors の実装は mmap を使い、private が増える挙動を確認（下記 E2・E5） | [実測] |
| `device_map=None`（従来経路） | `param_device="cpu"`。全パラメータが CPU 上に残る（マッピングを参照するテンソルとして） | [コード] |
| allocator の予約動作 | warmup が 1 回の大きな確保を行い、以降のパラメータの確保はそのキャッシュから切り出される（§3） | [コード] |
| 欠損キーの初期化 | 重みの読み込み**前**に、欠損キーを CPU に実体化して `initialize_weights()` を呼ぶ（`torch_cpu` の演算を使う）。`device_map` とは無関係に起きる | [コード] |
| PyTorch C++ 拡張 | `torch_cpu.dll` は、CPU のテンソル演算・コピー・確保を担う。障害はここで起きた。どの演算かは未特定 | [実測] |
| メモリマッピング | 有り（safetensors は mmap）。1 マッピングごとに commit がファイルサイズ分計上される（E5） | [実測] |

## 5. C. Windows の commit メモリ

### 用語

| 用語 | 意味 | 測り方 |
|---|---|---|
| private bytes | プロセスが確保した commit 量（物理に載っているとは限らない） | `psutil.memory_info().private` |
| RSS（ワーキングセット） | 物理 RAM に載っているページ | `memory_info().rss` |
| 物理の空き | 他プロセスが使える物理 RAM | `GlobalMemoryStatusEx.ullAvailPhys` |
| システムの commit 使用 / 余裕 | 全プロセスの commit の合計。上限（物理 + ページファイル）に近づくと、新しい確保が失敗しうる | `ullTotalPageFile − ullAvailPageFile` / `ullAvailPageFile` |

プロセスの private とシステム commit は**別の量**で、1 つのプロセスの private が増えれば、システム commit の使用も（そのプロセスが唯一の増加要因なら）同じだけ増える。

### CPU のみの実測（`scripts/measure_load_memory.py`、512MB の合成 safetensors）

| 操作 | private | システム commit | RSS |
|---|---|---|---|
| E5: `safe_open` を 1 回開く | +513MB | **+514MB** | +1MB |
| E5: 同じファイルを 2 重に開く | +1,026MB | +1,027MB | +1MB |
| E5: 3 重に開く | **+1,539MB（3 倍）** | +1,540MB | +1MB |
| E5: `get_slice` のみ / `[...]` で実体化 | 追加なし / +72MB（マッピングを参照するビュー） | 同様 | — |
| E5: 閉じる | 元へ戻る（+72MB 残） | 元へ戻る | — |
| E5b: transformers の `load_state_dict(map_location="meta")` | 0 | 0 | 0（ヘッダ読みのみで、マッピングを保持しない） |

- **safetensors の 1 マッピングは、物理 RAM を使わず、ファイルサイズ分の commit だけを計上する**。プロセスの private とシステム commit は 1 対 1 で動く（確認済み）
- したがって前回の急増は、**DiT ファイル（約 4.6GB）が最大 3 重にマップされた**ことで説明できる。ただし transformers のソースでは、ロード中に開いたままなのは `file_pointer` の 1 か所のみ。残りの 2 重分の出どころは**未特定**（`trust_remote_code` のモデルコードは safetensors を開いていない。acestep 側は `silence_latent.pt` を `torch.load` するのみ）
- 別の可能性として、CUDA の確保が commit を増やす（M2）。CUDA コンテキストの作成で private が +3,987MB 増えた事実と整合するが、warmup の確保で同様かは未測定

### クラッシュ時に発生しえた要因の整理

| 要因 | 根拠 | 反証・未確認点 | 強さ |
|---|---|---|---|
| システム commit の不足（確保失敗の連鎖） | 急増 +13.8GB が、開始前の余裕 16.3GB から CUDA 後の消費分を引いた約 9.8GB を超える | クラッシュは急増の 3〜4 秒後。クラッシュ時の余裕は未測定。システム commit の上限は動的（ページファイルの自動拡張）で 93.8〜96.2GB と変動 | 中 |
| CUDA allocator による過剰予約 | VRAM が 7,194MiB（全体 8,188MiB の約 88%）に達した。空き約 1GiB | 確保失敗は通常 PyTorch の OOM 例外になり、アクセス違反にはならない | 低〜中 |
| PyTorch 内部例外 | — | 例外なら Python のトレースバックが出る。出ていない（faulthandler は未設定だった） | 低 |
| `torch_cpu.dll` の不具合・ABI | 障害モジュールが `torch_cpu.dll`。torchao は C++ 拡張をスキップする警告を出している（ただし torchao は未使用の段階） | 従来の CPU 経路（同じ DLL）は本番で成功している | 中 |
| transformers との互換性 | 4.57.6 は固定。`device_map` の dict は一般的な使い方 | カスタムモデル（`trust_remote_code`）と flash attention の組み合わせは未確認 | 中 |
| メモリマッピング・ページファイル | マッピングで commit が最大 3 倍に計上される（E5）。`safe_open(device="cuda")` が、mmap の領域から GPU へコピーする際の問題の可能性 | 実機での再現なし。ページファイルの使用量は 8.89 → 8.95GB とほぼ不変 | 中 |

**結論**: アクセス違反の原因は未特定。最も効率よく切り分けるには、実 DiT を使わない小さな再現実験（§8 の D0）が有効。

## 6. 小型モデルによる検証（CPU・モックのみ、`scripts/test_load_investigation.py` 14 件）

| 検証 | 結果 |
|---|---|
| warmup の確保量（CUDA をモック） | 1 回、モデルのバイト数、空き VRAM − 1.2GiB で制限、未使用の予約分を差し引く、CPU では確保なし |
| `device_map` あり／なしでの warmup の呼び出し | なし → 呼ばれない / `{"": "cpu"}` → 呼ばれる（早期 return） |
| warmup 無効化 | 呼ばれなくなる。重みは完全一致、dtype 維持。元の関数は失われず復元可能 |
| 読み込み経路の重み | 従来（CPU）と `device_map={"":"cpu"}` で完全一致 |
| ロード失敗時 | 例外が伝わり、その後の通常のロードは影響を受けない |
| ロード計装（`safe_open` の開閉・warmup の開始終了） | イベントが順序どおりに記録され、開いたものは必ず閉じ、重みは不変。復元も確認 |
| 高頻度サンプラー | 全サンプルが JSONL に即時保存され、要約を作らなくても残る |

実 DiT・実 GPU では未確認。極小モデル（数百 KB）では、commit の急増もアクセス違反も再現しない。

## 7. 計測精度と監視処理の負荷

| 項目 | 修正前 | 修正後 |
|---|---|---|
| 子プロセスのサンプラー 1 回の処理時間 | 約 20ms（`psutil.virtual_memory()` が約 19.9ms） | **0.032ms**（`GlobalMemoryStatusEx` 1 回、約 2µs。psutil の available と同じ値） |
| 子プロセスのサンプル間隔 | 0.25 秒 | **0.1 秒**（負荷は CPU の約 0.03%。より細かくも可能） |
| ランナー側の 1 回の収集 | 約 270ms（nvidia-smi 165 + 子孫列挙 56 + psutil の virtual / swap 50） | 子孫列挙は 5 回に 1 回、pagefile は 10 回に 1 回、周期は処理時間を差し引いて維持 |
| ランナー側の実際の間隔 | 1〜**3 秒**（クラッシュ前後で欠測） | 最大 **1.0 秒** |
| クラッシュ時のデータ保全 | 要約を最後に 1 回保存するため、クラッシュで系列が失われた | **全サンプルを JSONL へ即時保存**。クラッシュしても残る |
| 記録する項目 | 空き RAM・private・RSS・VRAM（ランナーのみ） | 加えて **commit 余裕**、子プロセス内の VRAM、工程イベント（`safe_open` の開閉、warmup の開始終了、C / D の区切り） |

観測できないもの: 0.1 秒より短い瞬間的なピーク。前回の commit 急増は約 2 秒続いたため 0.1 秒間隔で捉えられるが、**1 秒未満で完了する急増は、事後の監視では防げない**。防御は、急増を見込んだ事前の許可条件（commit 余裕の要求）に頼る。

## 8. 次回の実 GPU 診断の設計

### 比較: D1 / D2 / D3

| 項目 | D1: GPU 直接 + warmup 有効（前回失敗） | D2: GPU 直接 + warmup 無効 | D3: 従来の CPU 経由ロード + GPU 転送 + INT8 |
|---|---|---|---|
| 実行オプション | `--path=gpu-direct`（既定） | `--path=gpu-direct --disable-warmup` | `--path=cpu-via` |
| 物理 RAM の増分 | **+0.86GB 以上**（クラッシュまでの観測。実測） | D1 と同程度の見込み [推定] | 不明。ファイルマップ由来の RSS が一時的に最大 約 4.5GB（再利用可能なページ）。A の初期化では空きが 8.0 → 0.81GB まで低下（CUDA・LM 等を含む）[実測] |
| commit の増分 | **+13.8GB（一時）/ +4.7GB（定常）**（実測） | マッピングが主因なら D1 と同じ。warmup の確保が主因なら一時分が消え +4.7GB 程度 [推定] | 不明（A の初期化では private を記録していない）。マッピング 1〜3 重分（+4.6〜13.8GB）に VRAM 分が加わる可能性 [推定] |
| VRAM | ピーク 約 5.4GiB（7,194MiB − 他アプリ 1,769）（実測） | 最終は同じ。先確保なし [推定] | DiT 4.46GiB + コンテキスト 0.84GiB。INT8 の適用時に層ごとの一時領域 [推定] |
| 既知の実測値 | 上記。アクセス違反 | なし | A の初期化（0.6B LM + DiT、CPU 経由）は本番で成功（旧環境） |
| 残るリスク | アクセス違反が再発する可能性 | アクセス違反の原因が warmup 以外なら再発。ロード時間の増加 | commit の急増は D1 と同様に起きうる。INT8 が実機 DiT で動くかは未確認 |
| 生成品質への影響 | 重みは同一（bf16）。影響なし | 同左 | INT8 の量子化誤差（小型層で相対誤差約 0.8%）。実機の品質は未確認 |
| 許可に必要な安全条件 | 空き RAM ≥ 4.4 + 3 = 7.4GB / **commit 余裕 ≥ 24.2GB** / 空き VRAM ≥ 5,943MiB | 同左（機構が不明のため、commit は D1 と同じ観測値で判定） | 同左（同上） |

**現状（commit 余裕 約 18.9GB）では、D1・D2・D3 のいずれも許可されない**（必要 約 24.2GB）。機構が分かるまで、観測値で保守的に判定する方針を維持する。

### D0: モデルなしの切り分け診断（新規・実装済み・未実行）

目的: 実 DiT を使わずに、commit の計上機構（M1: マッピング / M2: CUDA の確保）と、`safe_open(device="cuda")` が単独でアクセス違反を起こすかを確認する。

内容（`dit_diag.py --steps=d0`）:

1. CUDA コンテキストの作成（B2 で実測済みの構成）
2. 合成の CUDA 確保: 128MiB → 512MiB（warmup と同じ `torch.empty(fp16)` の形式）→ 解放 → `empty_cache`。各操作の前後で private / commit 余裕を記録
3. warmup 相当の確保（256MiB）→ 解放 → 合成 safetensors（256MB）を `safe_open(device="cuda:0")` で 1 テンソルずつ GPU へ（transformers と同じ手順）→ 閉じる
4. 結果の読み方: 確保で commit が増えれば M2、`safe_open` の開閉で増えれば M1。ここでクラッシュすれば、safetensors の CUDA 直接読み込みが原因の候補

D0 の規模: 物理の増分 約 0.5GB（見込み）、commit 最大 +3GB、VRAM 最大 約 1.3GiB、実 DiT は使わない。現状の資源で許可判定は通る（必要: 空き RAM 6.9GB / commit 余裕 13.4GB / VRAM 2,653MiB）。**ただし GPU を使うため、承認が必要**。

### 推奨する次の実 GPU 診断

1. **D0**（モデルなし。commit の機構とアクセス違反の再現性を切り分ける）
2. D0 の結果に応じて:
   - M2（CUDA の確保が commit を増やす）かつ `safe_open(cuda)` が正常 → **D2**（warmup 無効）を、commit 余裕の要求を実測値に更新して検討
   - M1（マッピング）が主因 → **D3**（従来の経路）を、マッピングの重複を実測してから検討
   - `safe_open(cuda)` でクラッシュ → GPU 直接ロードは採用せず、D3 のみ

### 次回に必要な安全条件（変更なし）

- ユーザーの明示承認。見込み消費量 + 安全余裕 3GB（空き RAM）を満たし、commit 余裕は観測値で判定する。**安全下限 3GB は引き下げない**
- 所有プロセスのみ停止。他プロジェクト・Chrome・WSL は停止しない（commit を空けるための停止もしない）
- 自動再試行なし。モデル再取得なし。1.7B LM・XL は使わない
- テレメトリ: 子プロセス 0.1 秒（private・RSS・空き RAM・commit 余裕・VRAM・工程イベント）、ランナー 1 秒、`-X faulthandler` でクラッシュ時のスタックを取得

## 9. 実装・変更したファイル

- `scripts/dit_probe.py` — 高頻度サンプラー（0.1 秒・JSONL ストリーム・commit/VRAM）、ロード計装（`safe_open`・warmup）、D0 ステップ、`--path`（D3）
- `scripts/dit_diag.py` — `--steps=`、`--path=`、`--disable-warmup`
- `scripts/lunaria_profiles.py` — 許可条件に前回の実測値を既定として採用、D0 の判定
- `scripts/phase0_run.py` — ランナー側の収集の軽量化（周期 1.0 秒、`fast_status`）
- `scripts/measure_load_memory.py` — E5（`safe_open` のマッピングと commit）、E5b
- `scripts/audit_acestep_memory.py` — warmup・`safe_open`・ロード順序の静的検証（37 → 44 項目）
- `scripts/test_load_investigation.py`（新規、14 件）、`scripts/test_dit_diag.py`（24 件）
