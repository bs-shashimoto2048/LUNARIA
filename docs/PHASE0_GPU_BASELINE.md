# PHASE0_GPU_BASELINE — GPU 環境と実測結果

最終更新: 2026-10-22 / 状態: **Phase 0 Stage A は未達（メモリ不足により中断）**

> 本書は実測できた事実のみを記載する。未実施・未確認の項目は「未検証」と明記する。

## 1. ハードウェア（実測）

| 項目 | 値 |
|---|---|
| OS | Windows 11 Pro 10.0.26100 |
| CPU | Intel Core i9-13900HX（24C/32T） |
| 物理 RAM | 31.7 GB |
| GPU | **NVIDIA GeForce RTX 4070 Laptop GPU**（仕様書の想定は RTX 4060。実機が異なる） |
| VRAM | 8 GB（8188 MiB）、compute capability 8.9 |
| NVIDIA Driver | 537.53（nvidia-smi 表示 CUDA 12.2） |
| ストレージ | C: 空き約 194 GB |
| ページファイル | 約 60.7 GB（使用 約 8〜11 GB） |

## 2. ソフトウェア（実測）

| 項目 | 値 |
|---|---|
| Python | 3.11.17（uv 管理、プロジェクト専用 `.venv-engine`） |
| uv | 0.13.5（`.tools/uv`、PATH 未変更） |
| PyTorch | 2.7.1+cu128 / Torch CUDA runtime 12.8 |
| `torch.cuda.is_available()` | True（行列積の実行も確認。ドライバー 537 でも cu128 が動作） |
| ACE-Step commit | `ca1e85fe9430179831e6bc6be790c332190a3866` |
| DiT | `acestep-v15-turbo` |
| LM | `acestep-5Hz-lm-0.6B` |

## 3. エンジン構成（ACE-Step 自動判定の実測ログ）

| 項目 | 値 | 根拠 |
|---|---|---|
| GPU tier | tier3 | `Configuration Tier: tier3`（VRAM 8.0GB） |
| LM 初期化 | 有効 | `init_llm=True`、`/health` で `llm_initialized: true` |
| LM backend | **PyTorch（pt）** | `vLLM disabled due to insufficient free VRAM (total=8.00GB, free=0.57GB, need>=2.0GB free) — falling back to PyTorch backend` |
| CPU offload | 有効（Auto） | `Auto-setting offload_to_cpu=True based on GPU memory` |
| DiT | cuda 常駐 | `Keeping main model on cuda (persistent)` |
| attention | flash_attention | ログ |
| quantization | **ログに記載なし（未確認）** | 抽出パターンに該当行なし |
| VAE デコード | **CPU タイル分割へ自動切替** | `Only 0.00 GB free VRAM; auto-enabling CPU VAE decode` |

## 4. 実測した性能

| 項目 | 値 |
|---|---|
| 初回モデル取得を含む初期化（旧環境・前回） | 1204.9 秒（取得 約 10GB を含む） |
| エンジン起動〜ポート待受〜モデルロード完了（今回・取得なし） | 175.3 秒（モデルロードは待受後 0.1 秒＝起動処理内で完了） |
| 初期化時の最大 VRAM（GPU 全体） | 7186 MiB（その後 7888 MiB まで上昇） |
| プロセスツリー最大 RSS | **11,740 MB** |
| DiT 拡散時間（30 秒インスト、LM 計画付き） | 約 6 秒（`diffusion_time_cost` ログ） |
| VAE デコード | **6 分以上かかり未完了のまま中断**（CPU デコード＋RAM 逼迫） |
| 生成完了 | **なし**（WAV は未生成） |

## 5. 実行前後の空きリソース

| 時点 | 空き RAM | VRAM 使用 |
|---|---|---|
| 今回の実行前（事前確認） | 8.03 GB | 約 1.76 GB（他アプリ分） |
| モデルロード中の最小 | 0.81 GB | 7.2〜7.9 GB |
| 生成中の下限（中断時） | 1.47 GB | 7.63 GB |
| 停止後 | 回復（11.5 GB） | 1.74 GB（実行前と同水準） |

## 6. 中断の原因（実測に基づく推定）

1. エンジンのプロセスツリーが **最大 約 11.7GB の RSS** を使う（CPU offload 有効＋LM/DiT/Text Encoder/VAE の出し入れ）。
2. 実行前の空き RAM が約 8GB しかなく、Chrome（約 105 プロセス・4.5GB）・WSL（約 1GB）・他プロジェクトの推論サーバー等と競合し、**空き 1.5〜2.5GB まで低下**。
3. VRAM も他アプリ約 1.7GB＋DiT 常駐で空きが 0 となり、VAE が CPU デコードに切り替わった。CPU デコードが RAM 逼迫下でページングし、完了しなかった。
4. 前回（旧環境）の強制停止も同じ構造的原因（RAM 逼迫）の可能性が高い。ただし前回は RAM 計測が無く、**確定ではない**。

自動ウォッチドッグ（空き RAM < 1.5GB が 3 回連続）により、自分のエンジンのみ正常停止して中断した。

## 7. 停止検証（今回・Stage A 中断時）

| 項目 | 結果 |
|---|---|
| 正常停止（CTRL_BREAK） | 成功（強制終了なし） |
| 停止対象 PID | 78260（親）, 96400, 23956（所有子孫のみ） |
| 残存プロセス | なし |
| ポート 8001 | 解放（待受なし） |
| GPU メモリ | 1737 MiB（実行前 1756 MiB と同水準） |

## 8. 未検証項目

- 30 秒インスト WAV の生成・WAV 検証（Stage A 本体）
- 30 秒／60 秒の日本語歌詞付き生成（Stage B / C。今回の承認範囲外）
- quantization の実状態
- 歌声の実音声確認
- 空きメモリが十分ある状態での性能
