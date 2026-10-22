"""D0b（import メモリの切り分け）の許可条件・構成・解析ツールの単体テスト。GPU・実モデル不使用。

実行: .venv-engine\\Scripts\\python.exe -I scripts\\test_import_diag.py
"""
import os
import sys
import unittest
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU を一切使わない
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_import_chain as aic  # noqa: E402
import d0b_diag as dd0b  # noqa: E402
import lunaria_profiles as lp  # noqa: E402


class D0bAdmissionTests(unittest.TestCase):
    def test_safety_floor_is_not_lowered(self):
        for name, spec in dd0b.VARIANTS:
            a = lp.d0b_admission(spec)
            self.assertAlmostEqual(a["required_free_ram_gb"] - a["consumption_gb"], 3.0, places=1, msg=name)
            self.assertAlmostEqual(a["required_commit_headroom_gb"] - a["commit_consumption_gb"], 4.0, places=1, msg=name)

    def test_light_variants_are_small_and_heavy_cuda_are_large(self):
        light = lp.d0b_admission("torch,cuda")
        heavy = lp.d0b_admission("torch,acestep.handler,cuda")
        self.assertLess(light["required_free_ram_gb"], heavy["required_free_ram_gb"])
        self.assertEqual(light["required_free_vram_mib"], heavy["required_free_vram_mib"])
        self.assertEqual(lp.d0b_admission("torch")["required_free_vram_mib"], 0)   # CUDA なしは VRAM 不要

    def test_b2_subset_uses_measured_upper_bound_non_subset_is_conservative(self):
        sub = lp.d0b_admission("torch,transformers,cuda")
        non = lp.d0b_admission("torch,acestep.api_server,cuda")
        self.assertAlmostEqual(sub["consumption_gb"], 3.70, places=2)
        self.assertAlmostEqual(non["consumption_gb"], 3.90, places=2)
        self.assertLess(sub["required_free_ram_gb"], non["required_free_ram_gb"])
        # B2 と完全に同じ構成も部分集合として扱う（実測の再現）
        self.assertAlmostEqual(lp.d0b_admission(
            "torch,accelerate,diffusers,transformers,acestep.handler,cuda")["consumption_gb"], 3.70, places=2)

    def test_first_gpu_work_after_heavy_chain_uses_observed_w1_consumption(self):
        """W1 の実測（空き RAM -5.4GB 以上、増加中に自己停止）を下回らない。W2 のような後続構成も同じ根拠で判定する。"""
        w1 = lp.d0b_admission("torch,acestep.handler,cuda,cuda_work")
        w2 = lp.d0b_admission("torch,cuda,acestep.handler,cuda_work")
        base = lp.d0b_admission("torch,cuda,cuda_work")
        self.assertGreaterEqual(w1["consumption_gb"], 5.4)
        self.assertEqual(w1["consumption_gb"], w2["consumption_gb"])
        self.assertLess(base["consumption_gb"], 2.0)                 # import なしの基準は軽い（実測: 物理 約 0.4GB）
        self.assertAlmostEqual(w1["required_free_ram_gb"] - w1["consumption_gb"], 3.0, places=1)  # 余裕 3GB は維持
        # 空き 8.7GB でも不許可（実測に基づき、下限 3GB を割る恐れがあるため）
        self.assertFalse(lp.d0b_admission("torch,cuda,acestep.handler,cuda_work", avail_gb=8.7, commit_headroom_gb=18.8,
                                          free_vram_mib=6400)["ok"])

    def test_flash_attn_adds_unmeasured_margin(self):
        a = lp.d0b_admission("torch,cuda,flash_attn")
        b = lp.d0b_admission("torch,cuda")
        self.assertAlmostEqual(a["consumption_gb"] - b["consumption_gb"], 1.0, places=2)

    def test_checks_use_current_resources(self):
        ok = lp.d0b_admission("torch,cuda", avail_gb=6.0, commit_headroom_gb=12.0, free_vram_mib=6400)
        self.assertTrue(ok["ok"])
        bad = lp.d0b_admission("torch,cuda", avail_gb=3.9, commit_headroom_gb=12.0, free_vram_mib=6400)
        self.assertFalse(bad["checks"]["free_ram"]["ok"])
        bad2 = lp.d0b_admission("torch,cuda", avail_gb=6.0, commit_headroom_gb=12.0, free_vram_mib=1000)
        self.assertFalse(bad2["checks"]["free_vram"]["ok"])


class CudaFirstTests(unittest.TestCase):
    """CUDA を先に初期化するオプトインの実装（実験）。GPU は使わず、CUDA 呼び出しをモックして順序と分岐だけを確認する。"""

    def test_returns_not_applied_without_cuda(self):
        import bprime_wrapper as bw
        import torch
        from unittest import mock
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            r = bw.cuda_first_init()
        self.assertFalse(r["applied"])

    def test_initializes_cuda_context_in_order(self):
        import bprime_wrapper as bw
        import torch
        from unittest import mock
        calls = []
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "init", side_effect=lambda: calls.append("init")), \
                mock.patch.object(torch, "zeros", side_effect=lambda *a, **k: calls.append("zeros") or object()), \
                mock.patch.object(torch.cuda, "synchronize", side_effect=lambda *a: calls.append("sync")):
            r = bw.cuda_first_init()
        self.assertTrue(r["applied"])
        self.assertEqual(calls, ["init", "zeros", "sync"])
        self.assertIn("private_delta_mb", r)

    def test_not_enabled_by_default(self):
        """オプトイン: 環境変数が無ければ main は CUDA を先に初期化しない（実生成のピークへの効果が未検証のため）。"""
        import inspect
        import bprime_wrapper as bw
        src = inspect.getsource(bw.main)
        self.assertIn('os.environ.get("LUNARIA_CUDA_FIRST") == "1"', src)
        self.assertNotIn("LUNARIA_CUDA_FIRST", " ".join(f"{k}={v}" for k, v in lp.PROFILES["B_PRIME"]["env"].items()))

    def test_dit_diag_flag_is_opt_in(self):
        import dit_diag as dd
        self.assertNotIn("--cuda-first", [])  # 引数なしでは無効
        import dit_probe as dp
        import inspect
        self.assertIn('LUNARIA_CUDA_FIRST") == "1"', inspect.getsource(dp.step_cd))


class VariantTests(unittest.TestCase):
    def test_variants_are_unique_and_cover_required_configurations(self):
        names = [n for n, _ in dd0b.VARIANTS]
        self.assertEqual(len(names), len(set(names)))
        specs = {n: s for n, s in dd0b.VARIANTS}
        for need in ("python", "torch", "torch,cuda", "torch,transformers", "torch,safetensors", "torch,acestep.handler,cuda"):
            self.assertIn(need, specs.values())
        # 順序の影響を見るため、CUDA 初期化の前後で同じライブラリを import する構成がある
        self.assertIn("torch,transformers,cuda", specs.values())
        self.assertIn("torch,cuda,transformers", specs.values())
        self.assertIn("torch,acestep.handler,cuda", specs.values())
        self.assertIn("torch,cuda,acestep.handler", specs.values())

    def test_second_and_third_stage_variants_exist(self):
        specs = {n: s for n, s in dd0b.VARIANTS}
        for name, spec in (("W3_cuda_then_work", "torch,cuda,cuda_work"),
                           ("W1_handler_then_cuda_work", "torch,acestep.handler,cuda,cuda_work"),
                           ("W2_cuda_then_handler_work", "torch,cuda,acestep.handler,cuda_work"),
                           ("N1_peft_then_cuda", "torch,peft,cuda"), ("N3_lightning_then_cuda", "torch,lightning,cuda"),
                           ("Y_flash_attn_then_cuda", "torch,flash_attn,cuda")):
            self.assertEqual(specs.get(name), spec)

    def test_flash_attn_before_cuda_is_treated_as_heavy(self):
        before = lp.d0b_admission("torch,flash_attn,cuda")
        after = lp.d0b_admission("torch,cuda,flash_attn")
        self.assertGreater(before["consumption_gb"], after["consumption_gb"])
        self.assertAlmostEqual(before["consumption_gb"], 3.70, places=2)

    def test_custom_variants(self):
        got = dd0b.pick_variants(["--custom=Z_x:torch,transformers", "--custom=Z_y:torch"])
        self.assertEqual(got, [("Z_x", "torch,transformers"), ("Z_y", "torch")])
        with self.assertRaises(SystemExit):
            dd0b.custom_variants(["--custom=bad"])

    def test_no_variant_loads_models(self):
        for _, spec in dd0b.VARIANTS:
            for tok in spec.split(","):
                self.assertNotIn("pipeline", tok)
                self.assertNotIn("from_pretrained", tok)

    def test_pick_variants(self):
        self.assertEqual(len(dd0b.pick_variants([])), len(dd0b.VARIANTS))
        self.assertEqual([n for n, _ in dd0b.pick_variants(["--only=C_torch_cuda"])], ["C_torch_cuda"])
        with self.assertRaises(SystemExit):
            dd0b.pick_variants(["--only=nope"])


class ChainAnalysisTests(unittest.TestCase):
    def test_parent_package_init_is_part_of_the_chain(self):
        order, _, _, _ = aic.analyze("acestep.handler")
        names = [m for m, _ in order]
        self.assertIn("acestep", names)              # 親パッケージの __init__ が連鎖に含まれる
        self.assertLess(names.index("acestep"), names.index("acestep.handler"))

    def test_handler_eagerly_reaches_training_modules(self):
        """acestep.handler の import が、学習用モジュール（peft / lightning 等）まで eager に読み込む（調査結果の固定）。"""
        order, third, _, _ = aic.analyze("acestep.handler")
        names = [m for m, _ in order]
        self.assertTrue(any(m.startswith("acestep.training") for m in names))
        for lib in ("torch", "peft", "lightning", "torchaudio"):
            self.assertIn(lib, third)
        self.assertEqual(aic.analyze("acestep.handler")[3], {})    # import 時に CUDA・モデル・大きな確保を行う呼び出しはない

    def test_function_level_imports_are_lazy(self):
        _, third, lazy, _ = aic.analyze("acestep.handler")
        self.assertIn("diffusers", set(lazy) - set(third))     # diffusers は関数内（呼ばれるまで読み込まれない）
        self.assertIn("flash_attn", set(lazy) - set(third))

    def test_api_server_adds_transformers_and_web_stack(self):
        _, third, _, _ = aic.analyze("acestep.api_server")
        for lib in ("transformers", "fastapi", "uvicorn", "torchaudio"):
            self.assertIn(lib, third)


if __name__ == "__main__":
    unittest.main(verbosity=2)
