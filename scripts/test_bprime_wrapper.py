"""B′ ラッパー / プロファイル定義の単体テスト（GPU・モデル・ネットワーク不使用）。

実行: .venv-engine\\Scripts\\python.exe -I scripts\\test_bprime_wrapper.py
"""
import os
import sys
import unittest
import warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU を一切使わない（torch の import 前に設定）
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bprime_wrapper as bw  # noqa: E402
import lunaria_profiles as lp  # noqa: E402


def tiny_quantized_model():
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    m = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256)).to(torch.bfloat16)
    quantize_(m, Int8WeightOnlyConfig())
    return m


class FakeHandler:
    """上流 AceStepHandler の initialize_service 呼び出し規約だけを模したテスト用クラス。"""
    calls = []
    behavior = "quantized"  # quantized | silent_fallback | fail

    def initialize_service(self, **kwargs):
        FakeHandler.calls.append(dict(kwargs))
        if FakeHandler.behavior == "fail":
            return "boom", False
        if FakeHandler.behavior == "silent_fallback":
            # 上流が torchao の ImportError で警告だけ出して量子化なしに進む挙動
            self.quantization = None
            self.model = nn.Linear(8, 8).to(torch.bfloat16)
        else:
            self.quantization = kwargs.get("quantization")
            self.model = tiny_quantized_model()
        return "ok", True


class ProfileTests(unittest.TestCase):
    def test_b_prime_env_pins_backend_and_keeps_offload_auto(self):
        env = lp.PROFILES[lp.PROFILE_B_PRIME]["env"]
        self.assertEqual(env["ACESTEP_LM_BACKEND"], "pt")
        self.assertEqual(env["ACESTEP_CONFIG_PATH"], "acestep-v15-turbo")
        self.assertEqual(env["ACESTEP_LM_MODEL_PATH"], "acestep-5Hz-lm-0.6B")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("ACESTEP_OFFLOAD_TO_CPU", env)         # offload 全体は無条件に無効化しない
        self.assertNotIn("ACESTEP_OFFLOAD_DIT_TO_CPU", env)     # DiT は既定 False = GPU 常駐

    def test_build_env_unsets_conflicting_keys(self):
        base = {"ACESTEP_OFFLOAD_DIT_TO_CPU": "true", "ACESTEP_COMPILE_MODEL": "1", "KEEP": "x"}
        env = lp.build_env(lp.PROFILE_B_PRIME, base)
        self.assertNotIn("ACESTEP_OFFLOAD_DIT_TO_CPU", env)
        self.assertNotIn("ACESTEP_COMPILE_MODEL", env)
        self.assertEqual(env["KEEP"], "x")

    def test_admission_is_derived_from_measurements(self):
        adm = lp.b_prime_admission()
        peak = lp.MEASURED_A_INIT_PEAK_RSS_GB
        self.assertAlmostEqual(adm["required_free_ram_gb"], round(peak + max(1.0, peak * 0.15) + 2.0, 2))
        self.assertEqual(adm["required_free_vram_mib"], lp.MEASURED_A_INIT_PEAK_VRAM_OURS_MIB + lp.VRAM_MARGIN_MIB)
        self.assertEqual(adm["hard_abort_free_ram_gb"], 3.0)
        self.assertEqual(lp.A_MIN_FREE_RAM_GB, 14.0)  # 従来構成 A の保護条件は維持

    def test_profile_c_is_defined_but_not_run(self):
        self.assertEqual(lp.PROFILES[lp.PROFILE_C]["status"], "defined_not_run")
        self.assertEqual(lp.PROFILES[lp.PROFILE_C]["env"]["ACESTEP_INIT_LLM"], "false")

    def test_assert_profile_env(self):
        env = lp.build_env(lp.PROFILE_B_PRIME, {})
        saved = dict(os.environ)
        try:
            os.environ.clear()
            os.environ.update(env)
            bw.assert_profile_env()  # 一致なら例外なし
            os.environ["ACESTEP_OFFLOAD_DIT_TO_CPU"] = "true"
            with self.assertRaises(RuntimeError):
                bw.assert_profile_env()
        finally:
            os.environ.clear()
            os.environ.update(saved)


class VerifyInt8Tests(unittest.TestCase):
    def test_quantized_model_passes(self):
        h = type("H", (), {})()
        h.quantization, h.model = bw.QUANT, tiny_quantized_model()
        r = bw.verify_int8(h, require_device="cpu")
        self.assertTrue(r["ok"], r["reasons"])
        self.assertEqual(r["n_quantized"], 2)
        self.assertEqual(r["quantized_devices"], {"cpu": 2})
        self.assertGreater(r["quantized_numel_fraction"], 0.9)

    def test_wrong_device_is_rejected(self):
        h = type("H", (), {})()
        h.quantization, h.model = bw.QUANT, tiny_quantized_model()
        r = bw.verify_int8(h, require_device="cuda")  # 重みは cpu 上 → B′ の前提（GPU 上で量子化）を満たさない
        self.assertFalse(r["ok"])
        self.assertTrue(any("cuda" in x for x in r["reasons"]))

    def test_silent_fallback_is_detected(self):
        h = type("H", (), {})()
        h.quantization, h.model = None, nn.Linear(8, 8).to(torch.bfloat16)
        r = bw.verify_int8(h, require_device="cpu")
        self.assertFalse(r["ok"])
        self.assertTrue(any("黙って" in x for x in r["reasons"]))
        self.assertTrue(any("1 つも無い" in x for x in r["reasons"]))

    def test_attr_set_but_weights_not_quantized_is_detected(self):
        h = type("H", (), {})()
        h.quantization, h.model = bw.QUANT, nn.Linear(8, 8).to(torch.bfloat16)  # 属性だけ立っていて実体は bf16
        r = bw.verify_int8(h, require_device="cpu")
        self.assertFalse(r["ok"])


class PatchTests(unittest.TestCase):
    def setUp(self):
        FakeHandler.calls.clear()
        FakeHandler.behavior = "quantized"
        bw.VERIFY_RESULTS.clear()
        bw.HANDLERS.clear()
        # 毎回クリーンなクラスにパッチする
        self.cls = type("FH", (FakeHandler,), {})
        bw.apply_patch(self.cls)

    def test_quantization_is_injected_and_verified(self):
        # テスト環境は CPU のため、検証の要求デバイスだけ cpu に緩める（ロジックは本番と同じ）
        orig = bw.verify_int8
        bw.verify_int8 = lambda h, require_device="cuda", cls=None: orig(h, require_device="cpu", cls=cls)
        try:
            status, ok = self.cls().initialize_service(project_root="x", config_path="acestep-v15-turbo")
        finally:
            bw.verify_int8 = orig
        self.assertTrue(ok, status)
        self.assertEqual(FakeHandler.calls[-1]["quantization"], bw.QUANT)
        self.assertTrue(bw.VERIFY_RESULTS[-1]["ok"])

    def test_offload_dit_is_refused(self):
        status, ok = self.cls().initialize_service(offload_dit_to_cpu=True)
        self.assertFalse(ok)
        self.assertEqual(FakeHandler.calls, [])  # 上流の初期化は呼ばれない

    def test_other_quantization_is_refused(self):
        status, ok = self.cls().initialize_service(quantization="fp8_weight_only")
        self.assertFalse(ok)
        self.assertEqual(FakeHandler.calls, [])

    def test_silent_fallback_makes_init_fail(self):
        FakeHandler.behavior = "silent_fallback"
        status, ok = self.cls().initialize_service(project_root="x")
        self.assertFalse(ok)
        self.assertIn("フォールバックしない", status)
        self.assertFalse(bw.VERIFY_RESULTS[-1]["ok"])

    def test_upstream_failure_is_propagated(self):
        FakeHandler.behavior = "fail"
        status, ok = self.cls().initialize_service(project_root="x")
        self.assertFalse(ok)
        self.assertFalse(bw.VERIFY_RESULTS[-1]["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
