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


class FakeAutoModel:
    """transformers.AutoModel.from_pretrained の呼び出し規約だけを模したクラス。"""
    calls = []

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        FakeAutoModel.calls.append((str(path), dict(kwargs)))
        return object()


class FakeHandlerForTE:
    """上流 AceStepHandler の Text Encoder ロードと _get_vae_dtype の関係だけを模したクラス。"""
    device = "cuda"
    seen_cpu_dtype = None
    raise_in_load = False

    def _get_vae_dtype(self, device=None):
        d = device or self.device
        return torch.float32 if d == "cpu" else torch.bfloat16

    def _load_text_encoder_and_tokenizer(self, **kwargs):
        FakeHandlerForTE.seen_cpu_dtype = self._get_vae_dtype("cpu")  # 上流: cpu_dtype = self._get_vae_dtype("cpu")
        if FakeHandlerForTE.raise_in_load:
            raise RuntimeError("load failed")
        return "te-path"


class FakeLLMModule:
    seen = []

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*a, **kw):
            FakeLLMModule.seen.append(dict(kw))
            return object()


class FakeLLMHandler:
    dtype = torch.bfloat16

    def _load_pytorch_model(self, model_path, device):
        # 上流: self.llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        FakeLLMModule.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        return True, "ok"


def fresh_lowpeak_env(avail=10.0):
    """毎回クリーンなクラス群にパッチを当てる。avail は RAM ガードが見る空き RAM（GB）のモック。"""
    am = type("AM", (FakeAutoModel,), {})
    th = type("TH", (FakeHandlerForTE,), {})
    lh = type("LH", (FakeLLMHandler,), {})
    FakeAutoModel.calls.clear()
    FakeLLMModule.seen.clear()
    FakeHandlerForTE.seen_cpu_dtype = None
    FakeHandlerForTE.raise_in_load = False
    bw.LOAD_EVENTS.clear()
    bw.LOWPEAK_APPLIED.clear()
    bw.apply_low_peak_patches(automodel_cls=am, handler_cls=th, llm_cls=lh, llm_module=FakeLLMModule,
                              cuda_available=lambda: True, avail_fn=lambda: avail)
    return am, th, lh


class LowPeakLoadTests(unittest.TestCase):
    def test_dit_load_gets_device_map_on_gpu(self):
        am, _, _ = fresh_lowpeak_env()
        am.from_pretrained("C:/x/models/acestep-v15-turbo", trust_remote_code=True, dtype=torch.bfloat16)
        path, kw = FakeAutoModel.calls[-1]
        self.assertEqual(kw["device_map"], {"": "cuda:0"})
        self.assertEqual(kw["dtype"], torch.bfloat16)          # 既存引数は保持
        self.assertTrue(kw["trust_remote_code"])

    def test_dit_device_map_not_overridden_if_given(self):
        am, _, _ = fresh_lowpeak_env()
        am.from_pretrained("acestep-v15-turbo", device_map="cpu")
        self.assertEqual(FakeAutoModel.calls[-1][1]["device_map"], "cpu")

    def test_no_device_map_without_cuda(self):
        am = type("AM2", (FakeAutoModel,), {})
        bw.LOWPEAK_APPLIED.clear()
        bw.apply_low_peak_patches(automodel_cls=am, handler_cls=type("T", (FakeHandlerForTE,), {}),
                                  llm_cls=type("L", (FakeLLMHandler,), {}), llm_module=FakeLLMModule,
                                  cuda_available=lambda: False)
        FakeAutoModel.calls.clear()
        am.from_pretrained("acestep-v15-turbo")
        self.assertNotIn("device_map", FakeAutoModel.calls[-1][1])

    def test_text_encoder_loaded_in_bf16(self):
        am, _, _ = fresh_lowpeak_env()
        am.from_pretrained("models/Qwen3-Embedding-0.6B")
        self.assertEqual(FakeAutoModel.calls[-1][1]["dtype"], torch.bfloat16)

    def test_other_models_untouched(self):
        am, _, _ = fresh_lowpeak_env()
        am.from_pretrained("models/some-other-model", foo=1)
        self.assertEqual(FakeAutoModel.calls[-1][1], {"foo": 1})

    def test_te_cpu_dtype_is_bf16_only_during_load(self):
        _, th, _ = fresh_lowpeak_env()
        h = th()
        h._load_text_encoder_and_tokenizer()
        self.assertEqual(FakeHandlerForTE.seen_cpu_dtype, torch.bfloat16)   # ロード中は bf16
        self.assertEqual(h._get_vae_dtype("cpu"), torch.float32)           # 終了後は本来の挙動（VAE 等は fp32）
        self.assertEqual(h._get_vae_dtype("cuda"), torch.bfloat16)

    def test_te_override_is_removed_on_exception(self):
        _, th, _ = fresh_lowpeak_env()
        h = th()
        FakeHandlerForTE.raise_in_load = True
        with self.assertRaises(RuntimeError):
            h._load_text_encoder_and_tokenizer()
        self.assertNotIn("_get_vae_dtype", h.__dict__)
        self.assertEqual(h._get_vae_dtype("cpu"), torch.float32)

    def test_lm_loaded_with_handler_dtype_and_module_restored(self):
        _, _, lh = fresh_lowpeak_env()
        real = FakeLLMModule.AutoModelForCausalLM
        lh()._load_pytorch_model("models/acestep-5Hz-lm-0.6B", "cuda")
        self.assertEqual(FakeLLMModule.seen[-1]["dtype"], torch.bfloat16)
        self.assertTrue(FakeLLMModule.seen[-1]["trust_remote_code"])
        self.assertIs(FakeLLMModule.AutoModelForCausalLM, real)           # プロキシは元に戻る

    def test_patch_is_idempotent(self):
        am, th, lh = fresh_lowpeak_env()
        n = len(bw.LOWPEAK_APPLIED)
        bw.apply_low_peak_patches(automodel_cls=am, handler_cls=th, llm_cls=lh, llm_module=FakeLLMModule,
                                  cuda_available=lambda: True)
        self.assertEqual(len(bw.LOWPEAK_APPLIED), n)

    def test_load_events_record_private_memory(self):
        am, _, _ = fresh_lowpeak_env()
        am.from_pretrained("acestep-v15-turbo")
        ev = bw.LOAD_EVENTS[-1]
        self.assertIn("private_mb", ev)
        self.assertEqual(ev["injected"], {"device_map": "cuda:0"})


class RamGuardTests(unittest.TestCase):
    """段階ごとの RAM ガード: 足りないときは、その段階の確保前に安全に失敗する（preflight 閾値とは別の保護）。"""

    def test_guard_formula(self):
        need = lp.STAGE_NEED_GB["text_encoder"]
        floor = lp.STAGE_GUARD_FLOOR_GB
        with self.assertRaises(MemoryError):
            bw.ram_guard("text_encoder", avail_fn=lambda: need + floor - 0.01)
        self.assertAlmostEqual(bw.ram_guard("text_encoder", avail_fn=lambda: need + floor), need + floor)

    def test_dit_stage_blocked_before_loading(self):
        am, _, _ = fresh_lowpeak_env(avail=3.2)  # 3.2 - 0.6 < 3.0
        with self.assertRaises(MemoryError):
            am.from_pretrained("models/acestep-v15-turbo")
        self.assertEqual(FakeAutoModel.calls, [])  # 上流の from_pretrained は呼ばれない（何も確保していない）

    def test_text_encoder_stage_blocked_before_loading(self):
        am, _, _ = fresh_lowpeak_env(avail=4.0)  # 4.0 - 1.6 < 3.0
        with self.assertRaises(MemoryError):
            am.from_pretrained("models/Qwen3-Embedding-0.6B")
        self.assertEqual(FakeAutoModel.calls, [])

    def test_lm_stage_blocked_returns_upstream_failure_shape(self):
        _, _, lh = fresh_lowpeak_env(avail=4.5)  # 4.5 - 1.8 < 3.0
        ok, msg = lh()._load_pytorch_model("models/acestep-5Hz-lm-0.6B", "cuda")
        self.assertFalse(ok)
        self.assertIn("RAM ガード", msg)
        self.assertEqual(FakeLLMModule.seen, [])

    def test_all_stages_pass_with_enough_ram(self):
        am, _, lh = fresh_lowpeak_env(avail=10.0)
        am.from_pretrained("models/acestep-v15-turbo")
        am.from_pretrained("models/Qwen3-Embedding-0.6B")
        ok, _ = lh()._load_pytorch_model("models/acestep-5Hz-lm-0.6B", "cuda")
        self.assertTrue(ok)
        self.assertEqual(len(FakeAutoModel.calls), 2)

    def test_untargeted_models_are_not_guarded(self):
        am, _, _ = fresh_lowpeak_env(avail=0.1)
        am.from_pretrained("models/some-other-model")  # 対象外のモデルはガードしない
        self.assertEqual(len(FakeAutoModel.calls), 1)

    def test_guard_floor_matches_hard_abort_floor(self):
        self.assertEqual(lp.STAGE_GUARD_FLOOR_GB, lp.HARD_ABORT_FLOOR_GB)
        self.assertEqual(lp.b_prime_admission()["required_free_ram_gb"], 13.49)  # preflight 閾値は引き下げていない


class RealTransformersTinyTests(unittest.TestCase):
    """実物の transformers（固定版）が、注入する引数を実際に受理するかを極小モデル（CPU）で確認する。"""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import transformers
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=2, head_dim=16)
        cls.te_dir = Path(cls.tmp.name) / bw.TE_DIR
        cls.dit_dir = Path(cls.tmp.name) / bw.DIT_DIR
        for d in (cls.te_dir, cls.dit_dir):
            transformers.Qwen3Model(cfg).to(torch.bfloat16).save_pretrained(d, safe_serialization=True)
        cls.lm_dir = Path(cls.tmp.name) / "acestep-5Hz-lm-0.6B"
        transformers.Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(cls.lm_dir, safe_serialization=True)
        cls.transformers = transformers

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_default_load_is_fp32_which_is_the_problem(self):
        """上流の Text Encoder / LM の呼び出し（dtype なし）は、bf16 保存でも fp32 に展開される（現状の再現）。"""
        m = self.transformers.AutoModel.from_pretrained(self.te_dir)
        self.assertEqual(next(m.parameters()).dtype, torch.float32)

    def test_patched_text_encoder_loads_bf16(self):
        am = type("AMReal", (self.transformers.AutoModel,), {})
        bw.apply_low_peak_patches(automodel_cls=am, handler_cls=type("T", (FakeHandlerForTE,), {}),
                                  llm_cls=type("L", (FakeLLMHandler,), {}), llm_module=FakeLLMModule,
                                  cuda_available=lambda: False)
        m = am.from_pretrained(self.te_dir)
        self.assertEqual(next(m.parameters()).dtype, torch.bfloat16)

    def test_device_map_dict_is_accepted_by_pinned_transformers(self):
        """DiT に注入する形式 {"": device} を固定版 transformers が受理し、dtype と値が保たれる（CPU で代用）。"""
        base = self.transformers.AutoModel.from_pretrained(self.dit_dir, dtype=torch.bfloat16)
        m = self.transformers.AutoModel.from_pretrained(self.dit_dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        self.assertEqual(next(m.parameters()).dtype, torch.bfloat16)
        for (n1, p1), (n2, p2) in zip(base.named_parameters(), m.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), n1)

    def test_lm_dtype_kwarg_is_accepted_and_values_equal_via_fp32_detour(self):
        """LM を最初から bf16 で読んでも、fp32 経由（現状）と値が完全一致する（bf16→fp32→bf16 は可逆）。"""
        direct = self.transformers.AutoModelForCausalLM.from_pretrained(self.lm_dir, dtype=torch.bfloat16)
        detour = self.transformers.AutoModelForCausalLM.from_pretrained(self.lm_dir).to("cpu").to(torch.bfloat16)
        for (n1, p1), (n2, p2) in zip(direct.named_parameters(), detour.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), n1)

    def test_real_classes_exist_and_patch_marks_applied(self):
        from acestep.handler import AceStepHandler
        import acestep.llm_inference as llm_mod
        self.assertTrue(hasattr(AceStepHandler, "_load_text_encoder_and_tokenizer"))
        self.assertTrue(hasattr(llm_mod.LLMHandler, "_load_pytorch_model"))
        self.assertTrue(hasattr(llm_mod, "AutoModelForCausalLM"))


class UpstreamAssumptionTests(unittest.TestCase):
    def test_real_upstream_matches_assumptions(self):
        self.assertEqual(bw.upstream_assumption_problems(), [])

    def test_tampered_upstream_is_detected(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src = bw.ROOT / "engine" / "ACE-Step-1.5" / "acestep"
            for rel in ("core/generation/handler/init_service_loader.py",
                        "core/generation/handler/init_service_loader_components.py",
                        "core/generation/handler/memory_utils.py", "llm_inference.py"):
                dst = Path(tmp) / "acestep" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src / rel, dst)
            self.assertEqual(bw.upstream_assumption_problems(tmp), [])
            p = Path(tmp) / "acestep/core/generation/handler/init_service_loader.py"
            p.write_text(p.read_text(encoding="utf-8").replace("AutoModel.from_pretrained(",
                                                                "AutoModel.from_pretrained(device_map='auto', "),
                         encoding="utf-8")
            self.assertTrue(any("DiT" in x for x in bw.upstream_assumption_problems(tmp)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
