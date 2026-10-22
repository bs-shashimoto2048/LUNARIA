"""DiT 単独診断（許可条件・自己監視・集計・DiT 単独パッチ）の単体テスト。GPU・実モデル・ネットワーク不使用。

実行: .venv-engine\\Scripts\\python.exe -I scripts\\test_dit_diag.py
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
import dit_probe as dp  # noqa: E402
import lunaria_profiles as lp  # noqa: E402


def tiny_quantized():
    from torchao.quantization import Int8WeightOnlyConfig, quantize_
    m = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128)).to(torch.bfloat16)
    quantize_(m, Int8WeightOnlyConfig())
    return m


class AdmissionTests(unittest.TestCase):
    def test_safety_floor_is_not_lowered(self):
        self.assertEqual(lp.SAFETY_FLOOR_GB, 3.0)
        for step in ("b1", "b2", "cd", "d0"):
            a = lp.dit_diag_admission(step)
            self.assertAlmostEqual(a["required_free_ram_gb"] - a["consumption_gb"], 3.0, places=1)

    def test_b1_uses_measured_import(self):
        a = lp.dit_diag_admission("b1", avail_gb=7.0, commit_headroom_gb=16.0, free_vram_mib=6400)
        self.assertAlmostEqual(a["consumption_gb"], 2.55, places=2)
        self.assertTrue(a["ok"])
        low = lp.dit_diag_admission("b1", avail_gb=5.4, commit_headroom_gb=16.0, free_vram_mib=6400)
        self.assertFalse(low["ok"])
        self.assertFalse(low["checks"]["free_ram"]["ok"])

    def test_b2_uses_previous_cuda_measurement(self):
        a = lp.dit_diag_admission("b2")
        self.assertAlmostEqual(a["consumption_gb"], lp.CUDA_BASE_PHYS_MEASURED_GB, places=2)       # 物理（実測 3.40GB）
        self.assertAlmostEqual(a["commit_consumption_gb"], lp.CUDA_BASE_PRIVATE_MEASURED_GB, places=2)  # commit（実測 6.38GB）
        self.assertEqual(lp.CUDA_BASE_PHYS_MEASURED_GB, 3.40)

    def test_d0_is_small_and_independent_of_dit(self):
        d0 = lp.dit_diag_admission("d0")
        cd = lp.dit_diag_admission("cd")
        self.assertLess(d0["required_commit_headroom_gb"], cd["required_commit_headroom_gb"])
        self.assertLess(d0["required_free_vram_mib"], cd["required_free_vram_mib"])
        # 現状の commit 余裕（約 18〜19GB）・空き RAM（約 7〜8GB）で許可される規模であること
        ok = lp.dit_diag_admission("d0", avail_gb=7.2, commit_headroom_gb=18.0, free_vram_mib=6400)
        self.assertTrue(ok["ok"], ok.get("checks"))

    def test_cd_uses_previous_b2_measurement_until_rerun(self):
        need = lp.STAGE_NEED_GB["dit_gpu_direct"]
        no_b2 = lp.dit_diag_admission("cd")
        self.assertAlmostEqual(no_b2["consumption_gb"], 3.40 + need, places=2)
        self.assertAlmostEqual(no_b2["commit_consumption_gb"], 6.38 + lp.DIT_COMMIT_PEAK_OBSERVED_GB, places=2)
        with_b2 = lp.dit_diag_admission("cd", measured={"b2_phys_after_cuda_gb": 1.5, "b2_private_after_cuda_gb": 2.9,
                                                        "ctx_vram_mib": 300})
        self.assertAlmostEqual(with_b2["consumption_gb"], 1.5 + need, places=2)           # 物理消費で空き RAM を判定
        # commit は private + 観測した DiT 段階の急増（+13.8GB）で判定
        self.assertAlmostEqual(with_b2["commit_consumption_gb"], 2.9 + lp.DIT_COMMIT_PEAK_OBSERVED_GB, places=2)
        self.assertLess(with_b2["required_free_ram_gb"], no_b2["required_free_ram_gb"])
        self.assertAlmostEqual(with_b2["required_free_ram_gb"] - with_b2["consumption_gb"], 3.0, places=2)  # 余裕は 3GB のまま
        self.assertEqual(with_b2["required_free_vram_mib"], lp.DIT_FILE_VRAM_MIB + 300 + lp.VRAM_MARGIN_MIB)

    def test_cd_not_allowed_when_free_ram_is_short(self):
        # 現在の空き約 7GB・推定上限（B2 未実測）では不許可になる（推定だけで許可しない）
        a = lp.dit_diag_admission("cd", avail_gb=7.0, commit_headroom_gb=16.0, free_vram_mib=6400)
        self.assertFalse(a["ok"])
        self.assertFalse(a["checks"]["free_ram"]["ok"])

    def test_cd_blocked_by_commit_headroom_even_with_ample_ram(self):
        """前回の失敗の再現防止: 物理 RAM が十分でも、観測した commit 急増を賄えない余裕では許可しない。"""
        m = {"b2_phys_after_cuda_gb": 3.4, "b2_private_after_cuda_gb": 6.38, "ctx_vram_mib": 861}
        a = lp.dit_diag_admission("cd", measured=m, avail_gb=8.5, commit_headroom_gb=16.3, free_vram_mib=6419)
        self.assertTrue(a["checks"]["free_ram"]["ok"])
        self.assertFalse(a["checks"]["commit_headroom"]["ok"])
        self.assertFalse(a["ok"])
        self.assertGreater(a["required_commit_headroom_gb"], 16.3)
        ok = lp.dit_diag_admission("cd", measured=m, avail_gb=8.5, commit_headroom_gb=a["required_commit_headroom_gb"],
                                   free_vram_mib=6419)
        self.assertTrue(ok["checks"]["commit_headroom"]["ok"])

    def test_dit_stage_ram_need_reflects_observation(self):
        self.assertGreaterEqual(lp.STAGE_NEED_GB["dit_gpu_direct"], 0.86)  # 観測した物理低下（約 0.86GB）を下回らない

    def test_vram_and_commit_checks(self):
        a = lp.dit_diag_admission("cd", measured={"b2_phys_after_cuda_gb": 1.5, "b2_private_after_cuda_gb": 2.8,
                                                  "ctx_vram_mib": 300},
                                  avail_gb=9.0, commit_headroom_gb=5.0, free_vram_mib=6400)
        self.assertFalse(a["checks"]["commit_headroom"]["ok"])
        self.assertTrue(a["checks"]["free_vram"]["ok"])

    def test_unknown_step_rejected(self):
        with self.assertRaises(ValueError):
            lp.dit_diag_admission("zz")


class SelfMonitorTests(unittest.TestCase):
    def make(self, avails, aborts, commits=None):
        it = iter(avails)
        cit = iter(commits) if commits is not None else None
        m = dp.SelfMonitor(floor_gb=3.0, consecutive=2, avail_fn=lambda: next(it), private_fn=lambda: 100 * dp.MB,
                           rss_fn=lambda: 50 * dp.MB, on_abort=aborts.append, interval=0.25,
                           commit_fn=(lambda: next(cit)) if cit is not None else (lambda: 50.0))
        return m

    def test_commit_headroom_floor_aborts_even_with_ample_ram(self):
        aborts = []
        m = self.make([8.0, 8.0, 8.0, 8.0], aborts, commits=[16.0, 2.9, 2.5, 16.0])
        for _ in range(4):
            m.tick()
        self.assertEqual(len(aborts), 1)
        self.assertIn("commit", aborts[0])
        self.assertEqual(m.summary()["min_commit_headroom_gb"], 2.5)

    def test_single_commit_dip_does_not_abort(self):
        aborts = []
        m = self.make([8.0] * 4, aborts, commits=[16.0, 2.9, 16.0, 2.9])
        for _ in range(4):
            m.tick()
        self.assertEqual(aborts, [])

    def test_two_consecutive_lows_abort(self):
        aborts = []
        m = self.make([5.0, 2.9, 2.8, 5.0], aborts)
        for _ in range(4):
            m.tick()
        self.assertEqual(len(aborts), 1)
        self.assertTrue(m.aborted)
        self.assertIn("下限", aborts[0])

    def test_single_dip_does_not_abort(self):
        aborts = []
        m = self.make([5.0, 2.9, 5.0, 2.9, 5.0], aborts)
        for _ in range(5):
            m.tick()
        self.assertEqual(aborts, [])

    def test_exactly_at_floor_is_allowed(self):
        aborts = []
        m = self.make([3.0, 3.0, 3.0], aborts)
        for _ in range(3):
            m.tick()
        self.assertEqual(aborts, [])

    def test_abort_fires_only_once_and_tracks_extremes(self):
        aborts = []
        m = self.make([2.0, 2.0, 1.0, 1.0], aborts)
        for _ in range(4):
            m.tick()
        self.assertEqual(len(aborts), 1)
        self.assertEqual(m.summary()["min_avail_gb"], 1.0)
        self.assertEqual(m.summary()["max_private_mb"], 100)

    def test_series_is_downsampled_to_one_per_second(self):
        m = self.make([5.0] * 8, [])
        for _ in range(8):
            m.tick()
        self.assertEqual(len(m.series), 2)  # 0.25 秒間隔 → 4 tick ごとに 1 点


class ParamSummaryTests(unittest.TestCase):
    def test_bf16_model_summary(self):
        m = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
        s = dp.param_summary(m, bw)
        self.assertEqual(s["n_params"], 1)
        self.assertEqual(s["devices"], {"cpu": 1})
        self.assertEqual(s["by_kind"]["bfloat16"]["count"], 1)
        self.assertEqual(s["bytes_total"], 64 * 64 * 2)

    def test_quantized_model_summary_counts_int8_bytes(self):
        s = dp.param_summary(tiny_quantized(), bw)
        self.assertIn("int8(AffineQuantizedTensor)", s["by_kind"])
        q = s["by_kind"]["int8(AffineQuantizedTensor)"]
        self.assertEqual(q["count"], 2)
        self.assertEqual(q["bytes"], 2 * 128 * 128)  # int8 は 1 バイト/要素


class DiagOptionTests(unittest.TestCase):
    def test_parse_steps_and_path(self):
        import dit_diag as dd
        self.assertEqual(dd.parse_steps([]), ("b1", "b2", "cd"))
        self.assertEqual(dd.parse_steps(["--steps=d0"]), ("d0",))
        self.assertEqual(dd.parse_steps(["--steps=b2,cd"]), ("b2", "cd"))
        with self.assertRaises(SystemExit):
            dd.parse_steps(["--steps=zz"])
        self.assertEqual(dd.parse_path([]), "gpu-direct")
        self.assertEqual(dd.parse_path(["--path=cpu-via"]), "cpu-via")
        with self.assertRaises(SystemExit):
            dd.parse_path(["--path=bogus"])

    def test_probe_accepts_d0_step_and_synthetic_helpers_exist(self):
        self.assertTrue(callable(dp.step_d0))
        self.assertTrue(callable(dp.install_load_instrumentation))


class DitOnlyPatchTests(unittest.TestCase):
    def test_dit_only_components_do_not_touch_te_or_lm(self):
        class AM:
            calls = []

            @classmethod
            def from_pretrained(cls, path, *a, **kw):
                AM.calls.append((str(path), dict(kw)))
                return object()

        am = type("AMD", (AM,), {})
        bw.LOWPEAK_APPLIED.clear()
        # handler_cls / llm_cls を渡さず components=("dit",) だけ → それらの import / パッチは起きない
        bw.apply_low_peak_patches(automodel_cls=am, cuda_available=lambda: True, avail_fn=lambda: 10.0,
                                  components=("dit",))
        self.assertEqual(len(bw.LOWPEAK_APPLIED), 1)
        self.assertIn("AutoModel.from_pretrained", bw.LOWPEAK_APPLIED[0])
        am.from_pretrained("models/acestep-v15-turbo", dtype=torch.bfloat16)
        self.assertEqual(AM.calls[-1][1]["device_map"], {"": "cuda:0"})

    def test_warmup_is_not_disabled_by_default_and_optional(self):
        import types
        wm = types.SimpleNamespace(caching_allocator_warmup=lambda *a, **k: "real")

        class AM:
            @classmethod
            def from_pretrained(cls, path, *a, **kw):
                return object()

        bw.LOWPEAK_APPLIED.clear()
        bw.apply_low_peak_patches(automodel_cls=type("A1", (AM,), {}), cuda_available=lambda: True,
                                  avail_fn=lambda: 10.0, components=("dit",), warmup_module=wm)
        self.assertEqual(wm.caching_allocator_warmup(), "real")  # 既定ではオフ（上流の挙動のまま）
        bw.apply_low_peak_patches(automodel_cls=type("A2", (AM,), {}), cuda_available=lambda: True,
                                  avail_fn=lambda: 10.0, components=("dit",), disable_warmup=True, warmup_module=wm)
        self.assertIsNone(wm.caching_allocator_warmup())            # 明示した場合のみ無効化
        self.assertTrue(any("warmup" in x for x in bw.LOWPEAK_APPLIED))

    def test_dit_stage_guard_blocks_when_ram_is_short(self):
        class AM:
            calls = []

            @classmethod
            def from_pretrained(cls, path, *a, **kw):
                AM.calls.append(path)
                return object()

        am = type("AMG", (AM,), {})
        bw.apply_low_peak_patches(automodel_cls=am, cuda_available=lambda: True, avail_fn=lambda: 3.3,
                                  components=("dit",))
        with self.assertRaises(MemoryError):
            am.from_pretrained("models/acestep-v15-turbo")
        self.assertEqual(AM.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
