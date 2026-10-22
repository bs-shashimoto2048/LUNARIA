"""DiT ロード時の commit 急増・warmup の調査用テスト。GPU・実モデル・ネットワーク不使用（CPU・モック・極小モデルのみ）。

確認すること:
  - transformers の caching_allocator_warmup の確保量の計算（CUDA 関数をモックして、torch.empty の引数を検証）
  - device_map を渡したときのロード経路で warmup が呼ばれる条件（CPU の device_map では早期 return）
  - warmup 無効化の仕組みと副作用（重み・dtype の一致、元への復元）
  - ロード計装（safe_open の開閉・warmup の開始終了）が挙動を変えないこと
  - 高頻度サンプラーの JSONL ストリームと、サンプラー自体の負荷
実行: .venv-engine\\Scripts\\python.exe -I scripts\\test_load_investigation.py
"""
import json
import os
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU を一切使わない（torch の import 前に設定）
warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import transformers  # noqa: E402
import transformers.modeling_utils as mu  # noqa: E402

import bprime_wrapper as bw  # noqa: E402
import dit_probe as dp  # noqa: E402

GIB = 1024**3


def make_tiny(dirpath, dtype=torch.bfloat16, hidden=64, layers=2):
    cfg = transformers.Qwen3Config(vocab_size=128, hidden_size=hidden, intermediate_size=hidden * 2,
                                   num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    m = transformers.Qwen3Model(cfg).to(dtype)
    m.save_pretrained(dirpath, safe_serialization=True)
    return m


class WarmupAllocationTests(unittest.TestCase):
    """caching_allocator_warmup が確保する量を、CUDA をモックして検証する（GPU は使わない）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "tiny"
        make_tiny(cls.dir)
        cls.model = transformers.AutoModel.from_pretrained(cls.dir)  # CPU
        cls.expected_keys = list(cls.model.state_dict().keys())
        cls.param_bytes = sum(p.numel() * p.element_size() for p in cls.model.parameters())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_warmup(self, free_gib, reserved_gib=0.0, allocated_gib=0.0, factor_quant=None):
        """モックした CUDA で warmup を呼び、torch.empty に渡された (要素数, dtype, device) を返す。"""
        calls = []
        dev_map = mu.expand_device_map({"": torch.device("cuda:0")}, self.expected_keys)
        fake_cuda = mock.MagicMock()
        fake_cuda.mem_get_info.return_value = (int(free_gib * GIB), 8 * GIB)
        fake_cuda.memory_reserved.return_value = int(reserved_gib * GIB)
        fake_cuda.memory_allocated.return_value = int(allocated_gib * GIB)
        fake_cuda.current_device.return_value = 0
        with mock.patch.object(mu.torch, "cuda", fake_cuda), \
                mock.patch.object(mu.torch, "empty", side_effect=lambda *a, **k: calls.append((a, k))):
            mu.caching_allocator_warmup(self.model, dev_map, None)
        return calls

    def test_warmup_allocates_full_model_bytes_in_one_call_when_vram_is_ample(self):
        calls = self.run_warmup(free_gib=6.0)
        self.assertEqual(len(calls), 1)  # 1 回の大きな確保（モデル全体分）
        (n_elem,), kw = calls[0]
        self.assertEqual(kw["dtype"], torch.float16)
        self.assertEqual(n_elem * 2, self.param_bytes)        # fp16 要素数 × 2 バイト = モデルのバイト数
        self.assertEqual(kw["device"].type, "cuda")

    def test_warmup_is_clamped_to_free_vram_minus_1_2_gib(self):
        # 空き 0.00001GiB 相当 → max(0, free - 1.2GiB) = 0。確保量は 0 要素
        calls = self.run_warmup(free_gib=1.0)
        (n_elem,), _ = calls[0]
        self.assertEqual(n_elem, 0)

    def test_unused_reserved_memory_reduces_the_allocation(self):
        full = self.run_warmup(free_gib=6.0)[0][0][0]
        reduced = self.run_warmup(free_gib=6.0, reserved_gib=0.0001, allocated_gib=0.0)[0][0][0]
        self.assertLess(reduced, full)  # すでに予約済み（未使用）の分だけ確保が減る

    def test_cpu_device_map_does_not_trigger_allocation(self):
        calls = []
        dev_map = mu.expand_device_map({"": torch.device("cpu")}, self.expected_keys)
        with mock.patch.object(mu.torch, "empty", side_effect=lambda *a, **k: calls.append(a)):
            mu.caching_allocator_warmup(self.model, dev_map, None)
        self.assertEqual(calls, [])  # アクセラレータが無ければ何もしない


class WarmupCallPathTests(unittest.TestCase):
    """from_pretrained の経路で warmup が呼ばれる条件と、無効化の仕組み・副作用（CPU の極小モデル）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "tiny"
        cls.ref = make_tiny(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.orig = mu.caching_allocator_warmup
        self.calls = []

        def spy(*a, **k):
            self.calls.append((len(a[1]) if len(a) > 1 else None))
            return self.orig(*a, **k)

        mu.caching_allocator_warmup = spy

    def tearDown(self):
        mu.caching_allocator_warmup = self.orig

    def test_warmup_called_only_when_device_map_is_given(self):
        transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16)
        self.assertEqual(self.calls, [])                          # device_map なし → warmup なし（従来の CPU 経路）
        transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        self.assertEqual(len(self.calls), 1)                      # device_map あり → warmup が呼ばれる（CPU なら早期 return）

    def test_disable_warmup_skips_the_call_and_keeps_weights_and_dtype(self):
        wm = mu

        class AM(transformers.AutoModel):
            pass

        before = transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        self.calls.clear()
        bw.LOWPEAK_APPLIED.clear()
        bw.apply_low_peak_patches(automodel_cls=AM, cuda_available=lambda: False, components=("dit",),
                                  disable_warmup=True, warmup_module=wm)
        # 注: cuda_available=False のため device_map は注入されない。無効化だけを確認する
        after = AM.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        self.assertEqual(self.calls, [])                          # spy は no-op に置き換わり、呼ばれない
        self.assertTrue(getattr(mu.caching_allocator_warmup, "_lunaria_noop", False))
        for (n1, p1), (n2, p2) in zip(before.named_parameters(), after.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertEqual(p1.dtype, p2.dtype)                  # dtype の維持
            self.assertTrue(torch.equal(p1, p2), n1)              # 重みの一致

    def test_weights_are_identical_across_load_paths(self):
        a = transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16)                          # 従来（CPU）
        b = transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})  # device_map
        for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), n1)

    def test_disable_is_a_noop_and_is_restorable(self):
        orig = mu.caching_allocator_warmup
        wm = type("W", (), {"caching_allocator_warmup": orig})()
        bw.apply_low_peak_patches(automodel_cls=type("A", (transformers.AutoModel,), {}), cuda_available=lambda: False,
                                  components=("dit",), disable_warmup=True, warmup_module=wm)
        self.assertIsNone(wm.caching_allocator_warmup())
        self.assertIsNotNone(orig)                                # 元の関数は失われていない（復元可能）

    def test_load_error_propagates_and_state_is_not_corrupted(self):
        with self.assertRaises(Exception):
            transformers.AutoModel.from_pretrained(Path(self.dir) / "does-not-exist", device_map={"": "cpu"})
        m = transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        self.assertEqual(next(m.parameters()).dtype, torch.bfloat16)  # 例外のあとも通常どおりロードできる


class InstrumentationTests(unittest.TestCase):
    """ロード計装（safe_open の開閉・warmup の開始終了）は挙動を変えず、時刻と commit 余裕つきで記録する。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name) / "tiny"
        make_tiny(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_events_are_recorded_in_order_and_weights_unchanged(self):
        dp.EVENTS.clear()
        ref = transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        uninstall = dp.install_load_instrumentation(mu)
        try:
            got = transformers.AutoModel.from_pretrained(self.dir, dtype=torch.bfloat16, device_map={"": "cpu"})
        finally:
            uninstall()
        names = [e["event"] for e in dp.EVENTS]
        self.assertIn("warmup_start", names)
        self.assertIn("warmup_end", names)
        self.assertGreaterEqual(names.count("safe_open"), 1)
        self.assertEqual(names.count("safe_open"), names.count("safe_open_close"))  # 開いたものは必ず閉じる
        self.assertLess(names.index("warmup_start"), names.index("warmup_end"))
        for e in dp.EVENTS:
            self.assertIn("commit_headroom_gb", e)
            self.assertIn("private_mb", e)
        for (n1, p1), (n2, p2) in zip(ref.named_parameters(), got.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), n1)
        self.assertFalse(getattr(mu.safe_open, "_lunaria_instr", False))             # 復元されている

    def test_install_is_idempotent(self):
        u1 = dp.install_load_instrumentation(mu)
        try:
            u2 = dp.install_load_instrumentation(mu)
            u2()  # 2 回目は何もしない
            self.assertTrue(getattr(mu.safe_open, "_lunaria_instr", False))
        finally:
            u1()
        self.assertFalse(getattr(mu.safe_open, "_lunaria_instr", False))


class SamplerStreamTests(unittest.TestCase):
    def test_every_tick_is_streamed_and_survives_without_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            stream = dp.JsonlStream(path)
            it = iter([5.0, 5.0, 5.0])
            m = dp.SelfMonitor(avail_fn=lambda: next(it), private_fn=lambda: 100 * dp.MB, rss_fn=lambda: 50 * dp.MB,
                               commit_fn=lambda: 16.0, stream=stream, vram_fn=lambda: 1234, interval=0.1)
            for _ in range(3):
                m.tick()
            # summary() を呼ばなくても（= クラッシュしても）ファイルに全サンプルが残る
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["vram_used_mb"], 1234)
            self.assertEqual(rows[0]["commit_headroom_gb"], 16.0)
            stream.close()

    def test_events_and_samples_share_the_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.jsonl"
            dp.STREAM = dp.JsonlStream(path)
            try:
                dp.event("marker", note="x")
            finally:
                dp.STREAM.close()
                dp.STREAM = None
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["event"], "marker")

    def test_sampler_overhead_is_small(self):
        """実際の psutil / GlobalMemoryStatusEx を使った 1 サンプルの処理時間。0.1 秒間隔に対して無視できること。"""
        m = dp.SelfMonitor(interval=0.1)
        for _ in range(200):
            m.tick()
        s = m.summary()
        self.assertLess(s["tick_cost_mean_ms"], 5.0)  # 実測は通常 1ms 未満。0.1 秒間隔に対して 5% 未満
        print(f"\n[計測] サンプラー 1 回の処理時間 平均 {s['tick_cost_mean_ms']} ms（目標間隔 100ms）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
