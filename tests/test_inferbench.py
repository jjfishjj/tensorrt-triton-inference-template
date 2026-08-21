"""Tests for the benchmark harness.

The harness exists to stop unmeasured numbers reaching a table, so most of
these check that it refuses to produce one: that an unavailable backend reports
a reason instead of a figure, that the model under test is the *same* model
everywhere, and that the Triton config it writes says what Triton needs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from inferbench.accuracy import compare  # noqa: E402
from inferbench.backends import REFERENCE, TensorRTBackend, TorchBackend, all_backends  # noqa: E402
from inferbench.bench import measure, run_backend  # noqa: E402
from inferbench.model import build_torch_model, export_onnx  # noqa: E402
from inferbench.triton_repo import render_config, write_repository  # noqa: E402

ART = ROOT / "artifacts"


class ModelTests(unittest.TestCase):
    def test_weights_are_deterministic(self):
        """Regression: without a fixed seed each call built a different random
        network, so the accuracy check compared two initialisations rather than
        two runtimes and reported 0% top-1 agreement for every backend."""
        import torch

        a = build_torch_model("tinycnn")
        b = build_torch_model("tinycnn")
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_export_keeps_the_batch_axis_dynamic(self):
        import onnx

        spec = export_onnx("tinycnn", ART)
        model = onnx.load(str(spec.onnx_path))
        dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertEqual(dim.WhichOneof("value"), "dim_param",
                         "batch must stay symbolic or every batch size needs its own engine")


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = export_onnx("tinycnn", ART)

    def test_unavailable_backend_reports_a_reason_not_a_number(self):
        r = run_backend(TensorRTBackend("fp16"), self.spec, 1, 2, 1)
        if not r.available:
            self.assertTrue(r.reason, "an unavailable backend must say why")
            self.assertIsNone(r.latency)

    def test_reference_backend_is_always_available(self):
        ref = next(b for b in all_backends() if b.name == REFERENCE)
        self.assertTrue(ref.available().ok)

    def test_backends_agree_with_the_reference(self):
        ref = TorchBackend("cpu")
        ref.load(self.spec, 4)
        x = self.spec.sample(4, seed=99)
        expected = ref.infer(x)
        ref.close()

        for b in all_backends():
            if b.name == REFERENCE or not b.available().ok:
                continue
            with self.subTest(backend=b.name):
                b.load(self.spec, 4)
                stats = compare(b.name, b.infer(x), expected, REFERENCE)
                b.close()
                self.assertEqual(stats.top1_agreement, 1.0)
                self.assertLess(stats.max_abs_diff, 1.0)

    def test_measurement_reports_ordered_percentiles(self):
        b = TorchBackend("cpu")
        b.load(self.spec, 1)
        s = measure(b, self.spec, batch=1, iterations=12, warmup=3)
        b.close()
        self.assertLessEqual(s.min_ms, s.p50_ms)
        self.assertLessEqual(s.p50_ms, s.p90_ms)
        self.assertLessEqual(s.p90_ms, s.p99_ms)
        self.assertLessEqual(s.p99_ms, s.max_ms)
        self.assertAlmostEqual(s.throughput_ips, 1 / (s.mean_ms / 1000), places=3)


class TritonRepoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = export_onnx("tinycnn", ART)

    def test_dims_exclude_the_implicit_batch_axis(self):
        """With max_batch_size > 0 Triton adds the batch axis itself; repeating
        it in dims yields a model that loads and then rejects every request."""
        cfg = render_config(self.spec, max_batch_size=32)
        dims_line = [l for l in cfg.splitlines() if "dims:" in l][0]
        self.assertNotIn("-1", dims_line)
        self.assertEqual(dims_line.count(","), len(self.spec.input_shape) - 1)

    def test_dynamic_batching_is_enabled(self):
        cfg = render_config(self.spec, max_batch_size=16)
        self.assertIn("dynamic_batching", cfg)
        self.assertIn("preferred_batch_size: [ 4, 8, 16 ]", cfg)

    def test_repository_layout(self):
        root = write_repository(self.spec, ROOT / "artifacts" / "repo_test")
        self.assertTrue((root / self.spec.name / "config.pbtxt").is_file())
        self.assertTrue((root / self.spec.name / "1" / "model.onnx").is_file())


if __name__ == "__main__":
    unittest.main()
