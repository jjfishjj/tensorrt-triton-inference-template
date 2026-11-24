"""Tests for the benchmark harness.

The harness exists to stop unmeasured numbers reaching a table, so most of
these check that it refuses to produce one: that an unavailable backend reports
a reason instead of a figure, that the model under test is the *same* model
everywhere, and that the Triton config it writes says what Triton needs.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import jsonschema  # noqa: E402

from inferbench.accuracy import AccuracyStats, compare  # noqa: E402
from inferbench.backends import REFERENCE, TensorRTBackend, TorchBackend, all_backends  # noqa: E402
from inferbench.bench import BenchResult, LatencyStats, measure, run_backend  # noqa: E402
from inferbench.model import build_torch_model, export_onnx  # noqa: E402
from inferbench.report import standard_payload  # noqa: E402
from inferbench.triton_repo import copy_plan, render_config, write_repository  # noqa: E402

ART = ROOT / "artifacts"
SCHEMA = json.loads((ROOT / "benchmark-results.schema.json").read_text())


class PlanCopyTests(unittest.TestCase):
    def test_copy_creates_parent_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "build" / "engine.plan"
            source.parent.mkdir()
            source.write_bytes(b"TensorRT plan bytes\x00\xff")
            destination = copy_plan(source, root / "repo" / "resnet" / "1" / "model.plan")
            self.assertEqual(destination.read_bytes(), source.read_bytes())

    def test_missing_source_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                copy_plan(Path(tmp) / "missing.plan", Path(tmp) / "out.plan")

    def test_existing_destination_directory_uses_source_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.plan"
            source.write_bytes(b"plan")
            destination_dir = root / "destination"
            destination_dir.mkdir()
            destination = copy_plan(source, destination_dir)
            self.assertEqual(destination, destination_dir / "source.plan")
            self.assertEqual(destination.read_bytes(), b"plan")

    def test_same_file_and_symlink_are_verified_no_ops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.plan"
            source.write_bytes(b"plan")
            self.assertEqual(copy_plan(source, source), source)
            link = root / "link.plan"
            link.symlink_to(source)
            self.assertEqual(copy_plan(source, link), link)
            self.assertTrue(link.is_symlink())

    def test_trt_repository_requires_and_installs_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "built.plan"
            plan.write_bytes(b"engine")
            spec = type("Spec", (), {"name": "tiny", "input_shape": (3, 4, 4),
                                      "output_size": 2, "onnx_path": plan})()
            repo = write_repository(spec, root / "repo", platform="tensorrt_plan",
                                    plan_path=plan)
            self.assertEqual((repo / "tiny" / "1" / "model.plan").read_bytes(), b"engine")
            with self.assertRaises(ValueError):
                write_repository(spec, root / "missing", platform="tensorrt_plan")


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


class ReportTests(unittest.TestCase):
    def test_standard_payload_uses_true_even_sample_median_and_population_stdev(self):
        samples = [0.001, 0.003, 0.009, 0.010]
        latency = LatencyStats(
            backend="fake", batch=2, iterations=len(samples), warmup=1,
            mean_ms=statistics.fmean(samples) * 1000, p50_ms=9.0,
            p90_ms=10.0, p99_ms=10.0, min_ms=1.0, max_ms=10.0,
            stdev_ms=statistics.pstdev(samples) * 1000,
            throughput_ips=statistics.fmean(samples) ** -1,
            samples_seconds=samples,
        )
        results = [
            BenchResult("fake", True, latency=latency),
            BenchResult("missing", False, reason="not installed"),
            BenchResult("broken", True, error="RuntimeError: fake failure"),
        ]
        accuracies = {"fake": AccuracyStats("fake", "fake", 2, 0.0, 0.0, 1.0)}
        payload = standard_payload(
            "tiny", 2, len(samples), 1,
            {"machine": "test", "processor": "fake", "platform": "test", "python": "3.12"},
            results, "fake", accuracies,
        )
        measurement = payload["measurements"][0]
        self.assertEqual(measurement["statistics"]["median"], statistics.median(samples))
        self.assertNotEqual(measurement["statistics"]["median"], 0.009)
        self.assertEqual(measurement["statistics"]["stdev"], statistics.pstdev(samples))
        self.assertEqual([m["status"] for m in payload["measurements"]],
                         ["completed", "skipped", "failed"])
        self.assertEqual(payload["measurements"][1]["samples"], [])
        self.assertIsNone(payload["measurements"][2]["statistics"]["mean"])
        jsonschema.validate(payload, SCHEMA, format_checker=jsonschema.FormatChecker())

    def test_independent_schema_fixture_validates(self):
        fixture = {
            "schema_version": "1.0.0", "benchmark_id": "fixture", "run_id": "fixture-run",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "git": {"commit": None, "dirty": False},
            "environment": {
                "os": "test", "architecture": "x86_64", "python": "3.12",
                "software": {}, "gpus": [],
            },
            "parameters": {"model": "fixture", "batch": 1},
            "measurements": [{
                "case_id": "fixture-b1", "backend": "fake", "status": "completed",
                "unit": "seconds", "input_rows": 1, "output_rows": 1,
                "warmup_iterations": 0, "samples": [0.001],
                "statistics": {"median": 0.001, "mean": 0.001, "stdev": 0.0,
                                "min": 0.001, "max": 0.001},
            }],
        }
        jsonschema.validate(fixture, SCHEMA, format_checker=jsonschema.FormatChecker())


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
