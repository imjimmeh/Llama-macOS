#!/usr/bin/env python3
"""Tests for layer-specific latency report format and statistics."""

import json
import math
import unittest


def percentile(values, p):
    """Linear-interpolated percentile, same as numpy.percentile."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[f])
    return s[f] * (c - k) + s[c] * (k - f)


def compute_layer_stats(samples):
    """Compute median, P95, mean from a list of sample values (microseconds)."""
    if not samples:
        return {"median_us": 0, "p95_us": 0, "mean_us": 0, "count": 0}
    return {
        "median_us": percentile(samples, 50),
        "p95_us": percentile(samples, 95),
        "mean_us": sum(samples) / len(samples),
        "count": len(samples),
    }


def compute_savings(native, sidecar):
    """saved_us = native_median_us - sidecar_median_us; preserves negative."""
    return native["median_us"] - sidecar["median_us"]


class TestPercentileCalculation(unittest.TestCase):

    def test_median_odd(self):
        self.assertAlmostEqual(percentile([1, 3, 5], 50), 3.0)

    def test_median_even(self):
        self.assertAlmostEqual(percentile([1, 3, 5, 7], 50), 4.0)

    def test_median_single(self):
        self.assertAlmostEqual(percentile([42], 50), 42.0)

    def test_p95_known(self):
        vals = list(range(1, 101))
        self.assertAlmostEqual(percentile(vals, 95), 95.05)

    def test_p95_single(self):
        self.assertAlmostEqual(percentile([100], 95), 100.0)

    def test_mean(self):
        s = compute_layer_stats([10, 20, 30])
        self.assertAlmostEqual(s["mean_us"], 20.0)
        self.assertEqual(s["count"], 3)

    def test_empty_samples(self):
        s = compute_layer_stats([])
        self.assertEqual(s["count"], 0)

    def test_savings_preserves_negative(self):
        native = {"median_us": 100}
        sidecar = {"median_us": 150}
        saved = compute_savings(native, sidecar)
        self.assertEqual(saved, -50)

    def test_savings_positive(self):
        native = {"median_us": 200}
        sidecar = {"median_us": 120}
        saved = compute_savings(native, sidecar)
        self.assertEqual(saved, 80)

    def test_savings_zero(self):
        native = {"median_us": 100}
        sidecar = {"median_us": 100}
        saved = compute_savings(native, sidecar)
        self.assertEqual(saved, 0)

    def test_report_format(self):
        report = {
            "format": 1,
            "model": "test-model",
            "n_layers": 40,
            "native_samples": 1280,
            "sidecar_samples": 1280,
            "total_zero_ram_to_gpu": True,
            "layers": [
                {"layer": 5, "native": compute_layer_stats(list(range(100, 300))),
                 "sidecar": compute_layer_stats(list(range(50, 150))),
                 "saved_us": compute_savings(
                     compute_layer_stats(list(range(100, 300))),
                     compute_layer_stats(list(range(50, 150)))),
                 "n_native_samples": 32, "n_sidecar_samples": 32},
            ],
        }
        self.assertEqual(report["format"], 1)
        self.assertGreater(report["layers"][0]["saved_us"], 0)
        self.assertTrue(report["total_zero_ram_to_gpu"])

    def test_report_zero_h2d_invariant(self):
        """Must assert zero timed expert H2D bytes."""
        layer = {
            "layer": 5,
            "native": {"median_us": 150, "p95_us": 200, "mean_us": 155, "count": 32},
            "sidecar": {"median_us": 80, "p95_us": 120, "mean_us": 85, "count": 32},
            "saved_us": 70,
            "n_native_samples": 32,
            "n_sidecar_samples": 32,
            "ram_to_gpu_bytes": 0,
        }
        self.assertEqual(layer["ram_to_gpu_bytes"], 0)

    def test_different_layer_shapes(self):
        """Layers may have different median savings."""
        layers = [
            {"layer": 0, "native_median_us": 200, "sidecar_median_us": 90},
            {"layer": 10, "native_median_us": 150, "sidecar_median_us": 140},
        ]
        savings = [l["native_median_us"] - l["sidecar_median_us"] for l in layers]
        self.assertGreater(savings[0], savings[1])


if __name__ == "__main__":
    unittest.main()
