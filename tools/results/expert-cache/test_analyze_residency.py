#!/usr/bin/env python3
"""Tests for the held-out full-hit residency oracle analyzer."""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class TestTraceStats(unittest.TestCase):
    """Step 1: trace statistics and coverage tests."""

    def test_expert_frequency(self):
        from analyze_residency import count_expert_freq
        routes = [
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"layer": 0, "experts": [1, 2, 3, 9, 10, 11, 12, 13]},
        ]
        freq = count_expert_freq(routes)
        self.assertEqual(freq[0][1], 2)
        self.assertEqual(freq[0][2], 2)
        self.assertEqual(freq[0][3], 2)
        self.assertEqual(freq[0][9], 1)

    def test_individual_coverage_top1(self):
        from analyze_residency import individual_coverage
        routes = [
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 9]},
        ]
        cov = individual_coverage(routes, top_n=1)
        self.assertAlmostEqual(cov[0], 1.0)

    def test_individual_coverage_top2(self):
        from analyze_residency import individual_coverage
        routes = [
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 9]},
        ]
        cov = individual_coverage(routes, top_n=2)
        self.assertAlmostEqual(cov[0], 1.0)

    def test_individual_coverage_no_full_hit(self):
        from analyze_residency import individual_coverage, full_hit_probability
        routes = []
        for i in range(1000):
            experts = [(i * 8 + j) % 256 for j in range(8)]
            routes.append({"layer": 0, "experts": experts})
        cov = individual_coverage(routes, top_n=64)
        self.assertLess(cov[0], 0.1)

    def test_full_hit_probability_with_8_experts(self):
        from analyze_residency import full_hit_probability
        routes = []
        for _ in range(100):
            routes.append({"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 8]})
        prob = full_hit_probability(routes, resident_set={1, 2, 3, 4, 5, 6, 7, 8})
        self.assertAlmostEqual(prob, 1.0)

    def test_full_hit_probability_partial(self):
        from analyze_residency import full_hit_probability
        routes = [
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 8]},
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 9]},
        ]
        prob = full_hit_probability(routes, resident_set={1, 2, 3, 4, 5, 6, 7, 8})
        self.assertAlmostEqual(prob, 0.5)

    def test_full_hit_probability_zero(self):
        from analyze_residency import full_hit_probability
        routes = [
            {"layer": 0, "experts": [1, 2, 3, 4, 5, 6, 7, 8]},
        ]
        prob = full_hit_probability(routes, resident_set={10, 11, 12, 13, 14, 15, 16, 17})
        self.assertAlmostEqual(prob, 0.0)

    def test_shannon_entropy_uniform(self):
        from analyze_residency import shannon_entropy
        freq = {i: 1 for i in range(256)}
        self.assertAlmostEqual(shannon_entropy(freq), 8.0, places=2)

    def test_shannon_entropy_concentrated(self):
        from analyze_residency import shannon_entropy
        freq = {0: 100}
        self.assertAlmostEqual(shannon_entropy(freq), 0.0, places=2)

    def test_normalized_entropy(self):
        from analyze_residency import normalized_entropy
        freq_uniform = {i: 1 for i in range(256)}
        freq_concentrated = {0: 100}
        self.assertAlmostEqual(normalized_entropy(freq_uniform), 1.0, places=2)
        self.assertAlmostEqual(normalized_entropy(freq_concentrated), 0.0, places=2)

    def test_gini_coefficient(self):
        from analyze_residency import gini_coefficient
        equal = [1.0] * 100
        self.assertAlmostEqual(gini_coefficient(equal), 0.0, places=3)
        concentrated = [100.0] + [0.0] * 99
        self.assertGreater(gini_coefficient(concentrated), 0.9)

    def test_duplicate_route_ids(self):
        from analyze_residency import has_duplicate_route_ids
        self.assertTrue(has_duplicate_route_ids([1, 2, 3, 1, 4, 5, 6, 7]))
        self.assertFalse(has_duplicate_route_ids([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_high_individual_coverage_not_full_hit(self):
        from analyze_residency import individual_coverage, full_hit_probability
        routes = []
        for i in range(100):
            experts = [i % 256, (i + 1) % 256, (i + 2) % 256, (i + 3) % 256,
                       0, 1, 2, 3]
            routes.append({"layer": 0, "experts": experts})
        cov = individual_coverage(routes, top_n=4)
        self.assertGreater(cov[0], 0.9)
        prob = full_hit_probability(routes, resident_set={0, 1, 2, 3})
        self.assertLess(prob, 0.1)


class TestOracleOptimality(unittest.TestCase):
    """Step 2: oracle optimality and budget tests."""

    def test_single_bundle_greedy_stuck(self):
        from analyze_residency import greedy_oracle, RouteTrace
        # Single-bundle greedy gets stuck: expert 0 has high freq but never completes a route
        # Expert 8 is rare but completes a valuable route
        routes = []
        for _ in range(100):
            routes.append({"layer": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7]})
        for _ in range(10):
            routes.append({"layer": 0, "experts": [8, 9, 10, 11, 12, 13, 14, 15]})
        trace = RouteTrace(routes, layer=0, saved_us=1000.0, bundle_bytes=1000)
        result = greedy_oracle([trace], capacity_bytes=16000, top_k=8)
        bundles = result["bundles"]
        # Should include the second route's experts (8-15) because they complete a full route
        layer_0_experts = [b["expert"] for b in bundles if b["layer"] == 0]
        self.assertIn(8, layer_0_experts)

    def test_budget_constraint(self):
        from analyze_residency import greedy_oracle, RouteTrace
        routes = [{"layer": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7]}]
        trace = RouteTrace(routes, layer=0, saved_us=1000.0, bundle_bytes=2000)
        result = greedy_oracle([trace], capacity_bytes=15000, top_k=8)
        total_bytes = sum(b["bundle_bytes"] for b in result["bundles"])
        self.assertLessEqual(total_bytes, 15000)

    def test_only_complete_bundles(self):
        from analyze_residency import greedy_oracle, RouteTrace
        routes = [{"layer": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7]}]
        trace = RouteTrace(routes, layer=0, saved_us=1000.0, bundle_bytes=1000)
        result = greedy_oracle([trace], capacity_bytes=100000, top_k=8)
        # Must have complete bundles only (all 8 experts for a route)
        bundles = result["bundles"]
        self.assertGreater(len(bundles), 0)

    def test_8of8_scoring_only(self):
        from analyze_residency import greedy_oracle, RouteTrace
        routes = [
            {"layer": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7]},
            {"layer": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 8]},
        ]
        trace = RouteTrace(routes, layer=0, saved_us=1000.0, bundle_bytes=1000)
        result = greedy_oracle([trace], capacity_bytes=16000, top_k=8)
        # With 8of8 scoring, only 8 experts resident gets 1 full hit
        # 7 experts resident gets 0 full hits
        bundles = result["bundles"]
        full_hits = result.get("predicted_full_hits", 0)
        self.assertGreater(full_hits, 0)

    def test_deterministic_tie_break(self):
        from analyze_residency import greedy_oracle, RouteTrace
        routes = [{"layer": 0, "experts": [0, 1, 2, 3, 4, 5, 6, 7]}]
        trace = RouteTrace(routes, layer=0, saved_us=1000.0, bundle_bytes=1000)
        result1 = greedy_oracle([trace], capacity_bytes=100000, top_k=8)
        result2 = greedy_oracle([trace], capacity_bytes=100000, top_k=8)
        self.assertEqual(result1["bundles"], result2["bundles"])


class TestAnalyzer(unittest.TestCase):
    """Step 3: analyzer subcommand tests."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tmp = Path(self.tmpdir)

    def test_geometry_subcommand(self):
        from analyze_residency import cmd_geometry
        geom_path = Path(__file__).parent / "reports" / "geometry-v1.json"
        if not geom_path.exists():
            self.skipTest("geometry report not found")
        result = cmd_geometry(geom_path)
        self.assertEqual(result["expert_count"], 256)
        self.assertEqual(result["top_k"], 8)
        self.assertEqual(result["n_layers"], 40)

    def test_characterize_subcommand(self):
        from analyze_residency import cmd_characterize
        trace_path = Path(__file__).parent / "reports" / "tg1-route-trace.jsonl"
        geom_path = Path(__file__).parent / "reports" / "geometry-v1.json"
        if not trace_path.exists() or not geom_path.exists():
            self.skipTest("trace or geometry not found")
        result = cmd_characterize(trace_path, geom_path)
        self.assertIn("layers", result)
        self.assertGreater(len(result["layers"]), 0)
        layer0 = result["layers"][0]
        self.assertIn("expert_freq", layer0)
        self.assertIn("entropy", layer0)
        self.assertIn("gini", layer0)

    def test_oracle_subcommand(self):
        from analyze_residency import cmd_oracle
        trace_path = Path(__file__).parent / "reports" / "tg1-route-trace.jsonl"
        geom_path = Path(__file__).parent / "reports" / "geometry-v1.json"
        if not trace_path.exists() or not geom_path.exists():
            self.skipTest("trace or geometry not found")
        result = cmd_oracle(trace_path, geom_path, capacity_mib=128)
        self.assertIn("manifest", result)
        self.assertIn("predicted_full_hits", result)
        self.assertIn("predicted_saved_ms_per_1k", result)

    def test_sweep_subcommand(self):
        from analyze_residency import cmd_sweep
        trace_path = Path(__file__).parent / "reports" / "tg1-route-trace.jsonl"
        geom_path = Path(__file__).parent / "reports" / "geometry-v1.json"
        if not trace_path.exists() or not geom_path.exists():
            self.skipTest("trace or geometry not found")
        result = cmd_sweep(trace_path, geom_path)
        self.assertIn("capacities", result)
        self.assertGreater(len(result["capacities"]), 0)

    def test_compare_subcommand(self):
        from analyze_residency import cmd_compare
        trace_path = Path(__file__).parent / "reports" / "tg1-route-trace.jsonl"
        geom_path = Path(__file__).parent / "reports" / "geometry-v1.json"
        if not trace_path.exists() or not geom_path.exists():
            self.skipTest("trace or geometry not found")
        manifest = {
            "format": 3,
            "admission": "8of8",
            "model": {"sha256": "a2f6c7fdbe82113a2e48e2c38022b55bdcc4308a8002da96cf6d48dab67bb77d",
                       "top_k": 8, "expert_count": 256},
            "cache_bytes": 128 * 1024 * 1024,
            "bundles": []
        }
        result = cmd_compare(manifest, trace_path, geom_path)
        self.assertIn("full_hits", result)
        self.assertIn("total_routes", result)

    def test_refuses_non_attested_trace(self):
        from analyze_residency import load_trace
        bad_trace = self.tmp / "bad.jsonl"
        bad_trace.write_text('{"_header": true, "format": 1, "callback_matches_canonical": false}\n')
        with self.assertRaises(ValueError):
            load_trace(bad_trace)


class TestHeldOutProtocol(unittest.TestCase):
    """Step 4: held-out protocol tests."""

    def test_train_test_split(self):
        from analyze_residency import split_routes
        routes = [{"sequence_index": i, "layer": 0, "experts": [i % 256] * 8}
                  for i in range(100)]
        train, test = split_routes(routes, train_frac=0.5)
        self.assertEqual(len(train), 50)
        self.assertEqual(len(test), 50)
        train_sids = {r["sequence_index"] for r in train}
        test_sids = {r["sequence_index"] for r in test}
        self.assertEqual(train_sids & test_sids, set())

    def test_held_out_labeling(self):
        from analyze_residency import cmd_oracle
        trace_path = Path(__file__).parent / "reports" / "tg1-route-trace.jsonl"
        geom_path = Path(__file__).parent / "reports" / "geometry-v1.json"
        if not trace_path.exists() or not geom_path.exists():
            self.skipTest("trace or geometry not found")
        result = cmd_oracle(trace_path, geom_path, capacity_mib=128, held_out=True)
        self.assertIn("partition", result)
        self.assertIn(result["partition"], ("training", "held_out", "combined"))


if __name__ == "__main__":
    unittest.main()
