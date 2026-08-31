import importlib.util
import json
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("analyze_tg_matrix.py")
SPEC = importlib.util.spec_from_file_location("analyze_tg_matrix", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
analyze_tg_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze_tg_matrix)


def record(avg_ts, **telemetry):
    base = {
        "n_gen": 128,
        "avg_ts": avg_ts,
        "avg_ns": int(1e9 * 128 / avg_ts) if avg_ts > 0 else 0,
        "expert_cache_route_ready_full_hits": 0,
        "expert_cache_route_ready_fallbacks": 0,
        "expert_cache_route_ready_fast_rejects": 0,
        "expert_cache_route_ready_route_id_us": 0,
        "expert_cache_bytes_ram_to_gpu": 0,
        "expert_cache_bytes_avoided": 0,
    }
    for key in range(9):
        base[f"expert_cache_route_ready_resident_bundles_{key}"] = 0
    base.update(telemetry)
    return base


class PairedComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_matrix(self, per_config):
        """per_config: {config: [record, ...]} writes B-<run>.jsonl style files."""
        index = {"capacity_mib": 128, "runs": []}
        for config, records in per_config.items():
            for run, rec in enumerate(records, start=1):
                p = self.dir / f"m-{config}-{run}.jsonl"
                p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
                index["runs"].append({
                    "config": config,
                    "run": run,
                    "jsonl": str(p),
                    "stderr": str(p) + ".log",
                })
        index_path = self.dir / "m.index.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
        return index_path

    def test_mean_delta_and_sign(self):
        index_path = self.write_matrix({
            "A": [record(10.0), record(10.0), record(10.0)],
            "C": [record(11.0), record(12.0), record(10.5)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        comp = analyze_tg_matrix.compare(matrix, "C", "A")
        # deltas: +10%, +20%, +5%
        self.assertAlmostEqual(comp["mean_delta_pct"], 35.0 / 3.0, places=6)
        self.assertAlmostEqual(comp["median_delta_pct"], 10.0, places=6)
        self.assertEqual(comp["positive_pairs"], 3)
        self.assertEqual(comp["n_pairs"], 3)

    def test_interval_matches_student_t(self):
        from scipy import stats
        index_path = self.write_matrix({
            "A": [record(10.0), record(10.0), record(10.0), record(10.0)],
            "C": [record(10.5), record(11.0), record(10.2), record(10.8)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        comp = analyze_tg_matrix.compare(matrix, "C", "A")
        deltas = [5.0, 10.0, 2.0, 8.0]
        lo, hi = stats.t.interval(0.95, len(deltas) - 1,
                                  loc=sum(deltas) / 4,
                                  scale=stats.sem(deltas))
        self.assertAlmostEqual(comp["ci95_low"], lo, places=5)
        self.assertAlmostEqual(comp["ci95_high"], hi, places=5)

    def test_single_pair_has_no_interval(self):
        index_path = self.write_matrix({
            "A": [record(10.0)],
            "C": [record(12.0)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        comp = analyze_tg_matrix.compare(matrix, "C", "A")
        self.assertIsNone(comp["ci95_low"])
        self.assertIsNone(comp["ci95_high"])

    def test_placement_attribution_uses_b_as_denominator_for_execution(self):
        # A=10 baseline, B=9 reserved-empty (placement cost), C=10.8 static
        index_path = self.write_matrix({
            "A": [record(10.0), record(10.0), record(10.0)],
            "B": [record(9.0), record(9.0), record(9.0)],
            "C": [record(10.8), record(10.8), record(10.8)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        attr = analyze_tg_matrix.placement_attribution(matrix)
        self.assertAlmostEqual(attr["placement_cost_pct"], -10.0, places=6)
        self.assertAlmostEqual(attr["execution_benefit_pct"], 20.0, places=6)
        self.assertAlmostEqual(attr["net_benefit_pct"], 8.0, places=6)
        # net must be static/baseline; execution must be static/reserved-empty
        self.assertNotAlmostEqual(attr["execution_benefit_pct"],
                                  attr["net_benefit_pct"], places=6)

    def test_zero_hit_violation_flagged(self):
        index_path = self.write_matrix({
            "A": [record(10.0), record(10.0)],
            "B": [record(9.0, expert_cache_route_ready_full_hits=5),
                  record(9.0, expert_cache_route_ready_full_hits=0)],
            "C": [record(10.0), record(10.0)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        checks = analyze_tg_matrix.gate_checks(matrix)
        self.assertFalse(checks["reserved_empty_has_zero_full_hits"])

    def test_timed_h2d_flagged(self):
        index_path = self.write_matrix({
            "A": [record(10.0), record(10.0)],
            "B": [record(10.0), record(10.0)],
            "C": [record(11.0, expert_cache_bytes_ram_to_gpu=4096),
                  record(11.0, expert_cache_bytes_ram_to_gpu=0)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        checks = analyze_tg_matrix.gate_checks(matrix)
        self.assertFalse(checks["static_has_zero_timed_h2d"])
        self.assertGreater(checks["static_timed_h2d_bytes_total"], 0)

    def test_resident_histogram_reported(self):
        index_path = self.write_matrix({
            "C": [record(10.0, **{"expert_cache_route_ready_resident_bundles_8": 40}),
                  record(10.0, **{"expert_cache_route_ready_resident_bundles_0": 80})],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        hist = analyze_tg_matrix.resident_histogram(matrix, "C")
        self.assertEqual(hist["8"], 40)
        self.assertEqual(hist["0"], 80)

    def test_pairing_requires_matching_run_index(self):
        index_path = self.write_matrix({
            "A": [record(10.0), record(20.0)],
            "C": [record(12.0)],
        })
        matrix = analyze_tg_matrix.load_matrix(index_path)
        comp = analyze_tg_matrix.compare(matrix, "C", "A")
        # only run 1 pairs: (12-10)/10 = +20%
        self.assertEqual(comp["n_pairs"], 1)
        self.assertAlmostEqual(comp["mean_delta_pct"], 20.0, places=6)

    def test_missing_config_returns_none(self):
        index_path = self.write_matrix({"A": [record(10.0)]})
        matrix = analyze_tg_matrix.load_matrix(index_path)
        self.assertIsNone(analyze_tg_matrix.compare(matrix, "C", "A"))


if __name__ == "__main__":
    unittest.main()
