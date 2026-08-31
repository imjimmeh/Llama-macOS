import importlib.util
import os
import pathlib
import unittest
from argparse import Namespace


MODULE_PATH = pathlib.Path(__file__).with_name("run-tg-matrix.py")
SPEC = importlib.util.spec_from_file_location("run_tg_matrix", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_tg_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_tg_matrix)


class BenchCommandTest(unittest.TestCase):
    def test_explicit_sidecar_placement_omits_fit_target(self):
        args = Namespace(
            bench=pathlib.Path("llama-bench"),
            model=pathlib.Path("model.gguf"),
            cache_mib=1024,
            cache_period=1,
            n_gen=128,
            load_mode="mmap",
            fit_target=0,
            gpu_layers=99,
            cpu_moe_layers=40,
            max_swaps=16,
            pinned_experts=pathlib.Path("pinned_experts_1024mb.json"),
        )

        command = run_tg_matrix.bench_command(args, cache_enabled=True)

        self.assertIn("-ngl", command)
        self.assertIn("-ncmoe", command)
        self.assertEqual(command[command.index("-ngl") + 1], "99")
        self.assertEqual(command[command.index("-ncmoe") + 1], "40")
        self.assertEqual(command[command.index("-lm") + 1], "mmap")
        self.assertEqual(command[command.index("-exc") + 1], "1024")
        self.assertEqual(command[command.index("-excp") + 1], "1")
        self.assertEqual(command[command.index("-excm") + 1], "16")
        self.assertNotIn("-fitt", command)
        self.assertIn("-pe", command)
        self.assertEqual(command[command.index("-pe") + 1], "pinned_experts_1024mb.json")
        control_command = run_tg_matrix.bench_command(args, cache_enabled=False)
        self.assertNotIn("-pe", control_command)


    def test_cache_first_run_order(self):
        self.assertEqual(run_tg_matrix.run_order(cache_first=True), (True, False))
        self.assertEqual(run_tg_matrix.run_order(cache_first=False), (False, True))


class BenchEnvironmentTest(unittest.TestCase):
    def test_environment_forces_experimental_off_and_selects_concurrent(self):
        args = Namespace(hetero_concurrent=1)
        environment = run_tg_matrix.bench_environment(args)

        self.assertEqual(environment["GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL"], "0")
        self.assertEqual(environment["GGML_EXPERT_CACHE_HETERO_CONCURRENT"], "1")

    def test_environment_defaults_concurrent_off(self):
        args = Namespace(hetero_concurrent=0)
        environment = run_tg_matrix.bench_environment(args)

        self.assertEqual(environment["GGML_EXPERT_CACHE_HETERO_EXPERIMENTAL"], "0")
        self.assertEqual(environment["GGML_EXPERT_CACHE_HETERO_CONCURRENT"], "0")

    def test_environment_inherits_parent_variables(self):
        args = Namespace(hetero_concurrent=1)
        environment = run_tg_matrix.bench_environment(args)

        self.assertIn("PATH", environment)



class MatrixCommandTest(unittest.TestCase):
    def test_three_way_rows_for_one_capacity(self):
        args = Namespace(
            bench=pathlib.Path("llama-bench"),
            model=pathlib.Path("model.gguf"),
            cache_mib=128,
            cache_period=256,
            n_gen=128,
            load_mode="mlock",
            fit_target=256,
            gpu_layers=None,
            cpu_moe_layers=None,
            max_swaps=0,
            pinned_experts=pathlib.Path("manifests/oracle-128m.json"),
            hetero_concurrent=0,
        )
        a = run_tg_matrix.matrix_command(args, "A")
        b = run_tg_matrix.matrix_command(args, "B")
        c = run_tg_matrix.matrix_command(args, "C")
        self.assertEqual(a[a.index("-exc") + 1], "0")
        self.assertNotIn("-pe", a)
        self.assertEqual(b[b.index("-exc") + 1], "128")
        self.assertEqual(b[b.index("-excm") + 1], "0")
        self.assertNotIn("-pe", b)
        self.assertEqual(c[c.index("-exc") + 1], "128")
        self.assertEqual(c[c.index("-excm") + 1], "0")
        self.assertEqual(c[c.index("-pe") + 1], os.fsdecode(pathlib.Path("manifests/oracle-128m.json")))
        # identical fixed flags across all three rows (pair flag tokens -> value)
        def fixed(cmd):
            it = iter(cmd[1:])  # drop bench binary path; rest are flag/value pairs
            flags = dict(zip(it, it))
            return {k: v for k, v in flags.items() if k not in ("-exc", "-pe", "-excp", "-excm")}
        fa, fb, fc = fixed(a), fixed(b), fixed(c)
        self.assertEqual(fa, fb)
        self.assertEqual(fa, fc)

    def test_matrix_order_alternates_by_pair(self):
        order = run_tg_matrix.matrix_order(pair=1, cache_first=False)
        self.assertEqual(order, ["A", "B", "C"])
        order = run_tg_matrix.matrix_order(pair=2, cache_first=False)
        self.assertEqual(order, ["C", "B", "A"])
        self.assertEqual(run_tg_matrix.matrix_order(pair=1, cache_first=True),
                         ["C", "B", "A"])


class MatrixIndexTest(unittest.TestCase):
    def test_index_records_one_row_per_child_process(self):
        rows = [
            {"config": "A", "pair": 1, "order": "control_first", "stdout": "o1.jsonl", "stderr": "e1.log"},
            {"config": "C", "pair": 2, "order": "static_first", "stdout": "o2.jsonl", "stderr": "e2.log"},
        ]
        index = run_tg_matrix.build_matrix_index(capacity_mib=128, rows=rows)
        self.assertEqual(index["capacity_mib"], 128)
        self.assertEqual(len(index["runs"]), 2)
        first = index["runs"][0]
        self.assertEqual(first["config"], "A")
        self.assertEqual(first["run"], 1)
        self.assertEqual(first["order"], "control_first")
        self.assertEqual(first["jsonl"], "o1.jsonl")
        self.assertEqual(first["stderr"], "e1.log")

if __name__ == "__main__":
    unittest.main()
