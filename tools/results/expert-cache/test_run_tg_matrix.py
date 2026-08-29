import importlib.util
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


if __name__ == "__main__":
    unittest.main()
