import contextlib
import importlib.util
import io
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("expert_cache_sweep.py")
SPEC = importlib.util.spec_from_file_location("expert_cache_sweep", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
expert_cache_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(expert_cache_sweep)


class CacheConfigurationTest(unittest.TestCase):
    def test_zero_cache_runs_once_and_each_nonzero_size_uses_each_period(self):
        configurations = expert_cache_sweep.build_configurations(
            cache_sizes_mib=[0, 64, 128],
            cache_periods=[32, 64],
        )

        self.assertEqual(
            configurations,
            [
                (0, 64),
                (64, 32),
                (64, 64),
                (128, 32),
                (128, 64),
            ],
        )


    def test_dry_run_does_not_copy_profile_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = pathlib.Path(temporary_directory)
            profile = root / "seed-profile.json"
            output_dir = root / "output"
            profile.write_text("{}", encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                returncode = expert_cache_sweep.main(
                    [
                        "--server", "llama-server",
                        "--model-path", "model.gguf",
                        "--model-name", "model",
                        "--profile-file", str(profile),
                        "--output-dir", str(output_dir),
                        "--dry-run",
                    ]
                )

            self.assertEqual(returncode, 0)
            self.assertFalse(list(output_dir.glob("*.profile.json")))

if __name__ == "__main__":
    unittest.main()
