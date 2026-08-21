#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def build_configurations(cache_sizes_mib: list[int], cache_periods: list[int]) -> list[tuple[int, int]]:
    if not cache_sizes_mib:
        raise ValueError("cache sizes must not be empty")
    if not cache_periods:
        raise ValueError("cache periods must not be empty")
    if any(size < 0 for size in cache_sizes_mib):
        raise ValueError("cache sizes must be non-negative")
    if any(period < 0 for period in cache_periods):
        raise ValueError("cache periods must be non-negative")

    disabled_period = 64 if 64 in cache_periods else cache_periods[0]
    configurations = []
    for size in cache_sizes_mib:
        if size == 0:
            configurations.append((size, disabled_period))
        else:
            configurations.extend((size, period) for period in cache_periods)
    return configurations


def parse_csv_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def wait_for_server(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f} seconds")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for row in data.get("summary", []):
        if row.get("category") == "overall":
            return row
    return None


def build_server_command(args: argparse.Namespace, size_mib: int, period: int, profile_path: Path | None) -> list[str]:
    command = [
        args.server,
        "-m", args.model_path,
        "-c", str(args.context_size),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-t", str(args.threads),
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-fa", "on",
        "-lm", "mlock",
        "-fit", "on",
        "-fitt", str(args.fit_target_mib),
        "-exc", f"{size_mib}M",
        "-excp", str(period),
        "--ffn-split", "0",
        "--no-mmproj",
        "-np", "1",
        "--port", str(args.port),
        "-excs",
    ]
    if profile_path is not None:
        command.extend(["-excf", str(profile_path)])
    return command


def build_benchmark_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    return [
        args.python,
        str(Path(__file__).with_name("speed_bench.py")),
        "--url", args.url,
        "--bench", args.bench,
        "--category", args.category,
        "--osl", str(args.osl),
        "--concurrency", str(args.concurrency),
        "--limit", str(args.limit),
        "--output", str(output_path),
        "--model", args.model_name,
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPEED-Bench for isolated expert-cache configurations.")
    parser.add_argument("--server", required=True, help="Path to llama-server")
    parser.add_argument("--model-path", required=True, help="Path to the GGUF model")
    parser.add_argument("--model-name", required=True, help="Model value sent to the OpenAI-compatible endpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for per-configuration results and logs")
    parser.add_argument("--url", default="localhost:9999", help="Server URL passed to speed_bench.py")
    parser.add_argument("--port", type=int, default=9999, help="Local llama-server port")
    parser.add_argument("--cache-sizes", type=parse_csv_ints, default=[0, 64, 128, 192, 256], metavar="MIB")
    parser.add_argument("--cache-periods", type=parse_csv_ints, default=[32, 64, 128], metavar="TOKENS")
    parser.add_argument("--profile-file", type=Path, help="Seed profile copied before every nonzero-cache run")
    parser.add_argument("--context-size", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--fit-target-mib", type=int, default=256)
    parser.add_argument("--bench", default="qualitative")
    parser.add_argument("--category", default="coding")
    parser.add_argument("--osl", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configurations = build_configurations(args.cache_sizes, args.cache_periods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.profile_file is not None and not args.profile_file.is_file():
        print(f"expert_cache_sweep: profile file not found: {args.profile_file}", file=sys.stderr)
        return 2

    manifest: list[dict[str, Any]] = []
    for size_mib, period in configurations:
        stem = f"exc-{size_mib}m-excp-{period}"
        result_path = output_dir / f"{stem}.json"
        log_path = output_dir / f"{stem}.server.log"
        profile_copy = output_dir / f"{stem}.profile.json" if size_mib > 0 and args.profile_file else None
        if profile_copy is not None and not args.dry_run:
            shutil.copyfile(args.profile_file, profile_copy)

        server_command = build_server_command(args, size_mib, period, profile_copy)
        benchmark_command = build_benchmark_command(args, result_path)
        entry: dict[str, Any] = {
            "cache_size_mib": size_mib,
            "cache_period": period,
            "result": str(result_path),
            "server_log": str(log_path),
            "server_command": server_command,
            "benchmark_command": benchmark_command,
        }
        print(f"\n=== EXC={size_mib} MiB, EXCP={period} ===")
        if args.dry_run:
            print("server:", subprocess.list2cmdline(server_command))
            print("benchmark:", subprocess.list2cmdline(benchmark_command))
            manifest.append(entry)
            continue

        process: subprocess.Popen[str] | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, text=True)
                wait_for_server(args.url if "://" in args.url else "http://" + args.url, args.startup_timeout)
                benchmark = subprocess.run(benchmark_command, text=True)
                entry["benchmark_returncode"] = benchmark.returncode
                entry["summary"] = load_summary(result_path)
        except Exception as exc:
            entry["error"] = str(exc)
            print(f"expert_cache_sweep: {stem} failed: {exc}", file=sys.stderr)
        finally:
            if process is not None:
                stop_process(process)
            if profile_copy is not None:
                profile_copy.unlink(missing_ok=True)
        manifest.append(entry)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    failed = [entry for entry in manifest if entry.get("error") or entry.get("benchmark_returncode", 0) != 0]
    print(f"\nexpert_cache_sweep: completed {len(manifest) - len(failed)}/{len(manifest)} configurations")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
