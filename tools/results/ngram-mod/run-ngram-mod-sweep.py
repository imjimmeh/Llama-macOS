#!/usr/bin/env python3
"""Sweep ngram-mod pool sizes and measure hit coverage cold/warm.

Drives examples/speculative-simple (same common_speculative machinery as the
server) across a set of --spec-ngram-mod-size values. For each size it runs:
  - cold: fresh cache file, pool starts empty
  - warm: reuse the cache file saved by the cold run (destructor save)

With --cold-sizes (tiering): each (hot size, cold size, fallback mode) is run
cold then warm; the cache file is per (size, cold size) so the warm run
reopens the same cold store.

Parses the "statistics ngram-mod" line and the decoded t/s from the log and
writes a JSONL file for later analysis.

Model flags default to the qwen3.6-35B-apex-compact preset
(G:\\qwen3.6-35b-a3b-presets-exc-latest.ini).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

STATS_RE = re.compile(
    r"#nm lookups=(\d+) hits=(\d+) miss_fp=(\d+) inserts=(\d+) overwrites=(\d+)"
    r"(?:, cold lookups=(\d+) hits=(\d+) promotes=(\d+) demotes=(\d+) flushes=(\d+)"
    r" hot_loaded=(\d+))?")
SPEED_RE = re.compile(
    r"decoded\s+(\d+)\s+tokens in\s+[\d.]+\s+seconds, speed:\s+([\d.]+)\s+t/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="GGUF model path")
    parser.add_argument("--simple", type=Path,
                        default=Path("build/bin/Release/llama-speculative-simple.exe"))
    parser.add_argument("--output-dir", type=Path, default=Path("tools/results/ngram-mod"))
    parser.add_argument("--prefix", help="output filename prefix; defaults to the current UTC timestamp")
    parser.add_argument("--sizes", default="256M,512M,1G,2G,4G", help="comma-separated pool sizes")
    parser.add_argument("--cache-dir", type=Path,
                        help="directory for cache files (default: the output dir); "
                             "cold stores can be large, so a big volume helps")
    parser.add_argument("--cold-sizes", default="off",
                        help="comma-separated cold store sizes, 'off' disables tiering")
    parser.add_argument("--fallbacks", default="on",
                        help="comma-separated cold fallback modes (on/off), tiered runs only")
    parser.add_argument("--prompt-file", type=Path,
                        default=Path("tools/results/ngram-mod/sweep-prompt.txt"))
    parser.add_argument("--n-predict", type=int, default=768)
    parser.add_argument("--ctx", type=int, default=2048, help="context size (KV size does not affect ngram-mod)")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--n-match", type=int, default=24)
    parser.add_argument("--n-min", type=int, default=24)
    parser.add_argument("--n-max", type=int, default=48)
    parser.add_argument("--load-mode", default="mlock")
    parser.add_argument("--fit-target", type=int, default=256)
    parser.add_argument("--exc", default="128M", help="expert cache size, '0' disables")
    parser.add_argument("--expert-cache-period", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0, help="sampling temperature (0 = greedy)")
    return parser.parse_args()


def simple_command(args: argparse.Namespace, size: str, cold: str, fallback: str,
                   cache: Path) -> list[str]:
    command = [
        str(args.simple),
        "-m", str(args.model),
        "-f", str(args.prompt_file),
        "-n", str(args.n_predict),
        "-c", str(args.ctx),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-t", str(args.threads),
        "-lm", args.load_mode,
        "-fit", "on",
        "-fitt", str(args.fit_target),
        "-lv", "4",
        "--temp", str(args.temp),
        "--spec-type", "ngram-mod",
        "--spec-ngram-mod-n-match", str(args.n_match),
        "--spec-ngram-mod-n-min", str(args.n_min),
        "--spec-ngram-mod-n-max", str(args.n_max),
        "--spec-ngram-mod-size", size,
        "--spec-ngram-mod-cache", str(cache),
    ]
    if cold != "off":
        command += ["--spec-ngram-mod-cold-size", cold,
                    "--spec-ngram-mod-cold-fallback", fallback]
    if args.exc != "0":
        command += ["-exc", args.exc, "-excp", str(args.expert_cache_period)]
    return command


def parse_log(text: str) -> dict:
    result = {}
    m = STATS_RE.search(text)
    if m:
        result["lookups"] = int(m.group(1))
        result["hits"] = int(m.group(2))
        result["miss_fp"] = int(m.group(3))
        result["inserts"] = int(m.group(4))
        result["overwrites"] = int(m.group(5))
        result["hit_rate"] = result["hits"] / result["lookups"] if result["lookups"] else 0.0
        result["overwrite_rate"] = result["overwrites"] / result["inserts"] if result["inserts"] else 0.0
        if m.group(6) is not None:
            result["cold_lookups"] = int(m.group(6))
            result["cold_hits"] = int(m.group(7))
            result["promotes"] = int(m.group(8))
            result["demotes"] = int(m.group(9))
            result["flushes"] = int(m.group(10))
            result["hot_loaded"] = int(m.group(11))
            result["cold_hit_rate"] = (result["cold_hits"] / result["cold_lookups"]
                                       if result["cold_lookups"] else 0.0)
    m = SPEED_RE.search(text)
    if m:
        result["decoded_tokens"] = int(m.group(1))
        result["tps"] = float(m.group(2))
    return result


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"error: model not found: {args.model}")
        return 1
    if not args.simple.exists():
        print(f"error: speculative-simple not found: {args.simple}")
        return 1
    if not args.prompt_file.exists():
        print(f"error: prompt file not found: {args.prompt_file}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.output_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = args.output_dir / f"{prefix}-ngram-mod-sweep.jsonl"
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    cold_sizes = [s.strip() for s in args.cold_sizes.split(",") if s.strip()]
    fallbacks = [s.strip() for s in args.fallbacks.split(",") if s.strip()]

    print(f"model: {args.model}")
    print(f"sizes: {', '.join(sizes)}")
    print(f"cold sizes: {', '.join(cold_sizes)}")
    print(f"fallbacks: {', '.join(fallbacks)}")
    print(f"jsonl: {jsonl_path}")
    print()

    rows = []
    for size in sizes:
        for cold in cold_sizes:
            for fallback in fallbacks:
                if cold == "off" and fallback != "on":
                    continue  # fallback only applies to tiered runs
                cache = cache_dir / f"cache-{prefix}-{size}-{cold}.ngrammod"
                if cache.exists():
                    cache.unlink()

                for mode in ("cold", "warm"):
                    log_path = args.output_dir / f"{prefix}-{size}-{cold}-{fallback}-{mode}.log"
                    command = simple_command(args, size, cold, fallback, cache)
                    print(f"running {mode} size={size} cold={cold} fallback={fallback} ...")
                    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
                        proc = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT,
                                              text=True, encoding="utf-8", errors="replace")
                    row = {
                        "size": size,
                        "cold_size": cold,
                        "fallback": fallback,
                        "mode": mode,
                        "exit_code": proc.returncode,
                        "pool_bytes": None,
                    }
                    if proc.returncode != 0:
                        print(f"  exit code {proc.returncode} (see {log_path})")
                        row["error"] = "non-zero exit"
                    else:
                        row.update(parse_log(log_path.read_text(encoding="utf-8", errors="replace")))
                        print(f"  lookups={row.get('lookups')} hits={row.get('hits')} "
                              f"hit_rate={row.get('hit_rate', 0):.4f} "
                              f"overwrite_rate={row.get('overwrite_rate', 0):.4f} "
                              f"tps={row.get('tps', 0):.2f}")
                    rows.append(row)

                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")

    print()
    print("summary:")
    print(f"  {'size':>8} {'cold':>5} {'fb':>4} {'mode':>5} {'lookups':>9} {'hits':>8} "
          f"{'hit_rate':>9} {'overwrite':>10} {'tps':>7}")
    for row in rows:
        if "error" in row:
            continue
        print(f"  {row['size']:>8} {row['cold_size']:>5} {row['fallback']:>4} {row['mode']:>5} "
              f"{row['lookups']:>9} {row['hits']:>8} {row['hit_rate']:>9.4f} "
              f"{row['overwrite_rate']:>10.4f} {row['tps']:>7.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
