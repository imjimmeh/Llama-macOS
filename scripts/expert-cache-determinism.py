#!/usr/bin/env python3

# Deterministic single-request token-hash check for the expert cache.
#
# Starts llama-server with a fixed set of preset performance args (from
# G:/qwen3.6-35b-a3b-presets-exc.ini, [*] block + [qwen3.6-35B-mtp] section,
# which targets the MTP-Quality model), streams the server log live so the
# slow model load is visible, waits for HTTP readiness, then issues ONE
# /completion request with deterministic sampling (temperature 0, top_k 1,
# fixed seed, ignore_eos, fixed token count) and records SHA-256 hashes of
# both the content bytes and the returned token IDs.
#
# Different expert-cache flags (rows) produce identical hashes iff the cache
# does not alter the sampled token stream.
#
# llama-server is used instead of llama-cli on purpose: llama-cli auto-enters
# conversational mode for chat-templated models and never exits after
# generating, so it hangs waiting on stdin. The server performs one
# completion per HTTP request and shuts down cleanly.
#
# Only the expert-cache flags differ between rows; those are passed via the
# --exc / --excp / --extra-args arguments on the command line.
#
# Usage (single run):
#   python scripts/expert-cache-determinism.py --exc 0
#   python scripts/expert-cache-determinism.py --exc 64M --excp 64
#   python scripts/expert-cache-determinism.py --exc 64M --excp 64 \
#       --extra-args "--mtp-dynamic-offload --spec-draft-n-max 2"
#
# Row A: --exc 0
# Row B: --exc 64M --excp 64            (must hash-identical to A)
# Row E: --exc 64M --excp 64 --extra-args "--spec-type draft-mtp --spec-draft-n-max 2 --mtp-dynamic-offload"

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from typing import Dict, List

MODEL_PATH = "G:/ai/models/Qwen3.6-35B-A3B-APEX-MTP-Quality.gguf"
SERVER_EXE = "build/bin/Release/llama-server.exe"
DEFAULT_PORT = 8099

# Performance args from the preset file (G:/qwen3.6-35b-a3b-presets-exc.ini).
# [*] block plus [qwen3.6-35B-mtp]. ctx 128000 is part of the MTP preset.
# Sampling defaults are NOT here: the deterministic sampler flags are applied
# in the /completion payload instead.
PRESET_BASE_ARGS = [
    "--jinja",
    "-t", "14",
    "-b", "4096",
    "-ub", "2048",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--flash-attn", "on",
    "--mlock",
    "--no-mmap",
    "--no-context-shift",
    "-cram", "1024",
    "--fit", "on",
    "--fit-target", "256",
    "--parallel", "1",
]

DEFAULT_PROMPT = (
    "Explain how a mixture-of-experts transformer routes tokens "
    "through expert layers, in exactly five sentences."
)
DEFAULT_SEED = 42
DEFAULT_N_PREDICT = 256

LOG_PREFIXES = ("llama_", "main:", "sampler", "sched_", "load_", "print_")


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_tokens(tokens: List[int]) -> str:
    buf = b"".join(struct.pack("<i", int(t)) for t in tokens)
    return hashlib.sha256(buf).hexdigest()


def build_server_command(
    exe: str,
    model: str,
    exc: str,
    excp: int,
    extra_args: str,
    ctx_size: int,
    port: int,
) -> List[str]:
    cmd = [
        exe,
        "--model", model,
        "--ctx-size", str(ctx_size),
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    cmd += PRESET_BASE_ARGS
    if exc:
        cmd += ["-exc", exc]
    if excp:
        cmd += ["-excp", str(excp)]
    if extra_args:
        cmd += extra_args.split()
    return cmd


def build_completion_payload(prompt: str, n_predict: int, seed: int) -> Dict:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "ignore_eos": True,
        "cache_prompt": True,
        "return_tokens": True,
        "samplers": ["top_k", "temperature"],
    }


def wait_for_health(base_url: str, log_path: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    cursor = 0
    # stream the log file live so a slow model load is visible, not a hang
    while time.time() < deadline:
        try:
            with open(log_path, "rb") as rf:
                rf.seek(cursor)
                chunk = rf.read()
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    cursor += len(chunk)
        except OSError:
            pass
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=3) as resp:
                if resp.status == 200:
                    # drain remaining log output
                    with open(log_path, "rb") as rf:
                        rf.seek(cursor)
                        rest = rf.read()
                        if rest:
                            sys.stdout.buffer.write(rest)
                            sys.stdout.buffer.flush()
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic token-hash check for the expert cache (llama-server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exe", default=SERVER_EXE,
                        help="path to the llama-server binary (default: %(default)s)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help="path to the GGUF model (default: %(default)s)")
    parser.add_argument("--ctx-size", type=int, default=128000,
                        help="context size (default: %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="server port (default: %(default)s)")
    parser.add_argument("--exc", default="",
                        help="expert cache size, e.g. 0 or 64M (empty = default)")
    parser.add_argument("--excp", type=int, default=0,
                        help="expert cache period, e.g. 64 (0 = on-demand)")
    parser.add_argument("--extra-args", default="",
                        help="extra llama-server args, e.g. MTP draft flags")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT,
                        help="fixed prompt (default: built-in 5-sentence prompt)")
    parser.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT,
                        help="tokens to generate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="sampler seed (default: %(default)s)")
    parser.add_argument("--json-out", default=None,
                        help="write the result JSON record to this file")
    args = parser.parse_args()

    if not os.path.isfile(args.exe):
        print(f"error: llama-server not found: {args.exe}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.model):
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    cmd = build_server_command(
        args.exe, args.model, args.exc, args.excp, args.extra_args,
        args.ctx_size, args.port,
    )
    print(f"running: {' '.join(cmd)}", flush=True)

    fd, log_path = tempfile.mkstemp(suffix=".log", prefix="expert-cache-determinism-")
    os.close(fd)
    print(f"log: {log_path}", flush=True)

    base_url = f"http://127.0.0.1:{args.port}"
    payload = build_completion_payload(args.prompt, args.n_predict, args.seed)

    proc = subprocess.Popen(
        cmd, stdout=open(log_path, "wb"), stderr=subprocess.STDOUT
    )
    try:
        if not wait_for_health(base_url, log_path, timeout_s=720):
            print("\nserver did not become healthy in time", file=sys.stderr)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("--- server log tail ---", file=sys.stderr)
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                sys.stderr.write(f.read()[-4000:])
            return 1

        req = urllib.request.Request(
            base_url + "/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        wall_s = time.time() - t0

        tokens = body.get("tokens", [])
        content = body.get("content", "")
        tok_s = body.get("timings", {}).get("predicted_per_second", 0.0)

        record = {
            "tool": "llama-server",
            "model": args.model,
            "ctx_size": args.ctx_size,
            "exc": args.exc,
            "excp": args.excp,
            "extra_args": args.extra_args,
            "prompt": args.prompt,
            "seed": args.seed,
            "n_predict": args.n_predict,
            "n_tokens": len(tokens),
            "sha256_tokens": sha256_of_tokens(tokens) if tokens else None,
            "sha256_content": sha256_of_bytes(content.encode("utf-8")) if content else None,
            "tok_s": round(tok_s, 3),
            "wall_s": round(wall_s, 1),
        }
        print(json.dumps(record, indent=2))
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        return 0
    finally:
        # graceful shutdown lets llama-server free contexts, which prints the
        # expert-cache stats block when --expert-cache-stats is on
        try:
            req = urllib.request.Request(
                base_url + "/shutdown",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            proc.wait(timeout=15)
            with open(log_path, "rb") as rf:
                rest = rf.read()
                if rest:
                    sys.stdout.buffer.write(b"\n--- server shutdown stats ---\n")
                    sys.stdout.buffer.write(rest)
                    sys.stdout.buffer.flush()
        except Exception:
            pass
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())