#!/usr/bin/env python3
"""
Reusable route-trace test harness for llama-server.

Launches llama-server with GGML_EXPERT_ROUTE_TRACE set (via subprocess env
propagation, the only reliable way on Windows), waits for /health, sends a
completion request with a long prompt to trigger the MUL_MAT_ID interception
path, then reports trace file size and terminates.

Usage:
    python tools/training_data/run_trace_server.py [--port 8137] [--max-tokens 32] [--prompt "..."] [--prompt-repeat 30]

Requirements:
    - Server binary built with `cmake --build build/tools/server --target llama-server --config Release`
    - Model path resolved below matches the user's lmstudio install
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_EXE = os.path.join(REPO_ROOT, "build", "bin", "Release", "llama-server.exe")
DEFAULT_MODEL = r"C:/Users/jimme/.lmstudio/models/mudler/Qwen3.6-35B-A3B-APEX-GGUF/Qwen3.6-35B-A3B-APEX-Compact.gguf"
DEFAULT_TRACE = os.path.join(REPO_ROOT, "tools", "training_data", "route_trace.bin")
DEFAULT_LOG = os.path.join(REPO_ROOT, "tools", "training_data", "server.log")


def parse_args():
    p = argparse.ArgumentParser(description="Run llama-server with route trace enabled")
    p.add_argument("--exe", default=DEFAULT_EXE, help="Path to llama-server.exe")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Path to .gguf model")
    p.add_argument("--port", type=int, default=8137)
    p.add_argument("--ctx-size", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--ubatch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. ")
    p.add_argument("--prompt-repeat", type=int, default=30, help="Repeat prompt N times to force prefill")
    p.add_argument("--trace-path", default=DEFAULT_TRACE)
    p.add_argument("--log-path", default=DEFAULT_LOG)
    p.add_argument("--exc", default="256M", help="Expert cache size (e.g. 64M, 256M)")
    p.add_argument("--excp", type=int, default=64, help="Expert cache period")
    p.add_argument("--fitt", type=int, default=256, help="fit-target (MiB). 0 disables fit.")
    p.add_argument("--no-fit", action="store_true", help="Disable --fit; use --ngl instead")
    p.add_argument("--ngl", type=int, default=20, help="GPU layers if --no-fit")
    p.add_argument("--wait-seconds", type=int, default=180)
    p.add_argument("--keep-log", action="store_true", help="Don't truncate log file")
    return p.parse_args()


def build_server_args(args):
    a = [
        args.exe,
        "-m", args.model,
        "--ctx-size", str(args.ctx_size),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--threads", str(args.threads),
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--no-context-shift",
        "-exc", args.exc,
        "-excp", str(args.excp),
        "--routing-predictor-horizon", "8",
        "--routing-predictor-stats",
        "--port", str(args.port),
        "--temp", "0",
        "--no-mmproj",
        "--no-mmap",
    ]
    if args.no_fit:
        a += ["-ngl", str(args.ngl)]
    else:
        a += ["-fitt", str(args.fitt)]
    return a


def wait_health(port, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            if json.loads(r.read()).get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionResetError, OSError, json.JSONDecodeError):
            pass
        time.sleep(1)
    return False


def send_completion(port, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    data = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return data.get("usage", {}).get("completion_tokens", 0)


def main():
    args = parse_args()

    if os.path.exists(args.trace_path):
        os.remove(args.trace_path)

    env = os.environ.copy()
    env["GGML_EXPERT_ROUTE_TRACE"] = args.trace_path

    log_mode = "a" if args.keep_log else "w"
    log_file = open(args.log_path, log_mode, encoding="utf-8", errors="replace")
    server_args = build_server_args(args)

    proc = subprocess.Popen(
        server_args,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=0x00000008,  # CREATE_NO_WINDOW on Windows
    )

    try:
        if not wait_health(args.port, args.wait_seconds):
            print("[FAIL] server did not become healthy", file=sys.stderr)
            return 1

        long_prompt = args.prompt * args.prompt_repeat
        n_tokens = send_completion(args.port, long_prompt, args.max_tokens)
        print(f"Tokens: {n_tokens}")

        time.sleep(3)
        if os.path.exists(args.trace_path):
            sz = os.path.getsize(args.trace_path)
        else:
            sz = -1
        print(f"Trace size: {sz}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log_file.close()


if __name__ == "__main__":
    sys.exit(main())
