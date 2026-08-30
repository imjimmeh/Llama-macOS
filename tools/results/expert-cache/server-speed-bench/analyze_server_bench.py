import json
import numpy as np

ctrl_summary = json.load(open('tools/results/expert-cache/server-speed-bench/2026-08-30-speedbench-control-10x-512-summary.json'))
dyn_summary = json.load(open('tools/results/expert-cache/server-speed-bench/2026-08-30-speedbench-dynamic128m-10x-512-summary.json'))

print("### Pass-by-Pass Throughput (predicted_per_second / tok/s):")
print("Pass | Control (Cache-Off) | Dynamic (128M) | Delta (%) | Latency Ctrl | Latency Dyn")
print("---- | ------------------- | -------------- | --------- | ------------ | -----------")

ctrl_t = []
dyn_t = []
deltas = []

for c, d in zip(ctrl_summary, dyn_summary):
    p = c['pass']
    c_speed = c['predicted_per_second']
    d_speed = d['predicted_per_second']
    c_lat = c['latency_s']
    d_lat = d['latency_s']
    delta = (d_speed - c_speed) / c_speed * 100.0
    ctrl_t.append(c_speed)
    dyn_t.append(d_speed)
    deltas.append(delta)
    print(f"{p:4d} | {c_speed:17.2f} | {d_speed:14.2f} | {delta:+8.2f}% | {c_lat:10.3f}s | {d_lat:9.3f}s")

print("-------------------------------------------------------------------------------------")
print(f"Mean | {np.mean(ctrl_t):17.2f} | {np.mean(dyn_t):14.2f} | {np.mean(deltas):+8.2f}% | {np.mean([c['latency_s'] for c in ctrl_summary]):10.3f}s | {np.mean([d['latency_s'] for d in dyn_summary]):9.3f}s")
print(f"Warm (Pass 2-10) Mean: Control={np.mean(ctrl_t[1:]):.2f} tok/s, Dynamic={np.mean(dyn_t[1:]):.2f} tok/s ({((np.mean(dyn_t[1:]) - np.mean(ctrl_t[1:])) / np.mean(ctrl_t[1:]) * 100.0):+.2f}%)")
