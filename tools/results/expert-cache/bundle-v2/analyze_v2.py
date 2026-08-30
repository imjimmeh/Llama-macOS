import json
import glob
import numpy as np
from scipy import stats

def analyze(prefix):
    ctrl_files = sorted(glob.glob(f'tools/results/expert-cache/bundle-v2/{prefix}*-control-*.jsonl'))
    cache_files = sorted(glob.glob(f'tools/results/expert-cache/bundle-v2/{prefix}*-cache-*.jsonl'))
    ctrl_ts = []
    cache_ts = []
    deltas = []
    actions = []
    full_hits = []
    fallbacks = []
    zero_copy = []
    avoided_mb = []
    
    for c_f, k_f in zip(ctrl_files, cache_files):
        c_data = json.loads(open(c_f).readline())
        k_data = json.loads(open(k_f).readline())
        c_t = 512.0 / (c_data['avg_ns'] / 1e9)
        k_t = 512.0 / (k_data['avg_ns'] / 1e9)
        ctrl_ts.append(c_t)
        cache_ts.append(k_t)
        deltas.append((k_t - c_t) / c_t * 100.0)
        actions.append(k_data['expert_cache_route_ready_actions'])
        full_hits.append(k_data['expert_cache_route_ready_full_hits'])
        fallbacks.append(k_data['expert_cache_route_ready_fallbacks'])
        zero_copy.append(k_data['expert_cache_zero_copy_hits'])
        avoided_mb.append(k_data['expert_cache_bytes_avoided'] / (1024*1024))
    
    return {
        'n': len(deltas),
        'ctrl_mean': np.mean(ctrl_ts),
        'cache_mean': np.mean(cache_ts),
        'delta_mean': np.mean(deltas),
        'delta_median': np.median(deltas),
        'delta_std': np.std(deltas, ddof=1) if len(deltas) > 1 else 0,
        'pos': sum(1 for d in deltas if d > 0),
        'deltas': deltas,
        'actions_mean': np.mean(actions),
        'full_hits_mean': np.mean(full_hits),
        'fallbacks_mean': np.mean(fallbacks),
        'zero_copy_mean': np.mean(zero_copy),
        'avoided_mb_mean': np.mean(avoided_mb)
    }

r_cf = analyze('2026-08-30-bundle-v2-1024m-control-first')
r_kf = analyze('2026-08-30-bundle-v2-1024m-cache-first')
r_all = analyze('2026-08-30-bundle-v2-1024m-')

print('### Control First (5 pairs):')
print(f"Control mean: {r_cf['ctrl_mean']:.3f} t/s, Cache mean: {r_cf['cache_mean']:.3f} t/s")
print(f"Mean paired delta: {r_cf['delta_mean']:+.2f}%, Median: {r_cf['delta_median']:+.2f}%, Pos: {r_cf['pos']}/{r_cf['n']}")
print('Deltas:', [f'{d:+.2f}%' for d in r_cf['deltas']])

print('\n### Cache First (5 pairs):')
print(f"Control mean: {r_kf['ctrl_mean']:.3f} t/s, Cache mean: {r_kf['cache_mean']:.3f} t/s")
print(f"Mean paired delta: {r_kf['delta_mean']:+.2f}%, Median: {r_kf['delta_median']:+.2f}%, Pos: {r_kf['pos']}/{r_kf['n']}")
print('Deltas:', [f'{d:+.2f}%' for d in r_kf['deltas']])

print('\n### Combined (10 pairs):')
print(f"Control mean: {r_all['ctrl_mean']:.3f} t/s, Cache mean: {r_all['cache_mean']:.3f} t/s")
print(f"Mean paired delta: {r_all['delta_mean']:+.2f}%, Median: {r_all['delta_median']:+.2f}%, Pos: {r_all['pos']}/{r_all['n']}")
print(f"Sample stddev: {r_all['delta_std']:.3f}%")
print(f"Actions mean: {r_all['actions_mean']:.1f}, Full Hits mean: {r_all['full_hits_mean']:.1f}, Fallbacks mean: {r_all['fallbacks_mean']:.1f}")
print(f"Zero Copy Hits mean: {r_all['zero_copy_mean']:.1f}, Avoided RAM->GPU mean: {r_all['avoided_mb_mean']:.1f} MB")

ci = stats.t.interval(0.95, df=len(r_all['deltas'])-1, loc=r_all['delta_mean'], scale=stats.sem(r_all['deltas']))
print(f'95% Student-t interval: [{ci[0]:+.3f}%, {ci[1]:+.3f}%]')
