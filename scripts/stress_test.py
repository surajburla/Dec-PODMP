"""
Robustness sweeps (no retraining; filter/policy NOT told about the degradation).
  python scripts/stress_test.py --policies voronoi ckpt:results/mv_belief_lik/ckpt.pt ckpt:results/mv_residual/ckpt.pt --episodes 50 --target_sigma 0.4
Sweeps: comm_drop, comm_radius, noise_scale, det_scale, target_scale (true target speed multiplier, filter assumes 1.0).
Writes results/stress.csv and results/stress.png
"""
import sys, os, argparse, csv, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from envs.extras import RobustEnv
from algos.policy_io import get_policy

ap = argparse.ArgumentParser()
ap.add_argument('--policies', nargs='+', default=['voronoi', 'goto_belief'])
ap.add_argument('--episodes', type=int, default=50); ap.add_argument('--seed', type=int, default=12345)
ap.add_argument('--target_sigma', type=float, default=None); ap.add_argument('--out', default='stress')
a = ap.parse_args(); B = a.episodes
SW = {'comm_drop': [0.0, 0.25, 0.5, 0.75, 1.0], 'comm_radius': [2.0, 4.0, 8.0, 16.0], 'noise_scale': [1.0, 1.5, 2.0, 3.0],
      'det_scale': [1.0, 0.7, 0.5, 0.3], 'target_scale': [1.0, 1.5, 2.0, 3.0]}
rows = []
for spec in a.policies:
    label, cfg0, act_fn = get_policy(spec, n_envs=B, seed=a.seed, target_sigma=a.target_sigma)
    for key, vals in SW.items():
        if key == 'target_scale' and cfg0.target_sigma <= 0: continue
        for v in vals:
            cfg = dataclasses.replace(cfg0, comm_radius=v) if key == 'comm_radius' else cfg0
            kw = {} if key == 'comm_radius' else {key: v}
            env = RobustEnv(cfg, **kw); obs = env.reset(); done = False
            while not done: obs, r, done, info = env.step(act_fn(env, obs))
            rows.append([label, key, v, info['ret'].mean(), 1.96 * info['ret'].std() / np.sqrt(B), info['map_err'].mean(), info['violation'].mean()])
            print(f"{label:16s} {key:12s} {v:5.2f} | ret {rows[-1][3]:6.1f} +- {rows[-1][4]:5.1f} | map_err {rows[-1][5]:.2f} | viol {rows[-1][6]:.2f}", flush=True)
os.makedirs('results', exist_ok=True)
with open(f'results/{a.out}.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['policy', 'param', 'value', 'ret', 'ret_ci', 'map_err', 'violation']); w.writerows(rows)
keys = [k for k in SW if any(r[1] == k for r in rows)]
fig, axs = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 7))
for c, key in enumerate(keys):
    for name in dict.fromkeys(r[0] for r in rows):
        rr = [r for r in rows if r[0] == name and r[1] == key]
        axs[0, c].errorbar([r[2] for r in rr], [r[3] for r in rr], [r[4] for r in rr], marker='o', capsize=3, label=name)
        axs[1, c].plot([r[2] for r in rr], [r[5] for r in rr], marker='o', label=name)
    axs[0, c].set_title(key); axs[0, c].set_ylabel('team return'); axs[1, c].set_ylabel('final MAP error'); axs[1, c].set_xlabel(key)
axs[0, 0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f'results/{a.out}.png', dpi=110); print(f'saved results/{a.out}.csv, results/{a.out}.png')
