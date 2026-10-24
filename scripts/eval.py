"""
Evaluate on IDENTICAL layouts / target trajectories / noise (common random numbers).
  python scripts/eval.py --policy voronoi --save results/voronoi.npz
  python scripts/eval.py --policy ckpt --ckpt results/belief_lik/ckpt.pt --save results/belief_lik.npz
  add --target_sigma 0.4 to evaluate ANY policy (also old static-trained ckpts) on the moving target.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from envs.search_env import SearchEnv
from algos.policy_io import get_policy

ap = argparse.ArgumentParser()
ap.add_argument('--policy', default='random', choices=['random', 'goto_belief', 'voronoi', 'ckpt'])
ap.add_argument('--ckpt', default=None); ap.add_argument('--fusion', default='lik')
ap.add_argument('--episodes', type=int, default=100); ap.add_argument('--seed', type=int, default=12345)
ap.add_argument('--target_sigma', type=float, default=None); ap.add_argument('--save', default=None)
a = ap.parse_args()
B = 50
over = dict(n_envs=B, seed=a.seed, target_sigma=a.target_sigma)
if a.policy != 'ckpt': over['fusion'] = a.fusion
label, cfg, act_fn = get_policy('ckpt:' + a.ckpt if a.policy == 'ckpt' else a.policy, **over)
env = SearchEnv(cfg); res = {}
for _ in range(a.episodes // B):
    obs = env.reset(); done = False
    while not done:
        obs, r, done, info = env.step(act_fn(env, obs))
    for k, v in info.items():
        if k != 'qp_levels': res.setdefault(k, []).append(v)
print(f"policy={label} episodes={a.episodes} seed={a.seed} target_sigma={cfg.target_sigma}")
for k, v in res.items():
    v = np.concatenate(v); print(f"  {k:12s} {v.mean():8.3f} +- {1.96 * v.std() / np.sqrt(len(v)):.3f}")
print("  QP levels (0/1/2):", info['qp_levels'].tolist())
if a.save:
    os.makedirs(os.path.dirname(a.save) or '.', exist_ok=True)
    np.savez(a.save, **{k: np.concatenate(v) for k, v in res.items()}); print('saved per-episode metrics ->', a.save)
