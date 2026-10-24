"""
Belief-MAPPO training.  Run from ...\\DP :
  python scripts/train.py --run belief_lik                                   (static target, as before)
  python scripts/train.py --run mv_belief_lik --target_sigma 0.4             (moving target)
  python scripts/train.py --run mv_residual   --target_sigma 0.4 --residual  (residual RL on the Voronoi controller)
Ablations: --fusion pool|none   --no_belief
"""
import sys, os, time, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from envs.search_env import SearchEnv, EnvConfig
from envs.extras import voronoi_action
from algos.mappo import Actor, Critic, RetStats, ppo_update
import gc
import torch

# Add inside the update loop in train.py:
gc.collect()
torch.cuda.empty_cache()
ap = argparse.ArgumentParser()
ap.add_argument('--run', default='belief_lik'); ap.add_argument('--updates', type=int, default=800)
ap.add_argument('--envs', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--fusion', default='lik', choices=['lik', 'pool', 'none'])
ap.add_argument('--no_belief', action='store_true'); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--target_sigma', type=float, default=0.0); ap.add_argument('--target_period', type=int, default=5)
ap.add_argument('--residual', action='store_true'); ap.add_argument('--res_scale', type=float, default=0.5)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()

np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device(a.device)
cfg = EnvConfig(n_envs=a.envs, fusion=a.fusion, use_belief=not a.no_belief, seed=a.seed,
                target_sigma=a.target_sigma, target_period=a.target_period)
env = SearchEnv(cfg); E, N, T = env.E, env.N, env.T
od = env.obs_dim + (2 if a.residual else 0)          # residual: actor also sees the base controller's action
actor, critic = Actor(od).to(dev), Critic(env.state_dim).to(dev)
opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=a.lr, eps=1e-5)
hp = dict(gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.003, vf_coef=0.5, epochs=8, minibatches=4)
stats = RetStats()
out = os.path.join(os.path.dirname(__file__), '..', 'results', a.run); os.makedirs(out, exist_ok=True)
f = open(os.path.join(out, 'log.csv'), 'w', newline=''); w = csv.writer(f)
w.writerow(['update', 'ret', 'violation', 'shield_dev', 'first_det', 'map_err', 'entropy', 'pg', 'vl', 'ent', 'kl_proxy', 'sec'])
tt = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
print(f"run={a.run} dev={dev} obs_dim={od} state_dim={env.state_dim} fusion={a.fusion} belief={not a.no_belief} "
      f"target_sigma={a.target_sigma} residual={a.residual}")

for upd in range(1, a.updates + 1):
    t0 = time.time()
    for g in opt.param_groups: g['lr'] = a.lr * (1 - 0.9 * (upd - 1) / a.updates)
    buf = dict(obs=torch.zeros(T, E, N, od, device=dev), act=torch.zeros(T, E, N, 2, device=dev),
               logp=torch.zeros(T, E, N, device=dev), rew=torch.zeros(T, E, device=dev),
               val=torch.zeros(T, E, device=dev), state=torch.zeros(T, E, env.state_dim, device=dev))
    obs = env.reset()
    for t in range(T):
        base = None
        if a.residual:
            base = voronoi_action(env); obs_in = np.concatenate([obs, base.astype(np.float32)], -1)
        else:
            obs_in = obs
        st = tt(env.state()); o = tt(obs_in)
        with torch.no_grad():
            d = actor.dist(o.reshape(E * N, -1)); act = d.sample()
            lp = d.log_prob(act).sum(-1).view(E, N)
            v = critic(st) * stats.std + stats.mean
        a_np = act.view(E, N, 2).cpu().numpy()
        env_act = np.clip(base + a.res_scale * a_np, -1, 1) if a.residual else a_np
        obs, r, done, info = env.step(env_act)
        buf['obs'][t] = o; buf['act'][t] = act.view(E, N, 2); buf['logp'][t] = lp
        buf['rew'][t] = tt(r); buf['val'][t] = v; buf['state'][t] = st
    pg, vl, ent, kl = ppo_update(actor, critic, opt, buf, stats, hp)
    row = [upd, info['ret'].mean(), info['violation'].mean(), info['shield_dev'].mean(), info['first_det'].mean(),
           info['map_err'].mean(), info['entropy'].mean(), pg, vl, ent, kl, time.time() - t0]
    w.writerow(row); f.flush()
    print(f"upd {upd:4d}/{a.updates} | ret {row[1]:7.2f} | viol {row[2]:.2f} | dev {row[3]:.3f} | 1st-det {row[4]:5.1f} | "
          f"mapErr {row[5]:.2f} | H {row[6]:.2f} | pi_ent {ent:.2f} | {row[-1]:.1f}s")
    if upd % 50 == 0 or upd == a.updates:
        torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), cfg=cfg.dict(), obs_dim=od,
                        state_dim=env.state_dim, stats=(stats.mean, stats.std), residual=a.residual, res_scale=a.res_scale),
                   os.path.join(out, 'ckpt.pt'))
print("done ->", out)
