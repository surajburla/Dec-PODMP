"""Moving-target checks.  python tests/test_moving.py
 M1 predict step: mass conserved; away from boundary a delta spreads with variance sigma^2 and unchanged mean (Chapman-Kolmogorov)
 M2 closed-loop calibration: filter that KNOWS the random-walk model has E[err^2]/E[var] ~ 1 on the true target motion
 M3 mismatch: if the true target moves faster than the filter assumes, the ratio exceeds 1 (overconfidence is detectable)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from envs.search_env import SearchEnv, EnvConfig
from envs.extras import RobustEnv

cfg = EnvConfig(n_envs=2, seed=1, target_sigma=0.4)
env = SearchEnv(cfg); env.reset()
lb = np.full((2, env.M), -1e30); i0 = np.argmin(np.linalg.norm(env.cen, axis=1)); lb[:, i0] = 0.0
env.filt.reset(lb); env.filt.predict(cfg.target_sigma / (2 * cfg.L / cfg.grid), cfg.grid)
_, mu, sd, _ = env.filt.stats()
print("M1 mass", np.exp(env.filt.lb).sum(-1).ravel()[:2], " mean", mu[0, 0].round(3), " std", sd[0, 0].round(3), "(target std 0.400, cell 0.5 => grid sees ~0.41 incl. quantisation)")
assert np.allclose(np.exp(env.filt.lb).sum(-1), 1.0) and np.abs(mu[0, 0] - env.cen[i0]).max() < 0.05 and np.abs(sd[0, 0] - 0.4).max() < 0.08
print("[PASS] M1")

def calib(scale, E=120):
    e = RobustEnv(EnvConfig(n_envs=E, seed=5, target_sigma=0.4, fusion='lik'), target_scale=scale, reference=True)
    rng = np.random.default_rng(0); e.reset(); a = rng.uniform(-1, 1, (E, 4, 2)); out = {}
    for t in range(e.T):
        a = 0.95 * a + 0.3 * rng.standard_normal(a.shape); e.step(np.clip(a, -1, 1))
        if t + 1 >= 100:
            for k in ['iso', 'lik']:
                _, mu, sd, _ = e.ref[k].stats()
                out.setdefault(k, []).append((((mu - e.theta[:, None]) ** 2).mean(), (sd ** 2).mean()))
    return {k: np.mean([a for a, b in v]) / np.mean([b for a, b in v]) for k, v in out.items()}
r1, r2 = calib(1.0), calib(2.5)
print("M2 matched model   MSE/E[var]:", {k: round(v, 3) for k, v in r1.items()})
print("M3 target 2.5x faster than filter assumes:", {k: round(v, 3) for k, v in r2.items()})
assert all(0.8 < v < 1.25 for v in r1.values()); print("[PASS] M2 (within 0.8..1.25)")
assert all(v > 1.3 for v in r2.values()); print("[PASS] M3 (mismatch detected as overconfidence)")
