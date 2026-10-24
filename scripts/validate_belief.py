"""
Validates the Bayes filter on IDENTICAL data using an exogenous (belief-independent) policy.
Checks:  (1) calibration: E[(theta-mu)^2] / E[sigma^2] ~ 1   (law of total variance; holds for an exact posterior)
         (2) information ordering:  H_central <= H_lik <= H_iso   (conditioning reduces entropy, on average)
Run:  python scripts/validate_belief.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from envs.search_env import SearchEnv, EnvConfig

E = int(sys.argv[1]) if len(sys.argv) > 1 else 48
_ts = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
env = SearchEnv(EnvConfig(n_envs=E, seed=7, fusion="lik", target_sigma=_ts), reference=True)
rng = np.random.default_rng(1)
env.reset()
a = rng.uniform(-1, 1, (E, 4, 2))
rec = {k: [] for k in ['iso', 'pool', 'lik', 'central']}
for t in range(env.T):
    a = 0.95 * a + 0.3 * rng.standard_normal(a.shape)          # OU exploration, independent of beliefs
    env.step(np.clip(a, -1, 1))
    if (t + 1) in (75, 150, 300):
        for k, f in env.ref.items():
            _, mu, sd, ent = f.stats()
            err2 = ((mu - env.theta[:, None, :]) ** 2)           # E,N,2
            rec[k].append((t + 1, ent.mean(), err2.mean() / (sd ** 2).mean(), np.linalg.norm(mu - env.theta[:, None], axis=-1).mean()))
print(f"{'filter':8s} {'t':>4s} {'mean H (nats)':>14s} {'MSE/E[var] (want~1)':>20s} {'mean |err|':>11s}")
for k in rec:
    for (t, h, ratio, e) in rec[k]:
        print(f"{k:8s} {t:4d} {h:14.3f} {ratio:20.3f} {e:11.3f}")
H = {k: rec[k][-1][1] for k in rec}
print("\nordering at t=T:  H_central=%.3f  H_lik=%.3f  H_pool=%.3f  H_iso=%.3f" % (H['central'], H['lik'], H['pool'], H['iso']))
print("H_central <= H_lik <= H_iso :", H['central'] <= H['lik'] + 1e-6 <= H['iso'] + 2e-6)
