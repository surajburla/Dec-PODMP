"""
Robust calibration test of the posteriors, grid vs continuously-refined, on IDENTICAL data.
  python scripts/validate_refined.py 200 300        (E envs, checkpoint step)
Statistic: HPD level alpha = posterior mass of {theta': p(theta'|data) > p(theta_true|data)}.
For an exact posterior alpha ~ Uniform(0,1):  mean 0.5,  P(alpha>0.9)=0.10,  P(alpha>0.99)=0.01.
Overconfidence shows as mass piling up near alpha=1.  Also prints the (outlier-sensitive) MSE/E[var] ratio.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.stats import kstest
from envs.refine import HistoryEnv, stack_hist, refined_moments
from envs.search_env import EnvConfig

E = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TC = int(sys.argv[2]) if len(sys.argv) > 2 else 300
env = HistoryEnv(EnvConfig(n_envs=E, seed=7, fusion='lik'), reference=True)
rng = np.random.default_rng(1); env.reset(); a = rng.uniform(-1, 1, (E, 4, 2)); t0 = time.time()
for t in range(TC):
    a = 0.95 * a + 0.3 * rng.standard_normal(a.shape); env.step(np.clip(a, -1, 1))
H = stack_hist(env)
print(f"E={E}, t={TC}.  Exact posterior => mean 0.50, P(>0.9)=0.10, P(>0.99)=0.01, KS p-value not tiny")
print(f"{'filter':8s} {'est.':8s} {'n':>5s} {'mean a':>7s} {'P(a>.9)':>8s} {'P(a>.99)':>9s} {'KS p':>8s} | {'MSE/Evar':>8s} {'min mass':>8s} {'mean r':>6s}")
for kind in ['iso', 'lik', 'central']:
    hr, hg, er, vr, cp, rr = [], [], [], [], [], []
    for e in range(E):
        for i in ([0] if kind == 'central' else range(env.N)):
            mu, var, k, r, cap, a_r, a_g = refined_moments(env, H, kind, e, i)
            hr.append(a_r); hg.append(a_g); er.append((env.theta[e] - mu) ** 2); vr.append(var); cp.append(cap); rr.append(r)
    er, vr = np.array(er), np.array(vr)
    for name, h in [('grid', hg), ('refined', hr)]:
        h = np.array(h)
        print(f"{kind:8s} {name:8s} {len(h):5d} {h.mean():7.3f} {(h>.9).mean():8.3f} {(h>.99).mean():9.3f} {kstest(h,'uniform').pvalue:8.4f} | "
              f"{(er.mean()/vr.mean() if name=='refined' else float('nan')):8.3f} {min(cp):8.4f} {np.mean(rr):6.1f}")
    print(f"   [{time.time()-t0:.0f}s]")
