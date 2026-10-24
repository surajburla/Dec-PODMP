"""
Empirical checks that the simulator has the DEC-POMDP information structure.  python tests/test_decpomdp_structure.py
 T1 common reward        : one scalar reward per environment (not per agent)
 T2 obs locality         : obs_i does not depend on theta, nor on other agents' private measurement/belief
 T3 comm-graph flow      : agent i's belief reacts to agent j's data iff the link i<-j is active
 T4 recursive == batch   : the recursive filter equals the batch Bayes posterior of the agent's received history (belief = sufficient statistic)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from scipy.special import logsumexp
from envs.refine import HistoryEnv
from envs.search_env import SearchEnv, EnvConfig

ok_all = True
def check(name, cond, extra=""):
    global ok_all; ok_all &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

rng = np.random.default_rng(0)
env = HistoryEnv(EnvConfig(n_envs=4, seed=11, fusion='lik'), reference=True); env.reset()
for _ in range(60): obs, r, done, info = env.step(rng.uniform(-1, 1, (4, 4, 2)))

# T1
check("T1 common reward: reward shape == (n_envs,)", r.shape == (4,), f"shape={r.shape}")

# T2a: theta absent from observations
o0 = env._obs(); th = env.theta.copy(); env.theta = th + 5.0; o1 = env._obs(); env.theta = th
check("T2a obs independent of true theta (given stored D,z,belief)", np.array_equal(o0, o1))
# T2b: other agents' private measurement and belief absent from obs_i
o0 = env._obs(); lb, D, z = env.filt.lb.copy(), env.D.copy(), env.z.copy()
res = []
for j in range(env.N):
    env.filt.lb[:, j] += rng.normal(size=env.filt.lb[:, j].shape); env.D[:, j] = ~env.D[:, j]; env.z[:, j] += 3.0
    o1 = env._obs()
    res.append(all(np.array_equal(o0[:, i], o1[:, i]) for i in range(env.N) if i != j))
    env.filt.lb, env.D, env.z = lb.copy(), D.copy(), z.copy()
check("T2b obs_i independent of other agents' private D, z and belief", all(res))

# T3: information flows only along the communication graph
from envs.belief import GridBayes
f = GridBayes(env.cen, 4, 'lik'); f.reset(np.zeros((1, env.M)))
ll = rng.normal(size=(1, 4, env.M)); adj = np.eye(4)[None].copy(); adj[0, 0, 1] = 1.0    # agent 0 hears agent 1 only
f.update(ll, adj); base = f.lb.copy()
ll2 = ll.copy(); ll2[0, 1] += rng.normal(size=env.M); f.reset(np.zeros((1, env.M))); f.update(ll2, adj)
changed = [not np.allclose(base[0, i], f.lb[0, i]) for i in range(4)]
check("T3 changing agent-1 data changes belief of agents {1, 0(hears 1)} only", changed == [True, True, False, False], f"changed={changed}")

# T4: recursive filter == batch Bayes on the received history
P = np.stack(env.h_pos); D_ = np.stack(env.h_D); Z = np.stack(env.h_z); A = np.stack(env.h_adj); c = env.cfg
def batch_logpost(e, kind, i):
    lp = env._logprior(env.sams)[e].copy()
    for t in range(P.shape[0]):
        for j in range(env.N):
            use = (kind == 'central') or (kind == 'iso' and j == i) or (kind == 'lik' and A[t, e, i, j] > 0)
            if not use: continue
            dg = np.linalg.norm(P[t, e, j] - env.cen, axis=-1); pd = c.p0 * np.exp(-(dg / c.rho_det) ** 2)
            lp += (np.log(pd + 1e-12) - 0.5 * ((Z[t, e, j] - env.cen) ** 2).sum(-1) / c.sigma_z ** 2) if D_[t, e, j] else np.log1p(-pd)
    return lp
worst = 0.0
for kind in ['iso', 'lik', 'central']:
    for e in range(2):
        for i in range(env.N):
            lp = batch_logpost(e, kind, i); fin = env._logprior(env.sams)[e] > -1e29
            lp = lp - logsumexp(lp[fin]); rec = env.ref[kind].lb[e, i]
            worst = max(worst, np.abs(lp[fin] - rec[fin]).max())
check("T4 recursive Bayes == batch Bayes (sufficient statistic)", worst < 1e-6, f"max |dlogp| = {worst:.2e}")
print("RESULT:", "ALL PASS" if ok_all else "SOME FAIL")
