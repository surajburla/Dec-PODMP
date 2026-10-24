import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from envs.search_env import SearchEnv, EnvConfig

env = SearchEnv(EnvConfig(n_envs=16, seed=3))
print("obs_dim", env.obs_dim, "state_dim", env.state_dim)
obs = env.reset(); assert obs.shape == (16, 4, env.obs_dim) and np.isfinite(obs).all()
b, m, s, h = env.belief_stats(); assert np.allclose(b.sum(-1), 1.0)
t0 = time.time(); rng = np.random.default_rng(0)
for t in range(env.T):
    obs, r, done, info = env.step(rng.uniform(-1, 1, (16, 4, 2)))
    assert np.isfinite(obs).all() and np.isfinite(r).all()
print(f"{env.T} steps x 16 envs in {time.time()-t0:.1f}s ({(time.time()-t0)/env.T*1e3:.1f} ms/step)")
assert done
print({k: (np.round(np.mean(v), 3) if k != 'qp_levels' else v.tolist()) for k, v in info.items()})
# determinism of layouts across policies
e1 = SearchEnv(EnvConfig(n_envs=4, seed=5)); e2 = SearchEnv(EnvConfig(n_envs=4, seed=5))
e1.reset(); e2.reset()
for _ in range(10): e1.step(np.zeros((4, 4, 2))); e2.step(np.ones((4, 4, 2)))
e1.reset(); e2.reset(); assert np.allclose(e1.theta, e2.theta) and np.allclose(e1.sams, e2.sams)
print("layout determinism across policies: OK")
