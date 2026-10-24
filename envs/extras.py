"""Extensions (original files untouched): stress-test env + decentralized scripted baselines. numpy only."""
import numpy as np
from envs.search_env import SearchEnv, EnvConfig


class RobustEnv(SearchEnv):
    """Test-time degradations that the filter and policy are NOT told about (model mismatch):
       comm_drop   : each undirected link is dropped i.i.d. per step with this probability (applies to belief fusion AND teammate features)
       noise_scale : true measurement std = noise_scale * sigma_z (filter still assumes sigma_z)
       det_scale   : true detection prob = det_scale * p0*exp(-(d/rho)^2) (filter still assumes nominal)
       NOTE: the safety shield still uses true neighbour states (assumed separate collision-avoidance sensing)."""
    def __init__(self, cfg, comm_drop=0.0, noise_scale=1.0, det_scale=1.0, target_scale=1.0, **kw):
        self.comm_drop, self.noise_scale, self.det_scale, self.target_scale = comm_drop, noise_scale, det_scale, target_scale
        self._keep = None
        self.rng_m = np.random.default_rng(cfg.seed + 20_000)      # link-drop stream: does NOT perturb the measurement/target stream
        super().__init__(cfg, **kw)

    def _new_mask(self):
        N = self.N
        if self.comm_drop <= 0: self._keep = None; return
        k = np.triu(self.rng_m.random((self.E, N, N)) >= self.comm_drop, 1)
        self._keep = k | k.transpose(0, 2, 1) | np.eye(N, dtype=bool)[None]

    def reset(self):
        self._new_mask(); return super().reset()

    def step(self, a):
        self._new_mask(); return super().step(a)

    def _adj(self, radius=None):
        A = super()._adj(radius)
        return A if self._keep is None else A * self._keep

    def _measure(self):
        c = self.cfg
        d = np.linalg.norm(self.pos - self.theta[:, None, :], axis=-1)
        pd = self.det_scale * c.p0 * np.exp(-(d / c.rho_det) ** 2)
        self.D = self.rng_n.random(pd.shape) < pd
        self.z = self.theta[:, None, :] + c.sigma_z * self.noise_scale * self.rng_n.standard_normal(self.pos.shape)


def goto_belief_action(env, kp=1.0, kd=1.5):
    _, mean, _, _ = env.belief_stats()
    return np.clip(kp * (mean - env.pos) - kd * env.vel, -1, 1)


def voronoi_action(env, kp=1.0, kd=1.5):
    """Decentralized Lloyd/Voronoi coverage on the agent's OWN belief density. Agent i partitions the arena among itself and the
    neighbours it can currently talk to, and heads to the belief-weighted centroid of its own cell. No learning."""
    b, _, _, _ = env.belief_stats(); adj = env._adj()
    d = np.linalg.norm(env.cen[None, None] - env.pos[:, :, None, :], axis=-1)           # E,N,M
    tgt = np.zeros_like(env.pos)
    for i in range(env.N):
        dn = np.where(adj[:, i, :, None] > 0, d, np.inf).min(1)                         # E,M  nearest among {i} U neighbours
        w = b[:, i, :] * (d[:, i, :] <= dn + 1e-12)
        s = w.sum(-1, keepdims=True)
        tgt[:, i] = np.where(s > 1e-9, (w @ env.cen) / np.maximum(s, 1e-12), env.pos[:, i])
    return np.clip(kp * (tgt - env.pos) - kd * env.vel, -1, 1)


SCRIPTED = {'goto_belief': goto_belief_action, 'voronoi': voronoi_action}
