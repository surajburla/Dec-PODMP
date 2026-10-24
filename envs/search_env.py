"""
Cooperative UAV search under SAM threat: a DEC-POMDP with
  latent state  : target location theta (static, unknown), prior uniform on box minus SAM exclusion zones
  actions       : a_i in [-1,1]^2 (nominal accel), executed u_i = HOCBF-QP(a_i)  (shield is part of the environment)
  observations  : own (p,v) exact; detection D_i~Bern(p0 exp(-(d/rho)^2)); if D_i=1, z_i=theta+N(0,sz^2 I);
                  SAM map (known a-priori threat map); teammates within comm radius
  reward (COMMON, state-action only): exp(-dmin^2/2sr^2) - lam_f sum|u|^2 - lam_s sum|a-u|^2
numpy only.  Batched over n_envs.  Episode = fixed horizon; no mid-episode resets.
"""
import numpy as np
from dataclasses import dataclass, asdict
from scipy.special import logsumexp
from safety.hocbf_shield import HOCBFShield, ShieldParams
from envs.belief import GridBayes, make_grid, NEG


@dataclass
class EnvConfig:
    n_envs: int = 16
    n_agents: int = 4
    n_sams: int = 3
    horizon: int = 300
    dt: float = 0.1
    L: float = 10.0
    drag: float = 0.5
    a_max: float = 1.0
    sam_radius: float = 1.5
    sam_margin: float = 0.2
    wall_margin: float = 0.3
    sep_dist: float = 1.0
    sep_margin: float = 0.1
    alpha: float = 0.5
    p0: float = 0.95
    rho_det: float = 3.0
    sigma_z: float = 0.7
    comm_radius: float = 8.0
    grid: int = 40
    pool: int = 8                 # coarse map = (grid/pool)^2 features
    fusion: str = "lik"           # lik | pool | none
    use_belief: bool = True       # False => actor sees only the CURRENT detection (memoryless ablation)
    prior_box: float = 0.85       # target ~ U([-prior_box*L, prior_box*L]^2) minus SAM exclusion
    sam_excl: float = 1.5         # extra exclusion radius around SAM for target
    sigma_r: float = 2.5
    lam_fuel: float = 0.01
    lam_shield: float = 0.1
    seed: int = 0
    target_sigma: float = 0.0     # >0: target does a Gaussian random walk (std per jump, world units)
    target_period: int = 5        # jump every this many steps (filter predicts at the same times)

    def dict(self): return asdict(self)


class SearchEnv:
    def __init__(self, cfg: EnvConfig, reference=False):
        self.cfg = c = cfg
        self.target_scale = getattr(self, 'target_scale', 1.0)   # TRUE speed multiplier unknown to the filter (stress tests)
        self.rng_l = np.random.default_rng(c.seed)           # layout stream (policy independent)
        self.rng_n = np.random.default_rng(c.seed + 10_000)  # measurement-noise stream
        self.E, self.N, self.S, self.T = c.n_envs, c.n_agents, c.n_sams, c.horizon
        self.cen = make_grid(c.L, c.grid); self.M = self.cen.shape[0]
        self.v_term = c.a_max / c.drag
        P = ShieldParams(drag=c.drag, a_max=c.a_max, alpha1=c.alpha, alpha2=c.alpha,
                         sam_r_eff=c.sam_radius + c.sam_margin, wall_L=c.L - c.wall_margin,
                         sep_eff=c.sep_dist + c.sep_margin)
        self.shield = HOCBFShield(P)
        self.filt = GridBayes(self.cen, self.N, c.fusion)
        self.ref = {}
        if reference:   # reference filters on IDENTICAL data, for validation only
            self.ref = {'iso': GridBayes(self.cen, self.N, 'none'), 'pool': GridBayes(self.cen, self.N, 'pool'),
                        'lik': GridBayes(self.cen, self.N, 'lik'), 'central': GridBayes(self.cen, self.N, 'lik')}
        self.others = np.array([[j for j in range(self.N) if j != i] for i in range(self.N)])
        self.obs_dim = self._obs().shape[-1] if hasattr(self, 'pos') else None
        self.reset()
        self.obs_dim = self._obs().shape[-1]
        self.state_dim = self.state().shape[-1]

    # ------------------------------------------------------------------ reset / layouts
    def _layout(self):
        c, r, L = self.cfg, self.rng_l, self.cfg.L
        R_eff = c.sam_radius + c.sam_margin
        sams = []
        for _ in range(10_000):
            if len(sams) == self.S: break
            p = r.uniform(-0.7 * L, 0.7 * L, 2)
            if all(np.linalg.norm(p - q) >= 2 * c.sam_radius + 3.0 for q in sams): sams.append(p)
        pos = []
        for _ in range(10_000):
            if len(pos) == self.N: break
            p = r.uniform(-L + 1.0, L - 1.0, 2)
            if all(np.linalg.norm(p - q) >= R_eff + 1.0 for q in sams) and \
               all(np.linalg.norm(p - q) >= c.sep_dist + 0.6 for q in pos): pos.append(p)
        sams = np.array(sams); assert len(sams) == self.S and len(pos) == self.N, "layout sampling failed"
        for _ in range(10_000):
            th = r.uniform(-c.prior_box * L, c.prior_box * L, 2)
            if np.all(np.linalg.norm(sams - th, axis=1) >= c.sam_radius + c.sam_excl): break
        return sams, np.array(pos), th

    def _logprior(self, sams):
        c = self.cfg
        inbox = (np.abs(self.cen) <= c.prior_box * c.L).all(-1)
        d = np.linalg.norm(self.cen[None, :, None, :] - sams[:, None, :, :], axis=-1)   # E,M,S
        ok = inbox[None] & (d >= c.sam_radius + c.sam_excl).all(-1)
        return np.where(ok, 0.0, NEG)

    def reset(self):
        c = self.cfg
        lay = [self._layout() for _ in range(self.E)]
        self.sams = np.stack([l[0] for l in lay]); self.pos = np.stack([l[1] for l in lay])
        self.theta = np.stack([l[2] for l in lay]); self.vel = np.zeros_like(self.pos)
        self.t = 0
        lp = self._logprior(self.sams)
        self.filt.reset(lp)
        for f in self.ref.values(): f.reset(lp)
        self.D = np.zeros((self.E, self.N), bool); self.z = np.zeros((self.E, self.N, 2))
        self.ep_ret = np.zeros(self.E); self.min_sam = np.full(self.E, np.inf)
        self.min_pair = np.full(self.E, np.inf); self.min_wall = np.full(self.E, np.inf)
        self.first_det = np.full(self.E, float(self.T)); self.dev_sum = np.zeros(self.E)
        self.lvl_count = np.zeros(3)
        return self._obs()

    # ------------------------------------------------------------------ target transition model (stochastic T)
    def _target_transition(self):
        c = self.cfg
        if c.target_sigma <= 0 or self.t % c.target_period != 0: return False
        th = self.theta + c.target_sigma * self.target_scale * self.rng_n.standard_normal(self.theta.shape)
        th = np.where(th > c.L, 2 * c.L - th, th); th = np.where(th < -c.L, -2 * c.L - th, th)
        self.theta = th
        return True

    def _filter_predict(self):
        c = self.cfg; s = c.target_sigma / (2 * c.L / c.grid)
        self.filt.predict(s, c.grid)
        for f in self.ref.values(): f.predict(s, c.grid)

    # ------------------------------------------------------------------ sensing
    def _adj(self, radius=None):
        r = self.cfg.comm_radius if radius is None else radius
        d = np.linalg.norm(self.pos[:, :, None] - self.pos[:, None], axis=-1)
        return (d <= r).astype(np.float64)                                   # diag = 1

    def _measure(self):
        c = self.cfg
        d = np.linalg.norm(self.pos - self.theta[:, None, :], axis=-1)
        pd = c.p0 * np.exp(-(d / c.rho_det) ** 2)
        self.D = self.rng_n.random(pd.shape) < pd
        self.z = self.theta[:, None, :] + c.sigma_z * self.rng_n.standard_normal(self.pos.shape)

    def _loglik(self):
        c = self.cfg
        dg = np.linalg.norm(self.pos[:, :, None, :] - self.cen[None, None], axis=-1)       # E,N,M
        pd = c.p0 * np.exp(-(dg / c.rho_det) ** 2)
        ll_det = np.log(pd + 1e-12) - 0.5 * ((self.z[:, :, None, :] - self.cen[None, None]) ** 2).sum(-1) / c.sigma_z ** 2
        ll_non = np.log1p(-pd)
        return np.where(self.D[:, :, None], ll_det, ll_non)

    # ------------------------------------------------------------------ observations / state
    def belief_stats(self):
        return self.filt.stats()

    def _obs(self):
        c, E, N, L = self.cfg, self.E, self.N, self.cfg.L
        f = [self.pos / L, self.vel / self.v_term,
             self.D[..., None].astype(float), (self.z - self.pos) / c.rho_det * self.D[..., None]]
        if c.use_belief:
            b, mean, std, ent = self.filt.stats()
            G = c.grid; q = G // c.pool
            coarse = b.reshape(E, N, q, c.pool, q, c.pool).sum((3, 5)).reshape(E, N, -1)
            f += [(mean - self.pos) / L, std / L, (ent / np.log(self.M))[..., None], coarse]
        f.append((self.sams[:, None, :, :] - self.pos[:, :, None, :]).reshape(E, N, -1) / L)
        if c.fusion != 'none':
            adj = self._adj()
            idx = self.others
            rel = (self.pos[:, idx] - self.pos[:, :, None, :]) / L
            relv = (self.vel[:, idx] - self.vel[:, :, None, :]) / self.v_term
            mk = adj[:, np.arange(N)[:, None], idx][..., None]
            f.append(np.concatenate([rel * mk, relv * mk, mk], -1).reshape(E, N, -1))
        else:
            f.append(np.zeros((E, N, (N - 1) * 5)))
        f.append(np.broadcast_to(np.eye(N)[None], (E, N, N)))
        f.append(np.full((E, N, 1), self.t / self.T))
        return np.concatenate(f, -1).astype(np.float32)

    def state(self):
        L = self.cfg.L
        s = [self.pos.reshape(self.E, -1) / L, self.vel.reshape(self.E, -1) / self.v_term,
             self.theta / L, self.sams.reshape(self.E, -1) / L, np.full((self.E, 1), self.t / self.T)]
        return np.concatenate(s, -1).astype(np.float32)

    # ------------------------------------------------------------------ step
    def step(self, a_raw):
        c = self.cfg
        a_nom = np.clip(a_raw, -1.0, 1.0) * c.a_max
        u, si = self.shield.filter(self.pos, self.vel, a_nom, self.sams)
        self.vel = self.vel + c.dt * (-c.drag * self.vel + u)
        self.pos = self.pos + c.dt * self.vel
        self.t += 1
        moved = self._target_transition()
        # reward (common; function of s_{t+1}... uses post-move positions, theta, and a,u)
        dmin = np.linalg.norm(self.pos - self.theta[:, None, :], axis=-1).min(1)
        r = np.exp(-dmin ** 2 / (2 * c.sigma_r ** 2)) \
            - c.lam_fuel * (u ** 2).sum((1, 2)) - c.lam_shield * (si['dev'] ** 2).sum(1)
        self.ep_ret += r
        # sensing + belief
        self._measure()
        ll = self._loglik()
        if moved: self._filter_predict()
        self.filt.update(ll, self._adj())
        if self.ref:
            ones = np.ones((self.E, self.N, self.N))
            for k, f in self.ref.items():
                f.update(ll, ones if k == 'central' else self._adj())
        # metrics
        dsam = np.linalg.norm(self.pos[:, :, None] - self.sams[:, None], axis=-1) - c.sam_radius
        self.min_sam = np.minimum(self.min_sam, dsam.min((1, 2)))
        dp = np.linalg.norm(self.pos[:, :, None] - self.pos[:, None], axis=-1) + np.eye(self.N)[None] * 1e9
        self.min_pair = np.minimum(self.min_pair, dp.min((1, 2)) - c.sep_dist)
        self.min_wall = np.minimum(self.min_wall, (c.L - np.abs(self.pos).max(-1)).min(1))
        newdet = self.D.any(1) & (self.first_det >= self.T)
        self.first_det = np.where(newdet, self.t, self.first_det)
        self.dev_sum += si['dev'].mean(1)
        for l in range(3): self.lvl_count[l] += (si['level'] == l).sum()
        done = self.t >= self.T
        info = {}
        if done:
            _, mean, std, ent = self.filt.stats()
            info = dict(ret=self.ep_ret.copy(), min_sam=self.min_sam.copy(), min_pair=self.min_pair.copy(),
                        min_wall=self.min_wall.copy(),
                        violation=((self.min_sam < 0) | (self.min_pair < 0) | (self.min_wall < 0)).astype(float),
                        first_det=self.first_det.copy(), shield_dev=self.dev_sum / self.T,
                        map_err=np.linalg.norm(mean - self.theta[:, None, :], axis=-1).mean(1),
                        entropy=ent.mean(1), qp_levels=self.lvl_count.copy())
        return self._obs(), r, done, info
