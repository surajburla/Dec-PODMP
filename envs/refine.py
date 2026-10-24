"""
Continuous refinement of the grid posterior (OFFLINE estimator; does not touch the policy's features).
The exact posterior is  p(theta | data) ∝ prior(theta) * prod_k lik_k(theta)   -- a function of the stored measurement history,
so it can be evaluated at ANY theta.  We subdivide the grid cells that carry the posterior mass into r x r subcells (r adaptive to the
posterior width), evaluate the exact log-posterior at subcell centres, and compute moments with the within-subcell variance hs^2/12.
Exact up to the subcell spacing hs; the grid is then the coarse proposal.   (Zoomed-grid / adaptive mesh refinement.)
"""
import numpy as np
from envs.search_env import SearchEnv


class HistoryEnv(SearchEnv):
    """SearchEnv that also records (pos, D, z, adjacency) of every step. Original classes are unchanged."""
    def reset(self):
        o = super().reset()
        self.h_pos, self.h_D, self.h_z, self.h_adj = [], [], [], []
        return o

    def step(self, a):
        out = super().step(a)
        self.h_pos.append(self.pos.copy()); self.h_D.append(self.D.copy())
        self.h_z.append(self.z.copy()); self.h_adj.append(self._adj())
        return out


def stack_hist(env):
    return (np.stack(env.h_pos), np.stack(env.h_D), np.stack(env.h_z), np.stack(env.h_adj))   # (t,E,N,..)


def refined_moments(env, H, kind, e, i, mass=0.99999, kmax=1000, rmax=9, chunk=2000):
    """Returns (mean(2), var(2), n_cells, r, captured_mass, hpd_refined, hpd_grid) for agent i in env e.
    hpd = posterior mass of {theta': p(theta') > p(theta_true)}; Uniform(0,1) if the posterior is exact."""
    c = env.cfg
    P, D, Z, A = H
    t = P.shape[0]
    if kind == 'iso':
        m = np.zeros((t, env.N), bool); m[:, i] = True
    elif kind == 'lik':
        m = A[:, e, i, :] > 0
    else:
        m = np.ones((t, env.N), bool)
    pm, Dm, zm = P[:, e][m], D[:, e][m], Z[:, e][m]
    b = np.exp(env.ref[kind].lb[e, i]); b = b / b.sum()
    order = np.argsort(-b); cs = np.cumsum(b[order])
    h = 2 * c.L / c.grid
    mu_g = b @ env.cen; sg = np.sqrt(np.maximum(b @ env.cen ** 2 - mu_g ** 2, 1e-9)).min()
    r = int(np.clip(np.ceil(h / (0.3 * sg)), 1, rmax))
    k = env.M if r == 1 else min(int(np.searchsorted(cs, mass)) + 1, kmax)   # diffuse => nothing to refine, keep ALL cells
    sel = order[:k]; cap = cs[k - 1]
    off = (np.arange(r) + 0.5) / r * h - h / 2
    ox, oy = np.meshgrid(off, off, indexing='ij')
    sub = np.stack([ox.ravel(), oy.ravel()], -1)                          # r^2,2
    pts = (env.cen[sel][:, None, :] + sub[None]).reshape(-1, 2)
    hs = h / r
    # exact prior indicator
    ok = (np.abs(pts) <= c.prior_box * c.L).all(-1)
    ok &= (np.linalg.norm(pts[:, None, :] - env.sams[e][None], axis=-1) >= c.sam_radius + c.sam_excl).all(-1)
    pts = pts[ok]
    def logpost(Q):
        out = np.empty(len(Q))
        for s0 in range(0, len(Q), chunk):
            q = Q[s0:s0 + chunk]
            d = np.linalg.norm(q[:, None, :] - pm[None], axis=-1)
            pd = c.p0 * np.exp(-(d / c.rho_det) ** 2)
            ll_det = np.log(pd + 1e-300) - 0.5 * ((q[:, None, :] - zm[None]) ** 2).sum(-1) / c.sigma_z ** 2
            out[s0:s0 + chunk] = np.where(Dm[None], ll_det, np.log1p(-pd)).sum(1)
        return out
    lp = logpost(pts)
    lq = logpost(env.theta[e][None])[0]
    w = np.exp(lp - lp.max()); w /= w.sum()
    mu = w @ pts
    var = w @ pts ** 2 - mu ** 2 + hs ** 2 / 12
    hpd_r = w[lp > lq].sum()
    G = c.grid; ix, iy = np.clip(((env.theta[e] + c.L) / h).astype(int), 0, G - 1)
    hpd_g = b[b > b[ix * G + iy]].sum()
    return mu, np.maximum(var, 1e-12), k, r, cap, hpd_r, hpd_g
