"""
Closed-form HOCBF safety shield (numpy only, no cvxpy, no torch).

Dynamics per UAV (drag double integrator):   p' = v,   v' = -c v + u,   |u|_inf <= a_max

Every constraint has the same relative-degree-2 form for a scalar barrier h(p):
    h'' + (a1+a2) h' + a1 a2 h >= 0,     a1, a2 > 0 REAL   (real roots => valid HOCBF)
which is LINEAR in u:    -n^T u  <=  (a1+a2-c) * hdot + kappa + a1 a2 h
  SAM i  : h = |p-s_i| - R_eff,  n=(p-s_i)/|p-s_i|, hdot=n.v, kappa=(|v|^2-hdot^2)/|p-s_i|
  wall   : h = n^T p + L_w,      n = inward normal,  hdot=n.v, kappa=0
  pair ij: h = |p_i-p_j| - d_eff; full constraint on (u_i-u_j) is split 50/50 (reciprocal split)
QP:  min 0.5|u-a|^2  s.t. G u <= g  (+ box).  Solved EXACTLY in 2-D by enumerating
{a, projections onto each constraint line, all pairwise line intersections}, keeping the
feasible candidate of least cost (optimum of a 2-D polyhedral QP is always one of these).
Infeasibility fallback (logged, never silent): level 1 drops pair rows, level 2 box only.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class ShieldParams:
    drag: float = 0.5
    a_max: float = 1.0
    alpha1: float = 0.5
    alpha2: float = 0.5
    sam_r_eff: float = 1.7        # true radius + margin
    wall_L: float = 9.7           # arena half-size minus margin
    sep_eff: float = 1.1          # true separation + margin


class HOCBFShield:
    def __init__(self, p: ShieldParams):
        self.p = p
        self.walls = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])  # inward normals

    # ---------------------------------------------------------------- constraint assembly
    def _rows(self, pos, vel, sams):
        p = self.p
        E, N, _ = pos.shape
        S = sams.shape[1]
        k = p.alpha1 + p.alpha2 - p.drag
        ab = p.alpha1 * p.alpha2
        Gs, gs, grp = [], [], []

        # box (group 0)
        Gb = np.broadcast_to(np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], float), (E, N, 4, 2))
        gb = np.full((E, N, 4), p.a_max)
        Gs.append(Gb); gs.append(gb); grp += [0] * 4

        # SAMs (group 1)
        r = pos[:, :, None, :] - sams[:, None, :, :]                       # E,N,S,2
        D = np.maximum(np.linalg.norm(r, axis=-1), 1e-6)
        n = r / D[..., None]
        h = D - p.sam_r_eff
        hd = (n * vel[:, :, None, :]).sum(-1)                              # n.v
        vt2 = np.maximum((vel ** 2).sum(-1)[:, :, None] - hd ** 2, 0.0)
        Gs.append(-n); gs.append(k * hd + vt2 / D + ab * h); grp += [1] * S

        # walls (group 1)
        nw = np.broadcast_to(self.walls, (E, N, 4, 2))
        hw = np.einsum('enkc,enc->enk', nw, pos) + p.wall_L
        hdw = np.einsum('enkc,enc->enk', nw, vel)
        Gs.append(-nw); gs.append(k * hdw + ab * hw); grp += [1] * 4

        # pairs (group 2), reciprocal split; diagonal rows inactive
        rij = pos[:, :, None, :] - pos[:, None, :, :]                      # E,N,N,2
        Dij = np.maximum(np.linalg.norm(rij, axis=-1), 1e-6)
        nij = rij / Dij[..., None]
        vij = vel[:, :, None, :] - vel[:, None, :, :]
        hdij = (nij * vij).sum(-1)
        vt2ij = np.maximum((vij ** 2).sum(-1) - hdij ** 2, 0.0)
        b = k * hdij + vt2ij / Dij + ab * (Dij - p.sep_eff)
        Gp = -nij.copy(); gp = 0.5 * b
        eye = np.eye(N, dtype=bool)[None]
        Gp = np.where(eye[..., None], 0.0, Gp)
        gp = np.where(eye, 1e3, gp)
        Gs.append(Gp); gs.append(gp); grp += [2] * N

        G = np.concatenate(Gs, axis=2)       # E,N,m,2
        g = np.concatenate(gs, axis=2)       # E,N,m
        return G, g, np.array(grp)

    # ---------------------------------------------------------------- exact 2-D QP
    @staticmethod
    def _solve(G, g, a, active, tol=1e-7):
        """G:(B,m,2) g:(B,m) a:(B,2) active:(B,m) bool. returns u (B,2), feasible (B,)"""
        B, m, _ = G.shape
        nrm = np.linalg.norm(G, axis=-1)
        valid = (nrm > 1e-9) & active
        s = np.maximum(nrm, 1e-9)
        Gn = np.where(valid[..., None], G / s[..., None], 0.0)
        gn = np.where(valid, g / s, 1e6)

        viol = np.einsum('bmc,bc->bm', Gn, a) - gn
        proj = a[:, None, :] - viol[..., None] * Gn                         # B,m,2
        iu, ju = np.triu_indices(m, 1)
        n1, n2, g1, g2 = Gn[:, iu], Gn[:, ju], gn[:, iu], gn[:, ju]
        det = n1[..., 0] * n2[..., 1] - n1[..., 1] * n2[..., 0]
        ok = np.abs(det) > 1e-8
        ds = np.where(ok, det, 1.0)
        x = (g1 * n2[..., 1] - g2 * n1[..., 1]) / ds
        y = (n1[..., 0] * g2 - n2[..., 0] * g1) / ds
        vert = np.where(ok[..., None], np.stack([x, y], -1), 1e9)

        C = np.concatenate([a[:, None, :], proj, vert], axis=1)             # B,K,2
        val = np.einsum('bkc,bmc->bkm', C, Gn) - gn[:, None, :]
        feas = (val <= tol).all(-1)
        cost = ((C - a[:, None, :]) ** 2).sum(-1)
        cost = np.where(feas, cost, np.inf)
        j = cost.argmin(1)
        any_f = feas.any(1)
        u = C[np.arange(B), j]
        return u, any_f

    # ---------------------------------------------------------------- public API
    def filter(self, pos, vel, a_nom, sams):
        """pos,vel,a_nom:(E,N,2)  sams:(E,S,2). Returns u (E,N,2), info dict."""
        E, N, _ = pos.shape
        G, g, grp = self._rows(pos.astype(np.float64), vel.astype(np.float64), sams.astype(np.float64))
        m = G.shape[2]
        B = E * N
        Gf, gf = G.reshape(B, m, 2), g.reshape(B, m)
        a = np.clip(a_nom, -self.p.a_max, self.p.a_max).reshape(B, 2).astype(np.float64)

        level = np.zeros(B, dtype=np.int8)
        act = np.ones((B, m), bool)
        u, f = self._solve(Gf, gf, a, act)
        bad = ~f
        if bad.any():                                   # level 1: drop pair constraints
            act1 = act[bad] & (grp != 2)[None, :]
            u1, f1 = self._solve(Gf[bad], gf[bad], a[bad], act1)
            idx = np.where(bad)[0]
            u[idx] = u1; level[idx] = 1
            bad2 = idx[~f1]
            if bad2.size:                               # level 2: box only (UNSAFE, logged)
                act2 = np.broadcast_to((grp == 0)[None, :], (bad2.size, m))
                u2, _ = self._solve(Gf[bad2], gf[bad2], a[bad2], act2)
                u[bad2] = u2; level[bad2] = 2
        dev = np.linalg.norm(u - a, axis=-1)
        info = dict(level=level.reshape(E, N), dev=dev.reshape(E, N))
        return u.reshape(E, N, 2), info
