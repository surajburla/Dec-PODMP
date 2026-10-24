"""Standalone invariance test of the HOCBF shield. No RL, no torch.  Run:  python tests/test_shield.py"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from safety.hocbf_shield import HOCBFShield, ShieldParams

def run(N=4, E=100, steps=2000, seed=0, adversary='seek', dt=0.1, L=10.0, R=1.5, d_sep=1.0, drag=0.5, verbose=True):
    rng = np.random.default_rng(seed)
    P = ShieldParams(drag=drag, sam_r_eff=R + 0.2, wall_L=L - 0.3, sep_eff=d_sep + 0.1)
    sh = HOCBFShield(P)
    S = 3
    sams = np.zeros((E, S, 2)); pos = np.zeros((E, N, 2))
    for e in range(E):
        c = []
        while len(c) < S:
            p = rng.uniform(-0.7 * L, 0.7 * L, 2)
            if all(np.linalg.norm(p - q) >= 2 * R + 3 for q in c): c.append(p)
        sams[e] = c
        q = []
        while len(q) < N:
            p = rng.uniform(-L + 1, L - 1, 2)
            if np.all(np.linalg.norm(sams[e] - p, axis=1) >= R + 1.2) and all(np.linalg.norm(p - r) >= d_sep + 0.6 for r in q): q.append(p)
        pos[e] = q
    vel = np.zeros((E, N, 2))
    min_sam = np.inf; min_pair = np.inf; min_wall = np.inf
    lv = np.zeros(3, int); t0 = time.time()
    for t in range(steps):
        if adversary == 'seek':      # worst case: full thrust at nearest SAM centre / nearest teammate alternately
            d = sams[:, None, :, :] - pos[:, :, None, :]
            j = np.linalg.norm(d, axis=-1).argmin(-1)
            tgt = np.take_along_axis(d, j[:, :, None, None], 2)[:, :, 0, :]
            a = tgt / (np.linalg.norm(tgt, axis=-1, keepdims=True) + 1e-9)
            if (t // 200) % 2 == 1 and N > 1:         # ram teammate
                dd = pos[:, (np.arange(N) + 1) % N] - pos
                a = dd / (np.linalg.norm(dd, axis=-1, keepdims=True) + 1e-9)
        else:                        # random bang-bang
            a = rng.choice([-1.0, 1.0], size=(E, N, 2))
        u, info = sh.filter(pos, vel, a, sams)
        for l in range(3): lv[l] += (info['level'] == l).sum()
        vel = vel + dt * (-drag * vel + u); pos = pos + dt * vel
        min_sam = min(min_sam, (np.linalg.norm(pos[:, :, None] - sams[:, None], axis=-1) - R).min())
        if N > 1:
            dij = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1) + np.eye(N)[None] * 1e9
            min_pair = min(min_pair, dij.min() - d_sep)
        min_wall = min(min_wall, (L - np.abs(pos).max(-1)).min())
    if verbose:
        print(f"[{adversary:6s} N={N}] steps={steps} envs={E} | min SAM clearance={min_sam:+.3f} | min pair slack={min_pair:+.3f} | "
              f"min wall clearance={min_wall:+.3f} | QP levels(0/1/2)={lv.tolist()} | {time.time()-t0:.1f}s")
    return min_sam, min_pair, min_wall, lv

if __name__ == '__main__':
    ok = True
    for adv in ['seek', 'random']:
        for N in [1, 4]:
            a, b, c, lv = run(N=N, adversary=adv)
            ok &= (a > 0) and (b > 0 or N == 1) and (c > 0)
    print("RESULT:", "PASS (zero violations)" if ok else "FAIL (violations found - inspect numbers above)")
