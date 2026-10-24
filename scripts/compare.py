"""Paired comparison (episodes share layouts/seeds). Usage: python scripts/compare.py A.npz B.npz [more.npz ...]
Prints mean(A-B) with 95% bootstrap CI for every pair vs the first file."""
import sys, numpy as np
fs = sys.argv[1:]; D = [np.load(f) for f in fs]; rng = np.random.default_rng(0)
for j in range(1, len(fs)):
    print(f"\n{fs[0]}  minus  {fs[j]}   (paired, n={len(D[0]['ret'])})")
    for k in ['ret', 'first_det', 'map_err', 'entropy', 'shield_dev']:
        d = D[0][k] - D[j][k]; n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {k:11s} diff {d.mean():+9.3f}   95% CI [{lo:+8.3f}, {hi:+8.3f}]   {'SIGNIFICANT' if lo > 0 or hi < 0 else 'not significant'}")
