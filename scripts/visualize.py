"""
Annotated mission replay of ONE episode (full length, 300 steps).
  python scripts/visualize.py --policy voronoi --seed 3 --out results/viz_voronoi
  python scripts/visualize.py --policy ckpt:results/mv_belief_lik/ckpt.pt --target_sigma 0.4 --out results/viz_mv
Outputs: <out>.gif (full animation), <out>_mission.png (one-picture overview), <out>_snap{0..3}.png (key moments), <out>.mp4 if ffmpeg exists.
"""
import sys, os, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from envs.extras import RobustEnv
from algos.policy_io import get_policy

ap = argparse.ArgumentParser()
ap.add_argument('--policy', default='voronoi'); ap.add_argument('--seed', type=int, default=3)
ap.add_argument('--out', default='results/viz'); ap.add_argument('--every', type=int, default=1)
ap.add_argument('--fps', type=int, default=12); ap.add_argument('--comm_drop', type=float, default=0.0)
ap.add_argument('--target_sigma', type=float, default=None)
a = ap.parse_args()
spec = a.policy if (a.policy.startswith('ckpt:') or a.policy in ('random', 'goto_belief', 'voronoi')) else 'ckpt:' + a.policy
label, cfg, act_fn = get_policy(spec, n_envs=1, seed=a.seed, target_sigma=a.target_sigma)
env = RobustEnv(cfg, comm_drop=a.comm_drop); obs = env.reset()
L, G, N, T = cfg.L, cfg.grid, env.N, env.T
COL = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:brown']

fr, done = [], False
while not done:
    act = act_fn(env, obs)
    a_nom = np.clip(act, -1, 1) * cfg.a_max
    u, _ = env.shield.filter(env.pos, env.vel, a_nom, env.sams)
    b, mean, std, ent = env.belief_stats()
    fr.append(dict(t=env.t, pos=env.pos[0].copy(), th=env.theta[0].copy(), D=env.D[0].copy(), z=env.z[0].copy(),
                   b=b[0].reshape(N, G, G).copy(), ent=ent[0].copy(), adj=env._adj()[0].copy(), a=a_nom[0].copy(), u=u[0].copy(),
                   mu=mean[0].copy()))
    obs, r, done, info = env.step(act); fr[-1]['r'] = float(r[0])
sams = env.sams[0]; pos0 = fr[0]['pos']; n = len(fr)
P = np.array([f['pos'] for f in fr]); TH = np.array([f['th'] for f in fr]); ENT = np.array([f['ent'] for f in fr])
R = np.cumsum([f['r'] for f in fr]); DEV = np.array([np.linalg.norm(f['u'] - f['a'], axis=-1) for f in fr])
DMIN = np.array([np.linalg.norm(f['pos'] - f['th'], axis=-1).min() for f in fr])
ANYD = np.cumsum([f['D'].any() for f in fr]) > 0
first_det = int(np.argmax(ANYD)) if ANYD.any() else None
NDET = np.cumsum([f['D'].sum() for f in fr]); NSH = np.cumsum((DEV > 0.35).sum(1))

def phase(k):
    if not ANYD[k]: return "PHASE 1: SEARCH — nobody has seen the target yet", 'tab:gray'
    if DMIN[k] <= 2.5 and np.median(ENT[k]) < 1.5: return "PHASE 3: ON STATION — a UAV is near the target and the team knows where it is", 'tab:green'
    return "PHASE 2: LOCALIZE — target detected, beliefs converging, UAVs closing in", 'tab:orange'

def world(w, k, big=True):
    f = fr[k]
    w.set_xlim(-L, L); w.set_ylim(-L, L); w.set_aspect('equal'); w.set_xticks([]); w.set_yticks([])
    w.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fill=False, lw=2, ec='k'))
    for s in sams:
        w.add_patch(Circle(s, cfg.sam_radius, color='red', alpha=0.35)); w.add_patch(Circle(s, cfg.sam_radius + cfg.sam_margin, fill=False, ls=':', ec='red'))
        w.text(s[0], s[1], 'SAM\nno-fly', ha='center', va='center', fontsize=6, color='darkred')
    if cfg.target_sigma > 0: w.plot(TH[:k + 1, 0], TH[:k + 1, 1], color='goldenrod', lw=1.2, ls='--', alpha=0.8)
    w.plot(*f['th'], marker='*', ms=20, color='gold', mec='k', ls='', zorder=5)
    w.text(f['th'][0], f['th'][1] - 1.0, 'TARGET (hidden from UAVs)', ha='center', fontsize=6, color='goldenrod', weight='bold')
    for i in range(N):
        p = f['pos'][i]
        w.plot(P[:k + 1, i, 0], P[:k + 1, i, 1], color=COL[i], lw=1.3, alpha=0.6)
        w.plot(*pos0[i], marker='s', ms=9, mfc='none', mec=COL[i], mew=1.5); w.text(pos0[i][0] + 0.3, pos0[i][1] + 0.3, f'S{i}', fontsize=7, color=COL[i])
        w.add_patch(Circle(p, cfg.rho_det, fc=COL[i] if f['D'][i] else 'none', ec=COL[i], alpha=0.30 if f['D'][i] else 0.25, lw=1))
        w.plot(*p, 'o', color=COL[i], ms=9, mec='k', zorder=6); w.text(p[0] + 0.35, p[1] - 0.6, f'U{i}', fontsize=8, color=COL[i], weight='bold')
        w.annotate('', xy=p + 1.6 * f['a'][i], xytext=p, arrowprops=dict(arrowstyle='->', color=COL[i], ls='--', lw=1))
        w.annotate('', xy=p + 1.6 * f['u'][i], xytext=p, arrowprops=dict(arrowstyle='->', color='k', lw=1.8))
        if DEV[k, i] > 0.35: w.add_patch(Circle(p, 0.7, fill=False, ec='red', lw=2.2, zorder=7))
        if f['D'][i]: w.plot(*f['z'][i], 'x', color=COL[i], ms=9, mew=2)
        for j in range(i + 1, N):
            if f['adj'][i, j] > 0: w.plot([p[0], f['pos'][j][0]], [p[1], f['pos'][j][1]], color='gray', lw=1, ls='--', alpha=0.6)

def draw(fig, k, hold_note=None):
    f = fr[k]; fig.clf()
    gs = fig.add_gridspec(4, 4, height_ratios=[0.24, 1, 1, 0.55], hspace=0.3, wspace=0.1)
    bn = fig.add_subplot(gs[0, :]); bn.axis('off')
    ph, pc = phase(k)
    bn.text(0.0, 1.0, "MISSION: 4 UAVs spawn at random safe points (squares S0-S3) and search a 20x20 arena for a hidden target. They share information only with UAVs in radio range\n"
            "and must never enter SAM zones or collide. No weapons are modelled: team reward = how close the nearest UAV is to the target (recon / tracking).",
            fontsize=7.5, va='top')
    bn.text(0.0, 0.0, ph, fontsize=11, color=pc, weight='bold', va='bottom')
    w = fig.add_subplot(gs[1:3, 0:2]); world(w, k)
    w.set_title(f"GOD VIEW (true state s_t, invisible to the UAVs) — step {f['t']}/{T}", fontsize=9)
    w.text(0.01, 0.01, "dashed arrow = policy action a_RL | black arrow = shielded action u* | red ring = shield intervening\n"
           "x = detection z | gray dashed = radio link | circle = sensor range (filled = detecting) | squares = spawn", transform=w.transAxes, fontsize=5.8,
           bbox=dict(fc='white', alpha=0.7, lw=0))
    bx = [fig.add_subplot(gs[1 + i // 2, 2 + i % 2]) for i in range(N)]
    for i, ax in enumerate(bx):
        ax.imshow(f['b'][i].T, origin='lower', extent=[-L, L, -L, L], cmap='viridis', vmin=0, vmax=max(f['b'][i].max(), 1e-3))
        for s in sams: ax.add_patch(Circle(s, cfg.sam_radius, fill=False, ec='red', lw=0.8))
        ax.plot(*f['pos'][i], 'o', color=COL[i], ms=7, mec='w'); ax.plot(*f['th'], marker='*', ms=10, color='gold', mec='k')
        nb = [j for j in range(N) if j != i and f['adj'][i, j] > 0]
        ax.set_title(f"U{i}'s OWN belief  H={f['ent'][i]:.2f}  hears {nb if nb else 'nobody'}", fontsize=7, color=COL[i])
        ax.set_xticks([]); ax.set_yticks([])
    ts = fig.add_subplot(gs[3, :])
    for i in range(N): ts.plot(ENT[:k + 1, i], color=COL[i], lw=1.2, label=f'U{i}')
    ts.set_xlim(0, T); ts.set_ylim(0, ENT.max() * 1.05); ts.axvline(k, color='k', lw=0.8)
    if first_det is not None and first_det <= k: ts.axvline(first_det, color='green', ls='--', lw=1); ts.text(first_det + 2, ENT.max() * 0.9, 'first detection', fontsize=7, color='green')
    ts.set_ylabel('belief entropy (nats)\nlow = knows where target is', fontsize=7); ts.set_xlabel('step', fontsize=7)
    t2 = ts.twinx(); t2.plot(R[:k + 1], 'k', lw=1.8); t2.set_ylim(0, max(R.max() * 1.05, 1)); t2.set_ylabel('team return (common R)', fontsize=7)
    ts.tick_params(labelsize=6); t2.tick_params(labelsize=6); ts.legend(loc='upper right', fontsize=6, ncol=4)
    ts.text(0.35, 0.78, f"nearest UAV to target: {DMIN[k]:.1f} | detections so far: {int(NDET[k])} | shield interventions: {int(NSH[k])}",
            transform=ts.transAxes, fontsize=7, bbox=dict(fc='white', alpha=0.8, lw=0))
    if hold_note: w.text(0.5, 0.5, hold_note, transform=w.transAxes, ha='center', va='center', fontsize=12, weight='bold',
                         bbox=dict(fc='white', ec='k', alpha=0.9))

os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
# mission overview picture
figm, wm = plt.subplots(figsize=(8.5, 8.5), dpi=100)
world(wm, n - 1)
for i in range(N):
    wm.plot(*P[-1, i], marker='P', ms=12, color=COL[i], mec='k'); wm.text(P[-1, i, 0] + 0.3, P[-1, i, 1] + 0.3, f'F{i}', fontsize=8, color=COL[i])
wm.set_title(f"Episode overview ({label}): S=spawn, F=final position, lines=flight paths, star=target"
             f"{' (dashed=target path)' if cfg.target_sigma > 0 else ''}\nfinal return {R[-1]:.1f}, first detection step {first_det}, SAM/pair violations: {int(info['violation'][0])}", fontsize=9)
figm.savefig(a.out + '_mission.png'); plt.close(figm)
fig = plt.figure(figsize=(14, 8.6), dpi=62)
idx = list(range(0, n, a.every))
frames = [(0, "t=0: UAVs have just spawned.\nThey do NOT know where the target is.")] * 18 + [(k, None) for k in idx] + [(n - 1, "END OF EPISODE")] * 24
anim = FuncAnimation(fig, lambda q: draw(fig, frames[q][0], frames[q][1]), frames=len(frames))
anim.save(a.out + '.gif', writer=PillowWriter(fps=a.fps))
if shutil.which('ffmpeg'):
    try: anim.save(a.out + '.mp4', writer=FFMpegWriter(fps=a.fps, bitrate=2500))
    except Exception as e: print('mp4 skipped:', e)
snaps = [0, first_det if first_det is not None else n // 4, n // 2, n - 1]
fig2 = plt.figure(figsize=(14, 8.6), dpi=100)
for q, k in enumerate(snaps): draw(fig2, k); fig2.savefig(f"{a.out}_snap{q}.png")
print(f"saved {a.out}.gif ({len(frames)} frames @ {a.fps} fps = {len(frames)/a.fps:.0f}s), _mission.png, 4 snapshots | final return {R[-1]:.1f} | first detection step {first_det}")
