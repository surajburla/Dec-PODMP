"""Uniform policy loader. spec = 'random' | 'goto_belief' | 'voronoi' | 'ckpt:<path>'.  torch is imported lazily (scripted policies need only numpy)."""
import os
import numpy as np
from envs.search_env import EnvConfig
from envs.extras import SCRIPTED, voronoi_action


class CkptPolicy:
    def __init__(self, path, **over):
        import torch
        from algos.mappo import Actor
        self.torch = torch
        ck = torch.load(path, map_location='cpu', weights_only=False)
        cd = dict(ck['cfg']); cd.update({k: v for k, v in over.items() if v is not None})
        self.cfg = EnvConfig(**cd)
        self.residual = ck.get('residual', False); self.res_scale = ck.get('res_scale', 0.5)
        self.actor = Actor(ck['obs_dim']); self.actor.load_state_dict(ck['actor']); self.actor.eval()

    def act(self, env, obs, stochastic=False):
        E, N = env.E, env.N; base, o = None, obs
        if self.residual:
            base = voronoi_action(env); o = np.concatenate([obs, base.astype(np.float32)], -1)
        with self.torch.no_grad():
            d = self.actor.dist(self.torch.as_tensor(o.reshape(E * N, -1), dtype=self.torch.float32))
            a = (d.sample() if stochastic else d.mean).view(E, N, 2).numpy()
        return np.clip(base + self.res_scale * a, -1, 1) if self.residual else a


def get_policy(spec, **over):
    """returns (label, cfg, act_fn(env, obs) -> (E,N,2))"""
    if spec.startswith('ckpt:'):
        path = spec[5:]; pol = CkptPolicy(path, **over)
        label = os.path.basename(os.path.dirname(os.path.abspath(path)))
        return label, pol.cfg, (lambda env, obs: pol.act(env, obs))
    cfg = EnvConfig(**{k: v for k, v in over.items() if v is not None})
    if spec == 'random':
        rng = np.random.default_rng(over.get('seed', 0) + 777)
        return spec, cfg, (lambda env, obs: rng.uniform(-1, 1, (env.E, env.N, 2)))
    return spec, cfg, (lambda env, obs: SCRIPTED[spec](env))
