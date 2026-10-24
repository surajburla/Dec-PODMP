"""Feed-forward MAPPO (CTDE): shared actor on local features (belief summary + comm), centralized critic on true state.
Common team reward -> one value per env.  Fixed-horizon episodes, V(s_T)=0 (time feature in both nets => exact finite-horizon MDP).
The shield lives INSIDE the environment: we train on the raw Gaussian sample a_RL (correct policy gradient), not on u*."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std); nn.init.constant_(layer.bias, bias); return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hid=256):
        super().__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(obs_dim, hid)), nn.Tanh(),
                                 layer_init(nn.Linear(hid, hid)), nn.Tanh())
        self.mu = layer_init(nn.Linear(hid, act_dim), std=0.01)
        self.logstd = nn.Parameter(torch.full((act_dim,), -0.7))

    def dist(self, obs):
        mu = self.mu(self.net(obs))
        return Normal(mu, self.logstd.clamp(-2.0, 0.0).exp().expand_as(mu))


class Critic(nn.Module):
    def __init__(self, state_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(state_dim, hid)), nn.Tanh(),
                                 layer_init(nn.Linear(hid, hid)), nn.Tanh(), layer_init(nn.Linear(hid, 1), std=1.0))

    def forward(self, s): return self.net(s).squeeze(-1)


class RetStats:
    def __init__(self): self.mean, self.std, self.init = 0.0, 1.0, False
    def update(self, ret):
        m, s = float(ret.mean()), float(ret.std()) + 1e-6
        if not self.init: self.mean, self.std, self.init = m, max(s, 1.0), True
        else: self.mean, self.std = 0.9 * self.mean + 0.1 * m, max(0.9 * self.std + 0.1 * s, 1.0)


def gae(rew, val, gamma, lam):
    """rew,val: (T,E) torch. V_T = 0."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew); last = torch.zeros_like(rew[0])
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else torch.zeros_like(val[0])
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val


def ppo_update(actor, critic, opt, buf, stats, hp):
    T, E, N, od = buf['obs'].shape
    adv, ret = gae(buf['rew'], buf['val'], hp['gamma'], hp['lam'])
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    obs = buf['obs'].reshape(T * E, N, od); act = buf['act'].reshape(T * E, N, -1)
    lp0 = buf['logp'].reshape(T * E, N); st = buf['state'].reshape(T * E, -1)
    A = adv_n.reshape(T * E); R = ret.reshape(T * E)
    idx = np.arange(T * E); logs = []
    for _ in range(hp['epochs']):
        np.random.shuffle(idx)
        for mb in np.array_split(idx, hp['minibatches']):
            mb = torch.as_tensor(mb, device=obs.device)
            d = actor.dist(obs[mb].reshape(-1, od))
            lp = d.log_prob(act[mb].reshape(-1, act.shape[-1])).sum(-1).view(-1, N)
            ratio = (lp - lp0[mb]).exp()
            a = A[mb][:, None]
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - hp['clip'], 1 + hp['clip'])).mean()
            ent = d.entropy().sum(-1).mean()
            vl = 0.5 * ((critic(st[mb]) - (R[mb] - stats.mean) / stats.std) ** 2).mean()
            loss = pg - hp['ent_coef'] * ent + hp['vf_coef'] * vl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            opt.step()
            logs.append((pg.item(), vl.item(), ent.item(), ((ratio - 1).abs().mean()).item()))
    stats.update(ret.cpu().numpy())
    return np.mean(logs, 0)
