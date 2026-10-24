"""Exact grid Bayes filter over a static latent target location theta in [-L,L]^2 (numpy only)."""
import numpy as np
from scipy.special import logsumexp
from scipy.ndimage import gaussian_filter

NEG = -1e30


def make_grid(L, G):
    h = 2 * L / G
    xs = np.linspace(-L + h / 2, L - h / 2, G)
    gx, gy = np.meshgrid(xs, xs, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel()], -1)          # (G*G, 2), index = ix*G + iy


class GridBayes:
    """log-belief lb[e,i,:] per agent i.  fusion in {'none','lik','pool'}
    'lik' : each agent multiplies in the one-step likelihoods of agents within comm range (exact, no double counting)
    'pool': local Bayes update, then log-linear (geometric) pooling of neighbours' beliefs (conservative)
    'none': isolated Bayes filters
    """
    def __init__(self, cen, n_agents, fusion):
        self.cen, self.N, self.fusion = cen, n_agents, fusion
        self.M = cen.shape[0]

    def reset(self, logprior):                                # logprior (E,M)
        lp = logprior - logsumexp(logprior, axis=-1, keepdims=True)
        self.lb = np.repeat(lp[:, None, :], self.N, axis=1)

    def update(self, ll, adj):                                # ll (E,N,M), adj (E,N,N) float, diag=1
        lb = self.lb
        if self.fusion == 'lik':
            lb = lb + np.einsum('eij,ejm->eim', adj, ll)
            lb = np.maximum(lb, NEG)
            lb = lb - logsumexp(lb, axis=-1, keepdims=True)
        elif self.fusion == 'none':
            lb = lb + ll
            lb = np.maximum(lb, NEG)
            lb = lb - logsumexp(lb, axis=-1, keepdims=True)
        elif self.fusion == 'pool':
            lb = lb + ll
            lb = np.maximum(lb, NEG)
            lb = lb - logsumexp(lb, axis=-1, keepdims=True)
            W = adj / adj.sum(-1, keepdims=True)
            lb = np.einsum('eij,ejm->eim', W, lb)
            lb = np.maximum(lb, NEG)
            lb = lb - logsumexp(lb, axis=-1, keepdims=True)
        else:
            raise ValueError(self.fusion)
        self.lb = lb

    def predict(self, sigma_cells, G):
        """Chapman-Kolmogorov step for a Gaussian random-walk target (reflecting boundary at the grid edge)."""
        E = self.lb.shape[0]
        b = np.exp(self.lb).reshape(E, self.N, G, G)
        b = gaussian_filter(b, sigma=(0, 0, sigma_cells, sigma_cells), mode='reflect', truncate=4.0)
        b = b.reshape(E, self.N, self.M); b = b / b.sum(-1, keepdims=True)
        self.lb = np.maximum(np.log(np.maximum(b, 1e-300)), NEG)

    def stats(self):
        b = np.exp(self.lb)
        mean = b @ self.cen
        var = np.maximum(b @ (self.cen ** 2) - mean ** 2, 1e-9)
        ent = -np.where(b > 0, b * self.lb, 0.0).sum(-1)
        return b, mean, np.sqrt(var), ent
