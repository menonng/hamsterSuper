"""Actor-critic network, observation normalization and deployment export."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(n_in, n_out, hidden=(256, 256), act=nn.ELU):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden=(256, 256), init_log_std: float = -0.5):
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, tuple(hidden)
        self.actor = mlp(obs_dim, act_dim, hidden)
        self.critic = mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class RunningNorm:
    def __init__(self, dim: int, clip: float = 10.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4
        self.clip = clip

    def update(self, x: np.ndarray):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = np.asarray(d["mean"]), np.asarray(d["var"]), d["count"]


class DeployPolicy(nn.Module):
    """obs (raw) -> wheel commands in [-1, 1]; normalization baked in."""

    def __init__(self, ac: ActorCritic, norm: RunningNorm):
        super().__init__()
        self.actor = ac.actor
        self.register_buffer("mean", torch.tensor(norm.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.sqrt(norm.var + 1e-8), dtype=torch.float32))
        self.clip = float(norm.clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.mean) / self.std, -self.clip, self.clip)
        return torch.clamp(self.actor(x), -1.0, 1.0)


def load_policy(path, device="cpu"):
    """Loads a TorchScript export, a training checkpoint, or a policy.npz.
    Returns a callable: torch tensor (B, obs) -> torch tensor (B, 2)."""
    if str(path).endswith(".npz"):
        npol = NumpyPolicy(path)
        return lambda obs: torch.as_tensor(npol(obs.numpy() if hasattr(obs, "numpy") else obs))
    try:
        return torch.jit.load(str(path), map_location=device).eval()
    except (RuntimeError, ValueError):
        ck = torch.load(str(path), map_location=device, weights_only=False)
        ac = ActorCritic(ck["obs_dim"], ck.get("act_dim", 2), ck.get("hidden", (256, 256)))
        ac.load_state_dict(ck["model"])
        norm = RunningNorm(ck["obs_dim"])
        norm.load_state_dict(ck["obs_norm"])
        return DeployPolicy(ac, norm).eval()


def export_numpy(ac: ActorCritic, norm: RunningNorm, path):
    arrs = {"mean": norm.mean.astype(np.float32), "std": np.sqrt(norm.var + 1e-8).astype(np.float32),
            "clip": np.float32(norm.clip)}
    lin = [m for m in ac.actor if isinstance(m, nn.Linear)]
    for i, m in enumerate(lin):
        arrs[f"W{i}"] = m.weight.detach().numpy()
        arrs[f"b{i}"] = m.bias.detach().numpy()
    np.savez(path, n_layers=len(lin), **arrs)


class NumpyPolicy:
    """Torch-free inference of a policy exported with export_numpy (ELU MLP)."""

    def __init__(self, path):
        d = np.load(path)
        self.mean, self.std, self.clip = d["mean"], d["std"], float(d["clip"])
        self.layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -self.clip, self.clip)
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return np.clip(x, -1.0, 1.0)
