"""Gaussian MLP actor-critic for continuous-control PPO (cleanrl style).

Kept in the package (not the train script) so the eval/viz path can import the
same network definition to load a checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        # state-independent log std (standard for PPO continuous control)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return action, logprob, entropy, self.critic(x)

    @torch.no_grad()
    def act_deterministic(self, x):
        """Mean action, clipped to [-1, 1] — for eval/viz."""
        return torch.clamp(self.actor_mean(x), -1.0, 1.0)
