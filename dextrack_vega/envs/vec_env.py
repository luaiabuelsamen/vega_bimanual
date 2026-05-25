"""In-process synchronous vectorized env.

Runs N independent VegaTrackingEnv copies in a plain Python loop. This is the
simplest robust option on Jetson — no subprocess pickling of MuJoCo objects —
and is fine for validating PPO on the single-trajectory tracking task. Swap in
mujoco_warp / MJX later for large-scale throughput (step 3).

Auto-resets each sub-env on `done` (cleanrl convention): the returned `obs` is
the *next* episode's first obs, and the terminal obs is stashed in
`infos[i]["terminal_obs"]`.
"""
from __future__ import annotations

import numpy as np

from .tracking_env import VegaTrackingEnv


class SyncVectorEnv:
    def __init__(self, num_envs: int = 8, seed: int = 0, **env_kwargs):
        self.num_envs = num_envs
        self.envs = [VegaTrackingEnv(seed=seed + i, **env_kwargs)
                     for i in range(num_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs]).astype(np.float32)

    def step(self, actions: np.ndarray):
        n = self.num_envs
        obs = np.zeros((n, self.obs_dim), np.float32)
        rews = np.zeros(n, np.float32)
        dones = np.zeros(n, np.bool_)
        infos: list[dict] = [{} for _ in range(n)]
        for i, env in enumerate(self.envs):
            o, r, d, comp = env.step(actions[i])
            rews[i] = r
            dones[i] = d
            infos[i] = comp
            if d:
                infos[i] = dict(comp)
                infos[i]["terminal_obs"] = o
                infos[i]["episode_return_term"] = comp  # last-step terms
                o = env.reset()
            obs[i] = o
        return obs, rews, dones, infos

    def close(self):
        self.envs.clear()
