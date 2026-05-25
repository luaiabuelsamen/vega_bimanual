"""Reference-trajectory format + loader + synthetic generator.

A *tracking command* (DexTrack terminology) is a time series of:
  - object 6-DoF pose:  obj_pos[T,3], obj_quat[T,4]  (wxyz)
  - hand joint targets: hand_qpos[T, n_ctrl]          (arm + finger joints)

This is the interface the env consumes and the retargeting pipeline produces.
Real data (GRAB/TACO retargeted to the f5d6 hand) plugs in via `load_npz`; a
synthetic generator is provided so the env is runnable/testable before the
retargeting workstream lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import transforms as T


@dataclass
class ReferenceTrajectory:
    """One kinematic reference to track. Times are in seconds from t=0."""
    obj_pos: np.ndarray      # [T, 3]
    obj_quat: np.ndarray     # [T, 4] wxyz
    hand_qpos: np.ndarray    # [T, n_ctrl]
    dt: float                # seconds between frames
    joint_names: list[str]   # length n_ctrl, matches hand_qpos columns

    def __post_init__(self):
        self.obj_pos = np.asarray(self.obj_pos, np.float64)
        self.obj_quat = T.quat_normalize(np.asarray(self.obj_quat, np.float64))
        self.hand_qpos = np.asarray(self.hand_qpos, np.float64)
        assert self.obj_pos.shape[0] == self.hand_qpos.shape[0], "frame mismatch"
        assert self.hand_qpos.shape[1] == len(self.joint_names)

    @property
    def n_frames(self) -> int:
        return self.obj_pos.shape[0]

    @property
    def duration(self) -> float:
        return (self.n_frames - 1) * self.dt

    def sample(self, t: float) -> dict:
        """Linear/slerp-interpolated reference at continuous time `t` (s)."""
        f = np.clip(t / self.dt, 0.0, self.n_frames - 1)
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, self.n_frames - 1)
        a = f - i0
        return {
            "obj_pos": (1 - a) * self.obj_pos[i0] + a * self.obj_pos[i1],
            "obj_quat": T.slerp(self.obj_quat[i0], self.obj_quat[i1], a),
            "hand_qpos": (1 - a) * self.hand_qpos[i0] + a * self.hand_qpos[i1],
        }

    def window(self, t: float, horizon: int) -> dict:
        """Stack `horizon` future references (one per dt) for the obs window."""
        samples = [self.sample(t + k * self.dt) for k in range(horizon)]
        return {
            "obj_pos": np.stack([s["obj_pos"] for s in samples]),
            "obj_quat": np.stack([s["obj_quat"] for s in samples]),
            "hand_qpos": np.stack([s["hand_qpos"] for s in samples]),
        }


def load_npz(path: str | Path, joint_names: list[str]) -> ReferenceTrajectory:
    """Load a reference saved with keys obj_pos, obj_quat, hand_qpos, dt."""
    d = np.load(path)
    return ReferenceTrajectory(
        obj_pos=d["obj_pos"], obj_quat=d["obj_quat"],
        hand_qpos=d["hand_qpos"], dt=float(d["dt"]),
        joint_names=list(d["joint_names"]) if "joint_names" in d else joint_names,
    )


def save_npz(path: str | Path, traj: ReferenceTrajectory) -> None:
    np.savez(path, obj_pos=traj.obj_pos, obj_quat=traj.obj_quat,
             hand_qpos=traj.hand_qpos, dt=traj.dt,
             joint_names=np.array(traj.joint_names))


def make_synthetic(
    joint_names: list[str],
    home_qpos: np.ndarray,
    n_frames: int = 200,
    dt: float = 0.02,
    start_obj: tuple[float, float, float] = (0.4, -0.2, 0.80),
    seed: int = 0,
) -> ReferenceTrajectory:
    """A smooth lift-and-translate object trajectory with a gentle finger
    flexion profile. Placeholder until retargeted GRAB/TACO data exists, but
    shaped like a real pick: object rises ~12 cm and slides while fingers close.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_frames)

    obj_pos = np.zeros((n_frames, 3))
    obj_pos[:, 0] = start_obj[0] + 0.10 * t
    obj_pos[:, 1] = start_obj[1] + 0.15 * np.sin(np.pi * t)
    obj_pos[:, 2] = start_obj[2] + 0.12 * np.sin(np.pi * np.clip(t, 0, 1))

    # object yaw ramps to ~30 deg about z
    yaw = 0.5 * t
    obj_quat = np.stack([np.cos(yaw / 2), np.zeros_like(yaw),
                         np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)

    # fingers (any joint that isn't an arm joint) flex from home toward closed
    hand_qpos = np.tile(home_qpos, (n_frames, 1)).astype(np.float64)
    flex = 0.6 * (1 - np.cos(np.pi * t)) / 2  # 0 -> 0.6
    for c, name in enumerate(joint_names):
        if "_arm_" not in name:  # finger/thumb joint
            hand_qpos[:, c] = home_qpos[c] + flex
    # add tiny noise so it isn't perfectly degenerate
    hand_qpos += 0.01 * rng.standard_normal(hand_qpos.shape)

    return ReferenceTrajectory(obj_pos, obj_quat, hand_qpos, dt, list(joint_names))
