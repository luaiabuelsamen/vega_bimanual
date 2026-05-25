"""DexTrack-style tracking reward terms.

DexTrack's reward drives the simulated state toward the kinematic reference:
object 6-DoF pose tracking + hand-pose (joint / keypoint) tracking, minus
control-effort penalties. We use exponential kernels exp(-w * err) so each term
is bounded in (0, 1] and the weights set their relative pull — the same shape
used across DexTrack's configs.

`compute_reward` returns (total, components_dict) so the env can log every term.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import transforms as T


@dataclass
class RewardWeights:
    # exponential kernel sharpness (higher = stricter)
    obj_pos: float = 50.0
    obj_rot: float = 5.0
    hand_qpos: float = 5.0
    fingertip: float = 30.0
    # term weights in the weighted sum
    w_obj_pos: float = 1.0
    w_obj_rot: float = 0.5
    w_hand_qpos: float = 0.5
    w_fingertip: float = 1.0
    # penalties (subtracted)
    w_action_rate: float = 0.005
    # NOTE: w_torque defaults to 0. actuator_force is dominated by the
    # gravity-holding torque the arm *must* apply to stay up — penalising it
    # swamps the bounded (0-3) tracking reward (tens/step) and blows up the
    # value function, and the policy can't reduce it anyway. Re-enable only with
    # a gravity-compensated baseline (penalise torque ABOVE gravcomp).
    w_torque: float = 0.0
    # task-completion shaping
    success_pos_thresh: float = 0.03    # m
    success_rot_thresh: float = 0.30    # rad
    success_bonus: float = 1.0


@dataclass
class RewardState:
    """Carries the previous action for the action-rate penalty."""
    prev_action: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


def compute_reward(
    *,
    obj_pos: np.ndarray, obj_pos_ref: np.ndarray,
    obj_quat: np.ndarray, obj_quat_ref: np.ndarray,
    hand_qpos: np.ndarray, hand_qpos_ref: np.ndarray,
    fingertips: np.ndarray | None = None,
    fingertips_ref: np.ndarray | None = None,
    action: np.ndarray | None = None,
    torque: np.ndarray | None = None,
    state: RewardState | None = None,
    w: RewardWeights | None = None,
) -> tuple[float, dict]:
    w = w or RewardWeights()
    state = state or RewardState()
    comp: dict[str, float] = {}

    # --- tracking terms (exp kernels) ---
    pos_err = float(np.linalg.norm(obj_pos - obj_pos_ref))
    comp["obj_pos"] = w.w_obj_pos * np.exp(-w.obj_pos * pos_err)

    rot_err = float(T.quat_geodesic_angle(obj_quat, obj_quat_ref))
    comp["obj_rot"] = w.w_obj_rot * np.exp(-w.obj_rot * rot_err)

    qpos_err = float(np.mean((hand_qpos - hand_qpos_ref) ** 2))
    comp["hand_qpos"] = w.w_hand_qpos * np.exp(-w.hand_qpos * qpos_err)

    if fingertips is not None and fingertips_ref is not None:
        ft_err = float(np.mean(np.linalg.norm(fingertips - fingertips_ref, axis=-1)))
        comp["fingertip"] = w.w_fingertip * np.exp(-w.fingertip * ft_err)

    # --- penalties ---
    if action is not None and state.prev_action is not None:
        comp["action_rate"] = -w.w_action_rate * float(
            np.sum((action - state.prev_action) ** 2))
    if torque is not None:
        comp["torque"] = -w.w_torque * float(np.sum(torque ** 2))

    # --- success bonus ---
    if pos_err < w.success_pos_thresh and rot_err < w.success_rot_thresh:
        comp["success"] = w.success_bonus

    total = float(sum(comp.values()))
    comp["_obj_pos_err"] = pos_err      # raw errors for logging (leading _)
    comp["_obj_rot_err"] = rot_err
    comp["_hand_qpos_err"] = qpos_err
    if action is not None:
        state.prev_action = np.asarray(action).copy()
    return total, comp
