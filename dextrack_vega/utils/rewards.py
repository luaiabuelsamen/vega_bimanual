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
    # GRIP term: directly reward fingertips being close to the object surface
    # (small fingertip-to-object distance). Without this the policy is rewarded
    # only for pose-tracking; it has no incentive to maintain contact force,
    # which is what physically holds a squeeze-lift. Bounded exp kernel:
    # reward peaks when nearest fingertip is at ~contact distance from the obj.
    grip: float = 30.0
    w_grip: float = 1.0
    # LIFT BONUS: dense, linear-in-height reward for the object being above its
    # starting z. The exp(-50*err) obj_pos term is sharp — at err~0.18m (box on
    # table, ref at top of lift) it's ~exp(-9) ~ 0, so there's no gradient
    # toward lifting during early training. lift_bonus gives a continuous
    # never-zero signal so PPO sees that "up is better." Off by default (only
    # turn on for tasks where lifting is the point, e.g. bimanual squeeze-lift).
    w_lift_bonus: float = 0.0
    lift_clip: float = 0.20  # saturates so it can't dominate; ~ target lift
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
    obj_pos_actual: np.ndarray | None = None,  # for grip term (actual obj pos)
    obj_pos_start: np.ndarray | None = None,   # for lift_bonus baseline
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

    # GRIP: reward fingertips being close to the actual (not reference) object.
    # This directly rewards "having a hand on the object" — independent of where
    # the reference says the obj should be. For bimanual, this is what makes
    # PPO want to maintain a squeeze rather than discover an "open & give up"
    # local optimum.
    if fingertips is not None and obj_pos_actual is not None:
        grip_err = float(np.mean(np.linalg.norm(
            fingertips - obj_pos_actual[None, :], axis=-1)))
        comp["grip"] = w.w_grip * np.exp(-w.grip * grip_err)

    # dense lift bonus (linear, saturated): rewards object height above start
    if w.w_lift_bonus > 0 and obj_pos_start is not None and obj_pos_actual is not None:
        lift = float(np.clip(obj_pos_actual[2] - obj_pos_start[2], 0.0, w.lift_clip))
        comp["lift_bonus"] = w.w_lift_bonus * lift

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
