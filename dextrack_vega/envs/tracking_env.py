"""Single-arm Vega + f5d6 manipulation *tracking* environment (MuJoCo).

Replicates DexTrack's single-trajectory tracking setting:
  - Goal: drive the f5d6 hand + object to follow a kinematic ReferenceTrajectory.
  - Action space (DexTrack primary): cumulative residual position targets with a
    kinematic bias. The actuator target for each controlled joint is
        target_t = ref_qpos_t  +  cumulative_residual_t
    where the policy outputs a bounded per-step residual delta. This keeps the
    policy anchored to the reference while letting it apply the corrections that
    make the motion dynamically feasible.
  - Observation: proprioception (controlled-joint pos/vel) + object pose &
    velocity + a future window of the reference (the "tracking command").
  - Reward: DexTrack-style object-pose + hand-pose tracking (see utils.rewards).

Design is per-arm modular (`side="R"`), so bimanual = two coordinated instances
later. Plain numpy gym-style API (reset/step) — no gym dependency required, and
trivially wrappable for vectorised backends (MJX / mujoco_warp) once correct.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .. import assets
from .. import config as C
from ..utils import rewards as rw
from ..utils import trajectory as tj


class VegaTrackingEnv:
    def __init__(
        self,
        side: str = "R",
        sides: list[str] | None = None,
        ref: tj.ReferenceTrajectory | None = None,
        ref_horizon: int = 5,
        max_seconds: float | None = None,
        seed: int = 0,
        weights: rw.RewardWeights | None = None,
        obj_type: str = "box",
        obj_dims: tuple[float, ...] | None = None,
        obj_pos: tuple[float, float, float] = (0.6, -0.15, 0.80),
        obj_mass: float = 0.08,
        rebuild: bool = True,
    ):
        # `sides` (list) drives single- vs bi-manual; `side` (str) kept for the
        # common single-arm call. Bimanual = sides=["R","L"]: the controlled
        # joints, fingertips, action and obs simply concatenate over sides, and
        # both hands track the one shared object.
        self.sides = list(sides) if sides else [side]
        self.side = self.sides[0]
        self.ref_horizon = ref_horizon
        self.rng = np.random.default_rng(seed)
        self.weights = weights or rw.RewardWeights()

        # --- model / data ---
        # `rebuild=False` skips regenerating the shared scene XML — used by the
        # subprocess vector env, where the parent builds the (identical) scene
        # once and the workers just load it, so N processes don't race writing
        # the same file.
        if rebuild:
            assets.build_scene(sides=self.sides, obj_type=obj_type, obj_dims=obj_dims,
                               obj_pos=obj_pos, obj_mass=obj_mass)
        self.model = mujoco.MjModel.from_xml_path(str(C.GENERATED_SCENE))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * C.CONTROL_DECIMATION

        # --- index bookkeeping ---
        self.ctrl_joints = [j for s in self.sides for j in C.controlled_joints(s)]
        self.n_ctrl = len(self.ctrl_joints)
        self._jnt_qposadr = np.array(
            [self.model.jnt_qposadr[self._jid(j)] for j in self.ctrl_joints])
        self._jnt_dofadr = np.array(
            [self.model.jnt_dofadr[self._jid(j)] for j in self.ctrl_joints])
        self._act_id = {j: self._aid(f"act_{j}") for j in self.ctrl_joints}
        self._act_ctrl_idx = np.array(
            [self._act_id[j] for j in self.ctrl_joints])
        self._posture_act = {j: self._aid(f"act_{j}") for j in C.POSTURE_JOINTS}

        # joint ranges for residual scaling
        ranges = self.model.jnt_range[[self._jid(j) for j in self.ctrl_joints]]
        self._jnt_span = np.clip(ranges[:, 1] - ranges[:, 0], 1e-3, None)
        self._jnt_lo, self._jnt_hi = ranges[:, 0], ranges[:, 1]

        # object freejoint addresses
        self._obj_qadr = self.model.jnt_qposadr[self._jid("object_free")]
        self._obj_dofadr = self.model.jnt_dofadr[self._jid("object_free")]
        self._obj_bid = self._bid("object")

        # fingertip body ids (keypoint reward) — concatenated over all sides
        self._ft_bids = np.array(
            [self._bid(b) for s in self.sides for b in C.FINGERTIP_BODIES[s]])

        # --- reference trajectory ---
        self.ref = ref or self._default_ref()
        self.max_seconds = max_seconds if max_seconds is not None else self.ref.duration
        # precompute reference fingertip positions (FK of each ref frame) so the
        # fingertip tracking term has a target — without this the largest reward
        # weight is silently dead.
        self._ref_fingertips = self._precompute_ref_fingertips()

        # action / obs sizes
        self.action_dim = self.n_ctrl
        obs0 = self.reset()
        self.obs_dim = obs0.shape[0]

    # ---- id helpers ----
    def _jid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
    def _aid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    def _bid(self, n): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

    def _default_ref(self) -> tj.ReferenceTrajectory:
        home = np.zeros(self.n_ctrl)
        return tj.make_synthetic(self.ctrl_joints, home, n_frames=200, dt=0.02)

    def _precompute_ref_fingertips(self) -> np.ndarray:
        """FK the controlled-joint reference (+ home posture) at every frame to
        get reference fingertip world positions [n_frames, n_ft, 3]."""
        fts = np.zeros((self.ref.n_frames, len(self._ft_bids), 3))
        saveq = self.data.qpos.copy()
        for f in range(self.ref.n_frames):
            self.data.qpos[self._jnt_qposadr] = self.ref.hand_qpos[f]
            for j, v in C.HOME_POSTURE.items():
                self.data.qpos[self.model.jnt_qposadr[self._jid(j)]] = v
            mujoco.mj_forward(self.model, self.data)
            fts[f] = self.data.xpos[self._ft_bids]
        self.data.qpos[:] = saveq
        mujoco.mj_forward(self.model, self.data)
        return fts

    def _ref_fingertips_at(self, t: float) -> np.ndarray:
        """Linearly interpolate the precomputed reference fingertips at time t."""
        f = np.clip(t / self.ref.dt, 0.0, self.ref.n_frames - 1)
        i0 = int(np.floor(f)); i1 = min(i0 + 1, self.ref.n_frames - 1)
        a = f - i0
        return (1 - a) * self._ref_fingertips[i0] + a * self._ref_fingertips[i1]

    # ---- core API ----
    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0.0
        self._cum_residual = np.zeros(self.n_ctrl)
        self._reward_state = rw.RewardState()

        # arms/hands start at the reference's first frame; posture at home.
        ref0 = self.ref.sample(0.0)
        self.data.qpos[self._jnt_qposadr] = ref0["hand_qpos"]
        for j, val in C.HOME_POSTURE.items():
            self.data.qpos[self.model.jnt_qposadr[self._jid(j)]] = val
            self.data.ctrl[self._posture_act[j]] = val

        # object at its reference start pose.
        self.data.qpos[self._obj_qadr:self._obj_qadr + 3] = ref0["obj_pos"]
        self.data.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = ref0["obj_quat"]

        # initialise actuator targets to the reference so we don't snap.
        self.data.ctrl[self._act_ctrl_idx] = ref0["hand_qpos"]
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        # cumulative residual position target with kinematic bias. The residual
        # is CLAMPED to a fraction of each joint's span so it can only *correct*
        # the reference, never drift away from it: an unbounded cumulative
        # residual let the bimanual policy walk the arms to their joint limits
        # (hands ended ~1.8m apart) instead of tracking the squeeze. DexTrack
        # likewise bounds the residual; this keeps the policy in the feasible
        # neighbourhood of the kinematic reference.
        self._cum_residual = np.clip(
            self._cum_residual + action * C.RESIDUAL_SCALE * self._jnt_span,
            -C.RESIDUAL_CLIP * self._jnt_span, C.RESIDUAL_CLIP * self._jnt_span)
        ref = self.ref.sample(self.t)
        target = np.clip(ref["hand_qpos"] + self._cum_residual,
                         self._jnt_lo, self._jnt_hi)
        self.data.ctrl[self._act_ctrl_idx] = target

        for _ in range(C.CONTROL_DECIMATION):
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt

        obs = self._obs()
        reward, comp = self._reward(action)
        done = self.t >= self.max_seconds
        # early termination if the object is dropped far below the table
        if self.data.qpos[self._obj_qadr + 2] < 0.3:
            done = True
            comp["dropped"] = 1.0
        return obs, reward, done, comp

    # ---- observation ----
    def _obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._jnt_qposadr]
        qvel = self.data.qvel[self._jnt_dofadr]
        obj_pos = self.data.qpos[self._obj_qadr:self._obj_qadr + 3]
        obj_quat = self.data.qpos[self._obj_qadr + 3:self._obj_qadr + 7]
        obj_vel = self.data.qvel[self._obj_dofadr:self._obj_dofadr + 6]

        win = self.ref.window(self.t, self.ref_horizon)
        ref_flat = np.concatenate([
            win["obj_pos"].ravel(), win["obj_quat"].ravel(),
            win["hand_qpos"].ravel(),
        ])
        # error-to-immediate-reference (helps the policy)
        ref_now = self.ref.sample(self.t)
        err = np.concatenate([
            obj_pos - ref_now["obj_pos"],
            qpos - ref_now["hand_qpos"],
        ])
        return np.concatenate([
            qpos, qvel, obj_pos, obj_quat, obj_vel,
            self._cum_residual, ref_flat, err,
        ]).astype(np.float32)

    # ---- reward ----
    def _reward(self, action):
        ref = self.ref.sample(self.t)
        obj_pos = self.data.qpos[self._obj_qadr:self._obj_qadr + 3].copy()
        obj_quat = self.data.qpos[self._obj_qadr + 3:self._obj_qadr + 7].copy()
        qpos = self.data.qpos[self._jnt_qposadr].copy()
        fingertips = self.data.xpos[self._ft_bids].copy()
        torque = self.data.actuator_force[self._act_ctrl_idx].copy()
        return rw.compute_reward(
            obj_pos=obj_pos, obj_pos_ref=ref["obj_pos"],
            obj_quat=obj_quat, obj_quat_ref=ref["obj_quat"],
            hand_qpos=qpos, hand_qpos_ref=ref["hand_qpos"],
            fingertips=fingertips, fingertips_ref=self._ref_fingertips_at(self.t),
            action=action, torque=torque,
            state=self._reward_state, w=self.weights,
        )
