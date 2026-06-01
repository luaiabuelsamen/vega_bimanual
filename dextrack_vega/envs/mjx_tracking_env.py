"""MJX port of VegaTrackingEnv (brax Env) for GPU-parallel PPO.

Same action space / reward / observation layout as tracking_env.py, written
functionally over MJX. Static indices are resolved from a CPU MjModel (MJX has
no name lookup); the reference is held as jnp arrays. Requires the
collision="primitive" scene build (mesh-free) for tractable compile times.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from brax.envs.base import Env, State

from .. import assets
from .. import config as C


def _quat_geodesic_angle(q1, q2):
    """Angle (rad) between two unit quaternions (wxyz)."""
    d = jnp.abs(jnp.sum(q1 * q2, axis=-1))
    d = jnp.clip(d, 0.0, 1.0)
    return 2.0 * jnp.arccos(d)


def _nlerp(q0, q1, a):
    """Normalized lerp between quaternions with sign alignment."""
    q1 = jnp.where(jnp.sum(q0 * q1) < 0.0, -q1, q1)
    q = (1.0 - a) * q0 + a * q1
    return q / (jnp.linalg.norm(q) + 1e-9)


class MjxTrackingEnv(Env):
    def __init__(
        self,
        sides: list[str],
        ref: dict,            # arrays: obj_pos[T,3], obj_quat[T,4], hand_qpos[T,n], hand_ctrl[T,n] or None, dt
        obj_type: str = "box",
        obj_dims: tuple[float, ...] = (0.04, 0.045, 0.07),
        obj_pos: tuple[float, float, float] = (0.55, 0.0, 0.80),
        obj_mass: float = 0.05,
        obj_friction: str = "2.0 0.05 0.002",
        ref_horizon: int = 5,
        episode_length: int | None = None,
        # reward weights (mirror rewards.RewardWeights defaults)
        k_obj_pos: float = 50.0, k_obj_rot: float = 5.0, k_hand: float = 5.0,
        k_ftip: float = 30.0, k_grip: float = 30.0,
        w_obj_pos: float = 1.0, w_obj_rot: float = 0.5, w_hand: float = 0.5,
        w_ftip: float = 1.0, w_grip: float = 1.0,
        w_lift_bonus: float = 0.0, lift_clip: float = 0.20,
        w_action_rate: float = 0.005,
    ):
        self.sides = list(sides)
        self.ref_horizon = ref_horizon

        # --- build the MJX-compatible (primitive) scene + CPU model for indices ---
        assets.build_scene(sides=self.sides, obj_type=obj_type, obj_dims=obj_dims,
                           obj_pos=obj_pos, obj_mass=obj_mass,
                           obj_friction=obj_friction, collision="primitive")
        m = mujoco.MjModel.from_xml_path(str(C.GENERATED_SCENE))
        self._mjx_model = mjx.put_model(m)
        self.sim_dt = float(m.opt.timestep)
        self.decimation = C.CONTROL_DECIMATION
        self.ctrl_dt = self.sim_dt * self.decimation

        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

        self.ctrl_joints = [j for s in self.sides for j in C.controlled_joints(s)]
        self.n_ctrl = len(self.ctrl_joints)
        self._jnt_qadr = np.array([m.jnt_qposadr[jid(j)] for j in self.ctrl_joints])
        self._jnt_dadr = np.array([m.jnt_dofadr[jid(j)] for j in self.ctrl_joints])
        self._act_idx = np.array([aid(f"act_{j}") for j in self.ctrl_joints])
        ranges = m.jnt_range[[jid(j) for j in self.ctrl_joints]]
        self._jnt_lo = jnp.array(ranges[:, 0])
        self._jnt_hi = jnp.array(ranges[:, 1])
        self._jnt_span = jnp.array(np.clip(ranges[:, 1] - ranges[:, 0], 1e-3, None))

        self._obj_qadr = int(m.jnt_qposadr[jid("object_free")])
        self._obj_dadr = int(m.jnt_dofadr[jid("object_free")])
        self._ft_bids = np.array(
            [bid(b) for s in self.sides for b in C.FINGERTIP_BODIES[s]])

        # posture joints held at home
        self._posture = [(int(m.jnt_qposadr[jid(j)]), aid(f"act_{j}"), float(v))
                         for j, v in C.HOME_POSTURE.items() if jid(j) >= 0]

        # qpos0 baseline (free-joint quats default to identity here)
        self._qpos0 = jnp.array(m.qpos0)

        # --- reference as jnp arrays ---
        self.ref_dt = float(ref["dt"])
        self._r_obj_pos = jnp.array(ref["obj_pos"])
        self._r_obj_quat = jnp.array(ref["obj_quat"])
        self._r_hand_q = jnp.array(ref["hand_qpos"])
        has_ctrl = ref.get("hand_ctrl", None) is not None
        self._r_hand_ctrl = jnp.array(ref["hand_ctrl"]) if has_ctrl else self._r_hand_q
        self.n_frames = int(self._r_obj_pos.shape[0])
        self._obj_pos_start = jnp.array(ref["obj_pos"][0])

        # precompute reference fingertip world positions (FK each ref frame on CPU)
        self._r_ftips = jnp.array(self._precompute_ref_fingertips(m, ref))

        dur_frames = self.n_frames - 1
        self._episode_length = episode_length or int(
            np.ceil(dur_frames * self.ref_dt / self.ctrl_dt))

        # weights
        self.kw = dict(k_obj_pos=k_obj_pos, k_obj_rot=k_obj_rot, k_hand=k_hand,
                       k_ftip=k_ftip, k_grip=k_grip, w_obj_pos=w_obj_pos,
                       w_obj_rot=w_obj_rot, w_hand=w_hand, w_ftip=w_ftip,
                       w_grip=w_grip, w_lift_bonus=w_lift_bonus,
                       lift_clip=lift_clip, w_action_rate=w_action_rate)

        # obs/action sizes (derive obs_dim by a dry reset on host)
        self._action_size = self.n_ctrl
        self._obs_size = int(self._dummy_obs_dim())

    # ---- reference sampling (continuous time, interpolated) ----
    def _ref_at(self, t):
        f = jnp.clip(t / self.ref_dt, 0.0, self.n_frames - 1)
        i0 = jnp.floor(f).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, self.n_frames - 1)
        a = f - i0
        op = (1 - a) * self._r_obj_pos[i0] + a * self._r_obj_pos[i1]
        oq = _nlerp(self._r_obj_quat[i0], self._r_obj_quat[i1], a)
        hq = (1 - a) * self._r_hand_q[i0] + a * self._r_hand_q[i1]
        hc = (1 - a) * self._r_hand_ctrl[i0] + a * self._r_hand_ctrl[i1]
        ft = (1 - a) * self._r_ftips[i0] + a * self._r_ftips[i1]
        return op, oq, hq, hc, ft

    def _precompute_ref_fingertips(self, m, ref):
        d = mujoco.MjData(m)
        n = ref["obj_pos"].shape[0]
        out = np.zeros((n, len(self._ft_bids), 3))
        for f in range(n):
            d.qpos[self._jnt_qadr] = ref["hand_qpos"][f]
            for qadr, _, v in self._posture:
                d.qpos[qadr] = v
            mujoco.mj_forward(m, d)
            out[f] = d.xpos[self._ft_bids]
        return out

    # ---- brax Env API ----
    @property
    def observation_size(self): return self._obs_size
    @property
    def action_size(self): return self._action_size
    @property
    def backend(self): return "mjx"
    @property
    def dt(self): return self.ctrl_dt

    def reset(self, rng) -> State:
        op, oq, hq, hc, ft0 = self._ref_at(0.0)
        qpos = self._qpos0
        qpos = qpos.at[self._jnt_qadr].set(hq)
        for qadr, _, v in self._posture:
            qpos = qpos.at[qadr].set(v)
        qpos = qpos.at[self._obj_qadr:self._obj_qadr + 3].set(op)
        qpos = qpos.at[self._obj_qadr + 3:self._obj_qadr + 7].set(oq)

        data = mjx.make_data(self._mjx_model)
        data = data.replace(qpos=qpos)
        ctrl = jnp.zeros_like(data.ctrl)
        ctrl = ctrl.at[self._act_idx].set(hc)
        for _, act, v in self._posture:
            ctrl = ctrl.at[act].set(v)
        data = data.replace(ctrl=ctrl)
        data = mjx.forward(self._mjx_model, data)

        # Tie zero-valued info/reward/done to rng so brax's vmap(reset) gives
        # every State leaf a batch axis (constants stay unbatched -> vmap(step)
        # would then mismatch). Metrics derive from `data` (already batched).
        z = jnp.sum(rng).astype(jnp.float32) * 0.0
        info = {"step": z.astype(jnp.int32),
                "cum_residual": jnp.zeros(self.n_ctrl) + z,
                "prev_action": jnp.zeros(self.n_ctrl) + z}
        obs = self._obs(data, 0.0, info["cum_residual"])

        obj_pos = data.qpos[self._obj_qadr:self._obj_qadr + 3]
        obj_quat = data.qpos[self._obj_qadr + 3:self._obj_qadr + 7]
        qp = data.qpos[self._jnt_qadr]
        metrics = {"obj_pos_err": jnp.linalg.norm(obj_pos - op),
                   "obj_rot_err": _quat_geodesic_angle(obj_quat, oq),
                   "hand_qpos_err": jnp.mean((qp - hq) ** 2),
                   "lift": jnp.clip(obj_pos[2] - self._obj_pos_start[2],
                                    0.0, self.kw["lift_clip"])}
        return State(data, obs, z, z, metrics, info)

    def step(self, state: State, action) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        cum = jnp.clip(
            state.info["cum_residual"] + action * C.RESIDUAL_SCALE * self._jnt_span,
            -C.RESIDUAL_CLIP * self._jnt_span, C.RESIDUAL_CLIP * self._jnt_span)
        t = state.info["step"].astype(jnp.float32) * self.ctrl_dt
        op, oq, hq, hc, ftref = self._ref_at(t)
        target = jnp.clip(hc + cum, self._jnt_lo, self._jnt_hi)

        data = state.pipeline_state
        ctrl = data.ctrl.at[self._act_idx].set(target)
        data = data.replace(ctrl=ctrl)
        # control decimation: `decimation` physics substeps per control step
        def sub(_, d):
            return mjx.step(self._mjx_model, d)
        data = jax.lax.fori_loop(0, self.decimation, sub, data)

        cum_t = (state.info["step"] + 1).astype(jnp.float32) * self.ctrl_dt
        obs = self._obs(data, cum_t, cum)
        reward, m = self._reward(data, op, oq, hq, ftref, action,
                                 state.info["prev_action"])

        # done = true termination only (object dropped). The episode time limit
        # is left to brax's EpisodeWrapper, which marks it as a truncation so the
        # value function keeps bootstrapping past it.
        obj_z = data.qpos[self._obj_qadr + 2]
        done = (obj_z < 0.3)

        # Merge (don't replace) info/metrics — brax's wrappers add their own
        # carry keys and the rollout scan requires the carry structure preserved.
        info = {**state.info, "step": state.info["step"] + 1,
                "cum_residual": cum, "prev_action": action}
        metrics = {**state.metrics, **m}
        return state.replace(pipeline_state=data, obs=obs,
                             reward=reward.astype(jnp.float32),
                             done=done.astype(jnp.float32),
                             metrics=metrics, info=info)

    # ---- observation (mirror tracking_env._obs layout) ----
    def _obs(self, data, t, cum_residual):
        qpos = data.qpos[self._jnt_qadr]
        qvel = data.qvel[self._jnt_dadr]
        obj_pos = data.qpos[self._obj_qadr:self._obj_qadr + 3]
        obj_quat = data.qpos[self._obj_qadr + 3:self._obj_qadr + 7]
        obj_vel = data.qvel[self._obj_dadr:self._obj_dadr + 6]

        # future reference window
        ops, oqs, hqs = [], [], []
        for k in range(self.ref_horizon):
            op, oq, hq, _, _ = self._ref_at(t + k * self.ref_dt)
            ops.append(op); oqs.append(oq); hqs.append(hq)
        ref_flat = jnp.concatenate(
            [jnp.concatenate(ops), jnp.concatenate(oqs), jnp.concatenate(hqs)])

        op0, _, hq0, _, _ = self._ref_at(t)
        err = jnp.concatenate([obj_pos - op0, qpos - hq0])
        return jnp.concatenate([qpos, qvel, obj_pos, obj_quat, obj_vel,
                                cum_residual, ref_flat, err]).astype(jnp.float32)

    def _dummy_obs_dim(self):
        # obs = qpos(n) + qvel(n) + objpos(3)+objquat(4)+objvel(6)
        #     + cum(n) + window(horizon*(3+4+n)) + err(3+n)
        n = self.n_ctrl
        return (n + n + 3 + 4 + 6 + n
                + self.ref_horizon * (3 + 4 + n) + (3 + n))

    # ---- reward (mirror rewards.compute_reward) ----
    def _reward(self, data, op_ref, oq_ref, hq_ref, ftref, action, prev_action):
        kw = self.kw
        obj_pos = data.qpos[self._obj_qadr:self._obj_qadr + 3]
        obj_quat = data.qpos[self._obj_qadr + 3:self._obj_qadr + 7]
        qpos = data.qpos[self._jnt_qadr]
        ftips = data.xpos[self._ft_bids]

        pos_err = jnp.linalg.norm(obj_pos - op_ref)
        rot_err = _quat_geodesic_angle(obj_quat, oq_ref)
        qpos_err = jnp.mean((qpos - hq_ref) ** 2)
        ft_err = jnp.mean(jnp.linalg.norm(ftips - ftref, axis=-1))
        grip_err = jnp.mean(jnp.linalg.norm(ftips - obj_pos[None, :], axis=-1))

        r = (kw["w_obj_pos"] * jnp.exp(-kw["k_obj_pos"] * pos_err)
             + kw["w_obj_rot"] * jnp.exp(-kw["k_obj_rot"] * rot_err)
             + kw["w_hand"] * jnp.exp(-kw["k_hand"] * qpos_err)
             + kw["w_ftip"] * jnp.exp(-kw["k_ftip"] * ft_err)
             + kw["w_grip"] * jnp.exp(-kw["k_grip"] * grip_err))
        lift = jnp.clip(obj_pos[2] - self._obj_pos_start[2], 0.0, kw["lift_clip"])
        r = r + kw["w_lift_bonus"] * lift
        r = r - kw["w_action_rate"] * jnp.sum((action - prev_action) ** 2)

        metrics = {"obj_pos_err": pos_err, "obj_rot_err": rot_err,
                   "hand_qpos_err": qpos_err, "lift": lift}
        return r, metrics
