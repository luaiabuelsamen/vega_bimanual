"""Generate a reachable, physically-consistent grasp demo as a ReferenceTrajectory.

Scripted phases, run *in the env's MuJoCo model* with position control, so the
recorded object motion is a genuine consequence of the hand motion:

    settle -> reach (DLS IK on fingertip centroid) -> descend -> close -> lift

We record controlled-joint qpos + object 6-DoF pose every control step and save
an .npz that `VegaTrackingEnv` / `scripts/train.py` can track. This is the
Option-B reference that makes the object-tracking reward learnable, without
waiting on human-data retargeting.

Run:
    PYTHONPATH=. python scripts/make_demo.py --out demos/pick_box.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import mujoco

from dextrack_vega.envs import VegaTrackingEnv
from dextrack_vega import config as C
from dextrack_vega.utils import trajectory as tj
from dextrack_vega.utils import transforms as T


# grasp target per controlled finger joint (closing pose), by name suffix.
CLOSED = {
    "th_j0": 1.35, "th_j1": 0.10, "th_j2": -0.40,
    "ff_j1": -1.05, "ff_j2": -1.15,
    "mf_j1": -1.05, "mf_j2": -1.15,
    "rf_j1": -1.00, "rf_j2": -1.10,
    "lf_j1": -1.00, "lf_j2": -1.10,
}
OPEN = {k: 0.0 for k in CLOSED}

# Power-wrap grasp: strong flexion on all fingers + thumb hard opposing, to
# encircle a pole/cylinder. Used by the kinematic lift reference.
WRAP = {
    "th_j0": 1.45, "th_j1": 0.5, "th_j2": -1.0,
    "ff_j1": -1.3, "ff_j2": -1.5,
    "mf_j1": -1.3, "mf_j2": -1.5,
    "rf_j1": -1.3, "rf_j2": -1.5,
    "lf_j1": -1.3, "lf_j2": -1.5,
}

# A cupped "scoop": fingers curled into a forward-facing C with the thumb
# opposing, so a small object is trapped against the sweep instead of squirting
# out between splayed fingers. This is what gives solid, sustained contact.
SCOOP = {
    "th_j0": 1.1, "th_j1": 0.3, "th_j2": -0.5,
    "ff_j1": -0.5, "ff_j2": -0.7,
    "mf_j1": -0.5, "mf_j2": -0.7,
    "rf_j1": -0.5, "rf_j2": -0.7,
    "lf_j1": -0.5, "lf_j2": -0.7,
}


class DemoGen:
    def __init__(self, side="R", seed=0, **env_kwargs):
        self.env = VegaTrackingEnv(side=side, seed=seed, **env_kwargs)
        self.m, self.d = self.env.model, self.env.data
        self.side = side
        self.arm = C.ARM_JOINTS[side]
        self.hand = C.HAND_JOINTS[side]
        self.ctrl_joints = self.env.ctrl_joints
        self.arm_dof = np.array([self.m.jnt_dofadr[self._jid(j)] for j in self.arm])
        self.arm_qadr = np.array([self.m.jnt_qposadr[self._jid(j)] for j in self.arm])
        arm_range = self.m.jnt_range[[self._jid(j) for j in self.arm]]
        self.arm_lo, self.arm_hi = arm_range[:, 0], arm_range[:, 1]
        self.ft_bids = self.env._ft_bids
        self.wrist_bid = self.env._bid(C.WRIST_BODY[side])
        self.obj_q = self.env._obj_qadr
        self.rec_q, self.rec_op, self.rec_oq = [], [], []
        self.rec_ctrl = []  # what drive_to *commanded* each control step
        self.rec_ftd = []  # per-frame nearest fingertip-to-object distance

    def _jid(self, n): return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)

    # ---- helpers ----
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for j, v in C.HOME_POSTURE.items():
            self.d.qpos[self.m.jnt_qposadr[self._jid(j)]] = v
            self.d.ctrl[self.env._posture_act[j]] = v
        # arms start at a raised home so IK has room
        self.set_ctrl(self.cur_qpos())
        # object at its scene spawn (already in keyframe-free qpos default)
        mujoco.mj_forward(self.m, self.d)

    def cur_qpos(self):
        return self.d.qpos[self.env._jnt_qposadr].copy()

    def set_ctrl(self, target):
        self.d.ctrl[self.env._act_ctrl_idx] = target

    def fingertip_centroid(self):
        return self.d.xpos[self.ft_bids].mean(0).copy()

    def obj_pos(self):
        return self.d.qpos[self.obj_q:self.obj_q + 3].copy()

    def finger_targets(self, table):
        """Map a {suffix: value} table to a full controlled-joint vector
        (arm part filled with current arm qpos, fingers from the table)."""
        tgt = self.cur_qpos()
        for i, j in enumerate(self.ctrl_joints):
            for suf, val in table.items():
                if j.endswith(suf):
                    tgt[i] = val
        return tgt

    # ---- IK: move arm so fingertip centroid reaches goal (position only) ----
    def ik_arm_to(self, goal, iters=300, damp=0.12, step_clip=0.05):
        # Plan on a COPY of the sim state: the iterations mutate qpos to probe
        # Jacobians, but we must NOT leave the live physics state teleported to
        # the solution — otherwise drive_to would start each phase from the
        # snapped IK pose and physics would jerk to it (abrupt, choppy phase
        # transitions). Save here, restore before returning.
        q_save = self.d.qpos.copy()
        v_save = self.d.qvel.copy()
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            cur = self.fingertip_centroid()
            err = goal - cur
            if np.linalg.norm(err) < 3e-3:
                break
            # Jacobian of the centroid = mean of fingertip-body Jacobians
            J = np.zeros((3, self.m.nv))
            jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
            for b in self.ft_bids:
                mujoco.mj_jacBody(self.m, self.d, jp, jr, b)
                J += jp
            J /= len(self.ft_bids)
            Ja = J[:, self.arm_dof]
            JJt = Ja @ Ja.T + (damp ** 2) * np.eye(3)
            dq = Ja.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -step_clip, step_clip)
            # respect joint limits — an out-of-range "solution" is infeasible
            # and the position servo would just clamp it, moving the hand
            # somewhere unintended.
            self.d.qpos[self.arm_qadr] = np.clip(
                self.d.qpos[self.arm_qadr] + dq, self.arm_lo, self.arm_hi)
        sol = self.d.qpos[self.arm_qadr].copy()
        # restore the live physics state; drive_to blends from the real pose
        self.d.qpos[:] = q_save
        self.d.qvel[:] = v_save
        mujoco.mj_forward(self.m, self.d)
        return sol

    @staticmethod
    def _quat_to_rotvec(q):
        """World-frame rotation vector (axis*angle) of a wxyz quaternion."""
        q = q / (np.linalg.norm(q) + 1e-12)
        if q[0] < 0:
            q = -q  # shortest rotation
        w = np.clip(q[0], -1.0, 1.0)
        s = np.sqrt(max(1.0 - w * w, 1e-12))
        if s < 1e-6:
            return np.zeros(3)
        return (2.0 * np.arccos(w)) * (q[1:] / s)

    # ---- 6-DoF IK: fingertip centroid -> goal_pos AND wrist orient -> goal_quat
    def ik_arm_pose(self, goal_pos, goal_quat, iters=400, damp=0.12,
                    step_clip=0.05, w_rot=0.5):
        """DLS IK over the arm joints driving the fingertip centroid to goal_pos
        and the wrist (R_arm_l7) orientation to goal_quat (wxyz). Orientation
        control is what lets the hand present the right face / curl plane to an
        object — and mirrors the human-wrist 6-DoF target in retargeting.
        Plans on a saved state and restores it (see ik_arm_to)."""
        goal_quat = T.quat_normalize(np.asarray(goal_quat, float))
        q_save = self.d.qpos.copy(); v_save = self.d.qvel.copy()
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            perr = goal_pos - self.fingertip_centroid()
            dq_quat = T.quat_mul(goal_quat,
                                 T.quat_conjugate(self.d.xquat[self.wrist_bid]))
            rerr = self._quat_to_rotvec(dq_quat)
            if np.linalg.norm(perr) < 3e-3 and np.linalg.norm(rerr) < 3e-2:
                break
            Jp = np.zeros((3, self.m.nv))
            for b in self.ft_bids:
                mujoco.mj_jacBody(self.m, self.d, jp, jr, b); Jp += jp
            Jp /= len(self.ft_bids)
            mujoco.mj_jacBody(self.m, self.d, jp, jr, self.wrist_bid)  # jr = wrist
            J = np.vstack([Jp, w_rot * jr])[:, self.arm_dof]      # 6 x n_arm
            err = np.concatenate([perr, w_rot * rerr])
            JJt = J @ J.T + (damp ** 2) * np.eye(6)
            dq = np.clip(J.T @ np.linalg.solve(JJt, err), -step_clip, step_clip)
            self.d.qpos[self.arm_qadr] = np.clip(
                self.d.qpos[self.arm_qadr] + dq, self.arm_lo, self.arm_hi)
        sol = self.d.qpos[self.arm_qadr].copy()
        self.d.qpos[:] = q_save; self.d.qvel[:] = v_save
        mujoco.mj_forward(self.m, self.d)
        return sol

    # ---- run controller toward a target qpos for `n` control steps ----
    def drive_to(self, target_qpos, n, record=True, blend=True):
        start = self.cur_qpos()
        for k in range(n):
            a = (k + 1) / n
            s = 3 * a ** 2 - 2 * a ** 3 if blend else 1.0
            ctrl = start + s * (target_qpos - start)
            # PRE-step recording: snapshot state BEFORE this step's mj_step, paired
            # with the ctrl we're about to apply. Then env.reset (qpos=rec_q[0],
            # qvel=0) matches the demo's true pre-state, and env.step applying
            # rec_ctrl[k] reproduces rec_q[k+1] bit-for-bit. (Post-step recording
            # made env start at a post-step pose with qvel=0 — i.e. without the
            # momentum that was actually present, so contact-sensitive replays
            # diverged.)
            if record:
                self.rec_q.append(self.cur_qpos())
                self.rec_ctrl.append(ctrl.copy())
                self.rec_op.append(self.obj_pos())
                self.rec_oq.append(
                    self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy())
                d_ft = np.linalg.norm(
                    self.d.xpos[self.ft_bids] - self.obj_pos(), axis=1).min()
                self.rec_ftd.append(d_ft)
            self.set_ctrl(ctrl)
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(self.m, self.d)

    # ---- the scripted demo ----
    def generate(self, task="push", push_vec=(0.0, -0.13, 0.0), obj_half=0.03):
        self.reset()
        obj0 = self.obj_pos()
        # settle (record so the policy sees a stable start)
        self.drive_to(self.cur_qpos(), n=20)

        if task == "lift":
            # NOTE: a top-down cage can't get under a box on a table — no force
            # closure with f5d6. Kept for experimentation; `push` is feasible.
            arm = self.ik_arm_to(obj0 + np.array([0.0, 0.0, 0.10]))
            tgt = self.finger_targets(OPEN); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=60)
            arm = self.ik_arm_to(obj0 + np.array([0.0, 0.0, -0.01]))
            tgt = self.finger_targets(OPEN); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=60)
            tgt = self.finger_targets(CLOSED); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=40)
            arm = self.ik_arm_to(obj0 + np.array([0.0, 0.0, 0.15]))
            tgt = self.finger_targets(CLOSED); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=80)
            return self.obj_pos() - obj0

        if task == "reorient":
            # Nonprehensile REORIENTATION: push the box off-center (tangentially)
            # so it yaws on the table. Contact a corner region, then sweep along
            # a line that doesn't pass through the box center -> torque about z
            # -> rotation. This is the feasible "dexterous" task for f5d6 (which
            # can't grasp): it tracks object *orientation*, not just position.
            contact = obj0 + np.array([obj_half + 0.02, obj_half + 0.02, 0.0])
            arm = self.ik_arm_to(contact + np.array([0.0, 0.0, 0.05]))
            tgt = self.finger_targets(SCOOP); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=60)
            arm = self.ik_arm_to(contact)
            tgt = self.finger_targets(SCOOP); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=50)
            n_steps = 16
            for k in range(n_steps):
                arm = self.ik_arm_to(contact + np.array([-0.13 * (k + 1) / n_steps, 0, 0]))
                tgt = self.finger_targets(SCOOP); tgt[:len(self.arm)] = arm
                self.drive_to(tgt, n=6, blend=False)
            return self.obj_pos() - obj0

        # PUSH: cup the hand, place it just off the box's near face at mid-
        # height, then sweep along the push direction in small constant-height
        # waypoints. Two fixes vs the old "float" bug:
        #   (a) cup the fingers (SCOOP) so the small cube is trapped against the
        #       sweep instead of slipping between splayed/open fingers — and the
        #       fingertip centroid we IK on actually sits at the contact surface
        #       (open fingers point away; a closed fist retracts the tips ~10cm
        #       toward the palm, which left the old demo floating ~14cm off).
        #   (b) sweep in many small IK steps at *constant z* — a single big IK
        #       jump to the far goal lets the arm reconfigure and lift the hand
        #       over the box, so the box never moves and the hand floats past it.
        push = np.array(push_vec, dtype=float)
        pdir = push / (np.linalg.norm(push) + 1e-9)
        contact0 = obj0 - pdir * (obj_half + 0.02)  # just off the near face

        # 1. cup the hand and reach above the contact point (no knocking)
        arm = self.ik_arm_to(contact0 + np.array([0.0, 0.0, 0.05]))
        tgt = self.finger_targets(SCOOP); tgt[:len(self.arm)] = arm
        self.drive_to(tgt, n=60)
        # 2. lower to box mid-height, cup just off the near face
        arm = self.ik_arm_to(contact0)
        tgt = self.finger_targets(SCOOP); tgt[:len(self.arm)] = arm
        self.drive_to(tgt, n=50)
        # 3. incremental sweep at constant height — keeps the cup on the box.
        # blend=False so the sub-segments chain at constant velocity (no
        # decelerate-to-zero at each waypoint), giving one smooth slide.
        n_steps = 18
        for k in range(n_steps):
            arm = self.ik_arm_to(contact0 + push * ((k + 1) / n_steps))
            tgt = self.finger_targets(SCOOP); tgt[:len(self.arm)] = arm
            self.drive_to(tgt, n=6, blend=False)

        return self.obj_pos() - obj0

    # ---- KINEMATIC lift reference (no physics): approach -> wrap -> lift ----
    def generate_lift_ref(self, wrap_deg=90.0, lift_h=0.15, grasp_dz=0.0):
        """Build a *kinematic* pick-and-lift ReferenceTrajectory for a vertical
        pole at the object's spawn, using the verified 6-DoF IK to place the
        hand at a feasible wrap orientation. Poses are computed kinematically and
        the object is locked to the hand during the lift — this is NOT a physics
        replay (open-loop physics can't grip), it's a kinematic *target* for the
        RL tracker to realize (the DexTrack premise). Records hand_qpos + object
        6-DoF per frame.

        Keyframes: above-pole(open) -> at-pole(open) -> at-pole(wrap) ->
        lifted(wrap, object rises with the grasp point).
        """
        self.reset()
        mujoco.mj_forward(self.m, self.d)
        obj0 = self.obj_pos()
        objq0 = self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy()
        cur = self.d.xquat[self.wrist_bid].copy()
        ang = np.radians(wrap_deg)
        wrap_q = T.quat_mul(np.array([np.cos(ang / 2), np.sin(ang / 2), 0, 0]), cur)
        grasp = obj0 + np.array([0.0, 0.0, grasp_dz])
        above = grasp + np.array([0.0, 0.0, 0.08])

        # arm IK at the three distinct positions (wrap orientation throughout)
        arm_above = self.ik_arm_pose(above, wrap_q)
        arm_grasp = self.ik_arm_pose(grasp, wrap_q)
        arm_lift = self.ik_arm_pose(grasp + np.array([0, 0, lift_h]), wrap_q)
        f_open = self.finger_targets(OPEN)[len(self.arm):]
        f_wrap = self.finger_targets(WRAP)[len(self.arm):]

        def lerp(a, b, s):
            return (1 - s) * np.asarray(a) + s * np.asarray(b)

        def smooth(s):
            return 3 * s ** 2 - 2 * s ** 3

        def seg(arm_a, arm_b, fa, fb, o_a, o_b, n):
            for k in range(n):
                s = smooth((k + 1) / n)
                arm = lerp(arm_a, arm_b, s)
                fin = lerp(fa, fb, s)
                self.rec_q.append(np.concatenate([arm, fin]))
                self.rec_op.append(lerp(o_a, o_b, s))
                self.rec_oq.append(objq0)

        # settle at approach, descend, wrap (object still), then lift it
        seg(arm_above, arm_above, f_open, f_open, obj0, obj0, 15)
        seg(arm_above, arm_grasp, f_open, f_open, obj0, obj0, 40)
        seg(arm_grasp, arm_grasp, f_open, f_wrap, obj0, obj0, 30)
        seg(arm_grasp, arm_lift, f_wrap, f_wrap, obj0,
            obj0 + np.array([0, 0, lift_h]), 50)
        return float(lift_h)

    def save(self, path):
        traj = tj.ReferenceTrajectory(
            obj_pos=np.array(self.rec_op), obj_quat=np.array(self.rec_oq),
            hand_qpos=np.array(self.rec_q), dt=self.env.dt,
            joint_names=self.ctrl_joints,
            hand_ctrl=(np.array(self.rec_ctrl) if self.rec_ctrl else None))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tj.save_npz(path, traj)
        return traj


# Inward squeeze grip used by the bimanual lift: fingers curl moderately so each
# hand cups the box's side face and wraps the top/bottom edges. Both hands do
# this in mirror, so the two palms provide the object opposition (the inter-hand
# "force closure") that a single f5d6 hand cannot — that's the whole point of the
# bimanual lift.
SQUEEZE = {
    "th_j0": 1.2, "th_j1": 0.4, "th_j2": -0.6,
    "ff_j1": -0.7, "ff_j2": -0.9,
    "mf_j1": -0.7, "mf_j2": -0.9,
    "rf_j1": -0.7, "rf_j2": -0.9,
    "lf_j1": -0.7, "lf_j2": -0.9,
}


class BimanualDemoGen:
    """Two-arm reference generator. Builds a *kinematic* DexTrack-style target
    (hand-joint trajectory + object 6-DoF) for a cooperative squeeze-and-lift:
    the R and L hands approach opposite faces of a box, close to apply an inward
    squeeze, then lift it together. Like generate_lift_ref this is a kinematic
    *target* (object locked to the hands during the lift), not a physics replay —
    the RL tracker is what must learn to realize the grip. This is the bimanual
    analogue of what DexTrack tracks, and it's the one manipulation that is
    impossible for a single f5d6 hand (thumb can't oppose < ~3.1cm)."""

    def __init__(self, seed=0, **env_kwargs):
        self.sides = ["R", "L"]
        self.env = VegaTrackingEnv(sides=self.sides, seed=seed, **env_kwargs)
        self.m, self.d = self.env.model, self.env.data
        self.ctrl_joints = self.env.ctrl_joints
        self.obj_q = self.env._obj_qadr
        # per-side index bookkeeping
        self.arm = {s: C.ARM_JOINTS[s] for s in self.sides}
        self.arm_qadr, self.arm_dof, self.arm_lo, self.arm_hi = {}, {}, {}, {}
        self.ft_bids, self.wrist_bid = {}, {}
        # ctrl_joints is [R arm7, R hand11, L arm7, L hand11]; fingertips are the
        # first 5 (R) then next 5 (L) of env._ft_bids.
        for k, s in enumerate(self.sides):
            self.arm_qadr[s] = np.array(
                [self.m.jnt_qposadr[self._jid(j)] for j in self.arm[s]])
            self.arm_dof[s] = np.array(
                [self.m.jnt_dofadr[self._jid(j)] for j in self.arm[s]])
            r = self.m.jnt_range[[self._jid(j) for j in self.arm[s]]]
            self.arm_lo[s], self.arm_hi[s] = r[:, 0], r[:, 1]
            self.ft_bids[s] = self.env._ft_bids[5 * k:5 * k + 5]
            self.wrist_bid[s] = self.env._bid(C.WRIST_BODY[s])
        # offset of each side's 18-joint block within the 36-vector
        self.block = {"R": 0, "L": 18}
        self.rec_q, self.rec_op, self.rec_oq = [], [], []
        self.rec_ctrl = []  # what drive_to commanded each control step

    def _jid(self, n): return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        for j, v in C.HOME_POSTURE.items():
            self.d.qpos[self.m.jnt_qposadr[self._jid(j)]] = v
        mujoco.mj_forward(self.m, self.d)

    def obj_pos(self):
        return self.d.qpos[self.obj_q:self.obj_q + 3].copy()

    def ft_centroid(self, side):
        return self.d.xpos[self.ft_bids[side]].mean(0).copy()

    # ---- per-side position IK: move `side` arm so its fingertip centroid -> goal
    def ik_side_to(self, side, goal, iters=300, damp=0.12, step_clip=0.05):
        q_save = self.d.qpos.copy(); v_save = self.d.qvel.copy()
        qadr, dof = self.arm_qadr[side], self.arm_dof[side]
        lo, hi = self.arm_lo[side], self.arm_hi[side]
        jp = np.zeros((3, self.m.nv)); jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            err = goal - self.ft_centroid(side)
            if np.linalg.norm(err) < 3e-3:
                break
            J = np.zeros((3, self.m.nv))
            for b in self.ft_bids[side]:
                mujoco.mj_jacBody(self.m, self.d, jp, jr, b); J += jp
            J /= len(self.ft_bids[side])
            Ja = J[:, dof]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + damp ** 2 * np.eye(3), err)
            dq = np.clip(dq, -step_clip, step_clip)
            self.d.qpos[qadr] = np.clip(self.d.qpos[qadr] + dq, lo, hi)
        sol = self.d.qpos[qadr].copy()
        self.d.qpos[:] = q_save; self.d.qvel[:] = v_save
        mujoco.mj_forward(self.m, self.d)
        return sol

    def _assemble(self, arms: dict, fingers: dict):
        """Build a 36-vec from per-side arm solutions + per-side finger tables.
        `fingers[side]` is a {suffix: value} dict; unset finger joints keep their
        current value."""
        full = self.d.qpos[self.env._jnt_qposadr].copy()
        for s in self.sides:
            b = self.block[s]
            full[b:b + 7] = arms[s]
            for i, j in enumerate(self.ctrl_joints[b + 7:b + 18]):
                for suf, val in fingers[s].items():
                    if j.endswith(suf):
                        full[b + 7 + i] = val
        return full

    # ---- physics-driven driver: pre-step record, set ctrl, mj_step
    def drive_to(self, target_qpos, n, record=True, blend=True):
        start = self.d.qpos[self.env._jnt_qposadr].copy()
        for k in range(n):
            a = (k + 1) / n
            s = 3 * a ** 2 - 2 * a ** 3 if blend else 1.0
            ctrl = start + s * (target_qpos - start)
            # PRE-step recording (see DemoGen.drive_to)
            if record:
                self.rec_q.append(self.d.qpos[self.env._jnt_qposadr].copy())
                self.rec_ctrl.append(ctrl.copy())
                self.rec_op.append(self.obj_pos())
                self.rec_oq.append(self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy())
            self.d.ctrl[self.env._act_ctrl_idx] = ctrl
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(self.m, self.d)

    def generate_handover(self, push_dy=0.18, lift_h=0.15):
        """BIMANUAL HANDOVER (two-phase coordination):
           1) R hand alone pushes the box from one side of the table to the
              midline (the L hand can't reach the start position).
           2) Both hands then squeeze-lift it together.

        Two arms playing two genuinely different roles. Physics-driven via
        drive_to so hand_ctrl is recorded; the BC pipeline can clone it
        afterwards. Requires the box to start off-midline (use HANDOVER_BOX
        preset which spawns at y=-0.18)."""
        self.reset()
        obj0 = self.obj_pos()
        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        hy = float(self.m.geom_size[gid][1])
        up = np.array([0.0, 0.0, 1.0])
        push_v = np.array([0.0, push_dy, 0.0])

        # L stays at its reset pose throughout phase 1 (out of the way).
        L_init = self.d.qpos[self.arm_qadr["L"]].copy()

        # --- phase 1: R alone pushes box from obj0 to obj0 + push_v ---
        # Contact point on the -y face of the box (R hand pushes from behind in
        # -y, sweeping in +y to drive box toward the midline).
        contact_R = obj0 + np.array([0.0, -(hy + 0.02), 0.0])
        above_R = contact_R + 0.18 * up   # high enough to clear the box top

        arm_R_above = self.ik_side_to("R", above_R)
        arm_R_contact = self.ik_side_to("R", contact_R)

        def tgt(armR, fingers_R, armL=None, fingers_L=None):
            return self._assemble(
                {"R": armR, "L": armL if armL is not None else L_init},
                {"R": dict(fingers_R),
                 "L": dict(fingers_L if fingers_L is not None else OPEN)})

        # Teleport arms to the approach pose so the recording starts with arms
        # already positioned (otherwise transit through the box knocks it off).
        approach_full = tgt(arm_R_above, OPEN)
        self.d.qpos[self.env._jnt_qposadr] = approach_full
        self.d.ctrl[self.env._act_ctrl_idx] = approach_full
        mujoco.mj_forward(self.m, self.d)
        L_init = self.d.qpos[self.arm_qadr["L"]].copy()
        # Phase 1 is physics-driven drive_to (single-hand push with SCOOP only
        # reliably translates into physics push when the orientation comes from
        # actual ctrl, not from interpolated kinematic poses). Env replay of
        # this phase achieves ~38% of the demo's push (8cm vs 21cm) — see the
        # env-replay-divergence bug logged in memory.
        self.drive_to(tgt(arm_R_above, SCOOP), n=15)
        self.drive_to(tgt(arm_R_above, SCOOP), n=40)
        self.drive_to(tgt(arm_R_contact, SCOOP), n=45)
        n_push = 22
        for k in range(n_push):
            frac = (k + 1) / n_push
            arm_R_step = self.ik_side_to("R", contact_R + frac * push_v)
            self.drive_to(tgt(arm_R_step, SCOOP), n=6, blend=False)
        post_R = contact_R + push_v + np.array([0.0, -0.04, 0.12])
        arm_R_retreat = self.ik_side_to("R", post_R)
        self.drive_to(tgt(arm_R_retreat, OPEN), n=40)

        # --- phase 2: bimanual squeeze-and-lift around the now-centred box ---
        # Kinematic seg-loop pattern (like generate_squeeze_lift): records the
        # ideal interpolated pose as both qpos AND ctrl so the env's ctrl-replay
        # reproduces the +21 cm lift behaviour. Starts at the physical obj_now.
        obj_now = self.obj_pos()
        objq_now = self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy()
        cR = obj_now + np.array([0.0, -hy, 0.0])
        cL = obj_now + np.array([0.0, +hy, 0.0])
        aboveR = cR + 0.08 * up
        aboveL = cL + 0.08 * up
        armA = {"R": self.ik_side_to("R", aboveR),
                "L": self.ik_side_to("L", aboveL)}
        armC = {"R": self.ik_side_to("R", cR),
                "L": self.ik_side_to("L", cL)}
        n_lift = 5
        lift_arms = []
        for i in range(1, n_lift + 1):
            dz = lift_h * i / n_lift
            lift_arms.append((i / n_lift, {
                "R": self.ik_side_to("R", cR + dz * up),
                "L": self.ik_side_to("L", cL + dz * up)}))
        f_open = {s: dict(OPEN) for s in self.sides}
        f_sq = {s: dict(SQUEEZE) for s in self.sides}

        def lerp(a, b, s): return (1 - s) * np.asarray(a) + s * np.asarray(b)
        def smooth(s): return 3 * s ** 2 - 2 * s ** 3

        # Bridge: drive arms (physics) to the bimanual approach pose (armA),
        # then switch to kinematic seg-loop for the lift.
        approach_armA = self._assemble(armA, f_open)
        self.drive_to(approach_armA, n=60)

        def seg(aA, aB, fA, fB, oA, oB, n):
            for k in range(n):
                s = smooth((k + 1) / n)
                arms = {sd: lerp(aA[sd], aB[sd], s) for sd in self.sides}
                figs = {sd: {suf: (1 - s) * fA[sd][suf] + s * fB[sd][suf]
                             for suf in fA[sd]} for sd in self.sides}
                pose = self._assemble(arms, figs)
                self.rec_q.append(pose)
                self.rec_ctrl.append(pose.copy())
                self.rec_op.append(lerp(oA, oB, s))
                self.rec_oq.append(objq_now)

        # kinematic squeeze-and-lift around the (physical) current box position
        seg(armA, armA, f_open, f_open, obj_now, obj_now, 15)
        seg(armA, armC, f_open, f_open, obj_now, obj_now, 40)
        seg(armC, armC, f_open, f_sq, obj_now, obj_now, 30)
        prev_arm, prev_o = armC, obj_now
        for frac, arm in lift_arms:
            o = obj_now + frac * lift_h * up
            seg(prev_arm, arm, f_sq, f_sq, prev_o, o, 14)
            prev_arm, prev_o = arm, o
        seg(prev_arm, prev_arm, f_sq, f_sq, prev_o, prev_o, 15)
        return self.obj_pos() - obj0

    def generate_reorient_kinematic(self, sweep_dx=0.035, n_sweep=24, yaw_deg=60):
        """All-kinematic variant of bimanual reorient (object yaw locked to
        interpolated trajectory). Both hands sweep tangentially as before, but
        rec_q / rec_ctrl are the ideal interpolated poses (not physics-driven),
        and the ref's obj_quat ramps from 0 to yaw_deg over the sweep. This is
        the bim_lift pattern applied to reorient — env zero-residual then
        reproduces the kinematic intent in physics (the two hands sweep, the box
        yaws via friction), and BC of the zero-residual rollout gives a policy
        that yaws the box."""
        self.reset()
        obj0 = self.obj_pos()
        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        hy = float(self.m.geom_size[gid][1])
        up = np.array([0.0, 0.0, 1.0])
        cR = obj0 + np.array([0.0, -hy, 0.0])
        cL = obj0 + np.array([0.0, +hy, 0.0])
        aboveR = cR + 0.08 * up
        aboveL = cL + 0.08 * up

        armA = {"R": self.ik_side_to("R", aboveR),
                "L": self.ik_side_to("L", aboveL)}
        armC = {"R": self.ik_side_to("R", cR),
                "L": self.ik_side_to("L", cL)}
        # Sweep direction: R hand sweeps +x, L hand sweeps -x. Empirically the
        # box yaws in the -z direction with this combo in env physics (the
        # opposite of the naive torque-couple intuition — frictional contact
        # dynamics outweigh the abstract couple). With ref obj_quat ramping
        # 0 -> -yaw_deg to match, the kinematic playback and env replay agree
        # in direction.
        sweep_arms = []
        for k in range(n_sweep + 1):
            frac = k / n_sweep
            sweep_arms.append({
                "R": self.ik_side_to("R", cR + np.array([+sweep_dx * frac, 0, 0])),
                "L": self.ik_side_to("L", cL + np.array([-sweep_dx * frac, 0, 0]))})

        # Teleport to approach pose so seg's "start" matches env's reset state.
        approach_full = self._assemble(armA, {s: dict(SCOOP) for s in self.sides})
        self.d.qpos[self.env._jnt_qposadr] = approach_full
        self.d.ctrl[self.env._act_ctrl_idx] = approach_full
        mujoco.mj_forward(self.m, self.d)

        f_scoop = {s: dict(SCOOP) for s in self.sides}
        ang = np.radians(yaw_deg)
        # yaw quaternion around z, interpolated from identity to full yaw
        def yawq(a):
            return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])

        def lerp(a, b, s): return (1 - s) * np.asarray(a) + s * np.asarray(b)
        def smooth(s): return 3 * s ** 2 - 2 * s ** 3

        def seg(aA, aB, fA, fB, oA, oB, qA, qB, n):
            for k in range(n):
                s = smooth((k + 1) / n)
                arms = {sd: lerp(aA[sd], aB[sd], s) for sd in self.sides}
                figs = {sd: {suf: (1 - s) * fA[sd][suf] + s * fB[sd][suf]
                             for suf in fA[sd]} for sd in self.sides}
                pose = self._assemble(arms, figs)
                self.rec_q.append(pose)
                self.rec_ctrl.append(pose.copy())
                self.rec_op.append(lerp(oA, oB, s))
                # slerp two unit quats (small-angle so lerp+normalize is fine)
                q = (1 - s) * qA + s * qB
                q = q / (np.linalg.norm(q) + 1e-9)
                self.rec_oq.append(q)

        q_id = yawq(0.0)
        q_end = yawq(ang)

        # settle (above box)
        seg(armA, armA, f_scoop, f_scoop, obj0, obj0, q_id, q_id, 15)
        # descend to face contact (obj static, identity quat)
        seg(armA, armC, f_scoop, f_scoop, obj0, obj0, q_id, q_id, 50)
        # tangential sweep: hands move outward, OBJECT YAWS kinematically.
        for k in range(n_sweep):
            qa = yawq(ang * k / n_sweep)
            qb = yawq(ang * (k + 1) / n_sweep)
            seg(sweep_arms[k], sweep_arms[k + 1], f_scoop, f_scoop,
                obj0, obj0, qa, qb, 8)
        # hold
        seg(sweep_arms[-1], sweep_arms[-1], f_scoop, f_scoop,
            obj0, obj0, q_end, q_end, 15)
        return self.obj_pos() - obj0

    def generate_reorient(self, sweep_dx=0.06, n_sweep=24):
        """BIMANUAL cooperative reorient: both hands contact opposite ±y faces of
        a central box and sweep TANGENTIALLY in *opposite* x directions while
        maintaining radial contact. The two opposing tangential drags form a
        couple about z -> the box yaws. Nonprehensile (no force-closure needed),
        so the grip-margin problem that limits the squeeze-lift doesn't apply.
        Physics-driven (records real qpos + real obj pose)."""
        self.reset()
        obj0 = self.obj_pos()
        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        hy = float(self.m.geom_size[gid][1])
        # settle
        self.drive_to(self.d.qpos[self.env._jnt_qposadr].copy(), n=20)
        up = np.array([0.0, 0.0, 1.0])
        # contact points: at each ±y face, midline in x (so tangential sweep
        # passes through the centre line)
        cR = obj0 + np.array([0.0, -hy, 0.0])
        cL = obj0 + np.array([0.0, +hy, 0.0])
        aboveR, aboveL = cR + 0.06 * up, cL + 0.06 * up

        # 1. cup hands and reach above each face
        armR = self.ik_side_to("R", aboveR); armL = self.ik_side_to("L", aboveL)
        tgt = self._assemble({"R": armR, "L": armL},
                             {"R": dict(SCOOP), "L": dict(SCOOP)})
        self.drive_to(tgt, n=60)
        # 2. descend to face contact
        armR = self.ik_side_to("R", cR); armL = self.ik_side_to("L", cL)
        tgt = self._assemble({"R": armR, "L": armL},
                             {"R": dict(SCOOP), "L": dict(SCOOP)})
        self.drive_to(tgt, n=50)
        # 3. tangential sweep in OPPOSITE x directions -> couple about z.
        # Small constant-position sub-steps re-IK each waypoint and chain at
        # constant velocity, like the single-arm reorient sweep.
        for k in range(n_sweep):
            frac = (k + 1) / n_sweep
            armR = self.ik_side_to("R", cR + np.array([-sweep_dx * frac, 0, 0]))
            armL = self.ik_side_to("L", cL + np.array([+sweep_dx * frac, 0, 0]))
            tgt = self._assemble({"R": armR, "L": armL},
                                 {"R": dict(SCOOP), "L": dict(SCOOP)})
            self.drive_to(tgt, n=6, blend=False)
        return self.obj_pos() - obj0

    def generate_squeeze_lift(self, lift_h=0.18, face_inset=0.0):
        """approach -> squeeze opposite faces -> lift together. The reference is
        a KINEMATIC target trajectory (object locked to the hands during lift);
        the env then realizes it in physics via the residual policy. Both
        hand_qpos AND hand_ctrl record the same ideal interpolated pose — the
        env's ctrl-replay mode then uses those ideal poses as actuator targets
        (the historically working open-loop +21cm came from exactly this: the
        OLD env's qpos-as-ctrl replay was effectively applying these ideal
        poses as ctrl; making it explicit via hand_ctrl preserves that behavior
        while also fixing contact-rich tracks like bim_reorient)."""
        self.reset()
        obj0 = self.obj_pos()
        objq0 = self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy()
        gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        hy = float(self.m.geom_size[gid][1])
        cR = obj0 + np.array([0.0, -(hy - face_inset), 0.0])
        cL = obj0 + np.array([0.0, +(hy - face_inset), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        aboveR, aboveL = cR + 0.08 * up, cL + 0.08 * up

        armA = {"R": self.ik_side_to("R", aboveR), "L": self.ik_side_to("L", aboveL)}
        armC = {"R": self.ik_side_to("R", cR), "L": self.ik_side_to("L", cL)}
        n_lift = 5
        lift_arms = []
        for i in range(1, n_lift + 1):
            dz = lift_h * i / n_lift
            lift_arms.append((i / n_lift, {
                "R": self.ik_side_to("R", cR + dz * up),
                "L": self.ik_side_to("L", cL + dz * up)}))
        f_open = {s: dict(OPEN) for s in self.sides}
        f_sq = {s: dict(SQUEEZE) for s in self.sides}

        def lerp(a, b, s): return (1 - s) * np.asarray(a) + s * np.asarray(b)
        def smooth(s): return 3 * s ** 2 - 2 * s ** 3

        def seg(aA, aB, fA, fB, oA, oB, n):
            for k in range(n):
                s = smooth((k + 1) / n)
                arms = {sd: lerp(aA[sd], aB[sd], s) for sd in self.sides}
                figs = {sd: {suf: (1 - s) * fA[sd][suf] + s * fB[sd][suf]
                             for suf in fA[sd]} for sd in self.sides}
                pose = self._assemble(arms, figs)
                self.rec_q.append(pose)
                self.rec_ctrl.append(pose.copy())  # ideal pose = ctrl target
                self.rec_op.append(lerp(oA, oB, s))
                self.rec_oq.append(objq0)

        seg(armA, armA, f_open, f_open, obj0, obj0, 15)    # settle above
        seg(armA, armC, f_open, f_open, obj0, obj0, 40)    # descend to faces
        seg(armC, armC, f_open, f_sq,  obj0, obj0, 30)     # squeeze closed
        prev_arm, prev_o = armC, obj0
        for frac, arm in lift_arms:
            o = obj0 + frac * lift_h * up
            seg(prev_arm, arm, f_sq, f_sq, prev_o, o, 14)
            prev_arm, prev_o = arm, o
        seg(prev_arm, prev_arm, f_sq, f_sq, prev_o, prev_o, 15)  # hold at top
        return float(lift_h)

    def save(self, path):
        traj = tj.ReferenceTrajectory(
            obj_pos=np.array(self.rec_op), obj_quat=np.array(self.rec_oq),
            hand_qpos=np.array(self.rec_q), dt=self.env.dt,
            joint_names=self.ctrl_joints,
            hand_ctrl=(np.array(self.rec_ctrl) if self.rec_ctrl else None))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tj.save_npz(path, traj)
        return traj


# Pole preset: a vertical cylinder at the feasible wrap-grasp pose found by the
# workspace orientation search (X+90 reachable, perr ~2mm).
POLE = dict(obj_type="cylinder", obj_dims=(0.016, 0.07), obj_pos=(0.55, -0.15, 0.84),
            obj_mass=0.05)

# Bimanual box preset: centred on the robot midline (y=0) so both arms reach it
# symmetrically, sized so the two palms have a face to press (8x9x14 cm). Mass is
# 0.05kg (a light package): at the heavier 0.12kg the inter-hand friction squeeze
# can't hold the box (open-loop ref playback lifts it only ~6cm then drops it),
# whereas at 0.05kg the squeeze physically lifts and holds it (+21cm open-loop) —
# i.e. the tracking target is physically realizable, which it must be for the RL
# tracker to have any chance. f5d6's weak opposition caps the liftable mass.
BIM_BOX = dict(obj_type="box", obj_dims=(0.04, 0.045, 0.07), obj_pos=(0.55, 0.0, 0.80),
               obj_mass=0.05)

# Bimanual reorient preset: the same midline box, slightly heavier (the hands push
# tangentially against the ±y faces, no lift, so weight makes it more stable
# under uneven push — and the couple of two opposing tangential drags still
# yaws it).
BIM_REORIENT_BOX = dict(obj_type="box", obj_dims=(0.045, 0.045, 0.06),
                        obj_pos=(0.55, 0.0, 0.80), obj_mass=0.1)

# Handover preset: same box as BIM_BOX but spawned off-midline (y=-0.18) so the
# L hand can't reach the start position. R hand has to deliver the box to the
# midline before the bimanual lift becomes feasible.
HANDOVER_BOX = dict(obj_type="box", obj_dims=(0.04, 0.045, 0.07),
                    obj_pos=(0.55, -0.18, 0.80), obj_mass=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demos/push_box.npz")
    ap.add_argument("--side", default="R")
    ap.add_argument("--task", default="push",
                    choices=["push", "lift", "lift_ref", "reorient",
                             "bim_lift", "bim_reorient", "bim_reorient_k",
                             "bim_handover"])
    args = ap.parse_args()

    if args.task == "bim_handover":
        gen = BimanualDemoGen(**HANDOVER_BOX)
        disp = gen.generate_handover()
        traj = gen.save(args.out)
        obj_dz = (traj.obj_pos[-1] - traj.obj_pos[0])[2] * 100
        obj_dy = (traj.obj_pos[-1] - traj.obj_pos[0])[1] * 100
        moved = float(np.linalg.norm(disp)) * 100
        print(f"[make_demo] task=bim_handover frames={traj.n_frames} dur={traj.duration:.2f}s "
              f"action_dim={gen.env.action_dim}  push +y {obj_dy:+.1f}cm  lift {obj_dz:+.1f}cm  "
              f"total moved {moved:.1f}cm  -> {args.out}")
        if obj_dy < 10 or obj_dz < 5:
            print("[make_demo] WARNING: short on push or lift — handover incomplete.")
        return

    if args.task == "bim_reorient_k":
        gen = BimanualDemoGen(**BIM_REORIENT_BOX)
        gen.generate_reorient_kinematic()
        traj = gen.save(args.out)
        def yaw(q): return np.degrees(2 * np.arctan2(q[3], q[0]))
        dyaw = yaw(traj.obj_quat[-1]) - yaw(traj.obj_quat[0])
        print(f"[make_demo] task=bim_reorient_k frames={traj.n_frames} dur={traj.duration:.2f}s "
              f"action_dim={gen.env.action_dim}  yaw change {dyaw:+.0f} deg "
              f"(kinematic target)  -> {args.out}")
        return

    if args.task == "bim_reorient":
        gen = BimanualDemoGen(**BIM_REORIENT_BOX)
        disp = gen.generate_reorient()
        traj = gen.save(args.out)
        def yaw(q): return np.degrees(2 * np.arctan2(q[3], q[0]))
        dyaw = yaw(traj.obj_quat[-1]) - yaw(traj.obj_quat[0])
        moved = float(np.linalg.norm(disp))
        print(f"[make_demo] task=bim_reorient frames={traj.n_frames} dur={traj.duration:.2f}s "
              f"action_dim={gen.env.action_dim}  yaw change {dyaw:+.0f} deg  "
              f"translation {moved*100:.1f}cm  -> {args.out}")
        if abs(dyaw) < 20:
            print("[make_demo] WARNING: little rotation — bim_reorient likely failed.")
        return

    if args.task == "bim_lift":
        # bimanual cooperative squeeze-and-lift (kinematic DexTrack-style target)
        gen = BimanualDemoGen(**BIM_BOX)
        lift_h = gen.generate_squeeze_lift()
        traj = gen.save(args.out)
        obj_dz = (traj.obj_pos[-1] - traj.obj_pos[0])[2]
        print(f"[make_demo] task=bim_lift frames={traj.n_frames} dur={traj.duration:.2f}s  "
              f"action_dim={gen.env.action_dim}  object lift={obj_dz*100:.1f}cm "
              f"(kinematic target)  -> {args.out}")
        print("[make_demo] NOTE: kinematic reference (object locked to the two "
              "hands during the lift); open-loop physics will NOT reproduce it — "
              "the RL tracker must learn the cooperative grip.")
        return

    if args.task == "lift_ref":
        # kinematic pick-and-lift reference on the pole (target for RL tracking)
        gen = DemoGen(side=args.side, **POLE)
        lift_h = gen.generate_lift_ref()
        traj = gen.save(args.out)
        obj_dz = (traj.obj_pos[-1] - traj.obj_pos[0])[2]
        print(f"[make_demo] task=lift_ref frames={traj.n_frames} dur={traj.duration:.2f}s  "
              f"object lift={obj_dz*100:.1f}cm (kinematic target)  -> {args.out}")
        print("[make_demo] NOTE: kinematic reference (object locked to hand during "
              "lift); open-loop physics will NOT reproduce it — RL must learn the grip.")
        return

    gen = DemoGen(side=args.side)
    disp = gen.generate(task=args.task)
    traj = gen.save(args.out)
    moved = float(np.linalg.norm(disp))

    if args.task == "reorient":
        def yaw(q):  # signed yaw (deg) of a wxyz quat
            return np.degrees(2 * np.arctan2(q[3], q[0]))
        dyaw = yaw(traj.obj_quat[-1]) - yaw(traj.obj_quat[0])
        print(f"[make_demo] task=reorient frames={traj.n_frames} dur={traj.duration:.2f}s  "
              f"yaw change {dyaw:.0f} deg  translation {moved*100:.1f}cm  -> {args.out}")
        if abs(dyaw) < 15:
            print("[make_demo] WARNING: little rotation — reorientation likely failed.")
        return
    # contact quality: nearest fingertip-to-object distance over the push phase
    ftd = np.array(gen.rec_ftd)
    push_d = ftd[-80:] if ftd.size >= 80 else ftd
    print(f"[make_demo] task={args.task} frames={traj.n_frames} dur={traj.duration:.2f}s  "
          f"object moved {moved*100:.1f} cm  disp={np.round(disp,3)}  -> {args.out}")
    print(f"[make_demo] fingertip-obj dist over push: min {push_d.min()*100:.1f}cm "
          f"mean {push_d.mean()*100:.1f}cm  (box half-size {3.0:.0f}cm)")
    if moved < 0.03:
        print("[make_demo] WARNING: object barely moved — demo likely failed.")
    if push_d.min() > 0.06:
        print("[make_demo] WARNING: hand never got within 6cm of the box — "
              "contact is glancing, not solid.")


if __name__ == "__main__":
    main()
