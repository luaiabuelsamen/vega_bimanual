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
            self.set_ctrl(start + s * (target_qpos - start))
            for _ in range(C.CONTROL_DECIMATION):
                mujoco.mj_step(self.m, self.d)
            if record:
                self.rec_q.append(self.cur_qpos())
                self.rec_op.append(self.obj_pos())
                self.rec_oq.append(
                    self.d.qpos[self.obj_q + 3:self.obj_q + 7].copy())
                d_ft = np.linalg.norm(
                    self.d.xpos[self.ft_bids] - self.obj_pos(), axis=1).min()
                self.rec_ftd.append(d_ft)

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
            joint_names=self.ctrl_joints)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tj.save_npz(path, traj)
        return traj


# Pole preset: a vertical cylinder at the feasible wrap-grasp pose found by the
# workspace orientation search (X+90 reachable, perr ~2mm).
POLE = dict(obj_type="cylinder", obj_dims=(0.016, 0.07), obj_pos=(0.55, -0.15, 0.84),
            obj_mass=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demos/push_box.npz")
    ap.add_argument("--side", default="R")
    ap.add_argument("--task", default="push",
                    choices=["push", "lift", "lift_ref", "reorient"])
    args = ap.parse_args()

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
