"""Paths, joint groups, and control constants for DexTrack-on-Vega.

This package replicates DexTrack's RL *tracking* approach (track a kinematic
reference of hand joints + object 6-DoF pose with residual position targets)
on the Dexmate Vega upper body fitted with the f5d6 5-finger hand, in MuJoCo.

We deliberately keep a single source of truth for joint names here so the
asset builder, the env, and the reward terms all agree.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PKG_ROOT = Path(__file__).resolve().parents[1]          # dextrack_vega/
PROJECTS = PKG_ROOT.parent                               # ~/projects
GEN_DIR = PKG_ROOT / "dextrack_vega" / "assets_gen"      # generated MJCF lives here

# Source URDF: full Vega 1U upper body with two f5d6 hands, OBJ collision meshes.
DEXMATE_URDF_ROOT = PROJECTS / "dexmate" / "dexmate-urdf" / "robots"
VEGA_F5D6_URDF = DEXMATE_URDF_ROOT / "humanoid" / "vega_1u" / "vega_1u_f5d6-obj.urdf"

GENERATED_SCENE = GEN_DIR / "vega_f5d6_scene.xml"

# --- Joint groups (verified by compiling the URDF in MuJoCo) -----------------
# Per-arm: 7 arm joints + 11 hand joints (thumb 3 + 4 fingers x 2).
ARM_JOINTS = {
    "R": ["R_arm_j1", "R_arm_j2", "R_arm_j3", "R_arm_j4",
          "R_arm_j5", "R_arm_j6", "R_arm_j7"],
    "L": ["L_arm_j1", "L_arm_j2", "L_arm_j3", "L_arm_j4",
          "L_arm_j5", "L_arm_j6", "L_arm_j7"],
}
HAND_JOINTS = {
    "R": ["R_th_j0", "R_th_j1", "R_th_j2",
          "R_ff_j1", "R_ff_j2", "R_mf_j1", "R_mf_j2",
          "R_rf_j1", "R_rf_j2", "R_lf_j1", "R_lf_j2"],
    "L": ["L_th_j0", "L_th_j1", "L_th_j2",
          "L_ff_j1", "L_ff_j2", "L_mf_j1", "L_mf_j2",
          "L_rf_j1", "L_rf_j2", "L_lf_j1", "L_lf_j2"],
}
# Torso/lift/head: frozen for tabletop tracking (kept as joints, no actuators
# unless requested). Listed so the env can hold them at a home pose.
POSTURE_JOINTS = ["Lift", "torso_flip", "head_j1", "head_j2", "head_j3"]

# Fingertip body names per hand — used for keypoint-based tracking reward.
# (Distal link of each finger in the f5d6 URDF.)
FINGERTIP_BODIES = {
    "R": ["R_th_l2", "R_ff_l2", "R_mf_l2", "R_rf_l2", "R_lf_l2"],
    "L": ["L_th_l2", "L_ff_l2", "L_mf_l2", "L_rf_l2", "L_lf_l2"],
}
# Wrist / end-effector reference frame per arm. The URDF's R_arm_l8 + hand base
# attach via fixed joints, so MuJoCo merges them into the last jointed link.
WRIST_BODY = {"R": "R_arm_l7", "L": "L_arm_l7"}


def controlled_joints(side: str) -> list[str]:
    """All actuated joints for one arm (arm + hand), in canonical order."""
    return ARM_JOINTS[side] + HAND_JOINTS[side]


# --- Control / actuator defaults --------------------------------------------
# Position actuators (PD). Arm joints are heavier / stiffer than fingers.
ARM_KP, ARM_KV = 600.0, 40.0
HAND_KP, HAND_KV = 8.0, 0.3
POSTURE_KP, POSTURE_KV = 800.0, 60.0

# Simulation: physics substeps per policy step (control decimation).
SIM_DT = 0.005          # MuJoCo timestep (s)
CONTROL_DECIMATION = 4  # policy acts at 50 Hz when SIM_DT=0.005

# DexTrack primary action space: cumulative residual position targets with a
# kinematic bias. Per-step residual is bounded to this fraction of each joint
# range so the policy nudges the kinematic reference rather than overriding it.
RESIDUAL_SCALE = 0.05
# Hard cap on the *cumulative* residual, as a fraction of each joint's range.
# The residual may only nudge the kinematic reference within this band — it
# cannot accumulate into a drift that walks the arms to their limits (which an
# unbounded residual did on the bimanual lift). DexTrack similarly bounds it.
RESIDUAL_CLIP = 0.12

# Home posture (radians / metres) for frozen joints; arms start at reference.
# NOTE: Lift is 0.0, not 0.1 — the torso-lift position actuator (POSTURE_KP)
# can't hold 0.1 against the full upper-body weight, so it sags to its lower
# limit (~0) during sim. Declaring 0.1 made open-loop playback force-set the
# torso 10cm higher than it actually sits, rendering the hand floating above
# the object while the (real, recorded) object stayed put. Keep this equal to
# where Lift physically settles so the kinematic reference is self-consistent.
HOME_POSTURE = {
    "Lift": 0.0, "torso_flip": 0.0,
    "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0,
}
