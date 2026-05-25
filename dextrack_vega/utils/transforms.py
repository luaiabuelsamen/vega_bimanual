"""Minimal quaternion / rotation helpers (numpy, MuJoCo wxyz convention).

MuJoCo stores quaternions as [w, x, y, z]. Everything here follows that order
to avoid silent bugs when reading qpos / body xquat.
"""
from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-12, None)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of wxyz quaternions (supports broadcasting)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Smallest rotation angle (rad) between two orientations."""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def quat_diff_rad(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Per-sample geodesic angle, vectorised over leading dims."""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternions."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:           # take the shorter arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:        # near-parallel: linear fallback
        return quat_normalize(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    theta = theta0 * t
    q2 = quat_normalize(q1 - q0 * dot)
    return q0 * np.cos(theta) + q2 * np.sin(theta)
