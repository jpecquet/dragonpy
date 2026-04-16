"""
Quaternion helpers. Scalar-first convention: q = (w, x, y, z).
All quaternions represent body-to-world rotations unless noted.
"""

import numpy as np


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Return R (3x3) such that v_world = R @ v_body."""
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """dq/dt for body-to-world quaternion under body-frame angular velocity."""
    wx, wy, wz = omega_body
    w, x, y, z = q
    return 0.5 * np.array([
        -x * wx - y * wy - z * wz,
         w * wx + y * wz - z * wy,
         w * wy - x * wz + z * wx,
         w * wz + x * wy - y * wx,
    ])
