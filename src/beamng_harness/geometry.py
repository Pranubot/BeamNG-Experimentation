# Rotation and pose utilities for converting between coordinate conventions.

from __future__ import annotations

import numpy as np

# BeamNG vehicle-space basis (forward, up) used for sensor mounts.
VEHICLE_FORWARD = np.array([0.0, -1.0, 0.0])
VEHICLE_UP = np.array([0.0, 0.0, 1.0])


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"cannot normalize near-zero vector {v}")
    return v / n


def orthonormal_basis(forward, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = normalize(forward)
    u = np.asarray(up, dtype=np.float64)
    u = u - np.dot(u, f) * f
    u = normalize(u)
    r = np.cross(f, u)
    return r, f, u


def frame_to_world(forward_w, up_w, forward_l=VEHICLE_FORWARD, up_l=VEHICLE_UP) -> np.ndarray:
    rw, fw, uw = orthonormal_basis(forward_w, up_w)
    rl, fl, ul = orthonormal_basis(forward_l, up_l)
    basis_w = np.column_stack([rw, fw, uw])
    basis_l = np.column_stack([rl, fl, ul])
    return basis_w @ basis_l.T


def camera_to_world_rotation(forward_w, up_w) -> np.ndarray:
    r, f, u = orthonormal_basis(forward_w, up_w)
    return np.column_stack([r, -u, f])


def nuscenes_box_rotation(forward_w, up_w) -> np.ndarray:
    r, f, u = orthonormal_basis(forward_w, up_w)
    return np.column_stack([f, -r, u])


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_to_rotmat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def compose_pose(r_parent: np.ndarray, t_parent, r_child: np.ndarray, t_child):
    t_parent = np.asarray(t_parent, dtype=np.float64)
    t_child = np.asarray(t_child, dtype=np.float64)
    return r_parent @ r_child, t_parent + r_parent @ t_child


def world_points_to_frame(points: np.ndarray, r_frame: np.ndarray, t_frame) -> np.ndarray:
    t_frame = np.asarray(t_frame, dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - t_frame) @ r_frame


def intrinsics_from_fov(width: int, height: int, fov_y_deg: float) -> dict:
    fy = height / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
    return {
        "fx": fy,
        "fy": fy,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
        "fov_y_deg": fov_y_deg,
    }
