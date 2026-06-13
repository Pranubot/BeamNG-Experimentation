"""Replay a recorded session for inspection.

Prefers rerun.io (interactive 3D timeline: cameras, depth, world-space LiDAR, ego
trajectory, actor boxes). Falls back to a matplotlib viewer (camera grid + BEV
LiDAR scatter, arrow keys to step frames) when rerun is not installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..dataset import RecordedSession
from ..geometry import rotmat_to_quat

log = logging.getLogger(__name__)


def replay(session_dir: str | Path, backend: str = "auto") -> None:
    sess = RecordedSession(session_dir)
    if backend in ("auto", "rerun"):
        try:
            _replay_rerun(sess)
            return
        except ImportError:
            if backend == "rerun":
                raise
            log.info("rerun-sdk not installed, falling back to matplotlib")
    _replay_matplotlib(sess)


def _replay_rerun(sess: RecordedSession) -> None:
    import rerun as rr

    rr.init("beamng_harness_replay", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    trajectory = []
    for frame in sess.frames:
        idx = frame["frame"]
        rr.set_time("sim_time", duration=frame["sim_time"])

        ego = sess.ego_pose(frame)
        trajectory.append(ego.pos)
        rr.log("world/ego", rr.Transform3D(translation=ego.pos, mat3x3=ego.rot))
        rr.log("world/trajectory", rr.LineStrips3D([np.array(trajectory)]))

        for cam in sess.camera_names:
            img_path = sess.image_path(cam, idx)
            if img_path.exists():
                rr.log(f"cameras/{cam}", rr.Image(np.asarray(Image.open(img_path))))
            depth_path = sess.depth_path(cam, idx)
            if depth_path.exists():
                rr.log(f"depth/{cam}", rr.DepthImage(np.load(depth_path), meter=1.0))

        if sess.has_lidar():
            try:
                pts = sess.lidar_points(idx)
                rr.log("world/lidar", rr.Points3D(pts, radii=0.03))
            except FileNotFoundError:
                pass

        centers, half_sizes, quats = [], [], []
        for actor in frame.get("actors", []):
            box = RecordedSession.actor_box(actor)
            w, l, h = box["size"]
            centers.append(box["center"])
            half_sizes.append([l / 2, w / 2, h / 2])  # rerun boxes: x/y/z half-extent in box frame
            qw, qx, qy, qz = rotmat_to_quat(box["rot"])
            quats.append([qx, qy, qz, qw])
        if centers:
            rr.log(
                "world/actors",
                rr.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    quaternions=[rr.Quaternion(xyzw=q) for q in quats],
                ),
            )


def _replay_matplotlib(sess: RecordedSession) -> None:
    import matplotlib.pyplot as plt

    cams = sess.camera_names
    n = len(cams)
    cols = min(n, 3) if n else 1
    rows = (n + cols - 1) // cols if n else 1
    extra = 1 if sess.has_lidar() else 0

    fig = plt.figure(figsize=(5 * (cols + extra), 4 * rows))
    state = {"i": 0}

    def draw():
        fig.clf()
        frame = sess.frames[state["i"]]
        idx = frame["frame"]
        for j, cam in enumerate(cams):
            ax = fig.add_subplot(rows, cols + extra, j + 1 + (j // cols) * extra)
            p = sess.image_path(cam, idx)
            if p.exists():
                ax.imshow(Image.open(p))
            ax.set_title(cam, fontsize=9)
            ax.axis("off")
        if sess.has_lidar():
            ax = fig.add_subplot(1, cols + extra, cols + extra)
            try:
                pts = sess.lidar_points(idx)
                ego = sess.ego_pose(frame)
                local = (pts - ego.pos) @ ego.rot
                ax.scatter(-local[:, 0], -local[:, 1], s=0.3, c=local[:, 2], cmap="viridis")
            except FileNotFoundError:
                pass
            ax.set_title("LiDAR BEV (ego frame)", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
        fig.suptitle(
            f"frame {idx}/{sess.frames[-1]['frame']}  t={frame['sim_time']:.2f}s   "
            "(←/→ to step, Home/End to jump)"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["i"] = min(state["i"] + 1, len(sess.frames) - 1)
        elif event.key in ("left", "a"):
            state["i"] = max(state["i"] - 1, 0)
        elif event.key == "home":
            state["i"] = 0
        elif event.key == "end":
            state["i"] = len(sess.frames) - 1
        else:
            return
        draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
