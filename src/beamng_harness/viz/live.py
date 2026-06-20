# Real-time LiDAR bird's-eye-view monitor streamed to rerun while you drive.
#
# Unlike the recorder this never pauses physics and never touches disk: the sim
# free-runs at wall-clock speed, you drive in the BeamNG window, and each tick we
# poll the LiDAR and stream it to a rerun viewer (3D world cloud + ego frame +
# trajectory trail + a 2D top-down BEV panel).

from __future__ import annotations

import logging
import os
import shutil
import sysconfig
import time
from collections import deque
from pathlib import Path

import numpy as np

from .. import geometry as geo
from ..rig import SensorRig
from ..session import HarnessSession

log = logging.getLogger(__name__)


def _ensure_viewer_on_path() -> None:
    """rerun's spawn() needs the viewer executable on PATH, but pip installs it into
    a Scripts directory that often isn't on PATH (same as our console scripts). Find
    it and prepend so ``live`` works without the user editing their environment."""
    if shutil.which("rerun"):
        return
    exe_name = "rerun.exe" if os.name == "nt" else "rerun"
    for scheme_dir in (sysconfig.get_path("scripts"),
                       sysconfig.get_path("scripts", os.name + "_user")):
        if scheme_dir and (Path(scheme_dir) / exe_name).exists():
            os.environ["PATH"] = scheme_dir + os.pathsep + os.environ.get("PATH", "")
            log.debug("added rerun viewer dir to PATH: %s", scheme_dir)
            return
    log.warning("could not locate the rerun viewer executable; spawn may fail")


def _live_points(point_cloud) -> np.ndarray:
    """LiDAR returns a flat buffer padded with zero rows; drop the padding."""
    arr = np.asarray(point_cloud, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    mask = ~np.all(arr == 0.0, axis=1)
    return arr[mask]


def _height_colors(z: np.ndarray) -> np.ndarray:
    """Jet-style colormap over the 2nd-98th height percentile, as uint8 RGB."""
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def live_view(
    session: HarnessSession,
    rig: SensorRig,
    hz: float = 15.0,
    view_range: float | None = None,
    trail: int = 600,
    max_seconds: float | None = None,
) -> None:
    import rerun as rr

    if rig.lidar is None:
        raise ValueError("live view requires a LiDAR in the rig")
    ego = session.ego
    assert ego is not None

    _ensure_viewer_on_path()
    rr.init("beamng_live_bev", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    trajectory: deque = deque(maxlen=trail)
    dt_target = 1.0 / hz
    t0 = time.monotonic()
    frame = 0
    log.info(
        "live BEV running ~%.0f Hz -- drive in the BeamNG window; Ctrl+C here to stop", hz
    )

    try:
        while True:
            loop_start = time.monotonic()
            rr.set_time("elapsed", duration=loop_start - t0)

            # Ego pose (real-time, not stepped).
            ego.sensors.poll()
            pos = np.asarray(ego.state["pos"], dtype=np.float64)
            rot = geo.frame_to_world(ego.state["dir"], ego.state["up"])  # vehicle->world

            points = _live_points(rig.lidar.poll()["pointCloud"])
            if view_range is not None and len(points):
                points = points[np.linalg.norm(points - pos, axis=1) <= view_range]
            colors = _height_colors(points[:, 2]) if len(points) else None

            # 3D world view: cloud + moving ego frame + trajectory trail.
            rr.log("world/lidar", rr.Points3D(points, colors=colors, radii=0.04))
            rr.log("world/ego", rr.Transform3D(translation=pos, mat3x3=rot))
            trajectory.append(pos)
            if len(trajectory) > 1:
                rr.log("world/trajectory", rr.LineStrips3D([np.array(trajectory)]))

            # 2D bird's-eye panel in the ego frame (same transform as replay.py).
            if len(points):
                local = (points - pos) @ rot                      # world -> vehicle space
                bev = np.stack([-local[:, 0], -local[:, 1]], axis=1)
                rr.log("bev/points", rr.Points2D(bev, colors=colors, radii=0.15))
                rr.log("bev/ego", rr.Points2D([[0.0, 0.0]], colors=[[255, 255, 255]], radii=0.6))

            frame += 1
            if frame % 30 == 0:
                log.info(
                    "frame %d | %d points | %.1f Hz",
                    frame, len(points), frame / (time.monotonic() - t0),
                )

            if max_seconds is not None and (time.monotonic() - t0) >= max_seconds:
                log.info("reached --seconds limit (%.0fs)", max_seconds)
                break

            elapsed = time.monotonic() - loop_start
            if elapsed < dt_target:
                time.sleep(dt_target - elapsed)
    except KeyboardInterrupt:
        log.info("live view stopped after %d frames", frame)
