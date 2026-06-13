"""Synchronized capture loop and on-disk session format.

Session layout::

    <output_dir>/<session_name>/
        metadata.json            # config snapshot, frame count, dt
        calib.json               # per-camera intrinsics + mounts, lidar mount
        frames.jsonl             # one JSON record per frame: time, ego pose, actor boxes
        images/<cam>/<frame:06d>.png
        depth/<cam>/<frame:06d>.npy      # float32 meters (or raw values, see metadata)
        lidar/<frame:06d>.npz            # points: float32 Nx3 world-space

Synchronization model: physics runs deterministically and *paused*; each frame we
advance exactly ``steps_per_frame`` physics steps and then poll every sensor, so
all modalities in a frame correspond to the same simulation time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import HarnessConfig
from .rig import SensorRig
from .session import HarnessSession

log = logging.getLogger(__name__)


def _to_list(v):
    return [float(x) for x in v]


def _image_to_pil(data) -> Image.Image:
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return Image.fromarray(arr.astype(np.uint8))


def _depth_to_array(data, near_far: tuple[float, float]) -> tuple[np.ndarray, str]:
    """Normalize whatever the Camera returns for depth into float32 meters.

    beamngpy may hand back a float buffer of distances or an 8-bit visualization
    normalized over the near/far range; we detect which and record the source in
    metadata so downstream consumers know the precision.
    """
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint8:
        near, far = near_far
        meters = near + (arr.astype(np.float32) / 255.0) * (far - near)
        return meters, "uint8_normalized"
    return arr.astype(np.float32), "float_distance"


def _points_to_array(point_cloud) -> np.ndarray:
    arr = np.asarray(point_cloud, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    # The sensor pads its buffer with zeros up to the max return count.
    mask = ~np.all(arr == 0.0, axis=1)
    return arr[mask]


class Recorder:
    def __init__(self, cfg: HarnessConfig, session: HarnessSession, rig: SensorRig):
        self.cfg = cfg
        self.session = session
        self.rig = rig
        name = cfg.capture.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = Path(cfg.capture.output_dir) / name
        self._depth_encoding: str | None = None

    def record(self) -> Path:
        bng = self.session.bng
        assert bng is not None
        cap = self.cfg.capture
        dt = 1.0 / cap.hz
        steps_per_frame = self.cfg.steps_per_frame

        self._prepare_dirs()

        # Let the scenario settle (AI starts driving, traffic disperses) in real time,
        # then freeze the world and hand control of time to the capture loop.
        if cap.warmup_seconds > 0:
            log.info("warmup: %.1fs realtime", cap.warmup_seconds)
            time.sleep(cap.warmup_seconds)
        bng.control.pause()

        frames_path = self.out_dir / "frames.jsonl"
        t_start = time.monotonic()
        with frames_path.open("w", encoding="utf-8") as frames_file:
            for i in range(cap.num_frames):
                bng.control.step(steps_per_frame, wait=True)
                record = self._capture_frame(i, i * dt)
                frames_file.write(json.dumps(record) + "\n")
                if (i + 1) % 25 == 0 or i == cap.num_frames - 1:
                    rate = (i + 1) / (time.monotonic() - t_start)
                    log.info("frame %d/%d (%.1f frames/s wall)", i + 1, cap.num_frames, rate)

        bng.control.resume()
        self._write_metadata()
        log.info("session written to %s", self.out_dir)
        return self.out_dir

    def _prepare_dirs(self) -> None:
        for cam in self.rig.cameras:
            (self.out_dir / "images" / cam).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "depth" / cam).mkdir(parents=True, exist_ok=True)
        if self.rig.lidar is not None:
            (self.out_dir / "lidar").mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "calib.json").write_text(
            json.dumps(self.rig.calibration(), indent=2), encoding="utf-8"
        )

    def _capture_frame(self, idx: int, sim_time: float) -> dict:
        ego = self.session.ego
        assert ego is not None
        stem = f"{idx:06d}"

        ego.sensors.poll()
        record: dict = {
            "frame": idx,
            "sim_time": sim_time,
            "ego": {
                "pos": _to_list(ego.state["pos"]),
                "dir": _to_list(ego.state["dir"]),
                "up": _to_list(ego.state["up"]),
                "vel": _to_list(ego.state["vel"]),
            },
            "actors": [],
        }

        cam_cfgs = {c.name: c for c in self.cfg.rig.cameras}
        for name, camera in self.rig.cameras.items():
            data = camera.poll()
            _image_to_pil(data["colour"]).save(self.out_dir / "images" / name / f"{stem}.png")
            if cam_cfgs[name].render_depth and data.get("depth") is not None:
                depth, encoding = _depth_to_array(data["depth"], cam_cfgs[name].near_far_planes)
                np.save(self.out_dir / "depth" / name / f"{stem}.npy", depth)
                self._depth_encoding = encoding

        if self.rig.lidar is not None:
            readings = self.rig.lidar.poll()
            points = _points_to_array(readings["pointCloud"])
            np.savez_compressed(self.out_dir / "lidar" / f"{stem}.npz", points=points)
            record["lidar_points"] = int(points.shape[0])

        for vid, actor in self.session.actors.items():
            try:
                actor.sensors.poll()
                bbox = actor.get_bbox()
            except Exception:
                log.debug("actor %s poll failed on frame %d", vid, idx, exc_info=True)
                continue
            record["actors"].append(
                {
                    "id": vid,
                    "pos": _to_list(actor.state["pos"]),
                    "dir": _to_list(actor.state["dir"]),
                    "up": _to_list(actor.state["up"]),
                    "vel": _to_list(actor.state["vel"]),
                    "bbox_corners": {k: _to_list(v) for k, v in bbox.items()},
                }
            )
        return record

    def _write_metadata(self) -> None:
        meta = {
            "created": datetime.now().isoformat(),
            "config": asdict(self.cfg),
            "num_frames": self.cfg.capture.num_frames,
            "dt": 1.0 / self.cfg.capture.hz,
            "depth_encoding": self._depth_encoding,
            "conventions": {
                "world": "BeamNG right-handed, Z up; positions in meters",
                "lidar_points": "world-space XYZ",
                "vehicle_space": "+X left, -Y forward, +Z up (sensor mounts)",
            },
        }
        (self.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
