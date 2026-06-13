# Read-side API for recorded sessions, shared by exporters and the replay viewer

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as geo


@dataclass
class FramePose:
    # Ego (or actor) pose: rotation maps BeamNG vehicle space to world.

    rot: np.ndarray  # 3x3, vehicle-to-world
    pos: np.ndarray  # 3, world


class RecordedSession:
    def __init__(self, session_dir: str | Path):
        self.dir = Path(session_dir)
        self.metadata = json.loads((self.dir / "metadata.json").read_text(encoding="utf-8"))
        self.calib = json.loads((self.dir / "calib.json").read_text(encoding="utf-8"))
        with (self.dir / "frames.jsonl").open(encoding="utf-8") as f:
            self.frames = [json.loads(line) for line in f if line.strip()]

    @property
    def camera_names(self) -> list[str]:
        return list(self.calib["cameras"].keys())

    def intrinsics(self, cam: str) -> dict:
        return self.calib["cameras"][cam]["intrinsics"]

    def image_path(self, cam: str, frame_idx: int) -> Path:
        return self.dir / "images" / cam / f"{frame_idx:06d}.png"

    def depth_path(self, cam: str, frame_idx: int) -> Path:
        return self.dir / "depth" / cam / f"{frame_idx:06d}.npy"

    def lidar_points(self, frame_idx: int) -> np.ndarray:
        return np.load(self.dir / "lidar" / f"{frame_idx:06d}.npz")["points"]

    def has_lidar(self) -> bool:
        return self.calib.get("lidar") is not None

    def ego_pose(self, frame: dict) -> FramePose:
        ego = frame["ego"]
        rot = geo.frame_to_world(ego["dir"], ego["up"])
        return FramePose(rot=rot, pos=np.asarray(ego["pos"], dtype=np.float64))

    def _mount_in_vehicle(self, mount: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Mount direction/up/pos vectors (vehicle space) as numpy arrays.
        return (
            np.asarray(mount["dir"], dtype=np.float64),
            np.asarray(mount["up"], dtype=np.float64),
            np.asarray(mount["pos"], dtype=np.float64),
        )

    def camera_world_pose(self, cam: str, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        # Camera-to-world (R, C) in OpenCV camera convention (X right, Y down, Z fwd)
        ego = self.ego_pose(frame)
        d_v, u_v, p_v = self._mount_in_vehicle(self.calib["cameras"][cam]["mount"])
        fwd_w = ego.rot @ d_v
        up_w = ego.rot @ u_v
        center = ego.pos + ego.rot @ p_v
        return geo.camera_to_world_rotation(fwd_w, up_w), center

    def lidar_world_pose(self, frame: dict) -> tuple[np.ndarray, np.ndarray]:
        # LiDAR frame-to-world (R, t); sensor frame is X forward, Y left, Z up
        ego = self.ego_pose(frame)
        d_v, u_v, p_v = self._mount_in_vehicle(self.calib["lidar"]["mount"])
        fwd_w = ego.rot @ d_v
        up_w = ego.rot @ u_v
        origin = ego.pos + ego.rot @ p_v
        return geo.nuscenes_box_rotation(fwd_w, up_w), origin

    @staticmethod
    def actor_box(actor: dict) -> dict:
        # Returns center (3,), size (w, l, h) and a box-to-world rotation built from the actor's dir/up vectors (nuScenes box frame: X fwd, Y left, Z up).
        corners = np.array(list(actor["bbox_corners"].values()), dtype=np.float64)
        center = corners.mean(axis=0)
        rot = geo.nuscenes_box_rotation(actor["dir"], actor["up"])
        local = (corners - center) @ rot  # corners in box frame
        extents = local.max(axis=0) - local.min(axis=0)
        length, width, height = extents[0], extents[1], extents[2]
        return {"center": center, "size": (width, length, height), "rot": rot}
