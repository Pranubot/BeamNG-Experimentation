# Typed configuration for the sensor harness, loaded from YAML session files 

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BeamNGConfig:
    home: str | None = None
    user: str | None = None
    host: str = "localhost"
    port: int = 25252
    launch: bool = True


@dataclass
class SpawnConfig:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)


@dataclass
class ScenarioConfig:
    level: str = "west_coast_usa"
    name: str = "harness_capture"
    vehicle_model: str = "etk800"
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    traffic: int = 0
    ai_mode: str = "span"  # span | random | manual | disabled
    ai_speed_kph: float | None = 40.0


@dataclass
class CameraConfig:
    name: str
    pos: tuple[float, float, float] = (0.0, -1.6, 1.2)  # vehicle space: +X left, -Y fwd, +Z up
    dir: tuple[float, float, float] = (0.0, -1.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    resolution: tuple[int, int] = (1280, 720)
    fov_y_deg: float = 70.0
    near_far_planes: tuple[float, float] = (0.05, 300.0)
    render_depth: bool = True
    render_annotations: bool = False
    render_instance: bool = False


@dataclass
class LidarConfig:
    name: str = "lidar_top"
    pos: tuple[float, float, float] = (0.0, 0.0, 2.0)
    dir: tuple[float, float, float] = (0.0, -1.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    vertical_resolution: int = 64
    vertical_angle: float = 26.9
    frequency: float = 20.0
    max_distance: float = 120.0
    is_360_mode: bool = True


@dataclass
class RigConfig:
    cameras: list[CameraConfig] = field(default_factory=list)
    lidar: LidarConfig | None = None
    shared_memory: bool = True


@dataclass
class CaptureConfig:
    hz: float = 10.0
    steps_per_second: int = 60
    num_frames: int = 100
    warmup_seconds: float = 2.0
    output_dir: str = "data/sessions"
    session_name: str | None = None


@dataclass
class HarnessConfig:
    beamng: BeamNGConfig = field(default_factory=BeamNGConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    rig: RigConfig = field(default_factory=RigConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    @property
    def steps_per_frame(self) -> int:
        steps = round(self.capture.steps_per_second / self.capture.hz)
        if steps < 1 or abs(steps * self.capture.hz - self.capture.steps_per_second) > 1e-6:
            raise ValueError(
                f"capture.hz ({self.capture.hz}) must evenly divide "
                f"capture.steps_per_second ({self.capture.steps_per_second})"
            )
        return steps


def _tup(value, n: int) -> tuple:
    value = tuple(value)
    if len(value) != n:
        raise ValueError(f"expected {n} values, got {value}")
    return value


def load_config(path: str | Path) -> HarnessConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    bng = BeamNGConfig(**raw.get("beamng", {}))

    sc_raw = dict(raw.get("scenario", {}))
    spawn_raw = sc_raw.pop("spawn", {})
    spawn = SpawnConfig(
        pos=_tup(spawn_raw.get("pos", (0, 0, 0)), 3),
        rot_quat=_tup(spawn_raw.get("rot_quat", (0, 0, 0, 1)), 4),
    )
    scenario = ScenarioConfig(spawn=spawn, **sc_raw)

    rig_raw = dict(raw.get("rig", {}))
    cameras = []
    for cam_raw in rig_raw.pop("cameras", []):
        cam_raw = dict(cam_raw)
        for key, n in (("pos", 3), ("dir", 3), ("up", 3), ("near_far_planes", 2), ("resolution", 2)):
            if key in cam_raw:
                cam_raw[key] = _tup(cam_raw[key], n)
        cameras.append(CameraConfig(**cam_raw))
    lidar_raw = rig_raw.pop("lidar", None)
    lidar = None
    if lidar_raw is not None:
        lidar_raw = dict(lidar_raw)
        for key in ("pos", "dir", "up"):
            if key in lidar_raw:
                lidar_raw[key] = _tup(lidar_raw[key], 3)
        lidar = LidarConfig(**lidar_raw)
    rig = RigConfig(cameras=cameras, lidar=lidar, **rig_raw)

    capture = CaptureConfig(**raw.get("capture", {}))

    cfg = HarnessConfig(beamng=bng, scenario=scenario, rig=rig, capture=capture)
    cfg.steps_per_frame  
    if not cfg.rig.cameras and cfg.rig.lidar is None:
        raise ValueError("rig must define at least one camera or a lidar")
    return cfg
