# Builds the configured sensor rig (cameras + LiDAR) on the ego vehicle

from __future__ import annotations

import logging

from beamngpy.sensors import Camera, Lidar

from .config import HarnessConfig
from .geometry import intrinsics_from_fov
from .session import HarnessSession

log = logging.getLogger(__name__)


class SensorRig:
    # Instantiated rig: holds live sensor handles plus their static calibration

    def __init__(self, cfg: HarnessConfig, session: HarnessSession):
        self.cfg = cfg
        self.session = session
        self.cameras: dict[str, Camera] = {}
        self.lidar: Lidar | None = None

    def attach(self) -> "SensorRig":
        bng, ego = self.session.bng, self.session.ego
        assert bng is not None and ego is not None, "scenario must be set up first"
        rig = self.cfg.rig
        update_time = 1.0 / self.cfg.capture.hz

        for cam in rig.cameras:
            log.info("attaching camera %s (%dx%d, fov_y=%.1f)", cam.name, *cam.resolution, cam.fov_y_deg)
            self.cameras[cam.name] = Camera(
                cam.name,
                bng,
                ego,
                requested_update_time=update_time,
                pos=tuple(cam.pos),
                dir=tuple(cam.dir),
                up=tuple(cam.up),
                resolution=tuple(cam.resolution),
                field_of_view_y=cam.fov_y_deg,
                near_far_planes=tuple(cam.near_far_planes),
                is_using_shared_memory=rig.shared_memory,
                is_render_colours=True,
                is_render_depth=cam.render_depth,
                is_render_annotations=cam.render_annotations,
                is_render_instance=cam.render_instance,
            )

        if rig.lidar is not None:
            li = rig.lidar
            log.info("attaching lidar %s (%d channels)", li.name, li.vertical_resolution)
            self.lidar = Lidar(
                li.name,
                bng,
                ego,
                requested_update_time=update_time,
                pos=tuple(li.pos),
                dir=tuple(li.dir),
                up=tuple(li.up),
                vertical_resolution=li.vertical_resolution,
                vertical_angle=li.vertical_angle,
                frequency=li.frequency,
                max_distance=li.max_distance,
                is_360_mode=li.is_360_mode,
                is_using_shared_memory=rig.shared_memory,
            )
        return self

    def calibration(self) -> dict:
        # Static calibration block written once per session (calib.json)
        calib: dict = {"cameras": {}, "lidar": None}
        for cam in self.cfg.rig.cameras:
            calib["cameras"][cam.name] = {
                "intrinsics": intrinsics_from_fov(cam.resolution[0], cam.resolution[1], cam.fov_y_deg),
                "near_far_planes": list(cam.near_far_planes),
                # Mount transform in BeamNG vehicle space (+X left, -Y forward, +Z up).
                "mount": {"pos": list(cam.pos), "dir": list(cam.dir), "up": list(cam.up)},
            }
        if self.cfg.rig.lidar is not None:
            li = self.cfg.rig.lidar
            calib["lidar"] = {
                "name": li.name,
                "mount": {"pos": list(li.pos), "dir": list(li.dir), "up": list(li.up)},
                "vertical_resolution": li.vertical_resolution,
                "vertical_angle": li.vertical_angle,
                "frequency": li.frequency,
                "max_distance": li.max_distance,
            }
        return calib

    def detach(self) -> None:
        for name, cam in self.cameras.items():
            try:
                cam.remove()
            except Exception:
                log.warning("failed to remove camera %s", name, exc_info=True)
        self.cameras.clear()
        if self.lidar is not None:
            try:
                self.lidar.remove()
            except Exception:
                log.warning("failed to remove lidar", exc_info=True)
            self.lidar = None
