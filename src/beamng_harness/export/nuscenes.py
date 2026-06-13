"""Export a recorded session to a nuScenes-style schema.

Generates the core nuScenes v1.0 relational tables (scene, sample, sample_data,
ego_pose, calibrated_sensor, sensor, sample_annotation, instance, category, log,
attribute, visibility, map) as JSON under ``v1.0-beamng/``, with camera JPEG/PNGs
under ``samples/<CHANNEL>/`` and LiDAR sweeps as nuScenes-format ``.pcd.bin``
(float32 x, y, z, intensity, ring — points in the *sensor* frame).

Scope notes (documented limitations, by design for Phase 1):
- every frame is a keyframe (no intermediate sweeps);
- a single ``vehicle.car`` category covers all dynamic actors;
- visibility is reported as the highest bin (ground truth, no occlusion model).
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path

import numpy as np

from .. import geometry as geo
from ..dataset import RecordedSession

log = logging.getLogger(__name__)


def _token() -> str:
    return uuid.uuid4().hex


def _write_table(root: Path, name: str, rows: list[dict]) -> None:
    (root / f"{name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


# Map harness camera names to nuScenes channel names where the rig matches.
DEFAULT_CHANNEL_MAP = {
    "cam_front": "CAM_FRONT",
    "cam_front_left": "CAM_FRONT_LEFT",
    "cam_front_right": "CAM_FRONT_RIGHT",
    "cam_back": "CAM_BACK",
    "cam_back_left": "CAM_BACK_LEFT",
    "cam_back_right": "CAM_BACK_RIGHT",
}


def export_nuscenes(session_dir: str | Path, out_dir: str | Path) -> Path:
    sess = RecordedSession(session_dir)
    out = Path(out_dir)
    table_root = out / "v1.0-beamng"
    table_root.mkdir(parents=True, exist_ok=True)

    channels = {
        name: DEFAULT_CHANNEL_MAP.get(name, name.upper()) for name in sess.camera_names
    }
    if sess.has_lidar():
        channels["__lidar__"] = "LIDAR_TOP"

    log_token, map_token, scene_token = _token(), _token(), _token()

    sensors, calibrated = [], []
    cs_tokens: dict[str, str] = {}
    for name in sess.camera_names:
        s_tok, cs_tok = _token(), _token()
        cs_tokens[name] = cs_tok
        k = sess.intrinsics(name)
        rot_v, pos_v = _mount_to_nusc(sess.calib["cameras"][name]["mount"], camera=True)
        sensors.append({"token": s_tok, "channel": channels[name], "modality": "camera"})
        calibrated.append(
            {
                "token": cs_tok,
                "sensor_token": s_tok,
                "translation": pos_v.tolist(),
                "rotation": geo.rotmat_to_quat(rot_v).tolist(),
                "camera_intrinsic": [
                    [k["fx"], 0.0, k["cx"]],
                    [0.0, k["fy"], k["cy"]],
                    [0.0, 0.0, 1.0],
                ],
            }
        )
    if sess.has_lidar():
        s_tok, cs_tok = _token(), _token()
        cs_tokens["__lidar__"] = cs_tok
        rot_v, pos_v = _mount_to_nusc(sess.calib["lidar"]["mount"], camera=False)
        sensors.append({"token": s_tok, "channel": "LIDAR_TOP", "modality": "lidar"})
        calibrated.append(
            {
                "token": cs_tok,
                "sensor_token": s_tok,
                "translation": pos_v.tolist(),
                "rotation": geo.rotmat_to_quat(rot_v).tolist(),
                "camera_intrinsic": [],
            }
        )

    samples, sample_datas, ego_poses, annotations = [], [], [], []
    instances: dict[str, dict] = {}
    category_token = _token()
    attribute_token = _token()
    visibility_token = "4"

    sample_tokens = [_token() for _ in sess.frames]
    for i, frame in enumerate(sess.frames):
        ts = int(frame["sim_time"] * 1e6)
        sample_token = sample_tokens[i]
        samples.append(
            {
                "token": sample_token,
                "timestamp": ts,
                "prev": sample_tokens[i - 1] if i > 0 else "",
                "next": sample_tokens[i + 1] if i < len(sess.frames) - 1 else "",
                "scene_token": scene_token,
            }
        )

        ego = sess.ego_pose(frame)
        # nuScenes ego frame is X forward, Y left, Z up.
        ego_rot = geo.nuscenes_box_rotation(frame["ego"]["dir"], frame["ego"]["up"])
        ego_pose_token = _token()
        ego_poses.append(
            {
                "token": ego_pose_token,
                "timestamp": ts,
                "translation": ego.pos.tolist(),
                "rotation": geo.rotmat_to_quat(ego_rot).tolist(),
            }
        )

        for name in sess.camera_names:
            src = sess.image_path(name, frame["frame"])
            if not src.exists():
                continue
            rel = Path("samples") / channels[name] / f"{frame['frame']:06d}.png"
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            k = sess.intrinsics(name)
            sample_datas.append(
                _sample_data_row(
                    sample_token, cs_tokens[name], ego_pose_token, ts, rel,
                    fileformat="png", is_key_frame=True,
                    width=k["width"], height=k["height"],
                )
            )

        if sess.has_lidar():
            try:
                points_world = sess.lidar_points(frame["frame"])
            except FileNotFoundError:
                points_world = None
            if points_world is not None:
                r_l2w, t_l = sess.lidar_world_pose(frame)
                pts_sensor = geo.world_points_to_frame(points_world, r_l2w, t_l)
                rel = Path("samples") / "LIDAR_TOP" / f"{frame['frame']:06d}.pcd.bin"
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                n = len(pts_sensor)
                buf = np.zeros((n, 5), dtype=np.float32)
                buf[:, :3] = pts_sensor
                buf[:, 3] = 1.0  # intensity placeholder (BeamNG returns geometry only)
                buf.tofile(dst)
                sample_datas.append(
                    _sample_data_row(
                        sample_token, cs_tokens["__lidar__"], ego_pose_token, ts, rel,
                        fileformat="pcd", is_key_frame=True, width=0, height=0,
                    )
                )

        for actor in frame.get("actors", []):
            box = RecordedSession.actor_box(actor)
            inst = instances.setdefault(
                actor["id"],
                {"token": _token(), "category_token": category_token, "anns": []},
            )
            ann_token = _token()
            inst["anns"].append(ann_token)
            annotations.append(
                {
                    "token": ann_token,
                    "sample_token": sample_token,
                    "instance_token": inst["token"],
                    "visibility_token": visibility_token,
                    "attribute_tokens": [attribute_token],
                    "translation": [float(x) for x in box["center"]],
                    "size": [float(x) for x in box["size"]],
                    "rotation": geo.rotmat_to_quat(box["rot"]).tolist(),
                    "num_lidar_pts": -1,
                    "num_radar_pts": 0,
                    "prev": "",
                    "next": "",
                }
            )

    _link_annotation_chains(annotations, instances)

    level = sess.metadata["config"]["scenario"]["level"]
    _write_table(table_root, "scene", [{
        "token": scene_token, "log_token": log_token, "nbr_samples": len(samples),
        "first_sample_token": sample_tokens[0] if sample_tokens else "",
        "last_sample_token": sample_tokens[-1] if sample_tokens else "",
        "name": f"beamng-{level}", "description": "Synthetic BeamNG.tech capture",
    }])
    _write_table(table_root, "sample", samples)
    _write_table(table_root, "sample_data", sample_datas)
    _write_table(table_root, "ego_pose", ego_poses)
    _write_table(table_root, "calibrated_sensor", calibrated)
    _write_table(table_root, "sensor", sensors)
    _write_table(table_root, "sample_annotation", annotations)
    _write_table(table_root, "instance", [
        {
            "token": v["token"], "category_token": v["category_token"],
            "nbr_annotations": len(v["anns"]),
            "first_annotation_token": v["anns"][0], "last_annotation_token": v["anns"][-1],
        }
        for v in instances.values()
    ])
    _write_table(table_root, "category", [{
        "token": category_token, "name": "vehicle.car",
        "description": "BeamNG traffic vehicle",
    }])
    _write_table(table_root, "attribute", [{
        "token": attribute_token, "name": "vehicle.moving", "description": "",
    }])
    _write_table(table_root, "visibility", [{
        "token": "4", "level": "v80-100", "description": "ground truth (no occlusion model)",
    }])
    _write_table(table_root, "log", [{
        "token": log_token, "logfile": "", "vehicle": "beamng-ego",
        "date_captured": sess.metadata["created"][:10], "location": level,
    }])
    _write_table(table_root, "map", [{
        "token": map_token, "log_tokens": [log_token], "category": "semantic_prior", "filename": "",
    }])

    log.info(
        "nuScenes-style export: %d samples, %d sample_data, %d annotations -> %s",
        len(samples), len(sample_datas), len(annotations), out,
    )
    return out


def _mount_to_nusc(mount: dict, camera: bool) -> tuple[np.ndarray, np.ndarray]:
    """Sensor-to-ego rotation/translation in the nuScenes ego frame (X fwd, Y left, Z up).

    Mounts are stored in BeamNG vehicle space; nuScenes expects the sensor pose
    relative to an X-forward ego frame, with cameras in OpenCV convention.
    """
    d_v = np.asarray(mount["dir"], dtype=np.float64)
    u_v = np.asarray(mount["up"], dtype=np.float64)
    p_v = np.asarray(mount["pos"], dtype=np.float64)

    # Rotation from BeamNG vehicle space to the nuScenes ego frame.
    # Rows: ego X = vehicle forward (0,-1,0); ego Y = vehicle left (+X); ego Z = up.
    r_bng_to_ego = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    d_e, u_e, p_e = r_bng_to_ego @ d_v, r_bng_to_ego @ u_v, r_bng_to_ego @ p_v
    if camera:
        rot = geo.camera_to_world_rotation(d_e, u_e)  # camera-to-ego, OpenCV axes
    else:
        rot = geo.nuscenes_box_rotation(d_e, u_e)  # lidar-to-ego, X-forward axes
    return rot, p_e


def _sample_data_row(sample_token, cs_token, ego_pose_token, ts, rel, *, fileformat, is_key_frame, width, height) -> dict:
    return {
        "token": _token(),
        "sample_token": sample_token,
        "ego_pose_token": ego_pose_token,
        "calibrated_sensor_token": cs_token,
        "timestamp": ts,
        "fileformat": fileformat,
        "is_key_frame": is_key_frame,
        "height": height,
        "width": width,
        "filename": rel.as_posix(),
        "prev": "",
        "next": "",
    }


def _link_annotation_chains(annotations: list[dict], instances: dict[str, dict]) -> None:
    by_token = {a["token"]: a for a in annotations}
    for inst in instances.values():
        anns = inst["anns"]
        for i, tok in enumerate(anns):
            if i > 0:
                by_token[tok]["prev"] = anns[i - 1]
            if i < len(anns) - 1:
                by_token[tok]["next"] = anns[i + 1]
