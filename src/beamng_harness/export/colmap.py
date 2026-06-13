"""Export a recorded session to COLMAP text format (cameras.txt / images.txt / points3D.txt).

Ground-truth poses come straight from the simulator, so the usual COLMAP SfM step
is skipped entirely — the output is ready for gsplat / Nerfstudio ingestion
(``ns-train ... colmap`` style loaders). Images are copied into ``images/`` next
to the ``sparse/0`` model so the export directory is self-contained.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

from .. import geometry as geo
from ..dataset import RecordedSession

log = logging.getLogger(__name__)


def export_colmap(
    session_dir: str | Path,
    out_dir: str | Path,
    cameras: list[str] | None = None,
    frame_stride: int = 1,
) -> Path:
    sess = RecordedSession(session_dir)
    out = Path(out_dir)
    sparse = out / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    img_root = out / "images"

    cam_names = cameras or sess.camera_names
    cam_ids = {name: i + 1 for i, name in enumerate(cam_names)}

    with (sparse / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for name in cam_names:
            k = sess.intrinsics(name)
            f.write(
                f"{cam_ids[name]} PINHOLE {k['width']} {k['height']} "
                f"{k['fx']:.10f} {k['fy']:.10f} {k['cx']:.10f} {k['cy']:.10f}\n"
            )

    image_id = 0
    with (sparse / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for frame in sess.frames[::frame_stride]:
            idx = frame["frame"]
            for name in cam_names:
                src = sess.image_path(name, idx)
                if not src.exists():
                    log.warning("missing image %s; skipping", src)
                    continue
                rel = Path(name) / src.name
                dst = img_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

                r_c2w, center = sess.camera_world_pose(name, frame)
                r_w2c = r_c2w.T
                t = -r_w2c @ center
                qw, qx, qy, qz = geo.rotmat_to_quat(r_w2c)
                image_id += 1
                f.write(
                    f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {cam_ids[name]} {rel.as_posix()}\n\n"
                )

    # No SfM points — pipelines that need a seed cloud can use the LiDAR export
    # or random initialization (gsplat supports both).
    (sparse / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        encoding="utf-8",
    )

    if sess.has_lidar():
        _write_lidar_seed_points(sess, sparse, frame_stride)

    log.info("COLMAP export: %d images across %d cameras -> %s", image_id, len(cam_names), out)
    return out


def _write_lidar_seed_points(sess: RecordedSession, sparse: Path, frame_stride: int) -> None:
    """Aggregate a subsampled world-space LiDAR cloud as points3D.ply (gsplat seed)."""
    clouds = []
    for frame in sess.frames[:: max(frame_stride * 5, 5)]:
        try:
            pts = sess.lidar_points(frame["frame"])
        except FileNotFoundError:
            continue
        if len(pts) > 4000:
            pts = pts[np.random.default_rng(frame["frame"]).choice(len(pts), 4000, replace=False)]
        clouds.append(pts)
    if not clouds:
        return
    cloud = np.concatenate(clouds).astype(np.float32)
    ply = sparse / "points3D.ply"
    with ply.open("wb") as f:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(cloud)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        grey = np.full((len(cloud), 3), 128, dtype=np.uint8)
        rec = np.empty(len(cloud), dtype=[("xyz", np.float32, 3), ("rgb", np.uint8, 3)])
        rec["xyz"], rec["rgb"] = cloud, grey
        f.write(rec.tobytes())
    log.info("wrote %d LiDAR seed points to %s", len(cloud), ply)
