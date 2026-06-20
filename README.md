# Autonomous Driving and Visualization in BeamNG Tech

Part 1 of an autonomous-driving perception & scene-reconstruction project.
A typed Python harness over [BeamNG.tech](https://beamng.tech) that records
synchronized multi-camera RGB, LiDAR point clouds, depth maps, 6-DoF ego pose and
ground-truth 3D bounding boxes, and exports to COLMAP (for 3DGS pipelines) and a
nuScenes-style schema (for BEV perception).

## How synchronization works

BeamNG sensors have different native rates, so the harness never records in free-running
realtime. Instead it puts physics in deterministic mode, pauses the simulation, and
per frame advances exactly `steps_per_second / hz` physics steps before polling every
sensor. Every modality in a frame therefore corresponds to the same simulation time,
no interpolation or drift.

## Install

```powershell
cd beamng-sensor-harness
pip install -e ".[viz,dev]"   # quote the brackets in PowerShell
```

Requirements: BeamNG.tech (research license) + `beamngpy >= 1.35`. Set the install path
in your session YAML (`beamng.home`).

## Quick start

The module form below works with no PATH setup. Run it from the `beamng-sensor-harness`
directory (or anywhere, since the package is installed).

```powershell
# 1. Check the environment (does not launch the game)
python -m beamng_harness.cli doctor --config configs/session_starter.yaml

# 2. Record 200 frames (20 s at 10 Hz): 3 cameras + LiDAR + traffic.
#    BeamNG.tech launches itself; do not interact with the game window.
python -m beamng_harness.cli record --config configs/session_starter.yaml

# 3. Inspect the session (rerun.io if installed, else matplotlib)
python -m beamng_harness.cli replay data/sessions/<name>

# 4. Export
python -m beamng_harness.cli export-colmap   data/sessions/<name>
python -m beamng_harness.cli export-nuscenes data/sessions/<name>
```

`pip install` also creates a `beamng-harness` console script. If its Scripts directory
is on your PATH, you can use `beamng-harness …` in place of `python -m beamng_harness.cli …`.
The global `-v/--verbose` flag, if used, must come before the subcommand
(`... cli -v record ...`). Each `record` run prints the output folder name to use as
`<name>` in steps 3–4 (or pass `--session-name`).

## Live LiDAR BEV (real-time)

Watch the LiDAR as a live bird's-eye view in [rerun](https://rerun.io) while you drive,
with no capture or disk writes:

```powershell
python -m beamng_harness.cli live --config configs/session_starter.yaml --hz 15 --range 80
```

This free-runs the sim (no deterministic stepping), attaches the LiDAR only (cameras
are skipped to keep the framerate up), and streams a 3D world cloud + ego frame +
trajectory trail + a 2D top-down BEV panel. You drive in the BeamNG window; press
`Ctrl+C` in the terminal to stop. Requires `pip install rerun-sdk` and a `lidar` block
in the config.

Flags: `--ai` lets the AI drive instead of you; `--traffic N` sets the number of other
vehicles (overrides the config, `0` = none); `--hz` the refresh rate; `--range N` clips
points beyond N meters; `--seconds N` auto-stops after N seconds. If it's heavy, lower
`--hz`, tighten `--range`, or reduce the LiDAR `vertical_resolution` in the config.

## Session format

```
data/sessions/<name>/
  metadata.json            config snapshot, dt, conventions, depth encoding
  calib.json               per-camera intrinsics + mounts, lidar mount
  frames.jsonl             per frame: sim_time, ego pose, actor GT boxes
  images/<cam>/000000.png  RGB
  depth/<cam>/000000.npy   float32 depth
  lidar/000000.npz         world-space XYZ point cloud
```

## Coordinate conventions

| Frame | Convention |
|---|---|
| BeamNG world | right-handed, Z up, meters |
| BeamNG vehicle space (sensor mounts) | **+X left, −Y forward, +Z up** |
| COLMAP export | OpenCV camera (X right, Y down, Z forward), `images.txt` stores world-to-camera |
| nuScenes export | ego/box frames X forward, Y left, Z up; LiDAR `.pcd.bin` in sensor frame |

All conversions live in [`geometry.py`](src/beamng_harness/geometry.py) and are unit-tested.

## Configuring a rig

Rigs are plain YAML — see [`configs/session_starter.yaml`](configs/session_starter.yaml)
(3 front cameras + LiDAR) and
[`configs/session_6cam_nuscenes.yaml`](configs/session_6cam_nuscenes.yaml)
(nuScenes-matching 6-camera layout whose names map to `CAM_FRONT`, `CAM_FRONT_LEFT`, …
channels on export). Cameras are pinhole with intrinsics derived from `fov_y_deg`;
near/far planes, resolution, depth/annotation/instance passes are all per-camera.

## Tests

```powershell
python -m pytest          # geometry + export tests run without the simulator
```

Validated end-to-end against BeamNG.tech v0.38.3.0 (beamngpy 1.35, Python 3.11).
