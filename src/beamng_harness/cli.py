# Command-line interface: record, export-colmap, export-nuscenes, replay, doctor

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_record(args) -> int:
    from .config import load_config
    from .recorder import Recorder
    from .rig import SensorRig
    from .session import HarnessSession

    cfg = load_config(args.config)
    if args.frames is not None:
        cfg.capture.num_frames = args.frames
    if args.session_name is not None:
        cfg.capture.session_name = args.session_name

    with HarnessSession(cfg) as session:
        session.setup_scenario()
        rig = SensorRig(cfg, session).attach()
        try:
            out = Recorder(cfg, session, rig).record()
        finally:
            rig.detach()
    print(f"Recorded session: {out}")
    return 0


def cmd_export_colmap(args) -> int:
    from .export.colmap import export_colmap

    out = export_colmap(
        args.session,
        args.out or Path(args.session) / "export_colmap",
        cameras=args.cameras.split(",") if args.cameras else None,
        frame_stride=args.stride,
    )
    print(f"COLMAP export: {out}")
    return 0


def cmd_export_nuscenes(args) -> int:
    from .export.nuscenes import export_nuscenes

    out = export_nuscenes(args.session, args.out or Path(args.session) / "export_nuscenes")
    print(f"nuScenes-style export: {out}")
    return 0


def cmd_replay(args) -> int:
    from .viz.replay import replay

    replay(args.session, backend=args.backend)
    return 0


def cmd_live(args) -> int:
    from .config import load_config
    from .rig import SensorRig
    from .session import HarnessSession
    from .viz.live import live_view

    cfg = load_config(args.config)
    if cfg.rig.lidar is None:
        print("error: live view requires a 'lidar' block in the rig config")
        return 1
    if not args.ai:
        cfg.scenario.ai_mode = "manual"  # you drive, "--ai" hands it to BeamNG's built-in AI
    if args.traffic is not None:  # override the config's traffic setting
        if str(args.traffic).lower() == "auto":
            cfg.scenario.traffic = "auto"
        else:
            try:
                cfg.scenario.traffic = int(args.traffic)
            except ValueError:
                print(f"error: --traffic must be a number or 'auto', got {args.traffic!r}")
                return 1

    with HarnessSession(cfg) as session:
        session.setup_scenario(deterministic=False)  # real-time, not stepped
        rig = SensorRig(cfg, session).attach(cameras=False)  # LiDAR only
        try:
            live_view(session, rig, hz=args.hz, view_range=args.range, max_seconds=args.seconds)
        finally:
            rig.detach()
    return 0


def cmd_doctor(args) -> int:
    """Sanity-check the environment without recording anything."""
    import importlib

    ok = True
    for mod, required in (("beamngpy", True), ("numpy", True), ("yaml", True),
                          ("PIL", True), ("rerun", False), ("matplotlib", False)):
        try:
            m = importlib.import_module(mod)
            print(f"  [ok] {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            ok &= not required
            print(f"  [{'MISSING' if required else 'optional'}] {mod}")

    if args.config:
        from .config import load_config

        try:
            cfg = load_config(args.config)
            print(f"  [ok] config loads ({len(cfg.rig.cameras)} cameras, "
                  f"lidar={'yes' if cfg.rig.lidar else 'no'}, {cfg.capture.hz} Hz)")
            home = cfg.beamng.home
            if home and not Path(home).exists():
                print(f"  [MISSING] beamng.home does not exist: {home}")
                ok = False
            elif home:
                print(f"  [ok] beamng.home: {home}")
        except Exception as e:
            print(f"  [ERROR] config: {e}")
            ok = False
    print("Environment looks good." if ok else "Problems found, see above.")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="beamng-harness",
                                     description="BeamNG.tech multi-modal sensor harness")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record", help="run a capture session")
    p.add_argument("--config", required=True, help="session YAML")
    p.add_argument("--frames", type=int, help="override capture.num_frames")
    p.add_argument("--session-name", help="override capture.session_name")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("export-colmap", help="export a session to COLMAP text format")
    p.add_argument("session", help="recorded session directory")
    p.add_argument("--out", help="output directory")
    p.add_argument("--cameras", help="comma-separated camera subset")
    p.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    p.set_defaults(func=cmd_export_colmap)

    p = sub.add_parser("export-nuscenes", help="export a session to nuScenes-style schema")
    p.add_argument("session", help="recorded session directory")
    p.add_argument("--out", help="output directory")
    p.set_defaults(func=cmd_export_nuscenes)

    p = sub.add_parser("replay", help="visualize a recorded session")
    p.add_argument("session", help="recorded session directory")
    p.add_argument("--backend", choices=["auto", "rerun", "matplotlib"], default="auto")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("live", help="real-time LiDAR BEV in rerun while you drive")
    p.add_argument("--config", required=True, help="session YAML (must include a lidar)")
    p.add_argument("--hz", type=float, default=15.0, help="target poll/refresh rate")
    p.add_argument("--range", type=float, default=None, help="clip points beyond N meters")
    p.add_argument("--seconds", type=float, default=None, help="auto-stop after N seconds")
    p.add_argument("--ai", action="store_true", help="let the AI drive instead of you")
    p.add_argument("--traffic", nargs="?", const="auto", default=None,
                   help="traffic override: bare flag = on (auto count), N = N vehicles, "
                        "0 = none, omitted = config value")
    p.set_defaults(func=cmd_live)

    p = sub.add_parser("doctor", help="check environment and config")
    p.add_argument("--config", help="session YAML to validate")
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
