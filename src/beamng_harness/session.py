# BeamNG.tech session management: launch/connect, scenario setup, traffic, teardown

from __future__ import annotations

import logging

from beamngpy import BeamNGpy, Scenario, Vehicle

from .config import HarnessConfig

log = logging.getLogger(__name__)


class HarnessSession:
    # Owns the BeamNG connection, the scenario, the ego vehicle and traffic actors.

    def __init__(self, cfg: HarnessConfig):
        self.cfg = cfg
        self.bng: BeamNGpy | None = None
        self.scenario: Scenario | None = None
        self.ego: Vehicle | None = None
        self.actors: dict[str, Vehicle] = {}

    def open(self) -> "HarnessSession":
        b = self.cfg.beamng
        self.bng = BeamNGpy(b.host, b.port, home=b.home, user=b.user)
        log.info("connecting to BeamNG.tech at %s:%s (launch=%s)", b.host, b.port, b.launch)
        self.bng.open(launch=b.launch)
        return self

    def setup_scenario(self, deterministic: bool = True) -> None:
        assert self.bng is not None, "call open() first"
        sc = self.cfg.scenario

        self.scenario = Scenario(sc.level, sc.name)
        self.ego = Vehicle("ego", model=sc.vehicle_model, license="HARNESS")
        self.scenario.add_vehicle(
            self.ego, pos=tuple(sc.spawn.pos), rot_quat=tuple(sc.spawn.rot_quat)
        )
        self.scenario.make(self.bng)

        # Deterministic physics is the backbone of sensor synchronization: the sim only advances when we step it, 
        # so every poll within a frame shares one simulation time.
        if deterministic:
            # Deterministic physics underpins synchronized capture (the sim only advances when stepped). 
            # Live/real-time driving skips it so the world runs at wall-clock speed and you can drive.
            self.bng.settings.set_deterministic(self.cfg.capture.steps_per_second)
        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()

        if sc.traffic > 0:
            log.info("spawning %d traffic vehicles", sc.traffic)
            self.bng.traffic.spawn(max_amount=sc.traffic)
            self._connect_traffic()

        self._configure_ego_ai()

    def _connect_traffic(self) -> None:
        # Connect to traffic vehicles so their state and bounding boxes can be polled.
        assert self.bng is not None
        current = self.bng.vehicles.get_current()
        for vid, vehicle in current.items():
            if vid == "ego":
                continue
            try:
                vehicle.connect(self.bng)
                self.actors[vid] = vehicle
            except Exception:
                log.warning("could not connect to traffic vehicle %s; skipping", vid)
        log.info("tracking %d dynamic actors", len(self.actors))

    def _configure_ego_ai(self) -> None:
        assert self.ego is not None
        sc = self.cfg.scenario
        if sc.ai_mode in ("manual", "disabled"):
            return
        self.ego.ai.set_mode(sc.ai_mode)
        if sc.ai_speed_kph is not None:
            self.ego.ai.set_speed(sc.ai_speed_kph / 3.6, mode="limit")

    def close(self) -> None:
        if self.bng is not None:
            try:
                self.bng.close()
            except Exception:
                log.warning("error while closing BeamNG connection", exc_info=True)
            self.bng = None

    def __enter__(self) -> "HarnessSession":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
