"""RaceTrack: Track Manager and Multi-Vehicle Fleet Orchestrator.

Manages track geometry, checkpoint gate progression, and orchestrates stepping,
resetting, CSV exporting, and process termination across dynamic fleets of Racers.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .racer import Racer
from .telemetry import TelemetrySnapshot

logger = logging.getLogger(__name__)


class RaceTrack:
    """Manages track geometry, checkpoint progression, and fleet of Racer instances."""

    def __init__(
        self,
        num_racers: int = 1,
        base_port: int = 4567,
        num_gates: int = 8,
        waypoints: Optional[np.ndarray] = None,
        simulator_path: Optional[Union[str, Path]] = None,
        auto_launch: bool = False,
        headless: bool = True,
    ) -> None:
        if num_racers < 1:
            raise ValueError(f"num_racers must be >= 1, got {num_racers}")

        self.num_racers = num_racers
        self.base_port = base_port
        self.num_gates = max(1, num_gates)
        self.simulator_path = Path(simulator_path) if simulator_path else None
        self.auto_launch = auto_launch
        self.headless = headless

        # Track Waypoints & Geometry
        self.waypoints = waypoints  # Shape (M, 2) or (M, 3) [x, y] or [x, y, z]
        self.track_length: float = 0.0
        self.gate_locations: List[float] = []
        self._init_track_geometry()

        # Fleet Management
        self.racers: Dict[int, Racer] = {}
        self.racer_gate_progress: Dict[int, int] = {}
        self.racer_laps_completed: Dict[int, int] = {}

        self._init_fleet()

    def _init_track_geometry(self) -> None:
        """Calculate cumulative track distance and checkpoint gate intervals."""
        if self.waypoints is None or len(self.waypoints) < 2:
            self.track_length = 0.0
            self.gate_locations = []
            return

        # Extract 2D (X, Y) coordinates
        pts = self.waypoints[:, :2] if self.waypoints.shape[1] >= 2 else self.waypoints
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.track_length = float(np.sum(seg_lengths))

        # Place evenly spaced gate checkpoints along track length
        if self.track_length > 0 and self.num_gates > 0:
            step = self.track_length / self.num_gates
            self.gate_locations = [i * step for i in range(self.num_gates)]

    def _init_fleet(self) -> None:
        """Instantiate Racer instances for the fleet."""
        for i in range(self.num_racers):
            if i > 0 and self.auto_launch and not self.headless:
                # Stagger launch by 3.5s so DirectX 12 driver cleanly creates each desktop window
                time.sleep(3.5)
            port = self.base_port + i
            racer = Racer(
                racer_id=i,
                port=port,
                simulator_path=self.simulator_path,
                auto_launch=self.auto_launch,
                headless=self.headless,
            )
            self.racers[i] = racer
            self.racer_gate_progress[i] = 0
            self.racer_laps_completed[i] = 0

    def get_frenet_progress(self, x: float, y: float) -> Tuple[float, float]:
        """Project world coordinate (x, y) onto track centerline.

        Returns:
            (s, d): Arc-length track distance s in [0, track_length] and signed lateral deviation d.
        """
        if self.waypoints is None or len(self.waypoints) < 2:
            return (0.0, 0.0)

        pts = self.waypoints[:, :2]
        pos = np.array([x, y])

        # Distance to each waypoint
        diffs = pts - pos
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        closest_idx = int(np.argmin(dists))

        # Approximate s distance
        if closest_idx == 0:
            s = 0.0
        else:
            diff_segs = np.diff(pts[: closest_idx + 1], axis=0)
            s = float(np.sum(np.hypot(diff_segs[:, 0], diff_segs[:, 1])))

        # Lateral deviation d (perpendicular distance)
        d = float(dists[closest_idx])
        return (s, d)

    def update_checkpoints(self, racer_id: int, s: float) -> Tuple[bool, bool]:
        """Validate ordered sequential checkpoint gate passage.

        Returns:
            (gate_passed, lap_completed)
        """
        if not self.gate_locations:
            return (False, False)

        current_gate = self.racer_gate_progress.get(racer_id, 0)
        target_s = self.gate_locations[current_gate]

        gate_passed = False
        lap_completed = False

        # Gate threshold window (e.g. within 3.0 meters)
        if abs(s - target_s) < 3.0:
            gate_passed = True
            next_gate = (current_gate + 1) % self.num_gates
            self.racer_gate_progress[racer_id] = next_gate

            # If wrapped around from last gate to gate 0, lap is completed
            if next_gate == 0:
                lap_completed = True
                self.racer_laps_completed[racer_id] = self.racer_laps_completed.get(racer_id, 0) + 1

        return (gate_passed, lap_completed)

    def step_single(self, racer_id: int, throttle: float, steering: float) -> TelemetrySnapshot:
        """Advance one specific vehicle."""
        if racer_id not in self.racers:
            raise KeyError(f"Racer ID {racer_id} not found in active fleet.")

        racer = self.racers[racer_id]
        snap = racer.step(throttle, steering)

        # Update track coordinates if waypoints exist
        if self.track_length > 0:
            s, d = self.get_frenet_progress(snap.position[0], snap.position[2])
            snap.frenet_s = s
            snap.frenet_d = d
            self.update_checkpoints(racer_id, s)

        return snap

    def step_all(
        self,
        actions: Union[List[Tuple[float, float]], Dict[int, Tuple[float, float]]],
    ) -> Dict[int, TelemetrySnapshot]:
        """Step all active vehicles in the fleet.

        Args:
            actions: List or dict of (throttle, steering) commands.

        Returns:
            Dictionary mapping racer_id to TelemetrySnapshot.
        """
        results: Dict[int, TelemetrySnapshot] = {}

        if isinstance(actions, list):
            for i, act in enumerate(actions):
                if i in self.racers and self.racers[i].is_alive:
                    th, st = act
                    results[i] = self.step_single(i, th, st)
        elif isinstance(actions, dict):
            for i, act in actions.items():
                if i in self.racers and self.racers[i].is_alive:
                    th, st = act
                    results[i] = self.step_single(i, th, st)

        return results

    def step(self, *args) -> Union[TelemetrySnapshot, Dict[int, TelemetrySnapshot]]:
        """Polymorphic stepping helper:
        - step(throttle, steering) for single-car setups.
        - step(actions) for multi-car fleet setups.
        """
        if len(args) == 2 and isinstance(args[0], (int, float)):
            return self.step_single(0, float(args[0]), float(args[1]))
        if len(args) == 1 and isinstance(args[0], (list, dict)):
            return self.step_all(args[0])
        raise ValueError("Invalid arguments to step(). Use step(th, st) or step(actions_list).")

    def reset_single(self, racer_id: int) -> TelemetrySnapshot:
        """Reset one specific vehicle back to grid."""
        if racer_id not in self.racers:
            raise KeyError(f"Racer ID {racer_id} not found.")

        self.racer_gate_progress[racer_id] = 0
        return self.racers[racer_id].reset()

    def reset_all(self) -> Dict[int, TelemetrySnapshot]:
        """Reset all active vehicles in the fleet."""
        results: Dict[int, TelemetrySnapshot] = {}
        for r_id, racer in self.racers.items():
            if racer.is_alive:
                self.racer_gate_progress[r_id] = 0
                results[r_id] = racer.reset()
        return results

    def kill_racer(self, racer_id: int) -> bool:
        """Terminate a specific racer child process, stop its server, and release its port."""
        if racer_id in self.racers:
            racer = self.racers[racer_id]
            logger.info(f"[RaceTrack] Killing racer {racer_id}...")
            racer.kill()
            return True
        logger.warning(f"[RaceTrack] Cannot kill racer {racer_id}: Not found.")
        return False

    def kill_all(self) -> None:
        """Terminate all simulator processes and shut down all fleet servers."""
        logger.info(f"[RaceTrack] Terminating entire fleet of {len(self.racers)} racers...")
        for r_id, racer in list(self.racers.items()):
            try:
                racer.kill()
            except Exception as e:
                logger.error(f"[RaceTrack] Error killing racer {r_id}: {e}")

    def save_all_trajectories(self, output_dir: Union[str, Path]) -> Dict[int, str]:
        """Export CSV trajectory files for all racers in one call.

        Args:
            output_dir: Directory where CSV files will be saved.

        Returns:
            Dict mapping racer_id to saved file path string.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        saved_paths: Dict[int, str] = {}

        for r_id, racer in self.racers.items():
            file_name = f"racer_{r_id}_trajectory.csv"
            dest = out_path / file_name
            saved_file = racer.save_trajectory(dest)
            saved_paths[r_id] = saved_file
            logger.info(f"[RaceTrack] Saved trajectory for racer {r_id} to {saved_file}")

        return saved_paths

    def get_fleet_telemetry(self) -> Dict[int, TelemetrySnapshot]:
        """Retrieve latest telemetry snapshot from all alive racers."""
        return {r_id: racer.telemetry for r_id, racer in self.racers.items() if racer.is_alive}
