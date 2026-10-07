"""
Layer 2 Gymnasium environment — AutoDriveEnv.

This is the ONLY class Layer 3 (PPO / Stable-Baselines3) talks to.

Contract every Gymnasium env must implement
-------------------------------------------
  reset() → (observation, info)
  step(action) → (observation, reward, terminated, truncated, info)
  close()

What "terminated" vs "truncated" means
--------------------------------------
                terminated  = episode ended because the car failed
                (collision when enabled or frontier stagnation).
  truncated   = episode ended because of an optional step / idle cutoff.

1 env = 1 car
-------------
Each AutoDriveEnv owns (or wraps) exactly one Layer 1 Racer on one Socket.IO
port. Parallel training later = many AutoDriveEnv instances, many ports.

Docker
------
Inside the container the Linux binary lives at:
  /app/simulator/AutoDRIVE Simulator.x86_64
entrypoint.sh starts Xvfb on DISPLAY=:99; headless=True still launches Unity
with -batchmode. Prefer AICAR_SIMULATOR_PATH or auto-detect (see
default_simulator_path).
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.layer1.racer import Racer
from src.layer1.ros2_racer import encoder_forward_speed_mps
from src.layer1.telemetry import TelemetrySnapshot

from .rewards import RewardConfig, compute_reward_components
from .lap_tracker import LapTracker, map_lap_gate_config
from .route_progress import RouteProgressTracker, map_centerline_path
from .spaces import (
    LIDAR_BEAMS,
    make_action_space,
    make_observation_space,
    map_policy_throttle,
    snapshot_to_obs,
)

# Nominal control rate used to scale reward/route time by frame_skip. This is
# not a guarantee of Unity physics steps or simulated seconds per Bridge return.
SIMULATION_HZ = 40.0

# Type alias: observation is always a dict with "lidar" and "state" arrays.
ObsType = Dict[str, np.ndarray]


def default_simulator_path() -> Path:
    """
    Pick a simulator executable that exists on this machine / container.

    Order:
      1. Env var AICAR_SIMULATOR_PATH (set this in Docker compose if needed)
      2. First existing candidate among Windows + Linux paths
      3. Platform fallback (even if missing — FileNotFoundError later is clear)
    """
    env_override = os.environ.get("AICAR_SIMULATOR_PATH")
    if env_override:
        return Path(env_override)

    candidates = [
        Path("simulator/windows/AutoDRIVE Simulator.exe"),
        Path("simulator/AutoDRIVE Simulator.x86_64"),
        Path("/app/simulator/AutoDRIVE Simulator.x86_64"),  # Docker WORKDIR layout
        Path("simulator/linux/AutoDRIVE Simulator.x86_64"),
    ]
    for path in candidates:
        if path.exists():
            return path

    if platform.system() == "Windows":
        return Path("simulator/windows/AutoDRIVE Simulator.exe")
    return Path("/app/simulator/AutoDRIVE Simulator.x86_64")


class AutoDriveEnv(gym.Env):
    """
    Gymnasium wrapper around a single Layer 1 Racer.

    Defaults (Layer 2 v1 / PLAN.md §6)
    ----------------------------------
    headless=True
        Launch Unity without a visible window (needed for Docker / servers).
    frame_skip=1
        Request one Bridge response per env.step(), repeating the same action
        for each response. Legacy mode has no fixed action duration. With
        action_interval_s set, each response advances that simulated duration.
        frame_skip cannot be 0.
        Raise to 2/4 to repeat the same action across more Bridge responses.
    max_episode_steps=0
        0 = DISABLED (no hard step cap). Episode only truncates on stagnation
        (or if you set a positive limit).
        A positive number (e.g. 20000) = hard truncate after that many env steps.
    stagnation_speed_threshold=0.15
        |v_long| below this (m/s) counts as "idle" for that step.
    stagnation_steps=200
        Consecutive idle steps before truncated=True (stuck / crashed stop).
    laps_per_episode=0
        Optional successful episode target. Zero disables lap-based termination.
    collision cost
        Lives on RewardConfig; see rewards.py / README.
    """

    # Gymnasium metadata (no custom render modes in v1).
    metadata = {"render_modes": []}

    def __init__(
        self,
        # Path to Unity AutoDRIVE binary. None → default_simulator_path()
        # (Windows .exe locally, Linux .x86_64 in Docker).
        simulator_path: Optional[Union[str, Path]] = None,
        # Socket.IO port for THIS car's Bridge. Must be unique per parallel env.
        port: int = 4567,
        # If True, Racer launches the simulator process for us.
        auto_launch: bool = True,
        # If True, Unity -batchmode / no visible window (Docker-friendly).
        headless: bool = True,
        # How many Bridge responses reuse the same action per env.step(). MUST be >= 1.
        # Default 1 = one Bridge response. NOT zero — zero is invalid.
        frame_skip: int = 1,
        # Hard episode length in env steps. 0 = no hard cap (stagnation only).
        max_episode_steps: int = 0,
        # |v_long| below this counts toward stagnation (m/s).
        stagnation_speed_threshold: float = 0.15,
        # Consecutive idle steps → truncated.
        stagnation_steps: int = 200,
        map_id: str = "none",
        laps_per_episode: int = 0,
        frontier_stagnation_seconds: float = 10.0,
        terminate_on_collision: bool = True,
        # Seconds to wait for Unity to connect on reset/init.
        connect_timeout: float = 60.0,
        # Reward weights. None → RewardConfig() defaults.
        reward_config: Optional[RewardConfig] = None,
        # Must stay at the canonical full 1081-beam scan (no downsampling).
        lidar_beams: int = LIDAR_BEAMS,
        # Optional: inject an already-built Racer (tests / shared process).
        # If provided, this env will NOT kill it on close().
        racer: Optional[Racer] = None,
        # Opt-in acknowledged simulation interval; None preserves legacy timing.
        action_interval_s: Optional[float] = None,
        # Scale the normalized steering command before sending it to Unity.
        steering_action_scale: float = 1.0,
        # Optional forward-only throttle mapping for racing policies.
        throttle_mode: str = "bidirectional",
        # Optional throttle gain when the executed steering command is small.
        straight_throttle_gain: float = 1.0,
        straight_throttle_steering_threshold: float = 0.15,
        observation_profile: str = "simulator",
    ) -> None:
        # Required Gymnasium base init (seeding hooks, etc.).
        super().__init__()

        # Guard: frame_skip=0 would wait for no Bridge responses.
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        # Guard: max_episode_steps < 0 is meaningless; 0 means "unlimited".
        if max_episode_steps < 0:
            raise ValueError(f"max_episode_steps must be >= 0, got {max_episode_steps}")
        if laps_per_episode < 0:
            raise ValueError(f"laps_per_episode must be >= 0, got {laps_per_episode}")
        # Guard: Layer 2 locks the full canonical LiDAR scan.
        if lidar_beams != LIDAR_BEAMS:
            raise ValueError(
                f"Layer 2 v1 requires full {LIDAR_BEAMS}-beam LiDAR "
                f"(no downsampling); got {lidar_beams}"
            )
        if observation_profile not in ("simulator", "official_sensors"):
            raise ValueError(
                "observation_profile must be 'simulator' or 'official_sensors'"
            )
        map_policy_throttle(0.0, throttle_mode)

        # Store knobs as instance attributes (used every step).
        self.frame_skip = int(frame_skip)
        self.observation_profile = observation_profile
        self.throttle_mode = throttle_mode
        self._observation_encoder_positions: Optional[Tuple[float, float]] = None
        injected_interval = getattr(racer, "action_interval_s", None)
        if racer is not None and action_interval_s is not None:
            if injected_interval is None or not np.isclose(
                float(injected_interval), float(action_interval_s), rtol=0, atol=1e-6
            ):
                raise ValueError("Injected Racer must use the requested action_interval_s")
        self.action_interval_s = (
            float(injected_interval) if injected_interval is not None
            else (float(action_interval_s) if action_interval_s is not None else None)
        )
        if self.action_interval_s is not None and (
            not np.isfinite(self.action_interval_s) or self.action_interval_s <= 0
        ):
            raise ValueError("action_interval_s must be finite and positive")
        self.steering_action_scale = float(steering_action_scale)
        if not np.isfinite(self.steering_action_scale) or not 0.0 <= self.steering_action_scale <= 1.0:
            raise ValueError("steering_action_scale must be finite and in [0, 1]")
        self.straight_throttle_gain = float(straight_throttle_gain)
        if not np.isfinite(self.straight_throttle_gain) or not 1.0 <= self.straight_throttle_gain <= 2.0:
            raise ValueError("straight_throttle_gain must be finite and in [1, 2]")
        self.straight_throttle_steering_threshold = float(straight_throttle_steering_threshold)
        if (
            not np.isfinite(self.straight_throttle_steering_threshold)
            or not 0.0 <= self.straight_throttle_steering_threshold <= 1.0
        ):
            raise ValueError(
                "straight_throttle_steering_threshold must be finite and in [0, 1]"
            )
        self.max_episode_steps = int(max_episode_steps)
        self.stagnation_speed_threshold = float(stagnation_speed_threshold)
        self.stagnation_steps = int(stagnation_steps)
        self.map_id = str(map_id or "none")
        # Zero disables the successful lap-count episode target.
        self.laps_per_episode = int(laps_per_episode)
        self.frontier_stagnation_seconds = max(0.0, float(frontier_stagnation_seconds))
        self.terminate_on_collision = bool(terminate_on_collision)
        self.route_progress: Optional[RouteProgressTracker] = None
        self._expected_spawn_xz: Optional[Tuple[float, float]] = None
        centerline_path = map_centerline_path(self.map_id)
        if self.map_id != "none" and centerline_path is None:
            raise FileNotFoundError(
                f"Map '{self.map_id}' has no occupancy/centerline.csv; "
                "generate and validate its centerline before training."
            )
        if centerline_path is not None:
            try:
                # Keep route projection continuous when frame_skip is raised.
                # Use the explicit duration when available, otherwise retain
                # the legacy nominal 40 Hz estimate.
                max_action_distance = 15.0 * self.frame_skip * (
                    self.action_interval_s if self.action_interval_s is not None
                    else 1.0 / SIMULATION_HZ
                )
                self.route_progress = RouteProgressTracker.from_csv(
                    centerline_path,
                    max_forward_m=max(2.0, max_action_distance),
                    backward_tolerance_m=max(0.5, max_action_distance),
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Cannot use map '{self.map_id}' for route-progress training: {exc}"
                ) from exc
            # Unity's map loader uses meta.spawn, falling back to the first
            # centerline point. Remember that authoritative XZ pose so reset()
            # never silently re-anchors a frontier to Unity's stale scene spawn.
            try:
                meta = json.loads(
                    (centerline_path.parent / "meta.json").read_text(encoding="utf-8")
                )
                spawn = meta.get("spawn") if isinstance(meta, dict) else None
                if isinstance(spawn, dict):
                    self._expected_spawn_xz = (
                        float(spawn["x"]), float(spawn["z"])
                    )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
            if self._expected_spawn_xz is None:
                point = self.route_progress.points[0]
                self._expected_spawn_xz = (float(point[0]), float(point[1]))
        lap_gate = map_lap_gate_config(self.map_id, route=self.route_progress)
        self.lap_tracker = LapTracker(
            self.route_progress,
            gate_progress_m=lap_gate.get("progress_m") if lap_gate.get("supported") else None,
        )
        self.connect_timeout = float(connect_timeout)
        self.reward_config = reward_config or RewardConfig()
        self.lidar_beams = int(lidar_beams)

        # Declare Gym spaces (SB3 reads these to build networks).
        self.action_space: spaces.Space = make_action_space()
        self.observation_space: spaces.Space = make_observation_space(self.lidar_beams)

        # Own the Racer unless the caller passed one in (then we don't kill it).
        self._owns_racer = racer is None
        if racer is not None:
            self.racer = racer
        else:
            # Resolve Windows vs Docker Linux binary.
            if simulator_path is None:
                simulator_path = default_simulator_path()
            sim_path = Path(simulator_path)
            self.racer = Racer(
                racer_id=0,
                port=port,
                simulator_path=sim_path if auto_launch else None,
                auto_launch=auto_launch and sim_path.exists(),
                headless=headless,
                action_interval_s=self.action_interval_s,
            )
            if auto_launch and not sim_path.exists():
                raise FileNotFoundError(
                    f"Simulator executable not found at: {sim_path.resolve()}\n"
                    f"Tip: set AICAR_SIMULATOR_PATH, or mount ./simulator into "
                    f"/app/simulator in Docker (see src/layer2/README.md)."
                )

        # Episode bookkeeping — reset() clears these each episode.
        self._episode_steps = 0          # env.step count this episode
        self._positive_episode_return = 0.0
        self._episode_frontier_distance_m = 0.0
        self._episode_simulated_seconds = 0.0
        self._idle_steps = 0             # consecutive |v_long| < threshold
        self._prev_throttle = 0.0        # last throttle (for obs + rewards)
        self._prev_steering = 0.0        # last steering (for obs + jerk term)
        self._prev_collision_count = 0   # to detect NEW collisions
        self._prev_collision_flag = False  # rising-edge fallback when count is unavailable
        self._pending_collision_event = False  # collision observed while reset settles
        self._last_snap: TelemetrySnapshot = TelemetrySnapshot()

        # The bridge listener is always owned by this Racer, including when
        # Unity is launched externally (for example, in Docker). Wait for the
        # simulator's first actual connection/frame before reset() sends a
        # command; otherwise cold Unity startup races the shorter per-step
        # timeout and SubprocVecEnv can fail while workers are initializing.
        self._wait_for_connection()

    def _wait_for_connection(self) -> None:
        """Block until the Unity client connects (or bridge frames arrive)."""
        deadline = time.time() + self.connect_timeout
        while time.time() < deadline:
            if self.racer.is_connected:
                return
            # Some builds mark progress via step_counter before is_connected.
            if self.racer.step_counter > 0:
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"Simulator did not connect on port {self.racer.port} "
            f"within {self.connect_timeout:.0f}s"
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Start a new episode.

        Returns
        -------
        obs : dict
            {"lidar": (1081,), "state": (9,)} float32 arrays.
        info : dict
            Diagnostics (NOT fed into the policy network). See _build_info.
        """
        # Gymnasium seeding protocol (affects self.np_random if we use it later).
        super().reset(seed=seed)

        # Layer 1 soft reset (pose / controls); does not restart Unity process.
        snap = self.racer.reset()
        reset_collision_pending = False
        # Nudge zero-action ticks so telemetry reflects the new spawn. Preserve
        # collisions that happen during this settling window instead of making
        # the final warm-up count the new baseline and silently swallowing them.
        for _ in range(max(1, self.frame_skip)):
            previous_count = int(snap.collision_count)
            previous_flag = bool(snap.collision)
            snap = self.racer.step(0.0, 0.0)
            current_count = int(snap.collision_count)
            if current_count > previous_count:
                reset_collision_pending = True
            elif current_count < previous_count and current_count > 0:
                # Unity may clear its cumulative count as part of the soft reset.
                reset_collision_pending = True
            if bool(snap.collision) and not previous_flag:
                reset_collision_pending = True

        expected_spawn = getattr(self, "_expected_spawn_xz", None)
        if expected_spawn is not None:
            actual_xz = (float(snap.position[0]), float(snap.position[2]))
            spawn_error = float(np.linalg.norm(np.asarray(actual_xz) - expected_spawn))
            if spawn_error > 0.75:
                raise RuntimeError(
                    f"Unity reset for map '{self.map_id}' landed at "
                    f"({actual_xz[0]:.3f}, {actual_xz[1]:.3f}), but the configured "
                    f"spawn is ({expected_spawn[0]:.3f}, {expected_spawn[1]:.3f}) "
                    f"({spawn_error:.2f} m away). Refusing to reset the frontier "
                    "from an incorrect pose."
                )

        # Clear episode counters / history.
        self._episode_steps = 0
        self._positive_episode_return = 0.0
        self._episode_frontier_distance_m = 0.0
        self._episode_simulated_seconds = 0.0
        self._idle_steps = 0
        self._prev_throttle = 0.0
        self._prev_steering = 0.0
        self._prev_collision_count = int(snap.collision_count)
        self._prev_collision_flag = bool(snap.collision)
        self._pending_collision_event = reset_collision_pending
        self._last_snap = snap
        self._observation_encoder_positions = (snap.encoder_left, snap.encoder_right)
        progress_state = None
        progress_now = self._progress_time(snap)
        if self.route_progress is not None:
            try:
                # Each Gym reset starts a new per-car frontier at that episode's
                # configured spawn, independent of small post-teleport physics
                # settling offsets in the returned telemetry pose.
                frontier_x, frontier_z = (
                    expected_spawn
                    if expected_spawn is not None
                    else (float(snap.position[0]), float(snap.position[2]))
                )
                progress_state = self.route_progress.reset(
                    float(frontier_x), float(frontier_z), progress_now
                )
            except ValueError as exc:
                raise ValueError(
                    f"Spawn for map '{self.map_id}' does not project onto its centerline: {exc}"
                ) from exc
        self.lap_tracker.reset(progress_state, progress_now)

        obs = self._policy_observation(
            snap, 0.0, 0.0, lidar_beams=self.lidar_beams, elapsed_s=0.0
        )
        info = self._build_info(snap, reward=0.0, collision_event=False)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Repeat one action across `frame_skip` Bridge responses; return Gym 5-tuple.

        Parameters
        ----------
        action : array-like shape (2,)
            [throttle, steering] each in [-1, 1].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # Coerce / clip to legal continuous controls.
        action = np.asarray(action, dtype=np.float32).reshape(2)
        throttle = map_policy_throttle(
            action[0], getattr(self, "throttle_mode", "bidirectional")
        )
        steering = float(
            np.clip(action[1], -1.0, 1.0) * self.steering_action_scale
        )
        # A small configurable gain lets the car use more of its available
        # throttle on straights while leaving cornering and reverse commands
        # untouched. With the default gain of 1.0 this mapping is neutral.
        if (
            throttle > 0.0
            and abs(steering) < self.straight_throttle_steering_threshold
        ):
            throttle = min(1.0, throttle * self.straight_throttle_gain)

        # Frame skip: repeat the SAME action for N Layer 1 Bridge responses.
        # Explicit mode acknowledges the same simulated interval per response.
        snap = self._last_snap
        sim_time_before = (
            float(self.racer.simulation_time())
            if getattr(self, "action_interval_s", None) is not None else None
        )
        for _ in range(self.frame_skip):
            snap = self.racer.step(throttle, steering)
        step_duration_s = self.frame_skip / SIMULATION_HZ
        if sim_time_before is not None:
            step_duration_s = float(self.racer.simulation_time()) - sim_time_before
            expected_duration_s = self.frame_skip * self.action_interval_s
            if not np.isfinite(step_duration_s) or not np.isclose(
                step_duration_s, expected_duration_s, rtol=0, atol=1e-6
            ):
                raise RuntimeError(
                    f"Invalid acknowledged simulation duration: {step_duration_s}; "
                    f"expected {expected_duration_s}"
                )

        # Prefer Unity's cumulative counter, but also accept a rising edge on
        # the collision flag for Bridge builds that only transmit a boolean.
        # Latching the previous flag prevents a sticky flag from ending every
        # subsequent episode after an autoreset.
        collision_count = int(snap.collision_count)
        collision_flag = bool(snap.collision)
        collision_event = (
            self._pending_collision_event
            or collision_count > self._prev_collision_count
            or (collision_flag and not self._prev_collision_flag)
        )
        self._pending_collision_event = False

        progress = None
        if self.route_progress is not None:
            progress = self.route_progress.update(
                float(snap.position[0]), float(snap.position[2]), self._progress_time(snap)
            )

        frontier_advanced_m = (
            max(0.0, float(progress["advanced_m"])) if progress is not None else 0.0
        )
        self._episode_frontier_distance_m = (
            getattr(self, "_episode_frontier_distance_m", 0.0) + frontier_advanced_m
        )
        self._episode_simulated_seconds = (
            getattr(self, "_episode_simulated_seconds", 0.0)
            + max(0.0, float(step_duration_s))
        )
        average_frontier_speed_mps = (
            self._episode_frontier_distance_m / self._episode_simulated_seconds
            if self._episode_simulated_seconds > 0.0 else 0.0
        )

        lap_state = self.lap_tracker.update(
            progress["progress_m"] if progress is not None else None,
            self._progress_time(snap),
        )
        lap_target_reached = (
            self.laps_per_episode > 0
            and int(lap_state.get("lap_count", 0)) >= self.laps_per_episode
        )

        # Bookkeeping for truncation rules.
        self._episode_steps += 1
        if abs(float(snap.v_long)) < self.stagnation_speed_threshold:
            self._idle_steps += 1
        else:
            self._idle_steps = 0

        # A terminal collision or a stalled frontier causes the vector worker
        # to reset this car alone; reset() reinitializes its frontier at spawn.
        terminated = bool(self.terminate_on_collision and collision_event)
        episode_won = False
        if progress is not None:
            no_push_s = progress["time_since_push_s"] or 0.0
            hit_frontier_stagnation = (
                self.frontier_stagnation_seconds > 0
                and no_push_s >= self.frontier_stagnation_seconds
            )
        else:
            hit_frontier_stagnation = False
        hit_idle_stagnation = (
            progress is None and self._idle_steps >= self.stagnation_steps
        )

        # On maps with route data, failure to advance the frontier is a car
        # failure, not merely an external timeout. This lets training learn
        # from the failed episode while the vector environment respawns it.
        termination_reason = (
            "collision" if self.terminate_on_collision and collision_event
            else None
        )
        if not terminated and lap_target_reached:
            terminated = True
            episode_won = True
            termination_reason = "lap_target"
        if not terminated and hit_frontier_stagnation:
            terminated = True
            termination_reason = "frontier_stagnation"
        # max_episode_steps == 0 → disabled (no hard cap).
        hit_max_steps = (
            self.max_episode_steps > 0
            and self._episode_steps >= self.max_episode_steps
        )
        truncated = bool(
            (hit_idle_stagnation or hit_max_steps) and not terminated
        )

        # Frontier stalls are reset after their timer, but do not receive an
        # extra terminal failure deduction: the per-second time cost already
        # penalizes waiting. Keep the configurable failure cost for other
        # failed endings (for example, an idle timeout or step cap).
        episode_failure = bool(
            ((terminated and not episode_won)
             and termination_reason != "frontier_stagnation")
            or truncated
        )

        # Reward only newly advanced frontier distance, plus time cost and
        # failure costs. Current route movement is telemetry, never a second
        # positive reward for recovering ground already pushed before.
        reward_components = compute_reward_components(
            v_long=float(snap.v_long),
            step_duration_s=step_duration_s,
            frontier_advanced_m=(progress["advanced_m"] if progress is not None else None),
            frontier_average_speed_mps=average_frontier_speed_mps,
            collision_event=collision_event,
            episode_failure=episode_failure,
            positive_episode_return=self._positive_episode_return,
            slip_angle=float(snap.slip_angle),
            prev_steering=self._prev_steering,
            steering=steering,
            cfg=self.reward_config,
        )
        reward = reward_components["total"]
        self._positive_episode_return += max(0.0, float(reward_components["route_progress"]))

        obs = self._policy_observation(
            snap,
            prev_throttle=throttle,
            prev_steering=steering,
            lidar_beams=self.lidar_beams,
            elapsed_s=step_duration_s,
        )
        info = self._build_info(
            snap,
            reward=reward,
            collision_event=collision_event,
            throttle_command=throttle,
            steering_command=steering,
        )
        info["reward_components"] = reward_components
        info["step_duration_s"] = step_duration_s
        info["laps_per_episode"] = getattr(self, "laps_per_episode", 0)
        info["episode_won"] = episode_won
        info["positive_episode_return"] = self._positive_episode_return
        if progress is not None:
            info.update({
                "frontier_line": progress["line"],
                "frontier_progress_m": progress["progress_m"],
                "frontier_advanced_m": progress["advanced_m"],
                "current_progress_line": progress["current_line"],
                "current_progress_m": progress["current_progress_m"],
                "signed_route_delta_m": progress["current_delta_m"],
                "current_route_speed_mps": (
                    progress["current_delta_m"] / step_duration_s
                ),
                "route_projection_valid": progress["current_projection_valid"],
                "time_since_frontier_push_s": progress["time_since_push_s"],
                "frontier_speed_mps": progress["speed_mps"],
                "average_frontier_speed_mps": average_frontier_speed_mps,
            })
        if truncated:
            info["truncate_reason"] = (
                "stagnation" if hit_idle_stagnation else "max_episode_steps"
            )
        if termination_reason is not None:
            info["termination_reason"] = termination_reason

        # Remember controls / collision count for the NEXT step.
        self._prev_throttle = throttle
        self._prev_steering = steering
        self._prev_collision_count = collision_count
        self._prev_collision_flag = collision_flag
        self._last_snap = snap

        return obs, reward, terminated, truncated, info

    def _policy_observation(
        self,
        snap: TelemetrySnapshot,
        prev_throttle: float,
        prev_steering: float,
        *,
        lidar_beams: int,
        elapsed_s: float,
    ) -> ObsType:
        if self.observation_profile == "simulator":
            return snapshot_to_obs(
                snap, prev_throttle, prev_steering, lidar_beams=lidar_beams
            )

        positions = (float(snap.encoder_left), float(snap.encoder_right))
        previous = self._observation_encoder_positions
        speed = (
            0.0 if previous is None else encoder_forward_speed_mps(
                positions, previous, elapsed_s
            )
        )
        self._observation_encoder_positions = positions
        return snapshot_to_obs(
            snap,
            prev_throttle,
            prev_steering,
            lidar_beams=lidar_beams,
            forward_speed_mps=speed,
            lateral_speed_mps=0.0,
        )

    def _progress_time(self, snap: TelemetrySnapshot) -> float:
        # Bridge receipt timestamps are wall-clock based. Exclude PPO's policy
        # optimization pauses so they cannot age frontier-stagnation timers or
        # make lap frontier-speed rewards depend on machine training speed.
        simulation_clock = getattr(getattr(self, "racer", None), "simulation_time", None)
        if callable(simulation_clock):
            try:
                sim_time = float(simulation_clock())
                if np.isfinite(sim_time):
                    return sim_time
            except (TypeError, ValueError, RuntimeError):
                pass
        stamp = float(snap.timestamp)
        return stamp if np.isfinite(stamp) and stamp > 0 else time.monotonic()

    def _build_info(
        self,
        snap: TelemetrySnapshot,
        *,
        reward: float,
        collision_event: bool,
        throttle_command: Optional[float] = None,
        steering_command: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build the Gym `info` dict (diagnostics for YOU / logging / callbacks).

        Important: `info` is NOT part of the observation. The policy network
        only sees `obs` (lidar + state). Stuff here is for humans, TensorBoard
        callbacks, debugging — not for the neural net unless you later copy
        fields into obs yourself.
        """
        info = {
            "step": self._episode_steps,
            "idle_steps": self._idle_steps,
            "reward": float(reward),
            "v_long": float(snap.v_long),
            "true_speed": float(snap.true_speed),
            "throttle_command": (
                None if throttle_command is None else float(throttle_command)
            ),
            "steering_command": (
                None if steering_command is None else float(steering_command)
            ),
            "collision": bool(snap.collision),
            "collision_event": bool(collision_event),
            "collision_count": int(snap.collision_count),
            "position": tuple(float(x) for x in snap.position),
            # Radians — TelemetrySnapshot.heading_yaw (for fleet canvas later)
            "yaw": float(snap.heading_yaw),
        }
        info.update(self.lap_tracker.sample(self._progress_time(snap)))
        return info

    def close(self) -> None:
        """Tear down the Unity process if this env owns the Racer."""
        if getattr(self, "racer", None) is not None and self._owns_racer:
            self.racer.kill()
        return None

    def set_simulation_paused(self, paused: bool) -> None:
        """Synchronize the simulator with PPO's rollout/update phases."""
        self.racer.set_simulation_paused(paused)

    def resume_simulation(self) -> None:
        """Resume the attached Unity player even if pause state predates this env."""
        self.racer.resume_simulation()

    def set_laps_per_episode(self, laps: int) -> None:
        """Update the episode target for a Layer 3 curriculum transition."""
        laps = int(laps)
        if laps < 0:
            raise ValueError("laps_per_episode must be >= 0")
        self.laps_per_episode = laps

    def get_current_info(self) -> Dict[str, Any]:
        """Return current diagnostics for Layer 3's demonstration labeler.

        This info remains outside the observation consumed by the learned
        policy. In particular, map route progress is never copied into obs.
        """
        info = self._build_info(
            self._last_snap,
            reward=0.0,
            collision_event=False,
            throttle_command=self._prev_throttle,
            steering_command=self._prev_steering,
        )
        if self.route_progress is not None:
            progress = self.route_progress.sample(self._progress_time(self._last_snap))
            info.update({
                "current_progress_m": progress["current_progress_m"],
                "frontier_progress_m": progress["progress_m"],
            })
        return info
