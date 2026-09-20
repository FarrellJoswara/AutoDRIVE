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
  terminated  = episode ended because the task is over (goal / failure rule).
                Layer 2 v1: ALWAYS False — crashing does not end the episode.
  truncated   = episode ended because of a time / safety cutoff we chose
                (stagnation, or optional max_episode_steps).

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

import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.layer1.racer import Racer
from src.layer1.telemetry import TelemetrySnapshot

from .rewards import RewardConfig, compute_reward
from .spaces import LIDAR_BEAMS, make_action_space, make_observation_space, snapshot_to_obs

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
        Apply the SAME action for 1 physics tick per env.step().
        This is NOT "skip frames off". 1 = act every tick (~40 Hz bridge).
        frame_skip cannot be 0 (that would mean "do zero physics" — invalid).
        Raise to 2/4 later if you want coarser control (same action repeated).
    max_episode_steps=0
        0 = DISABLED (no hard step cap). Episode only truncates on stagnation
        (or if you set a positive limit).
        A positive number (e.g. 20000) = hard truncate after that many env steps.
    stagnation_speed_threshold=0.15
        |v_long| below this (m/s) counts as "idle" for that step.
    stagnation_steps=200
        Consecutive idle steps before truncated=True (stuck / crashed stop).
    collision_penalty
        Lives on RewardConfig, default 0.0 — see rewards.py / README.
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
        # How many sim ticks reuse the same action per env.step(). MUST be >= 1.
        # Default 1 = every physics tick. NOT zero — zero is invalid.
        frame_skip: int = 1,
        # Hard episode length in env steps. 0 = no hard cap (stagnation only).
        max_episode_steps: int = 0,
        # |v_long| below this counts toward stagnation (m/s).
        stagnation_speed_threshold: float = 0.15,
        # Consecutive idle steps → truncated.
        stagnation_steps: int = 200,
        # Seconds to wait for Unity to connect on reset/init.
        connect_timeout: float = 60.0,
        # Reward weights. None → RewardConfig() defaults (forward_scale=1, …).
        reward_config: Optional[RewardConfig] = None,
        # Must stay 1080 in v1 (no LiDAR downsampling).
        lidar_beams: int = LIDAR_BEAMS,
        # Optional: inject an already-built Racer (tests / shared process).
        # If provided, this env will NOT kill it on close().
        racer: Optional[Racer] = None,
    ) -> None:
        # Required Gymnasium base init (seeding hooks, etc.).
        super().__init__()

        # Guard: frame_skip=0 would run zero physics per step — nonsense.
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        # Guard: max_episode_steps < 0 is meaningless; 0 means "unlimited".
        if max_episode_steps < 0:
            raise ValueError(f"max_episode_steps must be >= 0, got {max_episode_steps}")
        # Guard: Layer 2 v1 locks full 1080-beam LiDAR.
        if lidar_beams != LIDAR_BEAMS:
            raise ValueError(
                f"Layer 2 v1 requires full {LIDAR_BEAMS}-beam LiDAR "
                f"(no downsampling); got {lidar_beams}"
            )

        # Store knobs as instance attributes (used every step).
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.stagnation_speed_threshold = float(stagnation_speed_threshold)
        self.stagnation_steps = int(stagnation_steps)
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
            )
            if auto_launch and not sim_path.exists():
                raise FileNotFoundError(
                    f"Simulator executable not found at: {sim_path.resolve()}\n"
                    f"Tip: set AICAR_SIMULATOR_PATH, or mount ./simulator into "
                    f"/app/simulator in Docker (see src/layer2/README.md)."
                )

        # Episode bookkeeping — reset() clears these each episode.
        self._episode_steps = 0          # env.step count this episode
        self._idle_steps = 0             # consecutive |v_long| < threshold
        self._prev_throttle = 0.0        # last throttle (for obs + rewards)
        self._prev_steering = 0.0        # last steering (for obs + jerk term)
        self._prev_collision_count = 0   # to detect NEW collisions
        self._last_snap: TelemetrySnapshot = TelemetrySnapshot()

        # Block until Unity Socket.IO client is up (auto-connect or manual).
        if auto_launch or racer is not None:
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
            {"lidar": (1080,), "state": (8,)} float32 arrays.
        info : dict
            Diagnostics (NOT fed into the policy network). See _build_info.
        """
        # Gymnasium seeding protocol (affects self.np_random if we use it later).
        super().reset(seed=seed)

        # Layer 1 soft reset (pose / controls); does not restart Unity process.
        snap = self.racer.reset()
        # Nudge a few zero-action ticks so telemetry reflects post-reset state.
        for _ in range(max(1, self.frame_skip)):
            snap = self.racer.step(0.0, 0.0)

        # Clear episode counters / history.
        self._episode_steps = 0
        self._idle_steps = 0
        self._prev_throttle = 0.0
        self._prev_steering = 0.0
        self._prev_collision_count = int(snap.collision_count)
        self._last_snap = snap

        obs = snapshot_to_obs(snap, 0.0, 0.0, lidar_beams=self.lidar_beams)
        info = self._build_info(snap, reward=0.0, collision_event=False)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Apply one action for `frame_skip` physics ticks; return Gym 5-tuple.

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
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steering = float(np.clip(action[1], -1.0, 1.0))

        # Frame skip: repeat the SAME action for N Layer 1 ticks.
        # frame_skip=1 → one tick (default, full ~40 Hz control).
        snap = self._last_snap
        for _ in range(self.frame_skip):
            snap = self.racer.step(throttle, steering)

        # Collision "event": flag this tick OR collision_count increased.
        collision_event = bool(snap.collision) or (
            int(snap.collision_count) > self._prev_collision_count
        )

        # Score this step (all reward math lives in rewards.py).
        reward = compute_reward(
            v_long=float(snap.v_long),
            collision_event=collision_event,
            slip_angle=float(snap.slip_angle),
            prev_steering=self._prev_steering,
            steering=steering,
            cfg=self.reward_config,
        )

        # Bookkeeping for truncation rules.
        self._episode_steps += 1
        if abs(float(snap.v_long)) < self.stagnation_speed_threshold:
            self._idle_steps += 1
        else:
            self._idle_steps = 0

        # v1: never terminate on crash; only truncate on idle / optional max steps.
        terminated = False
        hit_stagnation = self._idle_steps >= self.stagnation_steps
        # max_episode_steps == 0 → disabled (no hard cap).
        hit_max_steps = (
            self.max_episode_steps > 0
            and self._episode_steps >= self.max_episode_steps
        )
        truncated = bool(hit_stagnation or hit_max_steps)

        obs = snapshot_to_obs(
            snap,
            prev_throttle=throttle,
            prev_steering=steering,
            lidar_beams=self.lidar_beams,
        )
        info = self._build_info(snap, reward=reward, collision_event=collision_event)
        if truncated:
            info["truncate_reason"] = (
                "stagnation" if hit_stagnation else "max_episode_steps"
            )

        # Remember controls / collision count for the NEXT step.
        self._prev_throttle = throttle
        self._prev_steering = steering
        self._prev_collision_count = int(snap.collision_count)
        self._last_snap = snap

        return obs, reward, terminated, truncated, info

    def _build_info(
        self,
        snap: TelemetrySnapshot,
        *,
        reward: float,
        collision_event: bool,
    ) -> Dict[str, Any]:
        """
        Build the Gym `info` dict (diagnostics for YOU / logging / callbacks).

        Important: `info` is NOT part of the observation. The policy network
        only sees `obs` (lidar + state). Stuff here is for humans, TensorBoard
        callbacks, debugging — not for the neural net unless you later copy
        fields into obs yourself.
        """
        return {
            "step": self._episode_steps,
            "idle_steps": self._idle_steps,
            "reward": float(reward),
            "v_long": float(snap.v_long),
            "true_speed": float(snap.true_speed),
            "collision": bool(snap.collision),
            "collision_event": bool(collision_event),
            "collision_count": int(snap.collision_count),
            "position": tuple(float(x) for x in snap.position),
            # Radians — TelemetrySnapshot.heading_yaw (for fleet canvas later)
            "yaw": float(snap.heading_yaw),
        }

    def close(self) -> None:
        """Tear down the Unity process if this env owns the Racer."""
        if getattr(self, "racer", None) is not None and self._owns_racer:
            self.racer.kill()
        return None
