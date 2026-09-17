"""Map loading + LiDAR raycasting + kinematic bicycle racing env (Gymnasium)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from .contracts import (
    ACTION_NVEC,
    COLLISION_PENALTY,
    D_WALL_M,
    LIDAR_MAX_M,
    N_LIDAR_DEFAULT,
    TIMEOUT_S,
    W_PROGRESS,
    W_SPEED,
    W_TIME,
    W_WALL,
    decode_action,
)
from .observation import build_observation


@dataclass
class OccupancyMap:
    grid: np.ndarray  # True = occupied
    resolution: float
    origin: np.ndarray  # [x, y, yaw]
    path: Path

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.origin[0]) / self.resolution)
        row = int(self.height - 1 - (y - self.origin[1]) / self.resolution)
        return row, col

    def occupied_world(self, x: float, y: float) -> bool:
        r, c = self.world_to_px(x, y)
        if r < 0 or c < 0 or r >= self.height or c >= self.width:
            return True
        return bool(self.grid[r, c])


def load_map(map_yaml: str | Path) -> OccupancyMap:
    map_yaml = Path(map_yaml)
    with map_yaml.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    image_name = meta["image"]
    img_path = map_yaml.parent / image_name
    if not img_path.exists():
        # try png twin
        alt = map_yaml.parent / (Path(image_name).stem + ".png")
        if alt.exists():
            img_path = alt
    img = np.array(Image.open(img_path).convert("L"))
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    negate = int(meta.get("negate", 0))
    if negate:
        occupied = img > int(255 * occupied_thresh)
    else:
        # ROS map_server: black/dark = occupied when negate=0
        occupied = img < int(255 * (1.0 - occupied_thresh))
        occupied = occupied | (img < 40)
    origin = np.array(meta["origin"], dtype=np.float64)
    resolution = float(meta["resolution"])
    return OccupancyMap(grid=occupied, resolution=resolution, origin=origin, path=map_yaml)


def load_centerline(map_dir: Path) -> np.ndarray | None:
    csv_path = map_dir / "centerline.csv"
    if not csv_path.exists():
        return None
    pts = []
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        return None
    return np.asarray(pts, dtype=np.float64)


def load_start_pose(map_dir: Path, centerline: np.ndarray | None) -> np.ndarray:
    start_path = map_dir / "start_pose.txt"
    if start_path.exists():
        vals = [float(x) for x in start_path.read_text(encoding="utf-8").split()]
        return np.array(vals[:3], dtype=np.float64)
    if centerline is not None and len(centerline) >= 2:
        p0, p1 = centerline[0], centerline[1]
        theta = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return np.array([p0[0], p0[1], theta], dtype=np.float64)
    return np.array([0.0, 0.0, 0.0], dtype=np.float64)


def cast_lidar(
    occ: OccupancyMap,
    x: float,
    y: float,
    theta: float,
    n_beams: int = 180,
    fov: float = 4.7,
    max_range: float = LIDAR_MAX_M,
) -> np.ndarray:
    """Fast grid raycast LiDAR (n_beams defaults to contract size for training speed)."""
    ranges = np.full(n_beams, max_range, dtype=np.float32)
    angles = theta + np.linspace(-fov / 2.0, fov / 2.0, n_beams, dtype=np.float64)
    step = max(float(occ.resolution), 0.08)
    max_steps = int(max_range / step) + 1
    grid = occ.grid
    h, w = grid.shape
    ox, oy = float(occ.origin[0]), float(occ.origin[1])
    res = float(occ.resolution)
    for i in range(n_beams):
        dx = math.cos(angles[i]) * step
        dy = math.sin(angles[i]) * step
        cx = x
        cy = y
        for s in range(1, max_steps + 1):
            cx += dx
            cy += dy
            col = int((cx - ox) / res)
            row = int(h - 1 - (cy - oy) / res)
            if row < 0 or col < 0 or row >= h or col >= w or grid[row, col]:
                ranges[i] = float(s * step)
                break
    return ranges


def nearest_centerline_progress(pose_xy: np.ndarray, centerline: np.ndarray) -> tuple[float, int]:
    """Approximate arc-length progress to nearest centerline point."""
    diffs = centerline[1:] - centerline[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    d = np.linalg.norm(centerline - pose_xy[None, :], axis=1)
    idx = int(np.argmin(d))
    return float(cum[idx]), idx


class RacingEnv(gym.Env):
    """
    Contract-compliant solo time-attack env on F1TENTH-style occupancy maps.

    Uses a kinematic bicycle model + grid LiDAR. Backend id for metrics: \"gym\".
    Official f1tenth_gym is optional (legacy deps); this env is the supported training path on Windows.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_yaml: str | Path,
        n_lidar: int = N_LIDAR_DEFAULT,
        dt: float = 0.05,
        timeout_s: float = TIMEOUT_S,
        v_max: float = 6.0,
        steer_max: float = 0.4189,
        wheelbase: float = 0.33,
        collision_radius: float = 0.2,
        seed: int | None = None,
    ):
        super().__init__()
        self.map_yaml = Path(map_yaml)
        self.occ = load_map(self.map_yaml)
        self.centerline = load_centerline(self.map_yaml.parent)
        self.start_pose = load_start_pose(self.map_yaml.parent, self.centerline)
        self.n_lidar = int(n_lidar)
        self.dt = float(dt)
        self.timeout_s = float(timeout_s)
        self.max_steps = int(math.ceil(self.timeout_s / self.dt))
        self.v_max = float(v_max)
        self.steer_max = float(steer_max)
        self.wheelbase = float(wheelbase)
        self.collision_radius = float(collision_radius)

        low = np.concatenate(
            [np.zeros(self.n_lidar, dtype=np.float32), np.array([-1.0, -1.0], dtype=np.float32)]
        )
        high = np.concatenate(
            [np.ones(self.n_lidar, dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(list(ACTION_NVEC))

        self._rng = np.random.default_rng(seed)
        self._state = None  # x, y, theta, v
        self._prev_action = (0.0, 0.0)
        self._step_count = 0
        self._progress = 0.0
        self._lap_progress_base = 0.0
        self._track_length = 0.0
        if self.centerline is not None:
            self._track_length = float(
                np.sum(np.linalg.norm(self.centerline[1:] - self.centerline[:-1], axis=1))
            )
        self._crossed_half = False
        self._raw_scan = None

    def _obs(self) -> np.ndarray:
        x, y, th, _v = self._state
        self._raw_scan = cast_lidar(self.occ, x, y, th, n_beams=self.n_lidar)
        return build_observation(
            self._raw_scan,
            self._prev_action[0],
            self._prev_action[1],
            n_lidar=self.n_lidar,
        )

    def _in_collision(self, x: float, y: float) -> bool:
        # sample a few points around vehicle
        for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            px = x + self.collision_radius * math.cos(a)
            py = y + self.collision_radius * math.sin(a)
            if self.occ.occupied_world(px, py):
                return True
        return self.occ.occupied_world(x, y)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        pose = self.start_pose.copy()
        self._state = np.array([pose[0], pose[1], pose[2], 0.0], dtype=np.float64)
        self._prev_action = (0.0, 0.0)
        self._step_count = 0
        self._crossed_half = False
        if self.centerline is not None:
            self._progress, _ = nearest_centerline_progress(self._state[:2], self.centerline)
            self._lap_progress_base = self._progress
        else:
            self._progress = 0.0
            self._lap_progress_base = 0.0
        obs = self._obs()
        info = {"collision": False, "timeout": False, "lap_time": None}
        return obs, info

    def step(self, action):
        throttle, steering = decode_action(action)
        x, y, th, v = self._state

        target_v = throttle * self.v_max
        # simple first-order speed tracking
        v = v + np.clip(target_v - v, -2.0, 2.0) * self.dt * 4.0
        v = float(np.clip(v, 0.0, self.v_max))
        delta = float(steering) * self.steer_max

        # kinematic bicycle
        x = x + v * math.cos(th) * self.dt
        y = y + v * math.sin(th) * self.dt
        th = th + (v / self.wheelbase) * math.tan(delta) * self.dt
        self._state = np.array([x, y, th, v], dtype=np.float64)

        collision = self._in_collision(x, y)
        self._step_count += 1
        t = self._step_count * self.dt

        progress_delta = 0.0
        lap_complete = False
        lap_time = None
        if self.centerline is not None:
            new_progress, idx = nearest_centerline_progress(np.array([x, y]), self.centerline)
            # unwrap small backward noise
            raw_delta = new_progress - self._progress
            if raw_delta < -0.5 * self._track_length:
                raw_delta += self._track_length
            if raw_delta > 0.5 * self._track_length:
                raw_delta -= self._track_length
            progress_delta = max(0.0, raw_delta)
            self._progress = new_progress
            frac = (new_progress - self._lap_progress_base) / max(self._track_length, 1e-6)
            if frac > 0.5:
                self._crossed_half = True
            if self._crossed_half and frac > 0.95:
                lap_complete = True
                lap_time = float(t)

        raw_min = float(np.min(self._raw_scan)) if self._raw_scan is not None else LIDAR_MAX_M
        # refresh scan after motion for reward wall term
        scan = cast_lidar(self.occ, x, y, th, n_beams=self.n_lidar)
        self._raw_scan = scan
        raw_min = float(np.min(scan))

        wall_penalty = 0.0
        if raw_min < D_WALL_M:
            wall_penalty = W_WALL * (D_WALL_M - raw_min)

        reward = (
            W_PROGRESS * progress_delta
            + W_SPEED * v
            - wall_penalty
            - W_TIME
        )
        terminated = False
        truncated = False
        if collision:
            reward -= COLLISION_PENALTY
            terminated = True
        if lap_complete:
            terminated = True
        if self._step_count >= self.max_steps or t >= self.timeout_s:
            truncated = True

        self._prev_action = (throttle, steering)
        obs = build_observation(scan, throttle, steering, n_lidar=self.n_lidar)
        info = {
            "collision": bool(collision),
            "timeout": bool(truncated and not terminated),
            "lap_time": lap_time,
            "speed": v,
            "progress": self._progress,
        }
        return obs, float(reward), terminated, truncated, info


def resolve_map_yaml(map_id: str | Path, maps_root: Path | None = None) -> Path:
    """Resolve a map id or path to a yaml file. Falls back to first generated map / demo."""
    p = Path(map_id)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p
    root = maps_root or (Path(__file__).resolve().parent / "maps")
    candidate = root / str(map_id) / f"{map_id}.yaml"
    if candidate.exists():
        return candidate
    # search
    matches = sorted(root.glob(f"**/{map_id}.yaml"))
    if matches:
        return matches[0]
    any_maps = sorted(root.glob("**/map*.yaml"))
    if any_maps:
        return any_maps[0]
    demo = root / "demo" / "demo.yaml"
    if demo.exists():
        return demo
    raise FileNotFoundError(f"No map found for '{map_id}' under {root}")
