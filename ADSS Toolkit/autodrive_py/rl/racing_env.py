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
    COLLISION_FIRST_PENALTY,
    COLLISION_PENALTY,
    D_WALL_M,
    LAP_MIN_PROGRESS_FRAC,
    LIDAR_DR_DROPOUT,
    LIDAR_DR_MAX_RANGE_SCALE,
    LIDAR_DR_NOISE_STD,
    LIDAR_MAX_M,
    N_LIDAR_DEFAULT,
    PROGRESS_EPS_M,
    SPAWN_JITTER_HEADING_RAD,
    SPAWN_JITTER_LATERAL_M,
    SPAWN_JITTER_LONG_M,
    SPEED_GATE_CRASH_RATE,
    SPEED_GATE_WINDOW,
    SPEED_MAX_MPS,
    STALL_TIMEOUT_S,
    TIMEOUT_S,
    TTC_FRONT_BEAMS,
    TTC_HORIZON_S,
    TTC_MIN_RANGE_M,
    W_PROGRESS,
    W_SPEED,
    W_TIME,
    W_WALL,
    decode_action,
)
from .observation import build_observation, observation_bounds


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


def _centerline_cumlen(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (seg_len[n-1], cum[n] with cum[0]=0, track_length)."""
    diffs = centerline[1:] - centerline[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    return seg_len, cum, float(cum[-1]) if len(cum) else 0.0


def nearest_centerline_progress(pose_xy: np.ndarray, centerline: np.ndarray) -> tuple[float, int]:
    """Project pose onto centerline polyline; return (arc_s, segment_index).

    Uses closest point on each segment (not just vertices). Global search — prefer
    ``project_centerline_s`` with a local window during ``step`` to avoid orbit farming.
    """
    return project_centerline_s(pose_xy, centerline, last_idx=None)


def project_centerline_s(
    pose_xy: np.ndarray,
    centerline: np.ndarray,
    *,
    last_idx: int | None = None,
    window: int = 48,
    seg_len: np.ndarray | None = None,
    cum: np.ndarray | None = None,
) -> tuple[float, int]:
    """Frenet-s projection onto centerline.

    If ``last_idx`` is set, only search segments in a local window (with wrap) so
    circling near the track cannot jump to a distant nearest vertex and farm Δs.
    """
    n = int(len(centerline))
    if n < 2:
        return 0.0, 0
    n_seg = n - 1
    if seg_len is None or cum is None:
        seg_len, cum, _ = _centerline_cumlen(centerline)

    if last_idx is None:
        cand = range(n_seg)
    else:
        w = max(1, int(window))
        base = int(last_idx) % n_seg
        # Local window with wrap so finish→start stays continuous on loop tracks.
        cand = ((base + d) % n_seg for d in range(-w, w + 1))

    px, py = float(pose_xy[0]), float(pose_xy[1])
    best_d2 = float("inf")
    best_s = 0.0
    best_i = 0
    for i in cand:
        p0 = centerline[i]
        p1 = centerline[i + 1]
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            t = 0.0
            qx, qy = float(p0[0]), float(p0[1])
        else:
            t = ((px - float(p0[0])) * dx + (py - float(p0[1])) * dy) / denom
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
            qx = float(p0[0]) + t * dx
            qy = float(p0[1]) + t * dy
        d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy)
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(cum[i]) + t * float(seg_len[i])
            best_i = int(i)
    return best_s, best_i


def _unwrap_ds(new_s: float, old_s: float, track_len: float) -> float:
    """Signed centerline delta with shortest-path unwrap on a loop of length track_len."""
    d = float(new_s) - float(old_s)
    if track_len <= 1e-6:
        return d
    half = 0.5 * track_len
    if d < -half:
        d += track_len
    elif d > half:
        d -= track_len
    return d


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
        stall_timeout_s: float = STALL_TIMEOUT_S,
        v_max: float = SPEED_MAX_MPS,
        steer_max: float = 0.4189,
        wheelbase: float = 0.33,
        collision_radius: float = 0.2,
        seed: int | None = None,
        *,
        spawn_jitter: bool = False,
        collision_first: bool = False,
        speed_gate: bool = False,
        ttc_truncate: bool = False,
        lidar_dr: bool = False,
    ):
        super().__init__()
        self.map_yaml = Path(map_yaml)
        self.occ = load_map(self.map_yaml)
        self.centerline = load_centerline(self.map_yaml.parent)
        self.start_pose = load_start_pose(self.map_yaml.parent, self.centerline)
        self.n_lidar = int(n_lidar)
        self.dt = float(dt)
        self.timeout_s = float(timeout_s)
        self.stall_timeout_s = float(stall_timeout_s)
        self.max_steps = int(math.ceil(self.timeout_s / self.dt))
        self.v_max = float(v_max)
        self.steer_max = float(steer_max)
        self.wheelbase = float(wheelbase)
        self.collision_radius = float(collision_radius)
        self.spawn_jitter = bool(spawn_jitter)
        self.collision_first = bool(collision_first)
        self.speed_gate = bool(speed_gate)
        self.ttc_truncate = bool(ttc_truncate)
        self.lidar_dr = bool(lidar_dr)

        low, high = observation_bounds(self.n_lidar)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(list(ACTION_NVEC))

        self._rng = np.random.default_rng(seed)
        self._state = None  # x, y, theta, v
        self._prev_action = (0.0, 0.0)
        self._yaw_rate = 0.0
        self._ax = 0.0
        self._ay = 0.0
        self._step_count = 0
        self._progress = 0.0  # geometric Frenet s on [0, L)
        self._s_unwrapped = 0.0  # integrated signed Δs this episode
        self._s_hw = 0.0  # high-water of _s_unwrapped (reward only credits new highs)
        self._seg_idx = 0
        self._lap_progress_base = 0.0
        self._track_length = 0.0
        self._seg_len: np.ndarray | None = None
        self._cum: np.ndarray | None = None
        self._time_since_progress = 0.0
        if self.centerline is not None:
            self._seg_len, self._cum, self._track_length = _centerline_cumlen(self.centerline)
        self._crossed_half = False
        self._raw_scan = None
        # Cap per-step credited Δs (m) so a bad projection cannot dump a huge reward.
        self._max_progress_step = max(self.v_max * self.dt * 2.5, 0.75)
        self._crash_hist: list[int] = []
        self._speed_unlocked = not self.speed_gate

    @property
    def progress_frac(self) -> float:
        """Fraction of one lap covered as genuine forward high-water progress."""
        if self._track_length <= 1e-6:
            return 0.0
        return float(self._s_hw / self._track_length)

    def _obs(self) -> np.ndarray:
        x, y, th, v = self._state
        self._raw_scan = cast_lidar(self.occ, x, y, th, n_beams=self.n_lidar)
        return build_observation(
            self._raw_scan,
            self._prev_action[0],
            self._prev_action[1],
            float(v),
            self._yaw_rate,
            self._ax,
            self._ay,
            n_lidar=self.n_lidar,
            speed_max=self.v_max,
            lidar_dr=self.lidar_dr,
            rng=self._rng if self.lidar_dr else None,
            noise_std=LIDAR_DR_NOISE_STD if self.lidar_dr else 0.0,
            dropout=LIDAR_DR_DROPOUT if self.lidar_dr else 0.0,
            max_range_scale=LIDAR_DR_MAX_RANGE_SCALE if self.lidar_dr else 1.0,
        )

    def _apply_spawn_jitter(self, pose: np.ndarray) -> np.ndarray:
        """Pose/heading/lateral jitter; reject samples that spawn in collision."""
        if not self.spawn_jitter:
            return pose
        base = pose.copy()
        for _ in range(24):
            lat = float(self._rng.uniform(-SPAWN_JITTER_LATERAL_M, SPAWN_JITTER_LATERAL_M))
            long = float(self._rng.uniform(-SPAWN_JITTER_LONG_M, SPAWN_JITTER_LONG_M))
            hdg = float(self._rng.uniform(-SPAWN_JITTER_HEADING_RAD, SPAWN_JITTER_HEADING_RAD))
            th = float(base[2])
            x = float(base[0]) + long * math.cos(th) + lat * math.cos(th + math.pi / 2.0)
            y = float(base[1]) + long * math.sin(th) + lat * math.sin(th + math.pi / 2.0)
            th2 = th + hdg
            if not self._in_collision(x, y):
                return np.array([x, y, th2], dtype=np.float64)
        return base

    def _frontal_ttc_truncate(self, scan: np.ndarray, speed: float) -> bool:
        """LiDAR-only frontal collapse / TTC soft truncate (no MPC)."""
        if not self.ttc_truncate or scan is None or len(scan) == 0:
            return False
        n = int(len(scan))
        half = max(1, int(TTC_FRONT_BEAMS) // 2)
        mid = n // 2
        lo = max(0, mid - half)
        hi = min(n, mid + half + 1)
        front = scan[lo:hi]
        if front.size == 0:
            return False
        front_min = float(np.min(front))
        if front_min < float(TTC_MIN_RANGE_M):
            return True
        # Closing fast: rough TTC ≈ range / speed
        if speed > 0.3 and front_min / max(speed, 1e-3) < float(TTC_HORIZON_S):
            return True
        return False

    def _note_episode_crash(self, crashed: bool) -> None:
        self._crash_hist.append(1 if crashed else 0)
        if len(self._crash_hist) > SPEED_GATE_WINDOW:
            self._crash_hist = self._crash_hist[-SPEED_GATE_WINDOW:]
        if self.speed_gate:
            rate = float(sum(self._crash_hist)) / max(len(self._crash_hist), 1)
            self._speed_unlocked = rate <= float(SPEED_GATE_CRASH_RATE)

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
        pose = self._apply_spawn_jitter(self.start_pose.copy())
        self._state = np.array([pose[0], pose[1], pose[2], 0.0], dtype=np.float64)
        self._prev_action = (0.0, 0.0)
        self._yaw_rate = 0.0
        self._ax = 0.0
        self._ay = 0.0
        self._step_count = 0
        self._crossed_half = False
        self._time_since_progress = 0.0
        self._s_unwrapped = 0.0
        self._s_hw = 0.0
        if self.centerline is not None:
            self._progress, self._seg_idx = project_centerline_s(
                self._state[:2],
                self.centerline,
                last_idx=None,
                seg_len=self._seg_len,
                cum=self._cum,
            )
            self._lap_progress_base = self._progress
        else:
            self._progress = 0.0
            self._seg_idx = 0
            self._lap_progress_base = 0.0
        obs = self._obs()
        info = {
            "collision": False,
            "timeout": False,
            "stall": False,
            "ttc": False,
            "lap_time": None,
            "progress_frac": 0.0,
            "speed_unlocked": bool(self._speed_unlocked),
            "crash_tag": None,
        }
        return obs, info

    def step(self, action):
        throttle, steering = decode_action(action)
        x, y, th, v = self._state
        v_prev = float(v)

        target_v = throttle * self.v_max
        # simple first-order speed tracking
        v = v + np.clip(target_v - v, -2.0, 2.0) * self.dt * 4.0
        v = float(np.clip(v, 0.0, self.v_max))
        delta = float(steering) * self.steer_max

        # kinematic bicycle + proprio / IMU proxies (legal race-time analogues)
        yaw_rate = (v / self.wheelbase) * math.tan(delta)
        self._ax = (v - v_prev) / max(self.dt, 1e-6)
        self._ay = v * yaw_rate
        self._yaw_rate = yaw_rate

        x = x + v * math.cos(th) * self.dt
        y = y + v * math.sin(th) * self.dt
        th = th + yaw_rate * self.dt
        self._state = np.array([x, y, th, v], dtype=np.float64)

        collision = self._in_collision(x, y)
        self._step_count += 1
        t = self._step_count * self.dt

        # Progress reward: forward centerline Δs only (high-water of unwrapped s).
        # No spin/yaw penalty. Orbiting / reverse / sideways → progress term 0.
        progress_delta = 0.0
        lap_complete = False
        lap_time = None
        stall = False
        if self.centerline is not None:
            new_progress, idx = project_centerline_s(
                np.array([x, y]),
                self.centerline,
                last_idx=self._seg_idx,
                seg_len=self._seg_len,
                cum=self._cum,
            )
            ds = _unwrap_ds(new_progress, self._progress, self._track_length)
            # Reject pathological jumps (projection glitch / window miss).
            if abs(ds) > self._max_progress_step:
                ds = 0.0
                # Re-anchor with a global project so we do not stay stuck off-track.
                new_progress, idx = project_centerline_s(
                    np.array([x, y]),
                    self.centerline,
                    last_idx=None,
                    seg_len=self._seg_len,
                    cum=self._cum,
                )
            self._progress = new_progress
            self._seg_idx = idx
            self._s_unwrapped += ds
            # Credit only new forward highs along the lap direction.
            gain = self._s_unwrapped - self._s_hw
            if gain > 0.0:
                progress_delta = float(gain)
                self._s_hw = self._s_unwrapped

            if progress_delta > PROGRESS_EPS_M:
                self._time_since_progress = 0.0
            else:
                self._time_since_progress += self.dt

            # Lap credit rides on high-water forward progress, never on geometric
            # proximity to the start line: _s_hw only grows on new forward highs and
            # each step's Δs is capped, so orbiting or a projection re-anchor cannot
            # manufacture a lap.
            lap_frac = self._s_hw / max(self._track_length, 1e-6)
            if lap_frac > 0.5:
                self._crossed_half = True
            if self._crossed_half and lap_frac >= LAP_MIN_PROGRESS_FRAC:
                lap_complete = True
                lap_time = float(t)

            if self._time_since_progress >= self.stall_timeout_s:
                stall = True

        raw_min = float(np.min(self._raw_scan)) if self._raw_scan is not None else LIDAR_MAX_M
        # refresh scan after motion for reward wall term
        scan = cast_lidar(self.occ, x, y, th, n_beams=self.n_lidar)
        self._raw_scan = scan
        raw_min = float(np.min(scan))

        wall_penalty = 0.0
        if raw_min < D_WALL_M:
            wall_penalty = W_WALL * (D_WALL_M - raw_min)

        speed_term = W_SPEED * v if self._speed_unlocked else 0.0
        reward = (
            W_PROGRESS * progress_delta
            + speed_term
            - wall_penalty
            - W_TIME
        )
        terminated = False
        truncated = False
        crash_tag = None
        ttc_hit = self._frontal_ttc_truncate(scan, v)
        if collision:
            pen = COLLISION_FIRST_PENALTY if self.collision_first else COLLISION_PENALTY
            reward -= pen
            terminated = True
            crash_tag = "wall"
            self._note_episode_crash(True)
        if lap_complete:
            terminated = True
            if not collision:
                self._note_episode_crash(False)
        timeout = self._step_count >= self.max_steps or t >= self.timeout_s
        if timeout or stall or ttc_hit:
            truncated = True
            if stall and crash_tag is None:
                crash_tag = "stall"
            elif ttc_hit and crash_tag is None:
                crash_tag = "ttc"
            elif timeout and crash_tag is None:
                crash_tag = "timeout"
            if not collision and (stall or timeout or ttc_hit):
                # Non-collision end still updates crash hist as clean if no wall hit.
                if not terminated:
                    self._note_episode_crash(False)

        self._prev_action = (throttle, steering)
        obs = build_observation(
            scan,
            throttle,
            steering,
            v,
            self._yaw_rate,
            self._ax,
            self._ay,
            n_lidar=self.n_lidar,
            speed_max=self.v_max,
            lidar_dr=self.lidar_dr,
            rng=self._rng if self.lidar_dr else None,
            noise_std=LIDAR_DR_NOISE_STD if self.lidar_dr else 0.0,
            dropout=LIDAR_DR_DROPOUT if self.lidar_dr else 0.0,
            max_range_scale=LIDAR_DR_MAX_RANGE_SCALE if self.lidar_dr else 1.0,
        )
        info = {
            "collision": bool(collision),
            "timeout": bool(timeout and not terminated),
            "stall": bool(stall and not terminated),
            "ttc": bool(ttc_hit and not terminated),
            "lap_time": lap_time,
            "speed": v,
            "yaw_rate": self._yaw_rate,
            "ax": self._ax,
            "ay": self._ay,
            "progress": self._progress,
            "progress_delta": float(progress_delta),
            "progress_frac": self.progress_frac,
            "obs_dim": int(obs.shape[-1]),
            "speed_unlocked": bool(self._speed_unlocked),
            "crash_tag": crash_tag,
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
