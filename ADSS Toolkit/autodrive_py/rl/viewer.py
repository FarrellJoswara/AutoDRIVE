"""Minimal top-down OpenCV viewer: map + car + a few live numbers.

Opt-in only — call from ``rl.watch``, never from the train loop.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .racing_env import OccupancyMap, RacingEnv


class MapViewer:
    """
    One lightweight window. Cost is near-zero when you do not call ``show()``.
    Use ``--every N`` in ``rl.watch`` to redraw sparsely.
    """

    def __init__(
        self,
        env: RacingEnv,
        max_size: int = 512,
        window: str = "F1TENTH RL",
        show_beams: bool = True,
    ):
        self.env = env
        self.window = window
        self.max_size = int(max_size)
        self.show_beams = bool(show_beams)
        self._base = self._build_base(env.occ)
        self._enabled = True

    def _build_base(self, occ: OccupancyMap) -> np.ndarray:
        gray = np.where(occ.grid, 30, 220).astype(np.uint8)
        h, w = gray.shape
        scale = min(self.max_size / max(h, 1), self.max_size / max(w, 1), 1.0)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _world_to_img(self, x: float, y: float) -> tuple[int, int]:
        occ = self.env.occ
        col = (x - occ.origin[0]) / occ.resolution
        row = occ.height - 1 - (y - occ.origin[1]) / occ.resolution
        sx = self._base.shape[1] / occ.width
        sy = self._base.shape[0] / occ.height
        return int(col * sx), int(row * sy)

    def render_frame(self, metrics: dict[str, Any] | None = None) -> np.ndarray:
        frame = self._base.copy()
        if self.env._state is None:
            self._draw_overlay(frame, metrics)
            return frame

        x, y, th, v = self.env._state
        px, py = self._world_to_img(x, y)

        # Car as a small heading triangle
        L = 8
        pts = []
        for ang, rad in ((th, L), (th + 2.5, L * 0.6), (th - 2.5, L * 0.6)):
            pts.append(
                [
                    int(px + rad * math.cos(ang)),
                    int(py - rad * math.sin(ang)),
                ]
            )
        cv2.fillConvexPoly(frame, np.array(pts, dtype=np.int32), (40, 40, 255))

        # Sparse LiDAR rays (optional; no camera)
        if self.show_beams:
            scan = self.env._raw_scan
            if scan is not None and len(scan) > 0:
                fov = 4.7
                n = len(scan)
                for i in range(0, n, 8):
                    a = th - fov / 2 + fov * (i / max(n - 1, 1))
                    r = float(scan[i])
                    ex = x + r * math.cos(a)
                    ey = y + r * math.sin(a)
                    qx, qy = self._world_to_img(ex, ey)
                    cv2.line(frame, (px, py), (qx, qy), (80, 200, 80), 1)

        if metrics is None:
            metrics = {}
        metrics.setdefault("speed", float(v))
        metrics.setdefault("step", int(self.env._step_count))
        self._draw_overlay(frame, metrics)
        return frame

    def _draw_overlay(self, frame: np.ndarray, metrics: dict[str, Any] | None) -> None:
        m = metrics or {}
        speed = m.get("speed")
        step = m.get("step", 0)
        ep = m.get("episode")
        ep_ret = m.get("ep_return")
        collisions = m.get("collisions")
        train_ts = m.get("train_timesteps")
        train_rew = m.get("train_ep_rew")
        train_eps = m.get("train_episodes")
        train_cols = m.get("train_collisions")
        run_id = m.get("run_id")

        lines: list[str] = []
        if run_id:
            rid = str(run_id)
            lines.append(f"run={rid[-24:]}" if len(rid) > 24 else f"run={rid}")
        if train_ts is not None:
            lines.append(f"train_ts={int(train_ts)}")
        if train_rew is not None:
            lines.append(f"train_ep_rew={float(train_rew):.1f}")
        if train_eps is not None:
            lines.append(f"train_eps={int(train_eps)}")
        if train_cols is not None:
            lines.append(f"train_cols={int(train_cols)}")
        if speed is not None:
            lines.append(f"v={float(speed):.1f} m/s")
        lines.append(f"twin_step={int(step)}" if train_ts is not None else f"step={int(step)}")
        if ep is not None:
            lines.append(f"twin_ep={int(ep)}" if train_ts is not None else f"ep={int(ep)}")
        if ep_ret is not None:
            lines.append(f"return={float(ep_ret):.1f}")
        if collisions is not None:
            lines.append(f"collisions={int(collisions)}")

        y = 18
        for line in lines:
            cv2.putText(
                frame,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            y += 18

    def show(self, metrics: dict[str, Any] | None = None, wait_ms: int = 1) -> bool:
        """Show frame. Returns False if user pressed q/Esc (caller may stop)."""
        if not self._enabled:
            return True
        frame = self.render_frame(metrics)
        cv2.imshow(self.window, frame)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (27, ord("q")):
            self._enabled = False
            cv2.destroyWindow(self.window)
            return False
        return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass
