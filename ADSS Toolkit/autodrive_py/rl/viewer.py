"""Minimal top-down OpenCV viewer: map + colored cars + race KPIs.

Opt-in only — call from ``rl.watch``, never from the train loop.
Follow mode draws a centered top-middle white label (iteration / timesteps),
a grey sub-label admitting the lag-behind replay, a bottom-left race-KPI block
(adjusted_time / collisions), a red banner for map or contracts mismatch, and
crash / stall markers where twins died.
Full train curves stay on the RL control UI / TensorBoard.

Quit (Windows note)
-------------------
``cv2.waitKey`` only receives keyboard events when the OpenCV window is focused.
Always: click the map window, then press **q** / **Esc**, or use the window **X**.

Critical: after the user closes the window, **never call ``cv2.imshow`` again** —
OpenCV will recreate the window (looks like Watch "reopens"). Once quit is
detected we latch ``_enabled=False`` and tear down; ``show``/``poll`` stay False.
Control UI **Stop Watch** kills the watch subprocess if the window is stuck.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

from .racing_env import OccupancyMap, RacingEnv

# Distinct BGR colors for parallel twin agents (env index → color)
AGENT_COLORS_BGR: list[tuple[int, int, int]] = [
    (40, 40, 255),  # red
    (40, 200, 40),  # green
    (255, 160, 40),  # blue-ish
    (0, 220, 220),  # yellow
    (220, 0, 220),  # magenta
    (220, 220, 0),  # cyan
    (40, 140, 255),  # orange
    (180, 80, 255),  # pink
    (80, 255, 180),  # mint
    (255, 80, 80),  # light blue
    (100, 100, 255),
    (100, 255, 100),
    (255, 100, 100),
    (200, 200, 80),
    (80, 200, 200),
    (200, 80, 200),
]


GHOST_COLOR_BGR = (235, 235, 235)  # FTG baseline ghost (hollow white car)
WARN_COLOR_BGR = (60, 60, 235)  # red banner text / crash marks

# Crash marker glyph per ``RacingEnv`` info["crash_tag"].
CRASH_MARKS: dict[str, str] = {
    "wall": "x",
    "stall": "o",
    "ttc": "t",
    "timeout": "o",
}


def agent_color(i: int) -> tuple[int, int, int]:
    return AGENT_COLORS_BGR[int(i) % len(AGENT_COLORS_BGR)]


def _dim(color: tuple[int, int, int], factor: float = 0.45) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)  # type: ignore[return-value]


class MapViewer:
    """
    One lightweight window. Cost is near-zero when you do not call ``show()``.
    Use ``--every N`` in ``rl.watch`` to redraw sparsely, but call ``poll()``
    every sim step so q/Esc / window-X still work between redraws.

    Multi-agent: pass ``agents=[{x,y,theta,speed,...}, ...]`` to draw several
    colored cars on the same map (viewer twins — not live Subproc poses).
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
        self._ever_shown = False
        # WINDOW_NORMAL so getWindowProperty / close-button detection works on Windows.
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        try:
            h, w = self._base.shape[:2]
            cv2.resizeWindow(self.window, max(320, w), max(240, h))
        except Exception:
            pass
        # Paint immediately — otherwise Windows shows a blank namedWindow until the
        # first ``show()`` (can be many seconds while PPO weights load / first rollout).
        try:
            boot = self.render_frame(banner="loading...")
            cv2.imshow(self.window, boot)
            cv2.waitKey(1)
            self._ever_shown = True
        except Exception:
            pass

    def _build_base(self, occ: OccupancyMap) -> np.ndarray:
        # Darker walls, slightly darker floor so centerline/cars pop.
        gray = np.where(occ.grid, 18, 200).astype(np.uint8)
        h, w = gray.shape
        scale = min(self.max_size / max(h, 1), self.max_size / max(w, 1), 1.0)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _draw_centerline(self, frame: np.ndarray) -> None:
        """Track centerline for loop-track context behind cars."""
        cl = getattr(self.env, "centerline", None)
        if cl is None or len(cl) < 2:
            return
        pts = [self._world_to_img(float(xy[0]), float(xy[1])) for xy in cl]
        arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame,
            [arr],
            isClosed=False,
            color=(160, 160, 160),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    def _world_to_img(self, x: float, y: float) -> tuple[int, int]:
        occ = self.env.occ
        col = (x - occ.origin[0]) / occ.resolution
        row = occ.height - 1 - (y - occ.origin[1]) / occ.resolution
        sx = self._base.shape[1] / occ.width
        sy = self._base.shape[0] / occ.height
        return int(col * sx), int(row * sy)

    def _draw_speed_label(
        self,
        frame: np.ndarray,
        px: int,
        py: int,
        speed: float,
        color: tuple[int, int, int],
        prefix: str = "",
    ) -> None:
        label = f"{prefix}{float(speed):.1f}" if prefix else f"{float(speed):.1f}"
        tx, ty = px - 10 - 5 * len(prefix), max(12, py - 14)
        # No plate: a dark box per car would checkerboard the track at 16 twins.
        self._draw_outlined_text(
            frame, label, (tx, ty), scale=0.35, color=color, plate=False
        )

    def _draw_car(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
        th: float,
        color: tuple[int, int, int],
        *,
        speed: float | None = None,
        beams: np.ndarray | None = None,
        ghost: bool = False,
        done: bool = False,
        label: str = "",
    ) -> None:
        px, py = self._world_to_img(x, y)
        # Sized for large occupancy maps scaled down to ~512px (tiny L=8 vanishes).
        L = 14
        pts = []
        for ang, rad in ((th, L), (th + 2.5, L * 0.65), (th - 2.5, L * 0.65)):
            pts.append(
                [
                    int(px + rad * math.cos(ang)),
                    int(py - rad * math.sin(ang)),
                ]
            )
        poly = np.array(pts, dtype=np.int32)
        if ghost:
            # Hollow outline: the baseline is context, not another contender.
            cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)
        else:
            cv2.fillConvexPoly(frame, poly, _dim(color) if done else color)
            # Dark outline so cars stay visible on light track surface.
            cv2.polylines(frame, [poly], True, (10, 10, 10), 1, cv2.LINE_AA)

        if speed is not None:
            self._draw_speed_label(frame, px, py, float(speed), color, prefix=label)

        if beams is not None and len(beams) > 0:
            fov = 4.7
            n = len(beams)
            beam_color = tuple(int(c * 0.55) for c in color)
            for i in range(0, n, 12):
                a = th - fov / 2 + fov * (i / max(n - 1, 1))
                r = float(beams[i])
                ex = x + r * math.cos(a)
                ey = y + r * math.sin(a)
                qx, qy = self._world_to_img(ex, ey)
                cv2.line(frame, (px, py), (qx, qy), beam_color, 1)

    def _draw_crash_marker(self, frame: np.ndarray, marker: dict[str, Any]) -> None:
        """Mark where a twin died: X = wall, O = stall/timeout, triangle = TTC."""
        px, py = self._world_to_img(float(marker["x"]), float(marker["y"]))
        tag = str(marker.get("tag") or "wall")
        glyph = CRASH_MARKS.get(tag, "x")
        r = 9
        if glyph == "x":
            cv2.line(frame, (px - r, py - r), (px + r, py + r), WARN_COLOR_BGR, 2, cv2.LINE_AA)
            cv2.line(frame, (px - r, py + r), (px + r, py - r), WARN_COLOR_BGR, 2, cv2.LINE_AA)
        elif glyph == "t":
            pts = np.array(
                [[px, py - r], [px - r, py + r], [px + r, py + r]], dtype=np.int32
            )
            cv2.polylines(frame, [pts], True, (40, 170, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (px, py), r, (40, 170, 255), 2, cv2.LINE_AA)
        n = int(marker.get("n", 1) or 1)
        text = f"{tag} x{n}" if n > 1 else tag
        self._draw_outlined_text(
            frame, text, (px + r + 2, py + 4), scale=0.35, color=(230, 230, 230)
        )

    def _draw_warn_banner(self, frame: np.ndarray, text: str) -> int:
        """Full-width red bar at the very top. Returns the height it consumed."""
        h = 26
        band = frame[0:h, :]
        cv2.addWeighted(band, 0.25, np.zeros_like(band), 0.0, 0.0, dst=band)
        text = str(text)[:78]
        scale = 0.48
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x = max(4, (frame.shape[1] - tw) // 2)
        self._draw_outlined_text(
            frame, text, (x, 18), scale=scale, color=(90, 90, 255), plate=False
        )
        return h

    def _draw_kpi_block(self, frame: np.ndarray, lines: Sequence[str]) -> None:
        """Bottom-left race-KPI rows (adjusted_time / collisions / crash tags)."""
        rows = [str(ln) for ln in lines if ln]
        if not rows:
            return
        scale, line_h = 0.42, 15
        # Small maps: keep the top label and the track visible, drop trailing rows.
        max_rows = max(1, (frame.shape[0] - 70) // line_h)
        rows = rows[:max_rows]
        pad, bottom = 6, frame.shape[0] - 28  # clear of the quit hint
        widths = [
            cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for r in rows
        ]
        top = bottom - line_h * len(rows) - pad
        x0, x1 = 4, min(frame.shape[1] - 4, 4 + max(widths) + 2 * pad)
        y0, y1 = max(0, top), min(frame.shape[0], bottom + pad)
        panel = frame[y0:y1, x0:x1]
        cv2.addWeighted(panel, 0.35, np.zeros_like(panel), 0.0, 0.0, dst=panel)
        for i, row in enumerate(rows):
            y = y0 + pad + line_h * (i + 1) - 4
            self._draw_outlined_text(frame, row, (x0 + pad, y), scale=scale, plate=False)

    @staticmethod
    def _watching_line(
        metrics: dict[str, Any] | None,
        banner: str | None = None,
    ) -> str | None:
        """Centered top label, e.g. ``watching iter 12 | ts=49152``."""
        if banner:
            return str(banner)
        if not metrics:
            return None
        # Explicit label from watch --follow takes priority.
        watch = metrics.get("watch_label") or metrics.get("watching")
        if watch:
            return str(watch)

        n_upd = metrics.get("n_updates", metrics.get("iter", metrics.get("rollouts")))
        train_ts = metrics.get("train_ts", metrics.get("timesteps"))
        ep = metrics.get("episode", metrics.get("ep"))
        step = metrics.get("step")

        bits: list[str] = ["watching"]
        if n_upd is not None:
            bits.append(f"iter {int(n_upd)}")
        if train_ts is not None:
            bits.append(f"ts={int(train_ts)}")
        if len(bits) == 1:
            # Standalone / no train status: fold twin ep/step into the same line.
            if ep is not None:
                bits.append(f"ep={int(ep)}")
            if step is not None:
                bits.append(f"step={int(step)}")
        if len(bits) == 1:
            return None
        # "watching iter 12 | ts=49152" when both present; else space-join.
        if n_upd is not None and train_ts is not None:
            return f"watching iter {int(n_upd)} | ts={int(train_ts)}"
        return " ".join(bits)

    @staticmethod
    def _draw_outlined_text(
        frame: np.ndarray,
        text: str,
        org: tuple[int, int],
        *,
        scale: float = 0.55,
        color: tuple[int, int, int] = (255, 255, 255),
        outline: tuple[int, int, int] = (0, 0, 0),
        thickness: int = 1,
        plate: bool = True,
    ) -> None:
        """Text on a darkened plate, readable over both walls and track.

        A halo of offset black copies smears shut at these font scales — the
        plate keeps thin glyphs crisp. Pass ``plate=False`` where the caller
        already darkened the region.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        x, y = org
        if plate:
            (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
            x0, y0 = max(0, x - 3), max(0, y - th - 3)
            x1 = min(frame.shape[1], x + tw + 3)
            y1 = min(frame.shape[0], y + base + 2)
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                cv2.addWeighted(roi, 0.3, np.zeros_like(roi), 0.0, 0.0, dst=roi)
        cv2.putText(frame, text, (x + 1, y + 1), font, scale, outline, thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _draw_centered_top_label(
        self,
        frame: np.ndarray,
        text: str,
        *,
        y_top: int = 0,
        scale: float = 0.55,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> int:
        """Centered label near the top-middle. Returns its baseline y."""
        text = str(text)[:90]
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        x = max(4, (frame.shape[1] - tw) // 2)
        y = y_top + max(th + 6, 22)
        self._draw_outlined_text(frame, text, (x, y), scale=scale, color=color)
        return y

    def render_frame(
        self,
        metrics: dict[str, Any] | None = None,
        agents: Sequence[dict[str, Any]] | None = None,
        banner: str | None = None,
        markers: Sequence[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        frame = self._base.copy()
        self._draw_centerline(frame)

        for marker in markers or ():
            self._draw_crash_marker(frame, marker)

        if agents:
            for i, ag in enumerate(agents):
                color = tuple(ag.get("color") or agent_color(i))
                speed = ag.get("speed")
                if speed is None and "v" in ag:
                    speed = ag["v"]
                self._draw_car(
                    frame,
                    float(ag["x"]),
                    float(ag["y"]),
                    float(ag.get("theta", ag.get("th", 0.0))),
                    color,
                    speed=None if speed is None else float(speed),
                    beams=ag.get("scan") if self.show_beams else None,
                    ghost=bool(ag.get("ghost")),
                    done=bool(ag.get("done")),
                    label=str(ag.get("label") or ""),
                )
        elif self.env._state is not None:
            x, y, th, v = self.env._state
            self._draw_car(
                frame,
                float(x),
                float(y),
                float(th),
                agent_color(0),
                speed=float(v),
                beams=self.env._raw_scan if self.show_beams else None,
            )

        warn = (metrics or {}).get("warn")
        y_top = self._draw_warn_banner(frame, warn) if warn else 0

        label = self._watching_line(metrics, banner=banner)
        if label:
            y = self._draw_centered_top_label(frame, label, y_top=y_top)
            sub = (metrics or {}).get("sub_label")
            if sub:
                # Grey second line: says out loud that these are replay twins.
                self._draw_centered_top_label(
                    frame, sub, y_top=y - 6, scale=0.4, color=(185, 185, 185)
                )

        self._draw_kpi_block(frame, (metrics or {}).get("kpi_lines") or ())

        # Tiny quit hint (not a stats HUD).
        self._draw_outlined_text(
            frame,
            "q/Esc or X to quit (click window first)",
            (8, frame.shape[0] - 8),
            scale=0.4,
            color=(150, 150, 150),
            plate=False,
        )
        return frame

    def _window_closed(self) -> bool:
        """True if user closed the window via the title-bar X (or it vanished)."""
        if not self._enabled:
            return True
        # Before the first successful imshow, property probes are unreliable on Windows.
        if not self._ever_shown:
            return False
        try:
            # OpenCV returns -1 when the window is gone. Do NOT treat prop==0 as
            # closed — that fires on minimize / focus quirks and latches quit forever.
            prop = float(cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE))
            if prop < 0.0:
                return True
        except Exception:
            return True
        return False

    def _handle_key(self, key: int) -> bool:
        """Return False if key means quit."""
        if key in (27, ord("q"), ord("Q")):
            return False
        return True

    def _quit_now(self) -> bool:
        """Latch quit and tear down. Always returns False (caller convenience)."""
        self.close()
        return False

    def _pump_quit(self, wait_ms: int = 1) -> bool:
        """Process OpenCV events. Returns False if user wants quit.

        Call **before** ``imshow`` so a pending close-X is honored without
        recreating the window (Windows OpenCV recreates on imshow after X).
        """
        if not self._enabled:
            return False
        key = cv2.waitKey(max(1, int(wait_ms))) & 0xFF
        if not self._handle_key(key):
            return self._quit_now()
        if self._window_closed():
            return self._quit_now()
        return True

    def poll(self, wait_ms: int = 1) -> bool:
        """Pump OpenCV events without redrawing. Returns False if user wants quit.

        Must be called every sim step when ``show()`` is skipped via ``--every N``,
        otherwise q/Esc and the window close button are ignored between frames.
        """
        return self._pump_quit(wait_ms)

    def show(
        self,
        metrics: dict[str, Any] | None = None,
        wait_ms: int = 1,
        agents: Sequence[dict[str, Any]] | None = None,
        banner: str | None = None,
        markers: Sequence[dict[str, Any]] | None = None,
    ) -> bool:
        """Show frame. Returns False if user pressed q/Esc or closed the window."""
        if not self._enabled:
            return False
        # Honor pending X/q from the previous frame *before* imshow — otherwise
        # OpenCV recreates the destroyed window and it looks like Watch reopens.
        # Skip the pre-pump on the very first frame so we never latch quit on a
        # blank namedWindow before content is painted.
        if self._ever_shown and not self._pump_quit(1):
            return False
        frame = self.render_frame(metrics, agents=agents, banner=banner, markers=markers)
        try:
            cv2.imshow(self.window, frame)
            self._ever_shown = True
        except Exception:
            return self._quit_now()
        if not self._pump_quit(max(1, int(wait_ms))):
            return False
        return True

    def close(self) -> None:
        if not self._enabled and not hasattr(self, "_destroy_done"):
            # Already latched; still try destroy once if needed.
            pass
        self._enabled = False
        if getattr(self, "_destroy_done", False):
            return
        self._destroy_done = True
        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            cv2.waitKey(1)  # flush destroy on Windows
        except Exception:
            pass
