"""Watch a policy on a top-down map (separate process — zero train overhead).

Modes
-----
- Default: roll out FTG or a fixed PPO zip in a local Gym env + OpenCV.
- ``--follow``: lag-behind trail — read ``live_status.json`` for train metrics and
  drive *N colored twin cars* with the latest saved weights on one map.
  Does **not** stream train Subproc poses; training stays headless.

Overlay
-------
Race KPIs (``adjusted_time = lap + 10·collisions``), a persistent FTG ghost to
race against, crash / stall markers where twins died, and a red banner when the
watched map or the run contracts do not match the training run. Every number is
**unofficial** — promote models from ``eval_cli`` / ``compare_models`` only.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from .contracts import CONTRACTS_VERSION, obs_dim
from .eval_protocol import protocol_timeout_s
from .ftg import FollowTheGap
from .live_status import pick_status_for_operator, read_live_status, resolve_latest_weights
from .racing_env import RacingEnv, nearest_centerline_progress, resolve_map_yaml
from .ui_ops import read_operator_run_pin
from .viewer import GHOST_COLOR_BGR, MapViewer, agent_color
from .watch_kpi import (
    FleetKpi,
    beat_line,
    contracts_warning,
    map_mismatch_warning,
    read_run_config,
    train_status_line,
)

# live_status older than this means train stopped / crashed, not just slow.
STATUS_STALE_S = 90.0
# Keep the map readable when many twins die in one episode.
MAX_CRASH_MARKERS = 40
# Crashes closer than this with the same tag collapse into one counted mark (m).
MARKER_MERGE_M = 0.8
# Hold the last frame of an episode so crash marks are actually seen.
EPISODE_HOLD_MS = 450
# Follow redraw default (higher = snappier on weak machines; override with --every / RL_WATCH_EVERY).
FOLLOW_EVERY_DEFAULT = 8
STANDALONE_EVERY_DEFAULT = 2


def _env_bool(name: str) -> bool | None:
    """Parse ``1/true/yes/on`` vs ``0/false/no/off``; unset → None."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    v = str(raw).strip().lower()
    if not v:
        return None
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _viewer_quit(viewer: MapViewer | None, code: int = 0) -> int:
    """Close viewer and end the process for good (no reopen / no follow loop)."""
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass
    return code


def _pose_along_centerline(
    centerline: np.ndarray,
    arc_s: float,
    *,
    lateral: float = 0.0,
) -> tuple[float, float, float]:
    """Sample (x, y, heading) at arc length ``arc_s`` with optional lateral offset (m)."""
    diffs = centerline[1:] - centerline[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    track_len = float(cum[-1]) if len(cum) else 0.0
    if track_len <= 1e-6:
        p0 = centerline[0]
        return float(p0[0]), float(p0[1]), 0.0
    s = float(arc_s) % track_len
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = max(0, min(i, len(centerline) - 2))
    seg = max(float(seg_len[i]), 1e-9)
    t = (s - float(cum[i])) / seg
    p0, p1 = centerline[i], centerline[i + 1]
    x = float(p0[0] + t * (p1[0] - p0[0]))
    y = float(p0[1] + t * (p1[1] - p0[1]))
    th = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
    if abs(lateral) > 1e-9:
        x += lateral * math.cos(th + math.pi / 2.0)
        y += lateral * math.sin(th + math.pi / 2.0)
    return x, y, th


def _reset_twin(env: RacingEnv, twin_idx: int, twin_n: int, *, seed: int | None = None):
    """Reset one viewer twin with a unique spawn so colored cars are not stacked.

    Training envs are untouched. Watch twins share one map; without offsets,
    identical starts + the same policy look like a single car.
    """
    obs, info = env.reset(seed=seed)
    n = max(1, int(twin_n))
    i = int(twin_idx)
    if n <= 1:
        return obs, info

    # Spread along track (~2.2 m) plus lateral / heading jitter so colors separate.
    spacing = 2.2
    lateral = (i - (n - 1) / 2.0) * 0.18
    heading_jit = (i - (n - 1) / 2.0) * 0.06

    if env.centerline is not None and len(env.centerline) >= 2:
        x, y, th = _pose_along_centerline(
            env.centerline, arc_s=i * spacing, lateral=lateral
        )
        th = th + heading_jit
        # Back off lateral if spawn lands in an obstacle.
        if env._in_collision(x, y):
            x, y, th = _pose_along_centerline(
                env.centerline, arc_s=i * spacing, lateral=0.0
            )
            th = th + heading_jit
        env._state = np.array([x, y, th, 0.0], dtype=np.float64)
        env._progress, env._seg_idx = nearest_centerline_progress(env._state[:2], env.centerline)
        env._lap_progress_base = env._progress
        env._s_unwrapped = 0.0
        env._s_hw = 0.0
        env._time_since_progress = 0.0
    else:
        pose = env.start_pose.copy()
        # No centerline: stagger laterally / forward from the shared start.
        th0 = float(pose[2])
        pose[0] += i * 0.35 * math.cos(th0) + lateral * math.cos(th0 + math.pi / 2.0)
        pose[1] += i * 0.35 * math.sin(th0) + lateral * math.sin(th0 + math.pi / 2.0)
        pose[2] = th0 + heading_jit
        env._state = np.array([pose[0], pose[1], pose[2], 0.0], dtype=np.float64)

    obs = env._obs()
    return obs, info


def _check_ppo_obs(model, env: RacingEnv, label: str) -> None:
    """Fail fast if a v1 zip is loaded against the v2 env."""
    expected = int(env.observation_space.shape[-1])
    got = int(model.observation_space.shape[-1])
    if got != expected:
        raise SystemExit(
            f"ERROR: {label} obs_dim={got} but env expects {expected} "
            f"(contracts {CONTRACTS_VERSION}). Retrain with v2 obs; old LiDAR-only zips are obsolete."
        )
    if expected != obs_dim(env.n_lidar):
        raise SystemExit(
            f"ERROR: env obs_dim={expected} != contracts obs_dim({env.n_lidar})={obs_dim(env.n_lidar)}"
        )


def _agent_snapshot(
    env: RacingEnv,
    idx: int,
    *,
    done: bool = False,
    with_scan: bool = False,
) -> dict:
    x, y, th, v = env._state
    return {
        "x": float(x),
        "y": float(y),
        "theta": float(th),
        "speed": float(v),
        "color": agent_color(idx),
        "scan": env._raw_scan if with_scan else None,
        "done": bool(done),
    }


def _crash_marker(env: RacingEnv, tag: str) -> dict:
    x, y = float(env._state[0]), float(env._state[1])
    return {"x": x, "y": y, "tag": str(tag)}


def _push_marker(markers: list[dict], marker: dict | None) -> None:
    """Add a crash mark, merging repeats at the same corner into one ``xN``.

    The FTG ghost respawns after every wreck, so a graveyard forms on whichever
    corner kills it — one counted mark says the same thing without hiding the map.
    """
    if marker is None:
        return
    for old in markers:
        if old["tag"] != marker["tag"]:
            continue
        if math.hypot(old["x"] - marker["x"], old["y"] - marker["y"]) <= MARKER_MERGE_M:
            old["n"] = int(old.get("n", 1)) + 1
            return
    marker["n"] = 1
    markers.append(marker)
    if len(markers) > MAX_CRASH_MARKERS:
        del markers[: len(markers) - MAX_CRASH_MARKERS]


class GhostRunner:
    """Persistent FTG baseline car drawn hollow-white next to the twins.

    It owns its own episode lifecycle (resets the moment it crashes or laps) so
    the KPI block always has a real classical time to compare against, even when
    every PPO twin dies in the first corner.

    Uses ``protocol_timeout_s()`` (official_v2: 400 s) — not train ``TIMEOUT_S``
    (60 s) — so the ghost can finish map0 (~205 s) instead of a false DNF.
    """

    def __init__(
        self,
        map_yaml: Path,
        *,
        seed: int = 991,
        n_lidar: int | None = None,
        timeout_s: float | None = None,
    ):
        kwargs: dict = {"timeout_s": float(timeout_s if timeout_s is not None else protocol_timeout_s())}
        if n_lidar:
            kwargs["n_lidar"] = int(n_lidar)
        self.env = RacingEnv(map_yaml=map_yaml, seed=seed, **kwargs)
        self.policy = FollowTheGap(**({"n_lidar": int(n_lidar)} if n_lidar else {}))
        self.kpi = FleetKpi(1, label="FTG")
        self.seed = int(seed)
        self.reset()

    def reset(self) -> None:
        self.env.reset(seed=self.seed)
        self.kpi.start_episode(1)

    def step(self) -> dict | None:
        """Advance one sim step; returns a crash marker when the ghost dies."""
        action = self.policy.act(self.env._raw_scan)
        _, _, terminated, truncated, info = self.env.step(action)
        tag = self.kpi.record(info)
        marker = _crash_marker(self.env, tag) if tag else None
        if terminated or truncated:
            self.kpi.note_finished()
            self.kpi.end_episode()
            self.reset()
        return marker

    def snapshot(self) -> dict:
        x, y, th, v = self.env._state
        return {
            "x": float(x),
            "y": float(y),
            "theta": float(th),
            "speed": float(v),
            "color": GHOST_COLOR_BGR,
            "ghost": True,
            "label": "FTG ",
        }

    def line(self) -> str:
        """Prefer the rolling window; fall back to the live episode. Matches ``beat_line``."""
        if self.kpi.history:
            return self.kpi.session_line()
        return self.kpi.line(suffix="ghost")


def _kpi_lines(
    twins: FleetKpi,
    ghost: GhostRunner | None,
    status: dict | None = None,
    *,
    compact: bool = False,
) -> list[str]:
    """Bottom-left overlay rows: race score first, train counters last.

    ``compact`` keeps the live score (+ beat vs FTG if ghost) and drops the
    rolling session dump / verbose train row clutter.
    """
    lines = [twins.line()]
    if ghost is not None:
        if not compact:
            lines.append(ghost.line())
        lines.append(beat_line(twins, ghost.kpi))
    if not compact:
        if twins.session.episodes > 0:
            lines.append(twins.session_line())
        train = train_status_line(status)
        if train:
            lines.append(train)
    elif status:
        # Honesty strip: phase + crash_rate + short ts|/s (no rew / session dump).
        try:
            bits = []
            phase = status.get("phase")
            if phase:
                bits.append(str(phase))
            ts = status.get("timesteps")
            if ts is not None:
                bits.append(f"ts {int(ts)}")
            sps = status.get("steps_per_sec")
            if sps is not None:
                bits.append(f"{float(sps):.0f}/s")
            rate = status.get("crash_rate_estimate")
            if rate is not None:
                bits.append(f"crash {100.0 * float(rate):.0f}%")
            if bits:
                lines.append("train " + " | ".join(bits))
        except (TypeError, ValueError):
            pass
    return lines


def _frame_payload(
    envs: list[RacingEnv],
    dones: list[bool],
    ghost: GhostRunner | None,
    twins: FleetKpi,
    status: dict | None,
    *,
    label: str,
    sub_label: str,
    warn: str | None = None,
    ep: int = 0,
    step: int = 0,
    compact: bool = False,
) -> tuple[list[dict], dict]:
    agents = [
        _agent_snapshot(env, i, done=dones[i], with_scan=(i == 0))
        for i, env in enumerate(envs)
    ]
    if ghost is not None:
        agents.append(ghost.snapshot())
    metrics: dict = {
        "watch_label": label,
        "sub_label": sub_label,
        "warn": warn,
        "episode": ep,
        "step": step,
        "kpi_lines": _kpi_lines(twins, ghost, status, compact=compact),
    }
    return agents, metrics


def _age_str(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 90:
        return f"{int(s)}s"
    if s < 5400:
        return f"{int(s // 60)}m"
    return f"{int(s // 3600)}h{int((s % 3600) // 60)}m"


def _watching_label(status: dict | None, *, using_ftg: bool = False) -> str:
    """Human-readable follow overlay, e.g. ``watching iter 12 | ts=49152``."""
    st = status or {}
    n_upd = st.get("n_updates")
    if n_upd is None:
        n_upd = st.get("rollouts")
    ts = st.get("timesteps")
    try:
        n_upd_i = int(n_upd) if n_upd is not None else None
    except (TypeError, ValueError):
        n_upd_i = None
    try:
        ts_i = int(ts) if ts is not None else None
    except (TypeError, ValueError):
        ts_i = None

    if using_ftg:
        if ts_i is not None:
            return f"watching FTG | waiting weights | ts={ts_i}"
        return "watching FTG | waiting for train weights"

    if n_upd_i is not None and ts_i is not None:
        return f"watching iter {n_upd_i} | ts={ts_i}"
    if n_upd_i is not None:
        return f"watching iter {n_upd_i}"
    if ts_i is not None:
        return f"watching ts={ts_i}"
    return "watching (no live_status yet)"


def _lag_sub_label(weights: Path | None, *, now: float | None = None) -> str:
    """Second line that admits Watch is a replay, not the training cars."""
    if weights is None:
        return "lag-behind replay - FTG placeholder until the first weights land"
    try:
        age = (now or time.time()) - weights.stat().st_mtime
        aged = f" ({_age_str(age)} old)"
    except OSError:
        aged = ""
    return f"lag-behind replay of {weights.name}{aged} - not the training cars"


def _status_stale_warning(status: dict | None, *, now: float | None = None) -> str | None:
    """Loud cue when the train side stopped writing live_status."""
    if not status:
        return None
    ts = status.get("unix_time")
    try:
        age = (now or time.time()) - float(ts)
    except (TypeError, ValueError):
        return None
    if age < STATUS_STALE_S:
        return None
    return f"TRAIN STALE: no live_status for {_age_str(age)} (train stopped or crashed)"


def _config_warning(
    config: dict | None,
    map_yaml: Path,
    run_id: str | None = None,
) -> str | None:
    """Banner from the run fingerprint: contracts first, then wrong/changed map.

    Hashes map files, so call it only when the followed run changes.
    """
    return contracts_warning(config) or map_mismatch_warning(
        map_yaml, config, run_id=run_id
    )


def _resolve_timeout_s(args) -> float:
    """Viewer episode budget: CLI override, else official protocol timeout."""
    raw = getattr(args, "timeout_s", None)
    if raw is not None and float(raw) > 0:
        return float(raw)
    return protocol_timeout_s()


def _env_kwargs_from_config(config: dict | None, *, timeout_s: float | None = None) -> dict:
    """Mirror the run's env flags so twins fail the same way training does.

    Always sets ``timeout_s`` from the protocol (or an explicit override) so
    viewer episodes can complete a lap — train's 60 s budget is shaping-only.
    """
    out: dict = {"timeout_s": float(timeout_s if timeout_s is not None else protocol_timeout_s())}
    if not config:
        return out
    for key in ("ttc_truncate", "spawn_jitter"):
        if key in config:
            out[key] = bool(config[key])
    n_lidar = config.get("n_lidar")
    if n_lidar:
        out["n_lidar"] = int(n_lidar)
    return out


def _make_envs(map_yaml: Path, n: int, seed: int, env_kwargs: dict) -> list[RacingEnv]:
    kwargs = {"timeout_s": protocol_timeout_s(), **(env_kwargs or {})}
    return [RacingEnv(map_yaml=map_yaml, seed=seed + i, **kwargs) for i in range(n)]


def _run_standalone(args) -> int:
    maps_root = Path(__file__).resolve().parent / "maps"
    map_yaml = resolve_map_yaml(args.map, maps_root)
    n_twins = max(1, int(args.n_envs))
    timeout_s = _resolve_timeout_s(args)
    envs = _make_envs(map_yaml, n_twins, args.seed, {"timeout_s": timeout_s})
    viewer = MapViewer(envs[0], show_beams=not args.no_beams)

    model = None
    ftg = None
    if args.policy == "ppo":
        from stable_baselines3 import PPO

        model_path = Path(args.model) if args.model else None
        if model_path is None or not model_path.exists():
            zips = sorted((Path(__file__).resolve().parent / "models").glob("*/best_model.zip"))
            if not zips:
                print("ERROR: no PPO model found; pass --model path/to/best_model.zip")
                return 2
            model_path = zips[-1]
        print(f"loading {model_path}")
        model = PPO.load(str(model_path), device="auto")
        _check_ppo_obs(model, envs[0], str(model_path))
    else:
        ftg = FollowTheGap()

    # Ghost only earns its pixels next to a learned policy.
    ghost = (
        GhostRunner(map_yaml, seed=args.seed + 991, timeout_s=timeout_s)
        if (args.policy == "ppo" and not args.no_ghost)
        else None
    )
    twins = FleetKpi(n_twins, label="PPO" if args.policy == "ppo" else "FTG")
    sub_label = f"offline rollout - {args.policy} on {map_yaml.stem} (not a training run)"

    print(
        f"Viewer: {n_twins} colored twin(s) on one map. "
        "Click the OpenCV window, then q/Esc or window X to quit. "
        "Independent of training (use TensorBoard for curves)."
    )
    # Stochastic twin actions so multi-car paths diverge (viewer-only; train unchanged).
    det = bool(getattr(args, "deterministic", False))
    for ep in range(args.episodes):
        states = [
            _reset_twin(env, i, n_twins, seed=args.seed + i) for i, env in enumerate(envs)
        ]
        obss = [s[0] for s in states]
        dones = [False] * n_twins
        twins.start_episode(n_twins)
        markers: list[dict] = []
        step = 0
        ep_returns = [0.0] * n_twins
        while not all(dones):
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                if args.policy == "ftg":
                    action = ftg.act(env._raw_scan)
                else:
                    action, _ = model.predict(obss[i], deterministic=det)
                obss[i], reward, terminated, truncated, info = env.step(action)
                ep_returns[i] += float(reward)
                tag = twins.record(info)
                if terminated or truncated:
                    dones[i] = True
                    twins.note_finished()
                    if tag:
                        _push_marker(markers, _crash_marker(env, tag))
            if ghost is not None:
                _push_marker(markers, ghost.step())
            step += 1
            every = max(1, args.every)
            if step % every == 0:
                agents, metrics = _frame_payload(
                    envs,
                    dones,
                    ghost,
                    twins,
                    None,
                    label=f"watching ep={ep} step={step}",
                    sub_label=sub_label,
                    ep=ep,
                    step=step,
                    compact=bool(getattr(args, "compact", False)),
                )
                if not viewer.show(agents=agents, metrics=metrics, markers=markers, wait_ms=1):
                    return _viewer_quit(viewer)
            else:
                # Keep waitKey / close-button responsive between sparse redraws.
                if not viewer.poll(wait_ms=1):
                    return _viewer_quit(viewer)

        twins.end_episode()
        # Hold the death frame: crashes usually land between sparse redraws.
        agents, metrics = _frame_payload(
            envs,
            dones,
            ghost,
            twins,
            None,
            label=f"watching ep={ep} step={step} | episode over",
            sub_label=sub_label,
            ep=ep,
            step=step,
            compact=bool(getattr(args, "compact", False)),
        )
        if not viewer.show(
            agents=agents, metrics=metrics, markers=markers, wait_ms=EPISODE_HOLD_MS
        ):
            return _viewer_quit(viewer)
        print(
            f"episode={ep} steps={step} mean_return={sum(ep_returns)/n_twins:.1f} "
            f"| {twins.session_line()}"
        )
    return _viewer_quit(viewer)


def _weights_newer(path: Path, loaded_path: Path | None, loaded_mtime: float) -> bool:
    """True if ``path`` is a different/newer zip than what is currently loaded."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return False
    if loaded_path is None:
        return True
    if path.resolve() != loaded_path.resolve():
        return True
    return mtime > loaded_mtime


def _run_follow(args) -> int:
    """Lag-behind twins: train stats from JSON + N local rollouts of latest weights."""
    from stable_baselines3 import PPO

    rl_root = Path(__file__).resolve().parent
    runs_root = rl_root / "runs"
    models_root = rl_root / "models"
    maps_root = rl_root / "maps"
    logs_root = rl_root / "logs"
    map_yaml = resolve_map_yaml(args.map, maps_root)
    pin_run = bool(args.run_id)
    operator_pin = None if pin_run else read_operator_run_pin(logs_root)

    status_path: Path | None = None
    status: dict | None = None
    if pin_run:
        status_path = runs_root / args.run_id / "live_status.json"
        status = read_live_status(status_path)
        if status is None:
            print(f"Waiting for {status_path} ...")
    else:
        found = pick_status_for_operator(runs_root, preferred_run_id=operator_pin)
        if found:
            status_path, status = found
            print(f"Following {status_path}")
        else:
            print(f"Waiting for any {runs_root}/*/live_status.json ...")

    # Twin count: CLI --n-envs, else live_status n_envs, else 8
    def _resolve_n_twins(st: dict | None) -> int:
        if int(args.n_envs) > 0:
            return max(1, int(args.n_envs))
        if st and st.get("n_envs"):
            return max(1, int(st["n_envs"]))
        return 8

    def _refresh_status() -> None:
        """Re-read live_status; when unpinned, prefer CURRENT_RUN / highest timesteps."""
        nonlocal status_path, status
        if pin_run:
            if status_path is not None:
                status = read_live_status(status_path) or status
            return
        found = pick_status_for_operator(
            runs_root, preferred_run_id=read_operator_run_pin(logs_root)
        )
        if found:
            new_path, new_status = found
            if status_path is None or new_path != status_path:
                print(f"Following {new_path}")
            status_path, status = new_path, new_status
        elif status_path is not None:
            status = read_live_status(status_path) or status

    cfg_run_id: str | None = None
    config: dict | None = None
    config_warn: str | None = None
    seen_config = False

    def _refresh_config() -> None:
        """Reload the run fingerprint (and its banner) when the run changes."""
        nonlocal cfg_run_id, config, config_warn, seen_config
        rid = (status or {}).get("run_id") or args.run_id or None
        if rid == cfg_run_id and seen_config:
            return
        cfg_run_id = rid
        seen_config = True
        config = read_run_config(models_root, rid)
        config_warn = _config_warning(config, map_yaml, rid)
        if config_warn:
            print(f"WARNING: {config_warn}")

    def _banner() -> str | None:
        """Highest-priority banner: contracts > wrong map > dead train."""
        return config_warn or _status_stale_warning(status)

    _refresh_config()
    timeout_s = _resolve_timeout_s(args)
    env_kwargs = _env_kwargs_from_config(config, timeout_s=timeout_s)
    n_twins = _resolve_n_twins(status)
    envs = _make_envs(map_yaml, n_twins, args.seed, env_kwargs)
    viewer = MapViewer(envs[0], show_beams=not args.no_beams, window="F1TENTH RL (follow)")
    ftg = FollowTheGap()
    ghost = (
        None
        if args.no_ghost
        else GhostRunner(
            map_yaml,
            seed=args.seed + 991,
            n_lidar=env_kwargs.get("n_lidar"),
            timeout_s=timeout_s,
        )
    )
    twins = FleetKpi(n_twins, label="PPO")

    model = None
    loaded_path: Path | None = None
    loaded_mtime = -1.0
    ep = 0
    print(
        f"Follow mode: {n_twins} colored viewer twins (offset spawns, stochastic actions). "
        "Always loads the newest weights for the current run (latest_model.zip / checkpoints). "
        "Until train writes weights, twins drive with FTG so the map is never empty. "
        "Overlay shows unofficial race KPIs (adjusted_time = lap + 10*collisions). "
        "Click the OpenCV window -> q/Esc or window X to quit. "
        "Or use control UI Stop Watch / Open Watch to reload. "
        "Train stays headless; stats on http://127.0.0.1:7860"
    )
    # Viewer-only: stochastic predict so twins diverge (train policy unchanged).
    det = bool(getattr(args, "deterministic", False))

    while True:
        # Never recreate a closed window (resize used to call MapViewer again).
        if viewer is None or not getattr(viewer, "_enabled", False):
            return _viewer_quit(viewer)

        _refresh_status()
        _refresh_config()

        # Grow/shrink twin fleet if status reports a different n_envs and CLI left default 0
        want = _resolve_n_twins(status)
        if want != n_twins and int(args.n_envs) <= 0:
            print(f"Resizing viewer twins {n_twins} -> {want}")
            n_twins = want
            envs = _make_envs(map_yaml, n_twins, args.seed, env_kwargs)
            # Rebuild only while still running — never after user quit.
            if not getattr(viewer, "_enabled", False):
                return _viewer_quit(viewer)
            viewer.close()
            viewer = MapViewer(envs[0], show_beams=not args.no_beams, window="F1TENTH RL (follow)")

        run_id = (status or {}).get("run_id") or args.run_id
        weights = None
        if run_id:
            weights = resolve_latest_weights(models_root / str(run_id), status)

        if weights is not None and _weights_newer(weights, loaded_path, loaded_mtime):
            print(f"loading twin weights {weights}")
            try:
                model = PPO.load(str(weights), device="auto")
                _check_ppo_obs(model, envs[0], str(weights))
                loaded_path = weights
                loaded_mtime = weights.stat().st_mtime
            except SystemExit:
                raise
            except Exception as exc:
                print(f"WARNING: failed to load {weights}: {exc}; staying on FTG")
                model = None
                loaded_path = None
                loaded_mtime = -1.0

        using_ftg = model is None
        watch_label = _watching_label(status, using_ftg=using_ftg)
        sub_label = _lag_sub_label(None if using_ftg else loaded_path)
        warn = _banner()
        # Twins already are FTG here — a second FTG car would be noise.
        ep_ghost = None if using_ftg else ghost
        label = "FTG" if using_ftg else "PPO"
        if label != twins.label:
            # Weights just landed (or failed): don't average FTG episodes into PPO's score.
            twins = FleetKpi(n_twins, label=label)

        states = [
            _reset_twin(env, i, n_twins, seed=args.seed + i) for i, env in enumerate(envs)
        ]
        obss = [s[0] for s in states]
        dones = [False] * n_twins
        twins.start_episode(n_twins)
        markers: list[dict] = []
        step = 0
        ep_returns = [0.0] * n_twins
        hot_swap = False
        every = max(1, args.every)
        while not all(dones):
            for i, env in enumerate(envs):
                if dones[i]:
                    continue
                if using_ftg or model is None:
                    action = ftg.act(env._raw_scan)
                else:
                    action, _ = model.predict(obss[i], deterministic=det)
                obss[i], reward, terminated, truncated, info = env.step(action)
                ep_returns[i] += float(reward)
                tag = twins.record(info)
                if terminated or truncated:
                    dones[i] = True
                    twins.note_finished()
                    if tag:
                        _push_marker(markers, _crash_marker(env, tag))
            if ep_ghost is not None:
                _push_marker(markers, ep_ghost.step())
            step += 1

            if step % 20 == 0:
                _refresh_status()
                _refresh_config()
                watch_label = _watching_label(status, using_ftg=using_ftg)
                sub_label = _lag_sub_label(None if using_ftg else loaded_path)
                warn = _banner()
                if run_id:
                    # Prefer run_id from refreshed status (may switch when unpinned).
                    run_id = (status or {}).get("run_id") or run_id
                    newer = resolve_latest_weights(models_root / str(run_id), status)
                    if newer is not None and _weights_newer(newer, loaded_path, loaded_mtime):
                        hot_swap = True
                        break

            if step % every == 0:
                agents, metrics = _frame_payload(
                    envs,
                    dones,
                    ep_ghost,
                    twins,
                    status,
                    label=watch_label,
                    sub_label=sub_label,
                    warn=warn,
                    ep=ep,
                    step=step,
                    compact=bool(getattr(args, "compact", False)),
                )
                if not viewer.show(agents=agents, metrics=metrics, markers=markers, wait_ms=1):
                    return _viewer_quit(viewer)
            else:
                if not viewer.poll(wait_ms=1):
                    return _viewer_quit(viewer)

            # While on FTG, re-check often so we pick up latest_model as soon as it lands.
            if using_ftg and step % max(every * 4, 20) == 0 and run_id:
                _refresh_status()
                run_id = (status or {}).get("run_id") or run_id
                maybe = resolve_latest_weights(models_root / str(run_id), status)
                if maybe is not None:
                    hot_swap = True
                    break

        twins.end_episode()
        if not hot_swap:
            # Hold the death frame; a hot-swap should cut to new weights at once.
            agents, metrics = _frame_payload(
                envs,
                dones,
                ep_ghost,
                twins,
                status,
                label=f"{watch_label} | episode over",
                sub_label=sub_label,
                warn=warn,
                ep=ep,
                step=step,
                compact=bool(getattr(args, "compact", False)),
            )
            if not viewer.show(
                agents=agents, metrics=metrics, markers=markers, wait_ms=EPISODE_HOLD_MS
            ):
                return _viewer_quit(viewer)
        print(
            f"twin ep={ep} steps={step} mean_return={sum(ep_returns)/n_twins:.1f} "
            f"twins={n_twins} train_ts={(status or {}).get('timesteps')} "
            f"policy={'ftg' if using_ftg else (loaded_path.name if loaded_path else None)}"
            + (" (hot-swap)" if hot_swap else "")
        )
        print(f"  {twins.session_line()}")
        if ep_ghost is not None:
            print(f"  {ep_ghost.line()}")
        ep += 1
        if args.episodes > 0 and ep >= args.episodes:
            break

    return _viewer_quit(viewer)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal live viewer (map + colored twin cars). Not used during training."
    )
    parser.add_argument("--map", type=str, default="map0")
    parser.add_argument("--policy", choices=["ftg", "ppo"], default="ftg")
    parser.add_argument("--model", type=str, default="", help="SB3 zip for --policy ppo")
    parser.add_argument(
        "--every",
        type=int,
        default=None,
        help=(
            "Redraw every N steps (higher = lighter / snappier on weak machines). "
            f"Default: {FOLLOW_EVERY_DEFAULT} in --follow, {STANDALONE_EVERY_DEFAULT} standalone; "
            "override with RL_WATCH_EVERY."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Episodes then exit (follow: 0 = run forever until q)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-beams",
        action="store_true",
        help="Hide sparse LiDAR rays (map + car + numbers only)",
    )
    parser.add_argument(
        "--no-ghost",
        action="store_true",
        help="Hide the hollow white FTG baseline ghost car",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help=(
            "Episode sim-time budget in seconds. Default: eval_protocol.yaml "
            "timeout_s (official_v2: 400) so FTG can finish a lap. Train still "
            "uses contracts.TIMEOUT_S=60 for shaping."
        ),
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Lag-behind trail: live_status.json overlay + twin rollouts of latest_model/checkpoint",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Follow a specific run (default: newest live_status.json under rl/runs/)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Seconds between status polls while waiting for weights (follow mode)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=0,
        help=(
            "Number of colored viewer twins on one map. "
            "0 in --follow = use live_status n_envs (else 8). "
            "Standalone default becomes 1 when left at 0."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic PPO actions (default: stochastic so twins diverge)",
    )
    parser.add_argument(
        "--compact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fewer overlay lines (live score + beat-FTG + short train ts). "
            "Default ON in --follow; OFF standalone. Env RL_WATCH_COMPACT=0|1 overrides when flag omitted."
        ),
    )
    args = parser.parse_args(argv)

    if not args.follow and int(args.n_envs) <= 0:
        args.n_envs = 1

    env_every = _env_int("RL_WATCH_EVERY")
    if args.every is None:
        if env_every is not None and env_every > 0:
            args.every = env_every
        else:
            args.every = FOLLOW_EVERY_DEFAULT if args.follow else STANDALONE_EVERY_DEFAULT
    args.every = max(1, int(args.every))

    env_compact = _env_bool("RL_WATCH_COMPACT")
    if args.compact is None:
        if env_compact is not None:
            args.compact = env_compact
        else:
            # Follow: declutter by default; standalone keeps the denser session dump.
            args.compact = bool(args.follow)

    if args.follow:
        if args.episodes == 5:
            # Sensible follow default: keep going until user quits
            args.episodes = 0
        return _run_follow(args)
    return _run_standalone(args)


if __name__ == "__main__":
    raise SystemExit(main())
