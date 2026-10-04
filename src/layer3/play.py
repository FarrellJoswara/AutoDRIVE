"""Load a trained PPO checkpoint and drive (no learning)."""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ReplayPublisher:
    """Best-effort async publisher so hub telemetry never stalls model stepping."""

    def __init__(self, hub_url: str, run_id: str) -> None:
        self.url = hub_url.rstrip("/") + "/telemetry"
        self.run_id = run_id
        self.items: queue.Queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def enqueue(self, payload: dict) -> None:
        payload["channel"] = "replay"
        try:
            self.items.put_nowait(payload)
        except queue.Full:
            try:
                self.items.get_nowait()
            except queue.Empty:
                pass
            try:
                self.items.put_nowait(payload)
            except queue.Full:
                pass

    def _run(self) -> None:
        try:
            import requests
        except ImportError:
            return
        session = requests.Session()
        while True:
            payload = self.items.get()
            if payload is None:
                return
            try:
                session.post(self.url, json=payload, timeout=(0.5, 1.0))
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.items.put_nowait(None)
        except queue.Full:
            try:
                self.items.get_nowait()
                self.items.put_nowait(None)
            except queue.Empty:
                pass
        self.thread.join(timeout=1.0)


def _telemetry(
    obs: dict,
    info: dict,
    *,
    step: int,
    episode: int,
    run_id: str,
    episode_return: float,
    best_10_lap_time_s: Optional[float] = None,
) -> dict:
    position = info.get("position")
    pose = [float(position[0]), float(position[2])] if position is not None and len(position) >= 3 else None
    lidar = []
    try:
        import numpy as np
        beams = np.asarray(obs["lidar"], dtype=np.float32).reshape(-1)
        target = min(120, len(beams))
        if target:
            group = max(1, len(beams) // target)
            lidar = [round(float(x), 3) for x in beams[:group * target].reshape(target, group).min(axis=1)]
    except Exception:
        pass
    car = {
        "env_id": 0,
        "pose": pose,
        "yaw": info.get("yaw"),
        "collision": bool(info.get("collision_event", False)),
        "speed": info.get("true_speed"),
        "episode_return": episode_return,
        "frontier_line": info.get("frontier_line"),
        "frontier_progress_m": info.get("frontier_progress_m"),
        "current_progress_line": info.get("current_progress_line"),
        "current_progress_m": info.get("current_progress_m"),
        "signed_route_delta_m": info.get("signed_route_delta_m"),
        "current_route_speed_mps": info.get("current_route_speed_mps"),
        "route_projection_valid": bool(info.get("route_projection_valid", False)),
        "reward_components": info.get("reward_components"),
        "time_since_frontier_push_s": info.get("time_since_frontier_push_s"),
        "frontier_speed_mps": info.get("frontier_speed_mps"),
        "lap_supported": bool(info.get("lap_supported", False)),
        "lap_count": int(info.get("lap_count", 0)),
        "lap_times_s": list(info.get("lap_times_s", [])),
        "last_lap_time_s": info.get("last_lap_time_s"),
        "best_lap_time_s": info.get("best_lap_time_s"),
        "best_10_lap_time_s": best_10_lap_time_s,
        "lap_elapsed_s": info.get("lap_elapsed_s"),
        "lidar": lidar,
        "reset": False,
    }
    return {
        "channel": "replay",
        "kind": "fleet",
        "step": step,
        "episode": episode,
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "lap_gate": info.get("lap_gate"),
        "cars": [car],
    }


def rollout(
    *,
    model_path: Path,
    port: int = 4567,
    steps: int = 0,
    seed: int = 0,
    device: str = "auto",
    env_kwargs: Optional[dict] = None,
    stop_file: Optional[Path] = None,
) -> int:
    import torch
    from stable_baselines3 import PPO

    from src.layer3.envs import build_env

    env_kwargs = dict(env_kwargs or {})
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"device: cuda ({torch.cuda.get_device_name(0)})")
        else:
            device = "cpu"
            print("WARN: CUDA unavailable — play on CPU (last resort)")
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback for --device cuda")

    env = build_env(port=port, seed=seed, **env_kwargs)
    stop_requested = False
    previous_sigterm = None
    if hasattr(signal, "SIGTERM"):
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: setattr(env, "_replay_stop_requested", True))
    hub_url = os.environ.get("HUB_URL", "").strip()
    run_id = f"replay_{int(time.time())}"
    publisher = ReplayPublisher(hub_url, run_id) if hub_url else None
    try:
        model = PPO.load(str(model_path), env=env, device=device)
        obs, info = env.reset(seed=seed)
        print(f"play: model={model_path} port={port} steps={steps or 'unlimited'} device={device}")

        i = 0
        episode = 0
        episode_return = 0.0
        lap_times_this_episode: list[float] = []
        observed_laps_this_episode = 0
        observed_episode = -1
        best_10_lap_time_s: Optional[float] = None
        while steps <= 0 or i < steps:
            if getattr(env, "_replay_stop_requested", False) or (stop_file and stop_file.exists()):
                stop_requested = True
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            i += 1
            episode_return += float(reward)
            if episode != observed_episode:
                observed_episode = episode
                lap_times_this_episode = []
                observed_laps_this_episode = 0
            raw_lap_times = info.get("lap_times_s")
            if isinstance(raw_lap_times, (list, tuple)):
                for raw_time in raw_lap_times[observed_laps_this_episode:]:
                    try:
                        lap_time = float(raw_time)
                    except (TypeError, ValueError):
                        continue
                    if lap_time > 0.0:
                        lap_times_this_episode.append(lap_time)
                        if len(lap_times_this_episode) >= 10:
                            total = sum(lap_times_this_episode[-10:])
                            best_10_lap_time_s = (
                                total if best_10_lap_time_s is None
                                else min(best_10_lap_time_s, total)
                            )
                observed_laps_this_episode = len(raw_lap_times)
            if publisher is not None:
                publisher.enqueue(_telemetry(
                    obs,
                    info,
                    step=i,
                    episode=episode,
                    run_id=run_id,
                    episode_return=episode_return,
                    best_10_lap_time_s=best_10_lap_time_s,
                ))
            if i == 0 or (i + 1) % 50 == 0:
                print(
                    f"  step {i + 1}: v_long={info.get('v_long', float('nan')):.3f} "
                    f"reward={float(reward):.3f} trunc={truncated}"
                )
            if terminated or truncated:
                reason = info.get("truncate_reason")
                print(f"  episode end at step {i + 1} reason={reason}; reset")
                obs, info = env.reset()
                episode += 1
                episode_return = 0.0
        print("play: stopped" if stop_requested else "play: done")
        return 0
    finally:
        if publisher is not None:
            publisher.close()
        env.close()
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Layer 3 PPO play / eval")
    p.add_argument("--model", type=Path, required=True, help="Path to PPO .zip")
    p.add_argument("--port", type=int, default=4567)
    p.add_argument("--steps", type=int, default=0, help="0 runs until stopped")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--auto-launch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-timeout", type=float, default=90.0)
    p.add_argument("--forward-scale", type=float, default=1.0)
    p.add_argument("--collision-penalty", type=float, default=0.0)
    p.add_argument("--map-id", type=str, default="none")
    p.add_argument("--stop-file", type=Path, default=None)
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.model.exists():
        # SB3 save() may be given without .zip suffix
        alt = Path(str(args.model) + ".zip")
        if alt.exists():
            args.model = alt
        else:
            print(f"ERROR: model not found: {args.model}")
            return 1

    from src.layer3.envs import env_kwargs_from_args

    return rollout(
        model_path=args.model,
        port=args.port,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        env_kwargs=env_kwargs_from_args(args),
        stop_file=args.stop_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
