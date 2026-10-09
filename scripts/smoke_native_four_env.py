"""End-to-end bounded test: four official Windows sims, native bridges, one PPO learner."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUB = "http://127.0.0.1:8091"


def request_json(path: str, method: str = "GET", payload=None, timeout: float = 30):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        HUB + path, data=body, method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    settings_path = ROOT / "logs" / "layer4" / "official_train_settings.json"
    original_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    smoke = {**original_settings}
    smoke.update({
        "simulator_backend": "windows_external",
        "n_envs": 4,
        "total_timesteps": int(os.environ.get("AICAR_SMOKE_TIMESTEPS", "256")),
        "stop_on_plateau": False,
        "device": "cuda",
        "resume": (
            original_settings.get("resume")
            if os.environ.get("AICAR_SMOKE_RESUME", "0") == "1"
            else None
        ),
        "checkpoint_every": int(os.environ.get("AICAR_SMOKE_TIMESTEPS", "256")),
        "n_steps": 64,
        "n_epochs": 1,
        "timeout_s": 180,
        "training_timeout_s": 600,
        "observation_profile": "official_sensors_camera",
    })
    run_id = None
    failure = None
    try:
        print("Starting four official simulator windows and one bounded CUDA PPO learner...", flush=True)
        _status, run = request_json(
            "/train/runs", "POST",
            {"config": smoke, "display_name": "Native four-environment stability smoke"},
            timeout=1800,
        )
        run_id = run["run_id"]
        print(f"Run {run_id} started with connections {run.get('simulator_connections')}", flush=True)

        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            _status, run = request_json(f"/train/runs/{run_id}")
            print(f"Run state: {run['state']}; step={((run.get('latest_metrics') or {}).get('step'))}", flush=True)
            if run["state"] in {"completed", "failed", "stopped"}:
                break
            time.sleep(5)
        else:
            raise TimeoutError("four-environment smoke run did not finish within 15 minutes")

        if run["state"] != "completed":
            raise RuntimeError(f"smoke training ended as {run['state']}: {run.get('error')}")
        _status, diagnostics = request_json(f"/train/runs/{run_id}/ppo-diagnostics")
        updates = diagnostics.get("updates", [])
        if not updates:
            raise RuntimeError("PPO completed without writing any learner diagnostics")
        camera_results = []
        for env_id in range(4):
            request = urllib.request.Request(f"{HUB}/train/runs/{run_id}/camera/{env_id}")
            with urllib.request.urlopen(request, timeout=20) as response:
                frame = response.read()
                if response.status != 200 or len(frame) < 1000 or not frame.startswith(b"\xff\xd8"):
                    raise RuntimeError(f"environment {env_id} returned an invalid camera JPEG ({len(frame)} bytes)")
                camera_results.append({"env_id": env_id, "bytes": len(frame)})
        print(json.dumps({
            "run_id": run_id,
            "state": run["state"],
            "connections": run.get("simulator_connections"),
            "ppo_updates": len(updates),
            "last_update": updates[-1],
            "camera_frames": camera_results,
        }, indent=2), flush=True)
    except Exception as exc:
        failure = exc
        if run_id:
            try:
                request_json(f"/train/runs/{run_id}/stop", "POST", {})
            except Exception:
                pass
        raise
    finally:
        try:
            request_json("/train/official-settings", "PUT", original_settings)
        except urllib.error.URLError:
            if failure is None:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
