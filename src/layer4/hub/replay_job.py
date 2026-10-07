"""Process lifecycle for a separate Layer 3 model replay session."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.layer4.settings import ROOT

StatusCallback = Callable[[Dict[str, Any]], None]
REPLAY_PORT = 4583


class ReplayJob:
    def __init__(self, *, hub_url: str, on_status: Optional[StatusCallback] = None) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.on_status = on_status
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._watcher: Optional[threading.Thread] = None
        self.state = "idle"
        self.pid: Optional[int] = None
        self.started_at: Optional[str] = None
        self.argv: list[str] = []
        self.exit_code: Optional[int] = None
        self.error: Optional[str] = None
        self.log_path: Optional[str] = None
        self.model: Optional[str] = None
        self.map_id: Optional[str] = None
        self.seed: Optional[int] = None
        self.run_id: Optional[str] = None
        self.stop_file: Optional[Path] = None
        self._log_file = None
        self._docker_sim = False
        self._last_config: Optional[dict] = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "pid": self.pid,
                "started_at": self.started_at,
                "argv": list(self.argv),
                "exit_code": self.exit_code,
                "error": self.error,
                "log_path": self.log_path,
                "model": self.model,
                "map_id": self.map_id,
                "seed": self.seed,
                "run_id": self.run_id,
            }

    def _broadcast(self) -> None:
        if self.on_status:
            try:
                self.on_status(self.status())
            except Exception:
                pass

    def start(
        self, *, model_path: Path, map_id: str, seed: int, device: str = "auto",
        simulator_env: Optional[Dict[str, str]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self.state in {"starting", "running", "stopping"}:
                raise RuntimeError(f"replay already {self.state}")
            self.state = "starting"
            self.exit_code = None
            self.error = None
            self.model = str(model_path)
            self.map_id = map_id
            self.seed = int(seed)
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.run_id = f"replay_{int(time.time())}"
            self._last_config = {
                "model_path": model_path,
                "map_id": map_id,
                "seed": int(seed),
                "device": device,
                "simulator_env": dict(simulator_env or {}),
                "env_kwargs": dict(env_kwargs or {}),
            }
            run_dir = ROOT / "logs" / "replay"
            run_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = str(run_dir / f"{self.run_id}.log")
            self.stop_file = run_dir / f"{self.run_id}.stop"
            self.stop_file.unlink(missing_ok=True)
            docker = os.environ.get("AICAR_IN_DOCKER", "").strip().lower() in {"1", "true", "yes", "on"}
            self._docker_sim = docker
        self._broadcast()

        sim_created = False
        log_file = None
        try:
            if docker:
                from src.layer4.hub.docker_control import create_replay_sim

                create_replay_sim(map_id, port=REPLAY_PORT, env_overrides=simulator_env)
                sim_created = True
            cmd = [
                sys.executable, "-m", "src.layer3.play",
                "--model", str(model_path),
                "--port", str(REPLAY_PORT),
                "--steps", "0",
                "--seed", str(int(seed)),
                "--device", device,
                "--map-id", map_id,
                "--env-kwargs-json", json.dumps(env_kwargs or {}, separators=(",", ":")),
                "--connect-timeout", "90",
                "--stop-file", str(self.stop_file),
                "--headless",
                "--auto-launch" if not docker else "--no-auto-launch",
            ]
            env = {
                **os.environ,
                "HUB_URL": self.hub_url,
                "AICAR_MAP_ID": map_id,
                "AICAR_FLEET_HZ": "12",
                **(simulator_env or {}),
            }
            log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
            popen_kwargs: Dict[str, Any] = {
                "args": cmd,
                "cwd": str(ROOT),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": log_file,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(**popen_kwargs)
        except Exception as exc:
            if log_file:
                log_file.close()
            if sim_created:
                self._remove_docker_sim()
            with self._lock:
                self.state = "error"
                self.error = str(exc)
            self._broadcast()
            raise

        with self._lock:
            self.argv = cmd
            self._log_file = log_file
            self._proc = proc
            self.pid = proc.pid
            self.state = "running"
            self._watcher = threading.Thread(target=self._watch_exit, daemon=True, name="replay-job-watcher")
            self._watcher.start()
        self._broadcast()
        return self.status()

    def _remove_docker_sim(self) -> None:
        try:
            from src.layer4.hub.docker_control import remove_replay_sim

            remove_replay_sim()
        except Exception as exc:
            with self._lock:
                self.error = f"replay simulator cleanup failed: {exc}"

    def _watch_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        with self._lock:
            self.exit_code = int(code)
            self.pid = None
            self._proc = None
            if self.state != "error":
                self.state = "exited"
            if self._log_file:
                self._log_file.close()
                self._log_file = None
        if self._docker_sim:
            self._remove_docker_sim()
        self._broadcast()

    def stop(self, grace_s: float = 20.0) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            if proc is None:
                return self.status()
            self.state = "stopping"
            stop_file = self.stop_file
        self._broadcast()
        if stop_file:
            stop_file.touch(exist_ok=True)
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        watcher = self._watcher
        if watcher and watcher.is_alive():
            watcher.join(timeout=2)
        return self.status()

    def reset(self) -> Dict[str, Any]:
        config = self._last_config
        if config is None:
            raise RuntimeError("start a replay before resetting it")
        self.stop()
        return self.start(**config)
