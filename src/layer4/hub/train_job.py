"""TrainJob — subprocess.Popen lifecycle for `python -m src.layer3.train`."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.layer4.settings import ROOT, Settings

StatusCallback = Callable[[Dict[str, Any]], None]


class TrainJob:
    """One active train child at a time."""

    def __init__(
        self,
        *,
        hub_url: str = "http://127.0.0.1:8090",
        on_status: Optional[StatusCallback] = None,
    ) -> None:
        self.hub_url = hub_url.rstrip("/")
        self.on_status = on_status
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._watcher: Optional[threading.Thread] = None
        self._log_file = None
        self.state: str = "idle"
        self.pid: Optional[int] = None
        self.started_at: Optional[str] = None
        self.argv: List[str] = []
        self.exit_code: Optional[int] = None
        self.log_path: Optional[str] = None
        self.error: Optional[str] = None
        self.cleanup_error: Optional[str] = None
        self.stopped_containers: List[str] = []
        self.last_settings: Optional[Settings] = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "pid": self.pid,
            "started_at": self.started_at,
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "log_path": self.log_path,
            "hub_url": self.hub_url,
            "error": self.error,
            "cleanup_error": self.cleanup_error,
            "stopped_containers": list(self.stopped_containers),
        }

    def _broadcast(self) -> None:
        """Must not be called while holding self._lock (status takes the lock)."""
        if self.on_status is not None:
            try:
                self.on_status(self.status())
            except Exception:
                pass

    def start(self, settings: Settings) -> Dict[str, Any]:
        with self._lock:
            if self.state in {"starting", "running", "stopping"}:
                raise RuntimeError(f"train already {self.state}")

            out_dir = settings.resolve_out()
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "hub_train.log"
            cmd = [sys.executable, "-m", "src.layer3.train", *settings.to_train_argv()]
            env = {**os.environ, "HUB_URL": self.hub_url, **settings.hub_env()}

            self.state = "starting"
            self.exit_code = None
            self.error = None
            self.cleanup_error = None
            self.stopped_containers = []
            self.argv = cmd
            self.log_path = str(log_path)
            self.last_settings = settings
            self.started_at = datetime.now(timezone.utc).isoformat()

        self._broadcast()

        log_file = None
        try:
            log_file = open(log_path, "w", encoding="utf-8", buffering=1)
            popen_kwargs: Dict[str, Any] = dict(
                args=cmd,
                cwd=str(ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            if sys.platform == "win32":
                # Avoid console-allocation stalls when hub has no attached console.
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(**popen_kwargs)
        except Exception as exc:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            with self._lock:
                self.state = "error"
                self.error = str(exc)
                self.pid = None
                self._proc = None
                self._log_file = None
            self._broadcast()
            raise

        with self._lock:
            # Cancelled by stop() while Popen was in flight?
            if self.state == "stopping":
                try:
                    proc.terminate()
                except Exception:
                    pass
            self._log_file = log_file
            self._proc = proc
            self.pid = proc.pid
            if self.state == "starting":
                self.state = "running"
            self._watcher = threading.Thread(
                target=self._watch_exit, name="train-job-watcher", daemon=True
            )
            self._watcher.start()

        self._broadcast()
        return self.status()

    def _watch_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        with self._lock:
            settings = self.last_settings
            cleanup_pending = self._docker_cleanup_enabled(settings)
            self.exit_code = int(code) if code is not None else None
            self.state = "stopping" if cleanup_pending else "exited"
            self.pid = None
            self._proc = None
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
        self._broadcast()
        if cleanup_pending:
            self._cleanup_docker_services(settings)
            with self._lock:
                self.state = "exited"
            self._broadcast()

    @staticmethod
    def _docker_cleanup_enabled(settings: Optional[Settings]) -> bool:
        if settings is None:
            return False
        in_docker = os.environ.get("AICAR_IN_DOCKER", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not (settings.docker_mode or in_docker):
            return False
        return settings.stop_sims_on_train_exit or settings.stop_stack_on_train_exit

    def _cleanup_docker_services(self, settings: Settings) -> None:
        try:
            from src.layer4.hub.docker_control import stop_compose_containers

            stopped = stop_compose_containers(
                full_stack=settings.stop_stack_on_train_exit
            )
            with self._lock:
                self.stopped_containers = stopped
        except Exception as exc:
            with self._lock:
                self.cleanup_error = str(exc)

    def stop(self, grace_s: float = 12.0) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            if self.state not in {"starting", "running", "stopping"}:
                return self._status_unlocked()
            self.state = "stopping"

        self._broadcast()

        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

            deadline = time.monotonic() + grace_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
            else:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass

            # watcher sets exited; give it a moment
            for _ in range(25):
                with self._lock:
                    if self.state == "exited" or self._proc is None:
                        break
                time.sleep(0.1)

        with self._lock:
            if self.state not in {"exited", "error", "idle"}:
                self.state = "exited"
                if proc is not None and self.exit_code is None and proc.poll() is not None:
                    self.exit_code = proc.poll()

        self._broadcast()
        return self.status()
