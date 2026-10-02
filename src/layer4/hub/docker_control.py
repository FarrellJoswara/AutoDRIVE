"""Minimal Docker Engine client for stopping compose-managed containers."""

from __future__ import annotations

import http.client
import json
import os
import socket
from typing import Any, Dict, List, Optional
from urllib.parse import quote

DOCKER_SOCKET = os.environ.get("AICAR_DOCKER_SOCKET", "/var/run/docker.sock")
COMPOSE_PROJECT = os.environ.get("AICAR_COMPOSE_PROJECT", "aicar")


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 15.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def _request(method: str, path: str, *, timeout_s: float = 30.0) -> Any:
    """Issue a Docker API request with enough time for the operation to finish.

    Docker's stop/restart query timeout is a grace period for the container;
    the HTTP client must wait longer than that period or it can report a false
    timeout while Docker is still completing the requested operation.
    """
    conn = _UnixHTTPConnection(DOCKER_SOCKET, timeout=timeout_s)
    try:
        conn.request(method, path, headers={"Host": "localhost"})
        response = conn.getresponse()
        body = response.read()
        if response.status not in range(200, 300) and response.status != 304:
            detail = body.decode("utf-8", errors="replace")
            raise RuntimeError(f"Docker API {method} {path}: {response.status} {detail}")
        if not body:
            return None
        return json.loads(body)
    finally:
        conn.close()


def _list_compose_containers(*, service: Optional[str] = "sim", all_containers: bool = False) -> List[Dict[str, Any]]:
    labels = [f"com.docker.compose.project={COMPOSE_PROJECT}"]
    if service:
        labels.append(f"com.docker.compose.service={service}")
    filters = quote(json.dumps({"label": labels}, separators=(",", ":")))
    all_flag = 1 if all_containers else 0
    return _request("GET", f"/containers/json?all={all_flag}&filters={filters}")


def stop_compose_containers(*, full_stack: bool, timeout_s: int = 10) -> List[str]:
    """Stop running sims, or the whole AiCar compose project when requested."""
    if full_stack:
        containers = _list_compose_containers(service=None, all_containers=False)
    else:
        containers = _list_compose_containers(service="sim", all_containers=False)

    # During full teardown, stop the brain (this container) last so requests to
    # the other services complete before the Docker daemon terminates the hub.
    self_id = os.environ.get("HOSTNAME", "")
    containers.sort(key=lambda item: str(item.get("Id", "")).startswith(self_id))

    stopped: List[str] = []
    for container in containers:
        container_id = str(container["Id"])
        names = container.get("Names") or [container_id[:12]]
        name = str(names[0]).lstrip("/")
        _request(
            "POST", f"/containers/{container_id}/stop?t={int(timeout_s)}",
            timeout_s=max(30.0, float(timeout_s) + 10.0),
        )
        stopped.append(name)
    return stopped


def _container_name(container: Dict[str, Any]) -> str:
    names = container.get("Names") or [str(container.get("Id", ""))[:12]]
    return str(names[0]).lstrip("/")


def ensure_compose_sims_running(*, timeout_s: int = 20) -> List[str]:
    """Start any exited compose `sim` containers. No-op if already running.

    Needed after stop_sims_on_train_exit (or a failed Activate restart) so the
    next Train / Activate has something to talk to.
    """
    containers = _list_compose_containers(service="sim", all_containers=True)
    started: List[str] = []
    for container in containers:
        state = (container.get("State") or "").lower()
        if state == "running":
            continue
        container_id = str(container["Id"])
        name = _container_name(container)
        _request("POST", f"/containers/{container_id}/start")
        started.append(name)
    return started


def restart_compose_sims(*, timeout_s: int = 20) -> List[str]:
    """Ensure sims are up and restart them so entrypoint re-reads active map.

    Starts exited containers first (restart alone only hits running ones).
    Active map id is read from simulator/maps/.active_map.json (volume-mounted).
    """
    ensure_compose_sims_running(timeout_s=timeout_s)
    containers = _list_compose_containers(service="sim", all_containers=False)
    restarted: List[str] = []
    for container in containers:
        container_id = str(container["Id"])
        name = _container_name(container)
        _request(
            "POST", f"/containers/{container_id}/restart?t={int(timeout_s)}",
            timeout_s=max(30.0, float(timeout_s) + 10.0),
        )
        restarted.append(name)
    return restarted
