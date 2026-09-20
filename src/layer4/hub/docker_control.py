"""Minimal Docker Engine client for stopping compose-managed containers."""

from __future__ import annotations

import http.client
import json
import os
import socket
from typing import Any, Dict, List
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


def _request(method: str, path: str) -> Any:
    conn = _UnixHTTPConnection(DOCKER_SOCKET)
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


def stop_compose_containers(*, full_stack: bool, timeout_s: int = 10) -> List[str]:
    """Stop running sims, or the whole AiCar compose project when requested."""
    labels = [f"com.docker.compose.project={COMPOSE_PROJECT}"]
    if not full_stack:
        labels.append("com.docker.compose.service=sim")
    filters = quote(json.dumps({"label": labels}, separators=(",", ":")))
    containers: List[Dict[str, Any]] = _request(
        "GET", f"/containers/json?all=0&filters={filters}"
    )

    # During full teardown, stop the brain (this container) last so requests to
    # the other services complete before the Docker daemon terminates the hub.
    self_id = os.environ.get("HOSTNAME", "")
    containers.sort(key=lambda item: str(item.get("Id", "")).startswith(self_id))

    stopped: List[str] = []
    for container in containers:
        container_id = str(container["Id"])
        names = container.get("Names") or [container_id[:12]]
        name = str(names[0]).lstrip("/")
        _request("POST", f"/containers/{container_id}/stop?t={int(timeout_s)}")
        stopped.append(name)
    return stopped
