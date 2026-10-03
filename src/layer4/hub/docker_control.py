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


def _request(
    method: str, path: str, *, timeout_s: float = 30.0, body: Optional[Dict[str, Any]] = None
) -> Any:
    """Issue a Docker API request with enough time for the operation to finish.

    Docker's stop/restart query timeout is a grace period for the container;
    the HTTP client must wait longer than that period or it can report a false
    timeout while Docker is still completing the requested operation.
    """
    conn = _UnixHTTPConnection(DOCKER_SOCKET, timeout=timeout_s)
    try:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Host": "localhost"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
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


def _create_sim_replica(template: Dict[str, Any], index: int, expected: int) -> str:
    """Create a Compose-compatible sim replica cloned from an existing one."""
    config = template["Config"]
    labels = dict(config.get("Labels") or {})
    labels["com.docker.compose.container-number"] = str(index)
    config = {
        key: config[key]
        for key in (
            "Image", "Env", "Cmd", "Entrypoint", "WorkingDir", "User",
            "Tty", "OpenStdin", "AttachStdin", "AttachStdout", "AttachStderr",
        )
        if key in config
    }
    env = [
        item for item in config.get("Env", [])
        if not item.startswith(("AICAR_EXPECTED_SIMS=", "PORT="))
    ]
    env.append(f"AICAR_EXPECTED_SIMS={expected}")
    env.append(f"PORT={4566 + index}")
    config["Env"] = env
    config["Labels"] = labels
    config["Hostname"] = f"{COMPOSE_PROJECT}-sim-{index}"

    host = template.get("HostConfig") or {}
    host_config = {
        key: host[key]
        for key in ("Binds", "NetworkMode", "RestartPolicy", "ShmSize")
        if key in host
    }
    network_settings = template.get("NetworkSettings", {}).get("Networks", {})
    endpoints = {}
    for network_name, endpoint in network_settings.items():
        endpoints[network_name] = {
            "Aliases": [f"{COMPOSE_PROJECT}-sim-{index}", "sim"]
        }

    name = f"{COMPOSE_PROJECT}-sim-{index}"
    response = _request(
        "POST",
        f"/containers/create?name={quote(name, safe='')}",
        body={
            **config,
            "HostConfig": host_config,
            "NetworkingConfig": {"EndpointsConfig": endpoints},
        },
    )
    container_id = str(response["Id"])
    _request("POST", f"/containers/{container_id}/start")
    return name


def reconcile_compose_sims(desired_count: int, *, timeout_s: int = 20) -> List[str]:
    """Make the Compose simulator pool match the requested training env count."""
    if not 1 <= int(desired_count) <= 16:
        raise ValueError("simulator count must be between 1 and 16")
    desired_count = int(desired_count)
    containers = _list_compose_containers(service="sim", all_containers=True)
    if not containers:
        raise RuntimeError("no Compose simulator exists to use as a replica template")

    template_id = str(containers[0]["Id"])
    template = _request("GET", f"/containers/{template_id}/json")

    # Recreate the pool with explicit per-replica ports. DNS ordering is not
    # stable when a pool shrinks, so deriving a port from DNS position can leave
    # a single remaining sim listening on (for example) 4568 instead of 4567.
    for container in containers:
        container_id = str(container["Id"])
        if (container.get("State") or "").lower() == "running":
            _request(
                "POST", f"/containers/{container_id}/stop?t={int(timeout_s)}",
                timeout_s=max(30.0, float(timeout_s) + 10.0),
            )
        _request("DELETE", f"/containers/{container_id}?force=1&v=1")

    names: List[str] = []
    for index in range(1, desired_count + 1):
        names.append(_create_sim_replica(template, index, desired_count))
    return names


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


def create_replay_sim(map_id: str, *, port: int = 4583) -> str:
    """Start an isolated simulator container for the Layer 3 replay process."""
    templates = _list_compose_containers(service="sim", all_containers=True)
    if not templates:
        raise RuntimeError("no Compose simulator exists to use as a replay template")
    name = f"{COMPOSE_PROJECT}-replay-sim"
    for old in _list_compose_containers(service="replay", all_containers=True):
        old_id = str(old["Id"])
        if (old.get("State") or "").lower() == "running":
            _request("POST", f"/containers/{old_id}/stop?t=5")
        _request("DELETE", f"/containers/{old_id}?force=1&v=1")

    template = _request("GET", f"/containers/{templates[0]['Id']}/json")
    config = template["Config"]
    env = [
        item for item in config.get("Env", [])
        if not item.startswith(("AICAR_EXPECTED_SIMS=", "PORT=", "BASE_PORT=", "AICAR_MAP_ID="))
    ]
    env.extend((
        "AICAR_EXPECTED_SIMS=1",
        f"PORT={int(port)}",
        f"BASE_PORT={int(port)}",
        "BRAIN_HOST=brain",
        f"AICAR_MAP_ID={map_id}",
    ))
    labels = dict(config.get("Labels") or {})
    labels["com.docker.compose.service"] = "replay"
    labels.pop("com.docker.compose.container-number", None)
    labels["aicar.role"] = "replay"
    image_config = {
        key: config[key]
        for key in (
            "Image", "Env", "Cmd", "Entrypoint", "WorkingDir", "User",
            "Tty", "OpenStdin", "AttachStdin", "AttachStdout", "AttachStderr",
        )
        if key in config
    }
    image_config["Env"] = env
    image_config["Labels"] = labels
    image_config["Hostname"] = name
    host = template.get("HostConfig") or {}
    host_config = {
        key: host[key]
        for key in ("Binds", "NetworkMode", "RestartPolicy", "ShmSize")
        if key in host
    }
    networks = template.get("NetworkSettings", {}).get("Networks", {})
    endpoints = {
        network: {"Aliases": [name, "replay"]}
        for network in networks
    }
    response = _request(
        "POST", f"/containers/create?name={quote(name, safe='')}",
        body={
            **image_config,
            "HostConfig": host_config,
            "NetworkingConfig": {"EndpointsConfig": endpoints},
        },
    )
    container_id = str(response["Id"])
    _request("POST", f"/containers/{container_id}/start")
    return name


def remove_replay_sim() -> Optional[str]:
    """Stop and remove the dedicated replay simulator, if it exists."""
    removed = None
    for container in _list_compose_containers(service="replay", all_containers=True):
        container_id = str(container["Id"])
        removed = _container_name(container)
        if (container.get("State") or "").lower() == "running":
            _request("POST", f"/containers/{container_id}/stop?t=5")
        _request("DELETE", f"/containers/{container_id}?force=1&v=1")
    return removed
