"""Server networking must retain run isolation and reach a private hub."""
from unittest.mock import patch

from src.layer4.hub import docker_control


def test_api_worker_joins_hub_network_without_shared_api_alias(monkeypatch):
    monkeypatch.setenv("AICAR_HUB_DOCKER_NETWORK", "aicar-server-control")
    with patch.object(docker_control, "_request", return_value={"Id": "worker"}) as request:
        docker_control.create_official_run_container(
            run_id="abc", role="api", image="runtime", network_name="aicar-run-abc"
        )
    config = request.call_args_list[0].kwargs["body"]
    endpoints = config["NetworkingConfig"]["EndpointsConfig"]
    assert endpoints["aicar-run-abc"]["Aliases"] == ["api", "aicar-run-abc-api"]
    assert endpoints["aicar-server-control"]["Aliases"] == ["aicar-run-abc-api"]
    assert config["HostConfig"]["NetworkMode"] == "aicar-run-abc"


def test_simulator_stays_on_its_private_run_network(monkeypatch):
    monkeypatch.setenv("AICAR_HUB_DOCKER_NETWORK", "aicar-server-control")
    with patch.object(docker_control, "_request", return_value={"Id": "sim"}) as request:
        docker_control.create_official_run_container(
            run_id="abc", role="sim", image="simulator", network_name="aicar-run-abc"
        )
    endpoints = request.call_args_list[0].kwargs["body"]["NetworkingConfig"]["EndpointsConfig"]
    assert list(endpoints) == ["aicar-run-abc"]
