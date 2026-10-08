"""Mock-only tests for isolated official training run orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.layer4.hub import docker_control
from src.layer4.hub.official_run_manager import (
    OfficialRunManager,
    normalize_official_config,
)


class FakeDocker:
    def __init__(self, host_logs: Path) -> None:
        self.host_logs = host_logs
        self.networks = {}
        self.containers = {}
        self.calls = []
        self._counter = 0

    def find_host_path_for_container_path(self, path: Path) -> Path:
        return self.host_logs

    def create_official_run_network(self, run_id: str) -> str:
        name = f"aicar-run-{run_id}"
        self.networks[run_id] = {"Id": f"net-{run_id}", "Name": name,
                                 "Labels": {"aicar.managed": "official-run", "aicar.run_id": run_id}}
        self.calls.append(("network", run_id))
        return f"net-{run_id}"

    def remove_official_run_network(self, network_id: str) -> None:
        self.networks = {key: value for key, value in self.networks.items()
                         if value["Id"] != network_id}

    def create_official_run_container(self, **kwargs) -> str:
        self._counter += 1
        cid = f"container-{self._counter}"
        run_id, role = kwargs["run_id"], kwargs["role"]
        labels = {
            "aicar.managed": "official-run", "aicar.run_id": run_id,
            "aicar.role": role, **kwargs.get("metadata_labels", {}),
        }
        self.containers[cid] = {
            "Id": cid, "Names": [f"/{kwargs['network_name']}-{role}"],
            "State": "running", "Labels": labels, "kwargs": kwargs,
            "logs": f"started {role} for {run_id}\n",
        }
        self.calls.append(("container", run_id, role))
        return cid

    def list_official_run_containers(self, run_id=None):
        self.calls.append(("list_containers", run_id))
        return [
            {key: value for key, value in item.items() if key != "kwargs" and key != "logs"}
            for item in self.containers.values()
            if run_id is None or item["Labels"].get("aicar.run_id") == run_id
        ]

    def list_official_run_networks(self, run_id=None):
        self.calls.append(("list_networks", run_id))
        return [value for key, value in self.networks.items()
                if run_id is None or key == run_id]

    def inspect_container(self, container_id: str):
        return {"State": {"ExitCode": 17}}

    def stop_official_run_container(self, container_id: str, **kwargs) -> None:
        if container_id in self.containers:
            self.containers[container_id]["State"] = "exited"
            self.calls.append(("stop", container_id))

    def remove_official_run_container(self, container_id: str) -> None:
        self.containers.pop(container_id, None)

    def official_run_container_logs(self, container_id: str) -> str:
        return self.containers.get(container_id, {}).get("logs", "")


class OfficialRunManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.host_logs = self.root / "host-logs"
        self.host_logs.mkdir()
        self.runs_root = self.host_logs / "rl"
        self.registry = self.root / "layer4" / "official_runs.json"
        self.backend = FakeDocker(self.host_logs)
        self.manager = OfficialRunManager(
            max_concurrent_runs=2, backend=self.backend,
            registry_path=self.registry, runs_root=self.runs_root,
            start_monitor=False,
        )

    def tearDown(self) -> None:
        self.manager.close()
        self.temp.cleanup()

    def test_two_runs_are_isolated_and_limit_is_enforced(self) -> None:
        first = self.manager.create_run({"learning_rate": 5e-5}, "candidate A")
        second = self.manager.create_run({"learning_rate": 2e-5}, "candidate B")
        self.assertNotEqual(first["network_name"], second["network_name"])
        self.assertEqual(first["state"], "running")
        with self.assertRaisesRegex(RuntimeError, "maximum concurrent"):
            self.manager.create_run({}, "candidate C")
        api_call = next(call for call in self.backend.calls
                        if call[0:3] == ("container", first["run_id"], "api"))
        self.assertEqual(api_call[0], "container")
        api = next(item for item in self.backend.containers.values()
                   if item["Labels"].get("aicar.run_id") == first["run_id"]
                   and item["Labels"].get("aicar.role") == "api")
        env = dict(entry.split("=", 1) for entry in api["kwargs"]["env"])
        self.assertEqual(env["AICAR_PPO_LEARNING_RATE"], "5e-05")
        self.assertEqual(env["AICAR_TRAIN_OUT"], f"/runs/rl/official_run_{first['run_id']}")
        self.assertEqual(api["kwargs"]["binds"], [f"{self.host_logs}:/runs"])
        sim = next(item for item in self.backend.containers.values()
                   if item["Labels"].get("aicar.run_id") == first["run_id"]
                   and item["Labels"].get("aicar.role") == "sim")
        self.assertEqual(
            sim["kwargs"]["command"],
            ['"./AutoDRIVE Simulator.x86_64" -batchmode -nographics -ip api -port 4568'],
        )

        stopped = self.manager.stop_run(first["run_id"])
        self.assertEqual(stopped["state"], "stopped")
        self.assertEqual(self.manager.get_run(second["run_id"])["state"], "running")
        self.assertEqual(len(self.manager.list_runs()["runs"]), 2)

    def test_stop_signals_trainer_before_stopping_simulator(self) -> None:
        run = self.manager.create_run({}, "graceful stop")
        api_container_id = next(
            cid for cid, item in self.backend.containers.items()
            if item["Labels"].get("aicar.run_id") == run["run_id"]
            and item["Labels"].get("aicar.role") == "api"
        )
        sim_container_id = next(
            cid for cid, item in self.backend.containers.items()
            if item["Labels"].get("aicar.run_id") == run["run_id"]
            and item["Labels"].get("aicar.role") == "sim"
        )

        self.manager.stop_run(run["run_id"])

        first_stop_calls = [
            call[1] for call in self.backend.calls if call[0] == "stop"
        ][:2]
        self.assertEqual(first_stop_calls, [api_container_id, sim_container_id])

    def test_registry_and_labels_recover_running_run_after_manager_restart(self) -> None:
        run = self.manager.create_run({"seed": 11}, "recover me")
        self.manager.close()
        recovered = OfficialRunManager(
            max_concurrent_runs=2, backend=self.backend,
            registry_path=self.registry, runs_root=self.runs_root,
            start_monitor=False,
        )
        try:
            item = recovered.get_run(run["run_id"])
            self.assertEqual(item["state"], "running")
            self.assertEqual(item["display_name"], "recover me")
            self.assertEqual(item["config"]["seed"], 11)
        finally:
            recovered.close()

    def test_telemetry_reads_only_requested_run_snapshot(self) -> None:
        run = self.manager.create_run({}, "telemetry")
        snapshot = Path(run["output_dir"]) / "watch_latest.json"
        snapshot.write_text(json.dumps({"kind": "fleet", "run_id": run["run_id"], "step": 123}))
        payload = self.manager.telemetry(run["run_id"])
        self.assertEqual(payload["fleet"]["step"], 123)
        self.assertEqual(payload["fleet"]["run_id"], run["run_id"])

    def test_list_get_and_telemetry_use_cached_state_without_docker_polling(self) -> None:
        run = self.manager.create_run({}, "cached")
        self.backend.calls.clear()
        self.manager.list_runs()
        self.manager.get_run(run["run_id"])
        self.manager.telemetry(run["run_id"])
        self.assertFalse(any(call[0].startswith("list_") for call in self.backend.calls))

    def test_reconciler_does_not_race_a_requested_stop(self) -> None:
        run = self.manager.create_run({}, "stop race")
        self.backend.calls.clear()
        with self.manager._lock:
            self.manager._stopping.add(run["run_id"])
            self.manager._runs[run["run_id"]]["state"] = "stopping"
        self.manager.reconcile(run["run_id"])
        self.assertFalse(any(call[0] in {"stop", "remove"} for call in self.backend.calls))
        with self.manager._lock:
            self.manager._stopping.discard(run["run_id"])

    def test_run_scoped_metrics_and_phase_are_returned_only_for_their_run(self) -> None:
        first = self.manager.create_run({}, "telemetry A")
        second = self.manager.create_run({}, "telemetry B")
        metrics = {"kind": "metrics", "run_id": first["run_id"], "step": 256, "ts": "now"}
        phase = {"kind": "phase", "run_id": first["run_id"], "phase": "ppo_update", "ts": "now"}
        self.assertTrue(self.manager.ingest_telemetry(metrics))
        self.assertTrue(self.manager.ingest_telemetry(phase))
        self.assertFalse(self.manager.ingest_telemetry({"kind": "metrics", "run_id": "missing"}))
        a = self.manager.telemetry(first["run_id"])
        b = self.manager.telemetry(second["run_id"])
        self.assertEqual(a["last_metrics"]["step"], 256)
        self.assertEqual(a["last_train_phase"]["phase"], "ppo_update")
        self.assertIsNone(b["last_metrics"])
        self.assertIsNone(b["last_train_phase"])

    def test_completed_run_is_distinguished_from_container_failure(self) -> None:
        run = self.manager.create_run({}, "complete")
        Path(run["output_dir"], "stop_reason.json").write_text(
            json.dumps({"reason": "timestep_limit"}), encoding="utf-8"
        )
        self.manager.reconcile(run["run_id"])
        summary = self.manager.get_run(run["run_id"])
        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["stop_reason"], "timestep_limit")

    def test_summary_contains_ui_contract_fields(self) -> None:
        run = self.manager.create_run({}, "contract")
        self.assertIn("pid", run)
        self.assertIsNone(run["pid"])
        self.assertIn("stop_reason", run)
        self.assertIn("cleanup_error", run)
        self.assertIsInstance(run["containers"], list)
        self.assertIn("config", run)
        self.assertIn("latest_telemetry", run)

    def test_invalid_and_non_official_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            normalize_official_config({"frame_skip": 4})
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            normalize_official_config({"learning_rate": 0})
        with self.assertRaisesRegex(ValueError, "observation_profile"):
            normalize_official_config({"observation_profile": "simulator"})

    def test_exited_container_marks_only_its_run_failed_and_cleans_sibling(self) -> None:
        run_a = self.manager.create_run({}, "A")
        run_b = self.manager.create_run({}, "B")
        api_a = next(item for item in self.backend.containers.values()
                     if item["Labels"].get("aicar.run_id") == run_a["run_id"]
                     and item["Labels"].get("aicar.role") == "api")
        api_a["State"] = "exited"
        self.manager.reconcile(run_a["run_id"])
        self.assertEqual(self.manager.get_run(run_a["run_id"])["state"], "failed")
        self.assertEqual(self.manager.get_run(run_b["run_id"])["state"], "running")

    def test_missing_sibling_after_hub_restart_marks_run_failed(self) -> None:
        run = self.manager.create_run({}, "partial")
        sim_id = next(
            key for key, value in self.backend.containers.items()
            if value["Labels"].get("aicar.run_id") == run["run_id"]
            and value["Labels"].get("aicar.role") == "sim"
        )
        del self.backend.containers[sim_id]
        recovered = OfficialRunManager(
            max_concurrent_runs=2, backend=self.backend,
            registry_path=self.registry, runs_root=self.runs_root,
            start_monitor=False,
        )
        try:
            self.assertEqual(recovered.get_run(run["run_id"])["state"], "failed")
        finally:
            recovered.close()

    def test_orphan_network_is_recovered_and_removed(self) -> None:
        self.backend.create_official_run_network("orphan123")
        recovered = OfficialRunManager(
            max_concurrent_runs=2, backend=self.backend,
            registry_path=self.registry, runs_root=self.runs_root,
            start_monitor=False,
        )
        try:
            row = recovered.get_run("orphan123")
            self.assertEqual(row["state"], "failed")
            self.assertEqual(self.backend.networks, {})
        finally:
            recovered.close()

    def test_config_normalization_includes_official_defaults(self) -> None:
        config = normalize_official_config({"total_timesteps": 5000})
        self.assertEqual(config["total_timesteps"], 5000)
        self.assertEqual(config["observation_profile"], "official_sensors_camera")
        self.assertEqual(config["throttle_mode"], "bidirectional")

    def test_resume_checkpoint_is_mapped_to_persistent_container_mount(self) -> None:
        source = self.root / "logs" / "rl" / "source.zip"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"checkpoint")
        with patch("src.layer4.hub.official_run_manager.ROOT", self.root):
            config = normalize_official_config({"resume": str(source)})
            self.assertEqual(config["resume"], "/runs/rl/source.zip")


class DockerControlOfficialRunTests(unittest.TestCase):
    def test_container_request_has_run_isolation_and_no_host_ports(self) -> None:
        with patch.object(docker_control, "_request", side_effect=[{"Id": "abc"}, None]) as request:
            cid = docker_control.create_official_run_container(
                run_id="run123", role="sim", image="official-sim",
                network_name="aicar-run-run123",
                entrypoint=["/bin/bash", "-lc"], command=["run simulator"],
            )
        self.assertEqual(cid, "abc")
        body = request.call_args_list[0].kwargs["body"]
        self.assertNotIn("PortBindings", body["HostConfig"])
        self.assertEqual(body["HostConfig"]["NetworkMode"], "aicar-run-run123")
        self.assertEqual(body["Labels"]["aicar.run_id"], "run123")
        self.assertEqual(body["NetworkingConfig"]["EndpointsConfig"]["aicar-run-run123"]["Aliases"],
                         ["sim", "aicar-run-run123-sim"])

    def test_gpu_container_request_uses_nvidia_device_request(self) -> None:
        with patch.object(docker_control, "_request", side_effect=[{"Id": "abc"}, None]) as request:
            docker_control.create_official_run_container(
                run_id="run123", role="api", image="official-api-cuda",
                network_name="aicar-run-run123", gpu=True,
            )
        host = request.call_args_list[0].kwargs["body"]["HostConfig"]
        self.assertEqual(host["DeviceRequests"], [{
            "Driver": "nvidia", "Count": -1, "DeviceIDs": [],
            "Capabilities": [["gpu"]], "Options": {},
        }])


if __name__ == "__main__":
    unittest.main()
