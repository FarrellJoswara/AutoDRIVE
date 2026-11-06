"""Exercise real Replay ASGI routes with synthetic telemetry and no lifespan."""

import asyncio
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


async def request_json(app, method, path, payload=None):
    sent = []
    consumed = False

    async def receive():
        nonlocal consumed
        if consumed:
            return {"type": "http.disconnect"}
        consumed = True
        return {"type": "http.request", "body": json.dumps(payload).encode() if payload else b"",
                "more_body": False}

    async def send(message):
        sent.append(message)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
               "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
               "client": ("127.0.0.1", 1000), "server": ("qa.local", 80)}, receive, send)
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return status, json.loads(body)


class ReplayLapRouteTests(unittest.TestCase):
    def test_post_replay_telemetry_status_fallback_recovery_and_run_isolation(self):
        with patch.dict(os.environ, {"AICAR_RUN_BACKEND": "test"}):
            hub = importlib.import_module("src.layer4.hub.app")
        from src.layer4.hub.lap_session import LapSessionStore
        from src.layer4.hub.telemetry import TelemetryBus

        class Runs:
            frame = None

            def get_run(self, run_id):
                return {"run_id": run_id, "map_id": "qa-track", "config": {
                    "layout_id": "qa-layout", "direction": "forward"}}

            def ingest_telemetry(self, frame):
                self.frame = frame

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = Runs()
            job = SimpleNamespace(status=lambda: {"run_id": "replay-a", "last_fleet": runs.frame})
            with patch.object(hub, "bus", TelemetryBus(lap_sessions=LapSessionStore(root))), \
                    patch.object(hub, "official_runs", runs), patch.object(hub, "replay_job", job):
                frame = {"run_id": "replay-a", "channel": "replay", "kind": "fleet", "episode": 0,
                         "cars": [{"env_id": 0, "lap_supported": True,
                                   "lap_times_s": [30.0], "lap_elapsed_s": 2.0}]}
                status, _ = asyncio.run(request_json(hub.app, "POST", "/telemetry", frame))
                self.assertEqual(status, 200)
                self.assertNotIn("lap_session", runs.frame["cars"][0])

                def assert_summary():
                    status, result = asyncio.run(request_json(hub.app, "GET", "/api/replay/status"))
                    self.assertEqual(status, 200)
                    self.assertEqual(result["last_fleet"]["cars"][0]["lap_session"]["completed_laps"], 1)

                assert_summary()
                job.status = lambda: {"run_id": "replay-a"}
                assert_summary()  # A matching bus fleet fills an absent status fleet.
                job.status = lambda: {"run_id": "replay-b", "last_fleet": runs.frame}
                _, different_run = asyncio.run(request_json(hub.app, "GET", "/api/replay/status"))
                self.assertNotIn("last_fleet", different_run)
                hub.bus = TelemetryBus(lap_sessions=LapSessionStore(root))
                job.status = lambda: {"run_id": "replay-a", "last_fleet": runs.frame}
                assert_summary()  # A raw recovered status frame uses durable history.
                self.assertEqual(len(hub.bus.lap_sessions.history("replay-a", "0")["laps"]), 1)


if __name__ == "__main__":
    unittest.main()
