"""In-process HTTP route checks for Watch and Replay lap enrichment.

Run this file in a dependency-complete hub environment. It dispatches requests
through the actual FastAPI ASGI app without starting lifespan/server workers.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

async def _get_json(app, path: str, query: str = "") -> tuple[int, dict]:
    messages: list[dict] = []
    first_receive = True

    async def receive() -> dict:
        nonlocal first_receive
        if first_receive:
            first_receive = False
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        messages.append(message)

    raw_path = path.encode("ascii")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": raw_path,
        "query_string": query.encode("ascii"),
        "root_path": "",
        "headers": [(b"host", b"qa.local")],
        "client": ("127.0.0.1", 32100),
        "server": ("127.0.0.1", 8090),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in messages
                    if message["type"] == "http.response.body")
    return int(start["status"]), json.loads(body or b"{}")


class LiveLapRouteIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("LIVE_LAP_ROUTE_QA_ISOLATED") == "1",
        "run this production-app import check as a standalone subprocess",
    )
    def test_watch_live_and_recovered_frames_and_replay_status_are_enriched(self):
        # Import the production app only inside the test process. This does not
        # enter FastAPI lifespan or launch the server, trainer, or replay job.
        try:
            from src.layer4.hub import app as hub_app
        except ModuleNotFoundError as exc:
            self.skipTest(f"full hub import dependencies unavailable: {exc}")

        from src.layer4.hub.lap_session import LapSessionStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = LapSessionStore(root / "sessions")
            previous_bus_store = hub_app.bus.lap_sessions
            previous_runs = hub_app.official_runs
            previous_replay_job = hub_app.replay_job
            previous_run_dir = hub_app._run_output_dir
            hub_app.bus.lap_sessions = store
            hub_app._run_output_dir = lambda _run_id: root / "empty-run-dir"

            watch_frame = {
                "kind": "fleet", "runtime": "custom", "run_id": "watch-run",
                "step": 20, "episode": 1, "map_id": "track", "layout_id": "layout",
                "direction": "forward", "ts": "2026-11-06T00:00:20Z",
                "cars": [{"env_id": 0, "lap_supported": True,
                          "lap_times_s": [30.0, 29.5], "lap_elapsed_s": 2.0}],
            }
            hub_app.bus.publish("telemetry", watch_frame)

            class Runs:
                def telemetry(self, run_id: str) -> dict:
                    if run_id != "watch-run":
                        raise KeyError(run_id)
                    return {"last_fleet": watch_frame, "last_metrics": None}

            hub_app.official_runs = Runs()
            status, live = asyncio.run(_get_json(
                hub_app.app, "/train/runs/watch-run/telemetry"
            ))
            self.assertEqual(status, 200)
            self.assertEqual(live["last_fleet"]["cars"][0]["lap_session"]["completed_laps"], 2)

            # Drop only the volatile bus frame. The Watch route must recover
            # persisted telemetry and enrich it from the durable session file.
            hub_app.bus.clear_run("watch-run")
            status, recovered = asyncio.run(_get_json(
                hub_app.app, "/train/runs/watch-run/telemetry"
            ))
            self.assertEqual(status, 200)
            self.assertEqual(recovered["last_fleet"]["cars"][0]["lap_session"]["best_lap_time_s"], 29.5)

            replay_frame = {
                **watch_frame, "run_id": "replay-run", "channel": "replay",
                "cars": [{"env_id": 0, "lap_supported": True,
                          "lap_times_s": [28.0], "lap_elapsed_s": 1.0}],
            }
            store.observe(replay_frame)
            # A stale live bus frame for another run must not replace the
            # replay status's authoritative raw last_fleet. This mirrors the
            # status shape returned by manager/evaluation service telemetry.
            hub_app.bus.last_replay_fleet = {
                **replay_frame, "run_id": "different-replay",
                "cars": [{"env_id": 0, "lap_times_s": [99.0]}],
            }
            hub_app.replay_job = SimpleNamespace(status=lambda: {
                "state": "running", "run_id": "replay-run",
                "last_fleet": replay_frame,
            })
            status, replay = asyncio.run(_get_json(hub_app.app, "/api/replay/status"))
            self.assertEqual(status, 200)
            self.assertEqual(replay["last_fleet"]["run_id"], "replay-run")
            self.assertEqual(replay["last_fleet"]["cars"][0]["lap_session"]["latest_lap_time_s"], 28.0)

            hub_app.bus.lap_sessions = previous_bus_store
            hub_app.official_runs = previous_runs
            hub_app.replay_job = previous_replay_job
            hub_app._run_output_dir = previous_run_dir


if __name__ == "__main__":
    unittest.main(verbosity=2)
