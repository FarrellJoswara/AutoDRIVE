"""Synthetic integration checks for live lap session aggregation.

Run with ``python tests/test_live_lap_integration.py``. These checks feed
synthetic telemetry into the hub session store; they do not launch a simulator.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_lap_session_store():
    # The session store itself is stdlib-only. Load it under a unique QA name
    # so collecting this test never imports or replaces the production src.*
    # package namespace.
    spec = importlib.util.spec_from_file_location(
        "_live_lap_qa_session", ROOT / "src" / "layer4" / "hub" / "lap_session.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LapSessionStore


LapSessionStore = _load_lap_session_store()


def _load_lap_tracker():
    # LapTracker only needs RouteProgressTracker for annotations and its
    # optional map-config helper. Use a unique temporary package namespace.
    package_name = "_live_lap_qa_layer2"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src" / "layer2")]
    sys.modules[package_name] = package
    route_module = types.ModuleType(f"{package_name}.route_progress")
    route_module.RouteProgressTracker = object
    sys.modules[route_module.__name__] = route_module
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.lap_tracker", ROOT / "src" / "layer2" / "lap_tracker.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.modules.pop(spec.name, None)
    sys.modules.pop(route_module.__name__, None)
    sys.modules.pop(package_name, None)
    return module.LapTracker


@contextmanager
def _temporary_modules(prefixes: tuple[str, ...]):
    """Restore module-cache entries after an isolated fallback import."""
    original = {name: module for name, module in sys.modules.items()
                if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)}
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(original)


LapTracker = _load_lap_tracker()


def _fleet(run_id: str, episode: int, cars: list[dict], step: int) -> dict:
    return {
        "kind": "fleet",
        "run_id": run_id,
        "episode": episode,
        "step": step,
        "ts": f"2026-11-06T00:00:{step:02d}+00:00",
        "map_id": "synthetic-track",
        "direction": "forward",
        "cars": cars,
    }


class LiveLapIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_vector_episode_counter_does_not_reset_another_cars_split_cursor(self):
        store = LapSessionStore(self.root)
        store.observe(_fleet("run-a", 1, [
            {"env_id": 0, "lap_times_s": [31.0], "lap_elapsed_s": 8.0},
            {"env_id": 1, "lap_times_s": [34.0], "lap_elapsed_s": 9.0},
        ], 1))
        # hub_callback's top-level episode counter increments when *any* env
        # completes. Env 1 has not reset, so its episode-scoped split array
        # continues and must not be re-appended from index zero.
        store.observe(_fleet("run-a", 2, [
            {"env_id": 0, "lap_times_s": [], "lap_elapsed_s": 2.0},
            {"env_id": 1, "lap_times_s": [34.0, 32.0], "lap_elapsed_s": 1.0},
        ], 2))
        store.observe(_fleet("run-b", 1, [
            {"env_id": 0, "lap_times_s": [27.0], "lap_elapsed_s": 3.0},
        ], 1))

        run_a = store.get("run-a")
        run_b = store.get("run-b")
        cars_a = {car["car_id"]: car for car in run_a["cars"]}

        self.assertEqual([lap["time_s"] for lap in cars_a["0"]["laps"]], [31.0])
        self.assertEqual(cars_a["0"]["best_lap_time_s"], 31.0)
        self.assertEqual([lap["time_s"] for lap in cars_a["1"]["laps"]], [34.0, 32.0])
        self.assertEqual([lap["episode"] for lap in cars_a["1"]["laps"]], [1, 1])
        self.assertEqual(cars_a["1"]["best_lap_time_s"], 32.0)
        self.assertEqual([lap["time_s"] for lap in run_b["cars"][0]["laps"]], [27.0])
        self.assertEqual(cars_a["0"]["laps"][0]["status"], "complete")
        self.assertEqual(cars_a["0"]["interrupted_attempts"][0]["status"], "interrupted")

    def test_public_history_is_a_deep_snapshot(self):
        store = LapSessionStore(self.root)
        store.observe(_fleet("run-snapshot", 1, [
            {"env_id": 0, "lap_times_s": [22.0], "lap_elapsed_s": 3.0},
        ], 1))

        exposed = store.get("run-snapshot")
        exposed["cars"][0]["laps"].clear()

        self.assertEqual(len(store.get("run-snapshot")["cars"][0]["laps"]), 1)

    def test_lap_tracker_requires_a_full_forward_crossing_and_interpolates_time(self):
        class ClosedRoute:
            closed = True
            length_m = 100.0

            @staticmethod
            def gate_line_at(_progress_m):
                return ((0.0, 0.0), (1.0, 0.0))

        tracker = LapTracker(ClosedRoute(), gate_progress_m=10.0)
        tracker.reset({"progress_m": 10.0}, now=50.0)

        before_crossing = tracker.update(109.9, now=149.0)
        complete = tracker.update(110.0, now=150.0)

        self.assertEqual(before_crossing["lap_count"], 0)
        self.assertEqual(complete["lap_count"], 1)
        self.assertAlmostEqual(complete["last_lap_time_s"], 100.0)
        self.assertEqual(complete["lap_times_s"], [100.0])

    def test_mid_lap_start_arms_at_first_gate_crossing_without_counting_partial_lap(self):
        class ClosedRoute:
            closed = True
            length_m = 100.0

            @staticmethod
            def gate_line_at(_progress_m):
                return ((0.0, 0.0), (1.0, 0.0))

        tracker = LapTracker(ClosedRoute(), gate_progress_m=0.0)
        tracker.reset({"progress_m": 10.0}, now=10.0)

        armed = tracker.update(100.0, now=20.0)
        partial = tracker.update(199.9, now=29.0)
        complete = tracker.update(200.0, now=30.0)

        self.assertEqual(armed["lap_count"], 0)
        self.assertIsNone(armed["last_lap_time_s"])
        self.assertEqual(partial["lap_count"], 0)
        self.assertEqual(complete["lap_count"], 1)
        self.assertAlmostEqual(complete["last_lap_time_s"], 10.0)

    def test_process_reload_does_not_duplicate_current_episode_splits(self):
        frame = _fleet("run-restart", 4, [
            {"env_id": 0, "lap_times_s": [24.0, 23.5], "lap_elapsed_s": 2.0},
        ], 12)
        first_process = LapSessionStore(self.root)
        first_process.observe(frame)

        restarted_process = LapSessionStore(self.root)
        restarted_process.observe(frame)

        self.assertEqual(len(restarted_process.get("run-restart")["cars"][0]["laps"]), 2)

    def test_incomplete_elapsed_time_never_becomes_a_completed_lap(self):
        store = LapSessionStore(self.root)
        store.observe(_fleet("run-partial", 1, [
            {"env_id": 0, "lap_times_s": [], "lap_elapsed_s": 45.0},
        ], 1))
        car = store.get("run-partial")["cars"][0]
        self.assertEqual(car["laps"], [])
        self.assertEqual(car["current_lap_elapsed_s"], 45.0)

    def test_terminal_reset_frame_keeps_existing_split_cursor_until_episode_array_resets(self):
        store = LapSessionStore(self.root)
        store.observe(_fleet("run-terminal", 1, [
            {"env_id": 0, "episode": 3, "lap_times_s": [31.0], "lap_elapsed_s": 8.0},
        ], 1))
        # hub_callback emits the terminal info with reset=True. Its split array
        # still belongs to the finished episode and may contain one new lap.
        store.observe(_fleet("run-terminal", 2, [
            {"env_id": 0, "episode": 3, "reset": True,
             "lap_times_s": [31.0, 29.0], "lap_elapsed_s": 0.0},
        ], 2))

        self.assertEqual(
            [lap["time_s"] for lap in store.get("run-terminal")["cars"][0]["laps"]],
            [31.0, 29.0],
        )

    def test_car_local_episode_advances_when_reset_episode_had_no_completed_splits(self):
        store = LapSessionStore(self.root)
        store.observe(_fleet("run-empty-episode", 0, [
            {"env_id": 0, "lap_times_s": [], "lap_elapsed_s": 45.0},
        ], 1))
        store.observe(_fleet("run-empty-episode", 1, [
            {"env_id": 0, "reset": True, "lap_times_s": [], "lap_elapsed_s": 0.0},
        ], 2))
        store.observe(_fleet("run-empty-episode", 1, [
            {"env_id": 0, "lap_times_s": [], "lap_elapsed_s": 0.0},
        ], 3))
        store.observe(_fleet("run-empty-episode", 1, [
            {"env_id": 0, "lap_times_s": [33.0], "lap_elapsed_s": 1.0},
        ], 4))

        laps = store.get("run-empty-episode")["cars"][0]["laps"]
        self.assertEqual([lap["time_s"] for lap in laps], [33.0])
        self.assertEqual([lap["episode"] for lap in laps], [1])

    def test_matching_run_comparison_requires_known_track_layout_and_direction(self):
        store = LapSessionStore(self.root)

        def publish(run_id: str, map_id: str | None, layout_id: str | None,
                    direction: str | None, lap: float) -> None:
            payload = _fleet(run_id, 1, [
                {"env_id": 0, "episode": 1, "lap_times_s": [lap]},
            ], 1)
            payload.update(map_id=map_id, layout_id=layout_id, direction=direction)
            store.observe(payload)

        publish("prior-match", "oval", "layout-a", "forward", 30.0)
        publish("prior-other-layout", "oval", "layout-b", "forward", 29.0)
        publish("current-match", "oval", "layout-a", "forward", 28.0)
        matched = store.history("current-match", env_id="0")
        self.assertEqual([row["run_id"] for row in matched["previous_sessions"]], ["prior-match"])

        publish("current-unknown", None, None, None, 27.0)
        self.assertEqual(store.history("current-unknown", env_id="0")["previous_sessions"], [])

    def test_lap_history_is_recorded_before_fleet_frames_are_coalesced(self):
        # The bare QA interpreter may omit FastAPI. Scope its import stubs to
        # this test and restore src.* / fastapi entries before returning.
        with _temporary_modules(("fastapi", "src")):
            try:
                __import__("fastapi")
            except ModuleNotFoundError:
                fastapi_stub = types.ModuleType("fastapi")
                fastapi_stub.WebSocket = object
                sys.modules["fastapi"] = fastapi_stub
            for name, path in (
                ("src", ROOT / "src"),
                ("src.layer4", ROOT / "src" / "layer4"),
                ("src.layer4.hub", ROOT / "src" / "layer4" / "hub"),
            ):
                if name not in sys.modules:
                    package = types.ModuleType(name)
                    package.__path__ = [str(path)]
                    sys.modules[name] = package
            session_module = types.ModuleType("src.layer4.hub.lap_session")
            session_module.LapSessionStore = LapSessionStore
            sys.modules[session_module.__name__] = session_module
            spec = importlib.util.spec_from_file_location(
                "src.layer4.hub.telemetry", ROOT / "src" / "layer4" / "hub" / "telemetry.py"
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            TelemetryBus, WSClient = module.TelemetryBus, module.WSClient

            class Socket:
                async def send_text(self, _text: str) -> None:
                    return None

            store = LapSessionStore(self.root)
            bus = TelemetryBus(lap_sessions=store)
            client = WSClient(Socket())
            bus._clients.add(client)
            for step, times in enumerate(([30.0], [30.0, 29.5], [30.0, 29.5, 29.0]), 1):
                bus.publish("telemetry", _fleet("run-coalesce", 1, [
                    {"env_id": 0, "lap_supported": True, "lap_times_s": list(times),
                     "lap_elapsed_s": 1.0},
                ], step))

            self.assertEqual(client.fleet_frames_coalesced, 2)
            summary = store.get("run-coalesce")["cars"][0]["lap_session"]
            self.assertEqual(summary["completed_laps"], 3)
            self.assertEqual(summary["latest_lap_time_s"], 29.0)
            self.assertEqual(summary["best_lap_time_s"], 29.0)
            self.assertEqual(summary["last_lap_delta_s"], -0.5)
            self.assertEqual(summary["current_lap_number"], 4)
            queued = json.loads(json.dumps(client._latest_fleet["telemetry"]))
            self.assertEqual(queued["payload"]["step"], 3)
            self.assertEqual(len(store.get("run-coalesce")["cars"][0]["laps"]), 3)
            self.assertEqual(queued["payload"]["cars"][0]["lap_session"], summary)
            client.close()
            asyncio.run(client.sender())


if __name__ == "__main__":
    unittest.main(verbosity=2)
