import tempfile
import unittest
from pathlib import Path

from src.layer4.hub.lap_session import LapSessionStore
from src.layer4.hub.telemetry import TelemetryBus


class LiveLapSessionTests(unittest.TestCase):
    def test_accumulates_completed_laps_across_episode_reset_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LapSessionStore(root)
            bus = TelemetryBus(lap_sessions=store)
            frame = {"kind": "fleet", "run_id": "run-a", "map_id": "oval", "direction": "forward",
                     "episode": 99, "cars": [{"env_id": 0, "episode": 1, "lap_supported": True,
                     "lap_times_s": [], "lap_elapsed_s": 12.0}]}
            bus.publish("telemetry", frame)
            frame["cars"][0]["lap_elapsed_s"] = 13.0
            bus.publish("telemetry", frame)
            frame["cars"][0].update(lap_times_s=[42.5], last_lap_time_s=42.5, lap_elapsed_s=1.0)
            bus.publish("telemetry", frame)
            frame["cars"][0]["episode"] = 2
            frame["cars"][0].update(lap_times_s=[], lap_elapsed_s=3.0)
            bus.publish("telemetry", frame)

            recovered = LapSessionStore(root).get("run-a")
            car = recovered["cars"][0]
            self.assertEqual(recovered["session_id"], "run-a")
            self.assertEqual((recovered["track_id"], recovered["direction"]), ("oval", "forward"))
            self.assertEqual([lap["time_s"] for lap in car["laps"]], [42.5])
            self.assertEqual(car["best_lap_time_s"], 42.5)
            self.assertEqual(car["interrupted_attempts"][0]["elapsed_s"], 1.0)
            self.assertEqual(car["lap_session"], {
                "completed_laps": 1, "latest_lap_time_s": 42.5,
                "best_lap_time_s": 42.5, "last_lap_delta_s": None,
                "current_lap_number": 2, "current_lap_elapsed_s": 3.0,
            })

    def test_cars_and_sessions_are_isolated_and_frames_are_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            bus = TelemetryBus(lap_sessions=LapSessionStore(Path(tmp)))
            frame = {"kind": "fleet", "run_id": "run-a", "episode": 1,
                     "cars": [{"env_id": 0, "lap_times_s": [40.0]},
                              {"env_id": 1, "lap_times_s": [50.0]}]}
            bus.publish("telemetry", frame)
            self.assertNotIn("lap_session", frame)
            self.assertEqual(len(bus.run_fleets["run-a"]["cars"]), 2)
            self.assertEqual(bus.run_fleets["run-a"]["cars"][0]["lap_session"]["latest_lap_time_s"], 40.0)
            self.assertNotIn("session_laps", bus.run_fleets["run-a"]["cars"][0])
            self.assertEqual(LapSessionStore(Path(tmp)).get("run-b")["cars"], [])

    def test_previous_sessions_only_match_known_track_layout_and_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LapSessionStore(Path(tmp))
            def publish(run_id, track, layout, direction, time):
                store.observe({"kind": "fleet", "run_id": run_id, "map_id": track,
                               "layout_id": layout, "direction": direction, "episode": 1,
                               "cars": [{"env_id": 0, "lap_times_s": [time]}]})
            publish("old-match", "track-a", "layout-1", "forward", 40.0)
            publish("old-layout", "track-a", "layout-2", "forward", 41.0)
            publish("current", "track-a", "layout-1", "forward", 39.0)
            history = store.history("current", env_id="0")
            self.assertEqual([row["run_id"] for row in history["previous_sessions"]], ["old-match"])
            self.assertEqual(history["previous_sessions"][0]["laps"][0]["time_s"], 40.0)
            self.assertEqual(history["previous_sessions"][0]["track_id"], "track-a")
            self.assertEqual(history["previous_sessions"][0]["layout_id"], "layout-1")
            self.assertEqual(history["previous_sessions"][0]["direction"], "forward")

    def test_sanitized_run_id_paths_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LapSessionStore(Path(tmp))
            self.assertNotEqual(store._path("a/b"), store._path("a_b"))

    def test_global_episode_counter_does_not_interrupt_other_cars(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LapSessionStore(Path(tmp))
            store.observe({"run_id": "multi", "episode": 1, "cars": [
                {"env_id": 0, "episode": 1, "lap_times_s": [], "lap_elapsed_s": 18.0},
                {"env_id": 1, "lap_times_s": [], "lap_elapsed_s": 27.0},
            ]})
            store.observe({"run_id": "multi", "episode": 2, "cars": [
                {"env_id": 0, "episode": 2, "lap_times_s": [], "lap_elapsed_s": 0.2},
                {"env_id": 1, "lap_times_s": [], "lap_elapsed_s": 28.0},
            ]})
            cars = {car["car_id"]: car for car in store.get("multi")["cars"]}
            self.assertEqual(cars["0"]["interrupted_attempts"][0]["elapsed_s"], 18.0)
            self.assertEqual(cars["1"]["interrupted_attempts"], [])

    def test_snapshot_laps_are_deep_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LapSessionStore(Path(tmp))
            store.observe({"kind": "fleet", "run_id": "copy", "episode": 1,
                           "cars": [{"env_id": 0, "lap_times_s": [40.0]}]})
            session = store.get("copy")
            session["cars"][0]["laps"][0]["time_s"] = 1.0
            self.assertEqual(store.get("copy")["cars"][0]["laps"][0]["time_s"], 40.0)

    def test_persisted_frame_enrichment_is_read_only_and_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LapSessionStore(root)
            store.observe({"run_id": "persisted", "map_id": "track", "layout_id": "layout",
                           "direction": "forward", "episode": 1,
                           "cars": [{"env_id": 0, "lap_supported": True,
                                     "lap_times_s": [40.0], "lap_elapsed_s": 2.0}]})
            reloaded = LapSessionStore(root)
            raw_frame = {"run_id": "persisted", "kind": "fleet", "cars": [
                {"env_id": 0, "lap_supported": True, "lap_count": 1,
                 "lap_times_s": [40.0], "lap_elapsed_s": 3.0}
            ]}
            enriched = reloaded.enrich_frame(raw_frame)
            self.assertEqual(enriched["cars"][0]["lap_session"]["completed_laps"], 1)
            self.assertEqual(enriched["cars"][0]["lap_session"]["current_lap_elapsed_s"], 3.0)
            self.assertNotIn("lap_session", raw_frame["cars"][0])
            self.assertEqual(reloaded.get("persisted")["cars"][0]["lap_session"]["completed_laps"], 1)

    def test_replay_status_enrichment_uses_persisted_run_summary_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LapSessionStore(Path(tmp))
            store.observe({"run_id": "replay-run", "episode": 1, "cars": [
                {"env_id": 0, "lap_supported": True, "lap_times_s": [35.0],
                 "lap_elapsed_s": 4.0}
            ]})
            raw_fleet = {"run_id": "replay-run", "kind": "fleet", "cars": [
                {"env_id": 0, "lap_supported": True, "lap_times_s": [35.0],
                 "lap_elapsed_s": 5.0}
            ]}
            raw_status = {"state": "running", "last_fleet": raw_fleet}
            enriched = LapSessionStore(Path(tmp)).enrich_status(raw_status)
            self.assertEqual(enriched["last_fleet"]["cars"][0]["lap_session"]["completed_laps"], 1)
            self.assertEqual(enriched["last_fleet"]["cars"][0]["lap_session"]["current_lap_elapsed_s"], 5.0)
            self.assertNotIn("lap_session", raw_status["last_fleet"]["cars"][0])


if __name__ == "__main__":
    unittest.main()
