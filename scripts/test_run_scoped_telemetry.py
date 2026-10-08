from __future__ import annotations

import unittest

from src.layer4.hub.telemetry import TelemetryBus


class RunScopedTelemetryTests(unittest.TestCase):
    def test_latest_values_are_isolated_by_run_id(self):
        bus = TelemetryBus()
        bus.publish("telemetry", {"kind": "fleet", "run_id": "run-a", "step": 10})
        bus.publish("telemetry", {"kind": "fleet", "run_id": "run-b", "step": 99})
        bus.publish("telemetry", {"kind": "metrics", "run_id": "run-a", "step": 11})

        self.assertEqual(bus.run_snapshot("run-a")["last_fleet"]["step"], 10)
        self.assertEqual(bus.run_snapshot("run-a")["last_metrics"]["step"], 11)
        self.assertEqual(bus.run_snapshot("run-b")["last_fleet"]["step"], 99)
        self.assertIsNone(bus.run_snapshot("run-b")["last_metrics"])

    def test_run_phase_and_evaluator_are_scoped_and_clearable(self):
        bus = TelemetryBus()
        bus.publish_train_phase({"run_id": "r1", "phase": "rollout"})
        bus.publish_evaluator_live({"run_id": "r1", "snapshot_timesteps": 100})
        bus.clear_run("r1")
        self.assertEqual(bus.run_snapshot("r1"), {
            "run_id": "r1",
            "last_metrics": None,
            "last_fleet": None,
            "last_train_phase": None,
            "last_evaluator_live": None,
        })


if __name__ == "__main__":
    unittest.main()
