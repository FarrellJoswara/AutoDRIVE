from __future__ import annotations

import json
import fcntl
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.monitor_speed_campaign import inspect, write_status


class CampaignMonitorTests(unittest.TestCase):
    def fixture(self, root: Path, phase="halted_production_startup"):
        (root / "state.json").write_text(json.dumps({"phase": phase, "runs": {
            "production:0": {"run_id": "r1", "display_name": "prod", "retry_count": 1,
                             "retry_exhausted": True}}}))
        (root / "runner.pid").write_text("99999999")
        (root / "runner.lock").touch()

    def test_failed_terminal_run_and_retry_exhaustion_are_alert_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            replies = {
                "http://hub/health": {"status": "ok", "hub_pid": 1},
                "http://hub/train/runs": {"runs": [{"run_id": "r1", "state": "failed"}]},
                "http://hub/api/replay/status": {"state": "complete", "validation_mode": "porto_practice_diagnostic"}}
            with patch("scripts.monitor_speed_campaign.get_json", side_effect=lambda u, **kw: replies[u]):
                status = inspect(root, "http://hub")
            self.assertIn("run_failed:production:0", status["alerts"])
            self.assertIn("retry_exhausted:production:0", status["alerts"])
            self.assertFalse(status["runner"]["live"])
            self.assertEqual(status["runs"]["production:0"]["retry_count"], 1)

    def test_inspect_does_not_create_runner_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "runner.lock").unlink()
            replies = {
                "http://hub/health": {"status": "ok", "hub_pid": 1},
                "http://hub/train/runs": {"runs": []},
                "http://hub/api/replay/status": {"state": "idle"},
            }
            with patch("scripts.monitor_speed_campaign.get_json", side_effect=lambda u, **kw: replies[u]):
                status = inspect(root, "http://hub")
            self.assertFalse(status["runner"]["lock_held"])
            self.assertFalse((root / "runner.lock").exists())

    def test_stale_recorded_pid_uses_live_runner_lock_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "production")
            held_lock = (root / "runner.lock").open("r+")
            fcntl.flock(held_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            replies = {
                "http://hub/health": {"status": "ok", "hub_pid": 1},
                "http://hub/train/runs": {"runs": []},
                "http://hub/api/replay/status": {"state": "idle"},
            }
            with patch("scripts.monitor_speed_campaign.get_json", side_effect=lambda u, **kw: replies[u]), \
                 patch("scripts.monitor_speed_campaign.lock_owner_pid", return_value=12345), \
                 patch("scripts.monitor_speed_campaign.pid_alive", side_effect=lambda pid: str(pid) == "12345"):
                status = inspect(root, "http://hub")
            self.assertEqual(status["runner"]["recorded_pid"], 99999999)
            self.assertEqual(status["runner"]["pid"], 12345)
            self.assertTrue(status["runner"]["pid_alive"])
            self.assertTrue(status["runner"]["lock_held"])
            self.assertTrue(status["runner"]["live"])
            self.assertNotIn("runner_stopped", status["alerts"])
            held_lock.close()

    def test_active_run_flags_stale_audit_telemetry_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, "production")
            out = root / "run"
            (out / "checkpoints").mkdir(parents=True)
            (out / "checkpoints" / "model.zip").touch()
            (out / "ppo_optimizer_audit.jsonl").write_text(json.dumps({"num_timesteps": 10,
                "metrics_finite": True, "gradients_finite": True, "parameters_finite": True,
                "optimizer_state_finite": True}) + "\n")
            (out / "telemetry.jsonl").touch()
            s = json.loads((root / "state.json").read_text())
            s["runs"]["production:0"]["output_dir"] = str(out)
            (root / "state.json").write_text(json.dumps(s))
            import os, time
            old = time.time() - 10000
            for p in [out / "ppo_optimizer_audit.jsonl", out / "telemetry.jsonl", out / "checkpoints" / "model.zip"]:
                os.utime(p, (old, old))
            replies = {
                "http://hub/health": {"status": "ok"},
                "http://hub/train/runs": {"runs": [{"run_id": "r1", "state": "running", "output_dir": str(out)}]},
                "http://hub/api/replay/status": {"state": "idle"}}
            with patch("scripts.monitor_speed_campaign.get_json", side_effect=lambda u, **kw: replies[u]):
                status = inspect(root, "http://hub", now=time.time(), audit_stall_s=30,
                                 telemetry_stall_s=30, checkpoint_stall_s=30)
            self.assertIn("audit_stalled:production:0", status["alerts"])
            self.assertIn("telemetry_stalled:production:0", status["alerts"])
            self.assertIn("checkpoint_stalled:production:0", status["alerts"])

    def test_alert_log_only_appends_new_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = {"checked_utc": "now", "alerts": ["hub_unavailable"], "phase": "x"}
            write_status(root, status)
            write_status(root, status)
            rows = (root / "monitor_alerts.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["new_alerts"], ["hub_unavailable"])
            self.assertTrue((root / "monitor_status.json").is_file())

    def test_repeated_step_and_audit_signature_stalls_after_window(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = {"alerts": [], "progress_watch": {
                "audit:production:0": {"signature": [100, 4], "unchanged_since": time.time() - 1000}}}
            (root / "monitor_status.json").write_text(json.dumps(prior))
            status = {"checked_utc": "now", "alerts": [], "runs": {"production:0": {
                "active": True, "progress": {"step": 100, "audit_count": 4},
                "telemetry_age_s": 2, "telemetry_mtime": 10,
                "checkpoint_count": 1, "newest_checkpoint_age_s": 2,
                "newest_checkpoint_mtime": 10}}, "evaluation": {}}
            write_status(root, status)
            self.assertIn("audit_progress_stalled:production:0", status["alerts"])

    def test_advancing_hub_telemetry_step_prevents_stall_without_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = {"alerts": [], "progress_watch": {
                "telemetry:production:0": {
                    "signature": [1_349_942, None], "unchanged_since": 0}}}
            (root / "monitor_status.json").write_text(json.dumps(prior))
            status = {"checked_utc": "now", "alerts": [], "runs": {"production:0": {
                "active": True, "progress": {}, "telemetry_step": 1_350_200,
                "telemetry_mtime": None, "checkpoint_count": 0,
                "newest_checkpoint_mtime": None}}, "evaluation": {}}
            write_status(root, status)
            self.assertNotIn("telemetry_progress_stalled:production:0", status["alerts"])
            saved = json.loads((root / "monitor_status.json").read_text())
            self.assertEqual(saved["progress_watch"]["telemetry:production:0"]["signature"],
                             [1_350_200, None])

    def test_unchanged_telemetry_step_and_mtime_stalls_after_window(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = {"alerts": [], "progress_watch": {
                "telemetry:production:0": {
                    "signature": [100, 10.0], "unchanged_since": time.time() - 1000}}}
            (root / "monitor_status.json").write_text(json.dumps(prior))
            status = {"checked_utc": "now", "alerts": [], "runs": {"production:0": {
                "active": True, "progress": {}, "telemetry_step": 100,
                "telemetry_mtime": 10.0, "checkpoint_count": 0,
                "newest_checkpoint_mtime": None}}, "evaluation": {}}
            write_status(root, status)
            self.assertIn("telemetry_progress_stalled:production:0", status["alerts"])


if __name__ == "__main__":
    unittest.main()
