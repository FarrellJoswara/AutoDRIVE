"""Checks that recovered Watch frames retain their original freshness."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.layer4.hub.app import _watch_snapshot_age_ms


class WatchSnapshotAgeTests(unittest.TestCase):
    def test_old_persisted_frame_is_not_reported_as_live(self) -> None:
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=4)).isoformat()

        age_ms = _watch_snapshot_age_ms({"ts": timestamp})

        self.assertIsNotNone(age_ms)
        self.assertGreaterEqual(age_ms, 3_900)
        self.assertLess(age_ms, 5_000)

    def test_missing_or_invalid_timestamp_has_unknown_age(self) -> None:
        self.assertIsNone(_watch_snapshot_age_ms(None))
        self.assertIsNone(_watch_snapshot_age_ms({"ts": "not-a-time"}))


if __name__ == "__main__":
    unittest.main()
