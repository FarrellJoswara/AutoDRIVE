"""Pure tests for deterministic model-evaluation summaries."""

from __future__ import annotations

import unittest

from src.layer3.evaluate import summarize_episodes


class EvaluationSummaryTests(unittest.TestCase):
    def test_summary_uses_only_clean_lap_paces_for_speed_statistics(self) -> None:
        summary = summarize_episodes([
            {"episode_won": True, "lap_count": 1, "best_lap_time_s": 12.0,
             "termination_reason": "lap_target"},
            {"episode_won": False, "lap_count": 0, "best_lap_time_s": 8.0,
             "termination_reason": "collision"},
            {"episode_won": True, "lap_count": 1, "best_lap_time_s": 10.0,
             "termination_reason": "lap_target"},
        ])
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["clean_episode_wins"], 2)
        self.assertEqual(summary["completed_laps"], 2)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3)
        self.assertEqual(summary["best_lap_time_s"], 10.0)
        self.assertEqual(summary["mean_clean_lap_time_s"], 11.0)
        self.assertEqual(summary["termination_reasons"], {"collision": 1, "lap_target": 2})


if __name__ == "__main__":
    unittest.main()
