"""Tests for the official simulator's fixed-command control-path diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

from src.layer3.official_policy import StraightLineDiagnostic, load_policy


class StraightLineDiagnosticTests(unittest.TestCase):
    def test_emits_fixed_forward_throttle_and_zero_steering(self) -> None:
        policy = StraightLineDiagnostic(throttle=0.35)

        action, state = policy.predict({"unused": np.zeros(1)}, deterministic=True)

        np.testing.assert_array_equal(action, np.asarray([0.35, 0.0], dtype=np.float32))
        self.assertIsNone(state)

    def test_rejects_invalid_throttle(self) -> None:
        for throttle in (0.0, -0.1, 1.1):
            with self.subTest(throttle=throttle), self.assertRaises(ValueError):
                StraightLineDiagnostic(throttle=throttle)

    def test_diagnostic_does_not_require_a_checkpoint(self) -> None:
        policy = load_policy("unused.zip", controller="straight")

        action, _ = policy.predict(None)

        self.assertGreater(float(action[0]), 0.0)
        self.assertEqual(float(action[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
