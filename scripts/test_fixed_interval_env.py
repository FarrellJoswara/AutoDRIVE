"""Environment timing regression tests; no Unity process required."""
import numpy as np
import pytest

from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.autodrive_env import AutoDriveEnv


class ClockRacer:
    is_connected = True
    step_counter = 1
    port = 0

    def __init__(self, interval=None):
        self.action_interval_s = interval
        self.clock = 0.0
        self.calls = 0
        self.snap = TelemetrySnapshot(lidar_ranges=np.full(1081, 4.0))

    def simulation_time(self):
        return self.clock

    def step(self, throttle, steering):
        self.calls += 1
        self.clock += self.action_interval_s or 0.025
        return self.snap

    def reset(self):
        return self.step(0, 0)


@pytest.mark.parametrize("frame_skip", [1, 2, 4])
def test_reward_uses_acknowledged_simulation_duration(frame_skip):
    racer = ClockRacer(0.086)
    env = AutoDriveEnv(racer=racer, frame_skip=frame_skip)
    env.reset()
    racer.calls = 0
    _, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert racer.calls == frame_skip
    assert info["step_duration_s"] == pytest.approx(frame_skip * 0.086)
    assert reward == pytest.approx(-5 * frame_skip * 0.086)
    assert not terminated and not truncated
    assert env._progress_time(racer.snap) == racer.clock


def test_legacy_reward_duration_preserved():
    racer = ClockRacer()
    env = AutoDriveEnv(racer=racer, frame_skip=2)
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(2))
    assert info["step_duration_s"] == 0.05
    assert reward == pytest.approx(-0.25)


def test_injected_racer_interval_must_match():
    with pytest.raises(ValueError, match="Injected Racer"):
        AutoDriveEnv(racer=ClockRacer(), action_interval_s=0.086)


def test_invalid_simulation_ack_cannot_create_reward():
    racer = ClockRacer(0.086)
    env = AutoDriveEnv(racer=racer)
    env.reset()
    racer.step = lambda throttle, steering: racer.snap
    with pytest.raises(RuntimeError, match="Invalid acknowledged"):
        env.step(np.zeros(2))


def test_evaluation_normalizes_by_reported_duration():
    from src.layer3.evaluate import evaluate_until_episode_end

    class Model:
        def predict(self, obs, deterministic):
            return np.zeros(2), None

    class VectorEnv:
        def reset(self):
            return [0]

        def step(self, action):
            return [0], [2.0], [True], [{
                "step_duration_s": 0.086,
                "frontier_advanced_m": 0.43,
                "termination_reason": "frontier_stagnation",
            }]

    result = evaluate_until_episode_end(Model(), VectorEnv(), frame_skip=1)
    assert result["simulated_seconds"] == pytest.approx(0.086)
    assert result["frontier_speed_mps"] == pytest.approx(5.0)
    assert result["reward_per_simulated_second"] == pytest.approx(2 / 0.086)
