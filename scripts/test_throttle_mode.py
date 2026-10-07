"""Throttle action semantics shared by training and the official runner."""

import numpy as np
import pytest

from src.layer1.telemetry import TelemetrySnapshot
from src.layer2.autodrive_env import AutoDriveEnv
from src.layer2.spaces import map_policy_throttle
from src.layer4.settings import Settings
from src.layer3.train import build_arg_parser
from src.layer3.envs import env_kwargs_from_args


class CommandRacer:
    is_connected = True
    step_counter = 1
    port = 0
    action_interval_s = None

    def __init__(self):
        self.last_command = None
        self.snap = TelemetrySnapshot(lidar_ranges=np.full(1081, 4.0))

    def step(self, throttle, steering):
        self.last_command = (throttle, steering)
        return self.snap

    def reset(self):
        return self.step(0.0, 0.0)


@pytest.mark.parametrize(
    ("action", "expected"),
    [(-1.0, 0.0), (-0.5, 0.25), (0.0, 0.5), (0.5, 0.75), (1.0, 1.0)],
)
def test_forward_only_maps_centered_policy_action_to_half_throttle(action, expected):
    assert map_policy_throttle(action, "forward_only") == pytest.approx(expected)


def test_bidirectional_throttle_preserves_existing_policy_semantics():
    assert map_policy_throttle(-0.7, "bidirectional") == pytest.approx(-0.7)


def test_unknown_throttle_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown throttle mode"):
        map_policy_throttle(0.0, "reverse_only")


def test_layer_two_executes_forward_only_mapped_action():
    racer = CommandRacer()
    env = AutoDriveEnv(racer=racer, throttle_mode="forward_only")
    env.reset()
    env.step(np.asarray([0.0, 0.0], dtype=np.float32))
    assert racer.last_command == pytest.approx((0.5, 0.0))


def test_layer_four_persists_forward_only_through_train_argv():
    settings = Settings(throttle_mode="forward_only")
    argv = settings.to_train_argv()
    index = argv.index("--throttle-mode")
    assert argv[index + 1] == "forward_only"
    parsed = build_arg_parser().parse_args(argv)
    assert env_kwargs_from_args(parsed)["throttle_mode"] == "forward_only"
