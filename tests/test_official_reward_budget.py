import math

import pytest
from pydantic import ValidationError

from src.layer4.official_settings import OfficialTrainSettings
from src.layer2.official_race_env import OfficialRaceEnv

from test_official_competition import FakeRacer, _snapshot


def test_settings_allow_full_pace_strength_with_250_point_penalties():
    settings = OfficialTrainSettings(
        frontier_pace_bonus_strength=1.0,
        centerline_shaping_strength=20.0,
        collision_penalty_magnitude=250.0,
        failed_episode_penalty=250.0,
    )

    assert settings.frontier_pace_bonus_strength == 1.0


@pytest.mark.parametrize("strength", [-0.01, math.inf, -math.inf, math.nan])
def test_settings_reject_invalid_reward_strengths(strength):
    with pytest.raises(ValidationError):
        OfficialTrainSettings(frontier_pace_bonus_strength=strength)


def test_settings_require_penalties_strictly_above_dynamic_reward_budget():
    with pytest.raises(ValidationError, match="strictly exceed"):
        OfficialTrainSettings(
            frontier_pace_bonus_strength=1.0,
            centerline_shaping_strength=20.0,
            collision_penalty_magnitude=220.0,
        )


def test_settings_preserve_legacy_strictly_above_120_point_minimum():
    with pytest.raises(ValidationError, match="strictly exceed"):
        OfficialTrainSettings(collision_penalty_magnitude=120.0)


def test_official_environment_centerline_strength_defaults_to_zero():
    env = OfficialRaceEnv(racer=FakeRacer([_snapshot(lap=0, last_lap=0, collisions=0)]))

    assert env.training_centerline_potential_cap == 0.0


def test_official_environment_uses_explicit_centerline_strength():
    env = OfficialRaceEnv(
        racer=FakeRacer([_snapshot(lap=0, last_lap=0, collisions=0)]),
        training_centerline_shaping_strength=20.0,
    )

    assert env.training_centerline_potential_cap == 20.0
