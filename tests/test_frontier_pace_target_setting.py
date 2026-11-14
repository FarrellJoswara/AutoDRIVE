import pytest
from pydantic import ValidationError

from src.layer2.rewards import RewardConfig, compute_reward_components
from src.layer4.official_settings import OfficialTrainSettings


def _frontier_reward(target_mps: float) -> float:
    components = compute_reward_components(
        v_long=1.0,
        frontier_advanced_m=1.0,
        frontier_average_speed_mps=1.0,
        collision_event=False,
        slip_angle=0.0,
        prev_steering=0.0,
        steering=0.0,
        cfg=RewardConfig(
            route_progress_scale=1.0,
            frontier_pace_target_mps=target_mps,
            frontier_pace_bonus_strength=1.0,
        ),
    )
    return components["route_progress"]


def test_frontier_target_changes_pace_multiplier():
    assert _frontier_reward(1.5) == pytest.approx(1.0 + (1.0 / 1.5) ** 2)
    assert _frontier_reward(6.0) == pytest.approx(1.0 + (1.0 / 6.0) ** 2)
    assert _frontier_reward(1.5) > _frontier_reward(6.0)


def test_frontier_pace_target_defaults_and_flows_to_trainer_environment():
    default = OfficialTrainSettings()
    configured = OfficialTrainSettings(frontier_pace_target_mps=1.5)

    assert default.frontier_pace_target_mps == 6.0
    assert configured.to_container_env(hub_url="http://hub", run_id="run")[
        "AICAR_TRAIN_FRONTIER_PACE_TARGET_MPS"
    ] == "1.5"


@pytest.mark.parametrize("target", [0.49, 10.01, float("inf"), float("nan")])
def test_frontier_pace_target_is_bounded_and_finite(target):
    with pytest.raises(ValidationError):
        OfficialTrainSettings(frontier_pace_target_mps=target)
