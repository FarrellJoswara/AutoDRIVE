import json
from unittest.mock import Mock

from src.layer3.hub_callback import HubTelemetryCallback


def test_official_timing_diagnostics_compare_action_elapsed_to_official_lap(tmp_path):
    callback = HubTelemetryCallback(
        hub_url="http://hub", run_id="timing-test", runtime="official",
        run_dir=tmp_path, fleet_hz=0.1,
    )
    callback._pub = Mock()
    callback._clock_lap_count_by_env[0] = 0
    callback.num_timesteps = 1
    callback.locals = {"infos": [{
        "control_interval_source": "ros_sensor_stamp",
        "control_interval_s": 0.05,
        "control_interval_wall_s": 0.06,
        "lap_count": 1,
        "last_lap_time_s": 12.5,
    }], "dones": [False], "rewards": [0.0]}
    assert callback._on_step()
    callback._on_training_end()
    result = json.loads((tmp_path / "timing_diagnostics.json").read_text())
    assert result["clock_sources"] == {"ros_sensor_stamp": 1}
    assert result["ros_to_wall_elapsed_ratio_by_env"] == {"0": 0.05 / 0.06}
    assert result["completed_lap_clock_comparisons"] == [{
        "env_id": 0, "action_elapsed_s": 0.05,
        "wall_elapsed_s": 0.06,
        "official_lap_time_s": 12.5, "ratio": 0.004,
    }]
