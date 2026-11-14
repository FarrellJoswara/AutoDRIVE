import pytest

from src.layer3.official_envs import wait_for_official_sensor_readiness


class FakeVecEnv:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.last = []

    def env_method(self, name, *args, **kwargs):
        self.last = next(self.responses, self.last)
        return self.last


def test_sensor_readiness_waits_for_all_domains():
    env = FakeVecEnv([
        [{"ready": True, "callback_count": {"scan": 3}}, {"ready": False}],
        [{"ready": True}, {"ready": True}],
    ])
    result = wait_for_official_sensor_readiness(env, timeout_s=1, poll_s=0.001)
    assert len(result) == 2
    assert all(domain["ready"] for domain in result)


def test_sensor_readiness_timeout_includes_domain_diagnostics():
    env = FakeVecEnv([[{
        "ready": False,
        "bridge_identity": {"ros_domain_id": "7"},
        "callback_count": {"scan": 0},
        "topic_age_s": {"scan": None},
    }]])
    with pytest.raises(TimeoutError, match='"ros_domain_id": "7"'):
        wait_for_official_sensor_readiness(env, timeout_s=0.01, poll_s=0.001)


def test_sensor_readiness_rejects_nonpositive_timeout():
    with pytest.raises(ValueError):
        wait_for_official_sensor_readiness(FakeVecEnv([]), timeout_s=0)
