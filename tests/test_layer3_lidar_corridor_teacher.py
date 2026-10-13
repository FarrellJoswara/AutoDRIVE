import numpy as np
import pytest

from src.layer3.lidar_corridor_teacher import LidarCorridorTeacher


def _scan_with_close_wall(low_deg=None, high_deg=None, distance_m=10.0):
    ranges_m = np.full(1081, 10.0, dtype=np.float32)
    if low_deg is not None:
        angles = np.linspace(-135.0, 135.0, ranges_m.size)
        ranges_m[(angles >= low_deg) & (angles <= high_deg)] = distance_m
    return np.clip((ranges_m - 0.06) / (10.0 - 0.06), 0.0, 1.0)


def _info(**values):
    return {
        "lidar_range_min": 0.06,
        "lidar_range_max": 10.0,
        "watch_speed_mps": 0.0,
        **values,
    }


def test_open_track_drives_slowly_and_stays_straight():
    teacher = LidarCorridorTeacher(max_speed_mps=0.3)

    action = teacher.action(_info(), {"lidar": np.ones(1081, dtype=np.float32)})

    assert 0.0 < action[0] <= 0.1
    assert abs(float(action[1])) < 0.02


def test_close_right_wall_steers_left_and_close_left_wall_steers_right():
    teacher = LidarCorridorTeacher()
    right_wall = _scan_with_close_wall(15.0, 85.0, distance_m=0.5)
    left_wall = _scan_with_close_wall(-85.0, -15.0, distance_m=0.5)

    steer_away_from_right = teacher.action(_info(), {"lidar": right_wall})[1]
    steer_away_from_left = teacher.action(_info(), {"lidar": left_wall})[1]

    assert steer_away_from_right > 0.1
    assert steer_away_from_left < -0.1


def test_teacher_steering_comes_from_lidar_not_privileged_route_pose():
    teacher = LidarCorridorTeacher()
    lidar = _scan_with_close_wall(15.0, 85.0, distance_m=0.5)

    first = teacher.action(_info(position=[0.0, 0.0], yaw=0.0), {"lidar": lidar})
    second = teacher.action(_info(position=[50.0, -20.0], yaw=2.3), {"lidar": lidar})

    np.testing.assert_allclose(first, second)


def test_immediate_front_obstacle_stops_forward_throttle():
    teacher = LidarCorridorTeacher()
    scan = _scan_with_close_wall(-10.0, 10.0, distance_m=0.2)

    action = teacher.action(_info(), {"lidar": scan})

    assert action[0] == 0.0


def test_close_bend_clearance_keeps_a_crawl_so_steering_can_escape():
    teacher = LidarCorridorTeacher()
    scan = _scan_with_close_wall(-10.0, 10.0, distance_m=0.3)

    action = teacher.action(_info(), {"lidar": scan})

    assert action[0] > 0.0


def test_batch_labels_require_only_layer2_lidar():
    teacher = LidarCorridorTeacher()
    batch = teacher.action_batch([_info()], {"lidar": np.ones((1, 1081), dtype=np.float32)})

    assert batch.shape == (1, 2)
    with pytest.raises(ValueError, match="LiDAR"):
        teacher.action_batch([_info()], {"state": np.zeros((1, 9), dtype=np.float32)})
