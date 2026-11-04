"""Geometry and isolation checks for the privileged map teacher."""
import numpy as np
import pytest

from src.layer2.route_progress import RouteProgressTracker
from src.layer3.expert import CenterlineExpert
from src.layer3.official_teacher import _verified_lap_goal_reached


def straight_teacher():
    return CenterlineExpert(RouteProgressTracker(np.array([[0., 0.], [10., 0.], [20., 0.]])), max_speed_mps=0.3)


def pose(x=2., y=0., yaw=0.):
    return dict(position=[x, y, 0.], yaw=yaw, true_speed=0., lidar_range_min=0.06, lidar_range_max=10.)


def test_aligned_straight_drives_forward_without_steering():
    action = straight_teacher().action(pose())
    assert action[0] > 0
    assert abs(action[1]) < 1e-6


@pytest.mark.parametrize('y,sign', [(0.3,-1),(-0.3,1)])
def test_lateral_error_steers_back_toward_line(y, sign):
    action = straight_teacher().action(pose(y=y))
    assert action[1] * sign > 0


@pytest.mark.parametrize('yaw,sign', [(0.2,1),(-0.2,-1)])
def test_heading_error_steers_back_toward_line(yaw, sign):
    assert straight_teacher().action(pose(yaw=yaw))[1] * sign > 0


def test_teacher_projects_actual_pose_even_when_reward_frontier_is_stale():
    teacher = straight_teacher()
    info = pose(y=0.3)
    baseline = teacher.action(info)
    info.update(current_progress_m=19., route_projection_valid=True)
    np.testing.assert_allclose(teacher.action(info), baseline)


def test_side_scan_imbalance_cannot_cancel_route_steering():
    teacher = straight_teacher()
    info = pose(y=0.3)
    baseline = teacher.action(info, np.ones(1081))
    for side in (slice(0,400), slice(681,1081)):
        scan = np.ones(1081)
        scan[side] = 0.01
        np.testing.assert_allclose(teacher.action(info, scan), baseline)


def test_front_contact_risk_stops_throttle_without_changing_target():
    teacher = straight_teacher()
    info = pose(y=0.3)
    baseline = teacher.action(info, np.ones(1081))
    scan = np.ones(1081)
    scan[480:601] = 0.0
    guarded = teacher.action(info, scan)
    assert guarded[0] == 0
    assert guarded[1] == baseline[1]


def test_missing_or_nonfinite_pose_returns_neutral():
    teacher = straight_teacher()
    for info in ({}, pose(x=float('nan')), pose(yaw=float('nan'))):
        np.testing.assert_array_equal(teacher.action(info), [0.,0.])


def test_closed_route_tracking_recovers_offset_and_crosses_seam():
    # A bicycle-model regression checks geometry/sign and seam handling only;
    # it is not evidence that the official simulator can complete laps.
    angle = np.linspace(0., 2*np.pi, 241)
    route = RouteProgressTracker(np.column_stack([3*np.cos(angle), 3*np.sin(angle)]))
    teacher = CenterlineExpert(route, max_speed_mps=0.3, lookahead_base_m=0.8)
    xy = np.array([3.15,0.])
    yaw = -np.pi/2
    bearings = []
    errors = []
    for _ in range(1100):
        action = teacher.action(dict(position=xy.tolist(), yaw=yaw, true_speed=0.3))
        xy += 0.04*np.array([np.cos(yaw),-np.sin(yaw)])
        yaw -= 0.04/teacher.wheelbase_m*np.tan(float(action[1])*teacher.max_steer_angle_rad)
        bearings.append(np.arctan2(xy[1],xy[0]))
        errors.append(abs(np.linalg.norm(xy)-3.))
    assert np.unwrap(bearings)[-1] > 4*np.pi
    assert max(errors[100:]) < 0.10


def test_route_diagnostics_apply_vehicle_clearance_to_corridor_width():
    widths = np.full(3, 0.5)
    route = RouteProgressTracker(
        np.array([[0., 0.], [10., 0.], [20., 0.]]), widths, widths
    )
    teacher = CenterlineExpert(route, vehicle_half_width_m=0.12, corridor_margin_m=0.05)
    diagnostics = teacher.route_diagnostics(pose(x=2., y=0.1))
    assert diagnostics["route_s_m"] == pytest.approx(2.)
    assert diagnostics["lateral_error_m"] == pytest.approx(0.1)
    assert diagnostics["lateral_bound_m"] == pytest.approx(0.33)


def test_route_diagnostics_keep_unwrapped_progress_across_lap_seam():
    angle = np.linspace(0., 2*np.pi, 241)
    points = np.column_stack([3*np.cos(angle), 3*np.sin(angle)])
    route = RouteProgressTracker(points)
    teacher = CenterlineExpert(route)
    diagnostics = [
        teacher.route_diagnostics({"position": (3*np.cos(t), 3*np.sin(t))})
        for t in (6.1, 6.2, 0.05)
    ]
    progress = [row["route_s_m"] for row in diagnostics]
    assert all(value is not None for value in progress)
    assert progress[1] > progress[0]
    assert progress[2] > route.length_m
    assert all(row["lateral_bound_m"] > 0 for row in diagnostics)


def test_teacher_lap_gate_requires_counter_crossings_and_full_route_coverage():
    kwargs = {
        "race_laps_completed": 3,
        "race_lap_times_s": [31.0, 30.5, 32.1],
        "current_route_frontier_m": 107.7,
        "initial_route_frontier_m": 15.3,
        "route_length_m": 31.1169,
        "lap_goal": 3,
    }
    assert _verified_lap_goal_reached(**kwargs)
    assert not _verified_lap_goal_reached(**{**kwargs, "race_laps_completed": 2})
    assert not _verified_lap_goal_reached(**{**kwargs, "race_lap_times_s": [31.0, 30.5]})
    assert not _verified_lap_goal_reached(**{**kwargs, "race_lap_times_s": [31.0, float("nan"), 32.1]})
    assert not _verified_lap_goal_reached(**{**kwargs, "current_route_frontier_m": 103.0})
    assert not _verified_lap_goal_reached(**{**kwargs, "current_route_frontier_m": None})
