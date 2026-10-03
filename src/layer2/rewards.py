"""
Layer 2 reward scoring — THIS is the file you will edit most often.

Mental model
------------
AutoDriveEnv.step() reads the car (Layer 1 telemetry), then calls compute_reward(...)
with plain facts (how fast forward, did we just hit something, …).

This module does NOT talk to Unity. It only turns those facts into ONE number:
the reward. PPO (later, Layer 3) tries to make that number as large as possible
over time.

@dataclass
-----------
In Python, @dataclass is a decorator that auto-builds a simple class for storing
fields. RewardConfig is just a bag of named numbers (weights). Without @dataclass
you would write a long __init__ by hand; with it, RewardConfig(forward_scale=2.0)
just works.

Why float?
----------
Speeds, angles, and scores are continuous real numbers (3.7 m/s, -0.2 steer),
not integers. float is the normal type for that in Python / NumPy / Gymnasium.

What is v_long?
---------------
"Longitudinal velocity" in the car's body frame: how fast the car is moving
FORWARD along its nose (meters/second). Positive ≈ going forward, negative ≈
going backward. Layer 1 computes this on TelemetrySnapshot.v_long from Unity's
velocity + heading.

What is collision_event?
------------------------
A boolean the ENV computes each step: True if this step looks like a NEW hit
(Unity collision flag True, OR collision_count went up vs last step).
It is NOT a raw Unity field named collision_event. Rewards only see True/False.

What is slip_angle?
-------------------
Sideslip β (radians) on TelemetrySnapshot.slip_angle — angle between the car's
heading and its velocity direction. Big |slip| ≈ sliding sideways. Layer 1
derives it from body-frame velocities.

What is steer jerk (steer_jerk_penalty)?
----------------------------------------
Not a sensor. It is |steering_now - steering_previous| * weight.
We penalize sudden steering changes if you set steer_jerk_penalty > 0.
Default weight is 0.0 (off).

Reward priorities
-----------------
On maps with route data, forward progress is the small shaping signal. Raw
forward velocity is only used on maps without route data and defaults to zero.
The main pace objective is one bonus at a clean target-lap finish, calculated
from average frontier speed across the entire attempt; single-lap speed never
adds reward by itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Tunable weights for compute_reward.

    Change these numbers to reshape what "good driving" means.
    Defaults favor validated route progress; pace is rewarded only at a clean
    configured lap target.
    """

    # Raw forward velocity is not a training objective on mapped tracks.
    forward_scale: float = 0.0

    # Small secondary cost for reversing relative to the car body. Signed route
    # progress supplies the primary penalty for traveling backward on the map.
    backward_speed_penalty_scale: float = 1.0
    backward_speed_deadband_mps: float = 0.1

    # One-time clean episode completion reward: scale × average frontier speed
    # across all target laps. Per-lap pace is telemetry only.
    lap_time_reward_scale: float = 1000.0

    # Signed shaping reward per metre of current route movement.
    route_progress_scale: float = 10.0

    # Collision penalty; termination is controlled independently by the env.
    collision_penalty: float = -100.0

    # Subtracted as: slip_penalty * abs(slip_angle). Default 0.0 = ignore slip.
    slip_penalty: float = 0.0

    # Subtracted as: steer_jerk_penalty * abs(steering - prev_steering).
    # Default 0.0 = allow twitchy steering without score cost.
    steer_jerk_penalty: float = 0.0


def compute_reward_components(
    *,
    v_long: float,
    step_duration_s: float = 0.0,
    route_progress_delta_m: float | None = None,
    clean_run_average_frontier_speed_mps: float | None = None,
    collision_event: bool,
    slip_angle: float,
    prev_steering: float,
    steering: float,
    cfg: RewardConfig,
) -> dict[str, float]:
    """
    Compute the scalar reward for one AutoDriveEnv step.

    Parameters (facts from the env — not read from Unity here)
    ----------
    v_long:
        Forward speed along the car nose (m/s), from TelemetrySnapshot.v_long.
    step_duration_s:
        Simulated time advanced by this environment step, in seconds.
    collision_event:
        True if this step counted as a new collision (env-detected).
    slip_angle:
        Sideslip β in radians, from TelemetrySnapshot.slip_angle.
    prev_steering:
        Steering command used on the previous env step, in [-1, 1].
    steering:
        Steering command on THIS env step, in [-1, 1].
    cfg:
        RewardConfig weights.

    Returns
    -------
    dict[str, float]
        Named reward components plus their scalar total.

    How to add a new term later
    ---------------------------
    1. Add a weight field on RewardConfig (default 0.0 if unused).
    2. Add a parameter here if you need a new fact.
    3. Have AutoDriveEnv.step pass that fact from telemetry / your own math.
    4. Add one component: components["my_term"] = cfg.my_weight * something.
    """
    components = {
        "route_progress": 0.0,
        "backward_motion": 0.0,
        "collision": 0.0,
        "lap_bonus": 0.0,
        "slip": 0.0,
        "steering_change": 0.0,
        "raw_forward_velocity": 0.0,
    }

    # Penalize reverse travel by distance, not by action sign: negative throttle
    # is braking and remains unpenalized when the car is still moving forward.
    reverse_speed = max(
        0.0, -float(v_long) - max(0.0, float(cfg.backward_speed_deadband_mps))
    )
    reverse_distance_m = reverse_speed * max(0.0, float(step_duration_s))
    components["backward_motion"] = (
        -float(cfg.backward_speed_penalty_scale) * reverse_distance_m
    )

    # Prefer actual along-route frontier movement when the environment supplies
    # it; raw velocity remains useful for the builtin track without a centerline.
    if route_progress_delta_m is None:
        components["raw_forward_velocity"] = cfg.forward_scale * float(v_long)
    else:
        components["route_progress"] = (
            cfg.route_progress_scale * float(route_progress_delta_m)
        )

    if clean_run_average_frontier_speed_mps is not None:
        components["lap_bonus"] = cfg.lap_time_reward_scale * max(
            0.0, float(clean_run_average_frontier_speed_mps)
        )

    # Optional wall tax (default weight 0 → this adds nothing).
    if collision_event:
        components["collision"] = float(cfg.collision_penalty)

    # Optional: discourage sideways sliding (default weight 0).
    components["slip"] = -float(cfg.slip_penalty) * abs(float(slip_angle))

    # Optional: discourage sudden steering changes (default weight 0).
    steering_delta = abs(float(steering) - float(prev_steering))
    components["steering_change"] = -float(cfg.steer_jerk_penalty) * steering_delta

    # One number back to Gym / eventually PPO.
    components["total"] = float(sum(components.values()))
    return components


def compute_reward(
    *,
    v_long: float,
    step_duration_s: float = 0.0,
    route_progress_delta_m: float | None = None,
    clean_run_average_frontier_speed_mps: float | None = None,
    collision_event: bool,
    slip_angle: float,
    prev_steering: float,
    steering: float,
    cfg: RewardConfig,
) -> float:
    """Return the scalar total of the individually inspectable reward terms."""
    return compute_reward_components(
        v_long=v_long,
        step_duration_s=step_duration_s,
        route_progress_delta_m=route_progress_delta_m,
        clean_run_average_frontier_speed_mps=clean_run_average_frontier_speed_mps,
        collision_event=collision_event,
        slip_angle=slip_angle,
        prev_steering=prev_steering,
        steering=steering,
        cfg=cfg,
    )["total"]
