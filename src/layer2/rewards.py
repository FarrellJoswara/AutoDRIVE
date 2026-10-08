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
New best-so-far frontier distance is the only positive driving reward, with a
quadratic episode-average pace multiplier capped at 1 + bonus strength. Time costs penalize
elapsed time; reverse travel, collisions, and non-stall failed episode endings
cost reward. Frontier stalls reset after their timeout without an extra
terminal cost. Laps are diagnostic and never change reward or episode life.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class RewardConfig:
    """
    Tunable weights for compute_reward.

    Change these numbers to reshape what "good driving" means.
    Defaults reward validated frontier progress more strongly as the car's
    episode-average frontier pace approaches the configured target.
    """

    # Deprecated compatibility field. Raw velocity never earns reward.
    forward_scale: float = 0.0

    # Small secondary cost for reversing relative to the car body. The frontier
    # cannot retreat, so reverse movement earns no progress reward.
    backward_speed_penalty_scale: float = 1.0
    backward_speed_deadband_mps: float = 0.1

    # Reward per metre of newly advanced high-water frontier.
    route_progress_scale: float = 10.0
    # Average frontier pace at which the progress reward reaches its 2x cap.
    frontier_pace_target_mps: float = 6.0
    # Strength of the quadratic pace bonus. 1 preserves a 2x cap; 2 yields 3x.
    frontier_pace_bonus_strength: float = 1.0
    frontier_pace_source: Literal["episode_average", "current_push"] = "episode_average"

    # Cost per simulated second, including while making progress. This makes
    # slower completion less profitable than faster completion over the same route.
    time_penalty_per_second: float = 5.0

    # Positive settings magnitudes; signs are applied here consistently.
    collision_penalty_magnitude: float = 100.0
    collision_reward_percent: float = 100.0

    # Applied once when a non-collision failure ends/truncates an episode.
    episode_failure_penalty_magnitude: float = 100.0
    episode_failure_reward_percent: float = 100.0

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
    frontier_advanced_m: float | None = None,
    frontier_average_speed_mps: float = 0.0,
    frontier_current_speed_mps: float = 0.0,
    positive_episode_return: float = 0.0,
    collision_event: bool,
    episode_failure: bool = False,
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
    frontier_advanced_m:
        Newly pushed high-water frontier distance; the only per-step positive
        driving signal. ``route_progress_delta_m`` is retained for callers that
        still pass it, but is not used for reward.
    frontier_average_speed_mps:
        Episode-to-date average new-frontier distance divided by simulated time.
        It scales the per-metre progress reward, capped at 2x at the configured
        target pace.
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
        "reverse_direction_gate": 0.0,
        "backward_motion": 0.0,
        "time_cost": 0.0,
        "collision": 0.0,
        "episode_failure": 0.0,
        "slip": 0.0,
        "steering_change": 0.0,
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

    # Only a new high-water frontier push earns step-wise progress reward.
    # Signed current-position movement is deliberately ignored: recovering or
    # retracing route distance must not pay a second time. Raw forward speed is
    # also never a reward, including on maps without route geometry.
    if frontier_advanced_m is not None:
        advanced_m = max(0.0, float(frontier_advanced_m))
        target_pace = max(1e-6, float(cfg.frontier_pace_target_mps))
        pace_speed = (
            frontier_current_speed_mps
            if cfg.frontier_pace_source == "current_push"
            else frontier_average_speed_mps
        )
        normalized_pace = min(
            1.0,
            max(0.0, float(pace_speed)) / target_pace,
        )
        pace_multiplier = 1.0 + max(
            0.0, float(cfg.frontier_pace_bonus_strength)
        ) * normalized_pace * normalized_pace
        route_reward = cfg.route_progress_scale * pace_multiplier * advanced_m
        reverse_deadband = max(0.0, float(cfg.backward_speed_deadband_mps))
        if route_reward > 0.0 and float(v_long) < -reverse_deadband:
            # Do not pay a new frontier push when the body-frame sensor says
            # the car is travelling backward (e.g. tail-first).
            components["reverse_direction_gate"] = -route_reward
        else:
            components["route_progress"] = route_reward

    components["time_cost"] = (
        -max(0.0, float(cfg.time_penalty_per_second))
        * max(0.0, float(step_duration_s))
    )

    # A charged terminal failure claws back a configurable share of this life’s
    # positive frontier reward, then applies its fixed failure cost. Frontier
    # stalls are excluded by AutoDriveEnv; their time cost still applies.
    positive_frontier_return = (
        max(0.0, float(positive_episode_return))
        + max(0.0, float(components["route_progress"]))
    )
    if collision_event:
        magnitude = max(0.0, float(cfg.collision_penalty_magnitude))
        percent = max(0.0, float(cfg.collision_reward_percent))
        components["collision"] = -(magnitude + positive_frontier_return * percent / 100.0)

    # Collision already receives its event penalty above. Other charged failed
    # endings (idle timeout or step cap) receive one terminal cost.
    if episode_failure and not collision_event:
        magnitude = max(0.0, float(cfg.episode_failure_penalty_magnitude))
        percent = max(0.0, float(cfg.episode_failure_reward_percent))
        components["episode_failure"] = -(
            magnitude + positive_frontier_return * percent / 100.0
        )

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
    frontier_advanced_m: float | None = None,
    frontier_average_speed_mps: float = 0.0,
    frontier_current_speed_mps: float = 0.0,
    positive_episode_return: float = 0.0,
    collision_event: bool,
    episode_failure: bool = False,
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
        frontier_advanced_m=frontier_advanced_m,
        frontier_average_speed_mps=frontier_average_speed_mps,
        frontier_current_speed_mps=frontier_current_speed_mps,
        positive_episode_return=positive_episode_return,
        collision_event=collision_event,
        episode_failure=episode_failure,
        slip_angle=slip_angle,
        prev_steering=prev_steering,
        steering=steering,
        cfg=cfg,
    )["total"]
