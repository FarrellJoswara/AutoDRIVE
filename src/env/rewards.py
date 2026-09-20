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

What is "scale" (forward_scale = 1.0)?
--------------------------------------
It multiplies v_long: reward += forward_scale * v_long.
If scale is 1 and v_long is 3.0, reward gets +3.0 from that term.
If scale is 100, the same speed gives +300 — much stronger pressure to go fast
relative to other terms. Start at 1.0 so numbers stay interpretable; raise it
only when you know you want speed to dominate other penalties.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """
    Tunable weights for compute_reward.

    Change these numbers to reshape what "good driving" means.
    Defaults match PLAN.md Layer 2 v1 (crash tax off; forward speed is primary).
    """

    # Multiplier on forward speed (v_long). 1.0 ≈ "reward equals m/s forward".
    # Raise (e.g. 10) if forward progress should dominate other terms.
    forward_scale: float = 1.0

    # Added once per step when collision_event is True.
    # Use a negative number for a wall tax (e.g. -5.0). Default 0.0 = no tax;
    # getting stuck after a crash is handled by stagnation truncation in the env.
    collision_penalty: float = 0.0

    # Subtracted as: slip_penalty * abs(slip_angle). Default 0.0 = ignore slip.
    slip_penalty: float = 0.0

    # Subtracted as: steer_jerk_penalty * abs(steering - prev_steering).
    # Default 0.0 = allow twitchy steering without score cost.
    steer_jerk_penalty: float = 0.0


def compute_reward(
    *,
    v_long: float,
    collision_event: bool,
    slip_angle: float,
    prev_steering: float,
    steering: float,
    cfg: RewardConfig,
) -> float:
    """
    Compute the scalar reward for one AutoDriveEnv step.

    Parameters (facts from the env — not read from Unity here)
    ----------
    v_long:
        Forward speed along the car nose (m/s), from TelemetrySnapshot.v_long.
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
    float
        Total step reward. PPO will try to maximize the sum of these over time.

    How to add a new term later
    ---------------------------
    1. Add a weight field on RewardConfig (default 0.0 if unused).
    2. Add a parameter here if you need a new fact.
    3. Have AutoDriveEnv.step pass that fact from telemetry / your own math.
    4. Add one line: r += cfg.my_weight * something (or -=).
    """
    # Start at zero; accumulate every term into r.
    r = 0.0

    # Primary v1 signal: going forward is good (scale * m/s).
    r += cfg.forward_scale * float(v_long)

    # Optional wall tax (default weight 0 → this adds nothing).
    if collision_event:
        r += float(cfg.collision_penalty)

    # Optional: discourage sideways sliding (default weight 0).
    r -= float(cfg.slip_penalty) * abs(float(slip_angle))

    # Optional: discourage sudden steering changes (default weight 0).
    steering_delta = abs(float(steering) - float(prev_steering))
    r -= float(cfg.steer_jerk_penalty) * steering_delta

    # One number back to Gym / eventually PPO.
    return float(r)
