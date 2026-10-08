# Official Route Diagnostic — 2026-10-07

> **Superseded coordinate interpretation:** The initial report below projected
> the ROS `/ips` position into X/Z, based on the Unity telemetry frame. That was
> the wrong frame for this field. Official ROS IPS uses X/Y for the track plane
> and Z for height. Use [the corrected X/Y report](goal_experiment_20261007_xy_route_diagnostic.md)
> for all route conclusions; the original JSON and commit are retained as history.

## Purpose

Diagnose why the deterministic `lidar_gap` baseline fails to finish its official
warm-up lap. This is a diagnostic run, not a candidate race score and not a
checkpoint promotion.

## Starting point and exact setup

- Starting checkpoint: none; `lidar_gap` is a fixed sensor-based diagnostic
  controller, not a learned model.
- API image: `aicar-iros2026:latest`, image ID
  `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`
  (built from the official IROS 2026 API image).
- Simulator image: `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`,
  image ID
  `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
- Controller: deterministic `lidar_gap`; one attempt; `official_sensors`.
- Action mapping: bidirectional throttle, normal steering, steering scale 1.0,
  straight-throttle gain 1.0, straight-steering threshold 0.15.
- Race protocol: one warm-up lap, then 10 race laps; 300-second local wall guard,
  150,000-step guard, 180-second sensor timeout.
- Trace: evaluator-only restricted IPS positions, reduced to at most 600 x/z
  points; policy input is unchanged. The existing first-250-step action trace is
  also enabled.
- Result: [official_sensor_lidar_gap_route_diag_20261007.json](official_sensor_lidar_gap_route_diag_20261007.json).

## Result

The simulator delivered 5,889 policy steps over 300.0 wall seconds at 19.63 Hz.
The attempt completed zero warm-up or race laps, so its race score is invalid.
No race collisions were scored because the warm-up never completed; the raw
simulator collision counter nevertheless reached 322.

The car traversed 489.94 m of accumulated path while ending only 2.23 m from its
initial position. Its maximum distance from the initial position was 3.03 m and
path efficiency was 0.0046. The sampled x/z route occupied only about 3.03 m in
x and 0.007 m in z. This is strong evidence that it repeatedly oscillated or
bounced in a small region instead of progressing around the course. It rules
out “the simulator was stationary” and identifies route-following/collision
behavior as the immediate failure; it does not establish a competitive lap
time.

## Change and verification

Added bounded, evaluator-only path summaries to the official evaluation report:
path length, net displacement, path efficiency, maximum distance from start,
and a downsampled x/z trace. The trace is not provided to the policy and does
not change official sensor inputs, controls, simulator behavior, or scoring.
All five new tests passed inside `aicar-iros2026:latest`, using the same Python
runtime as official evaluation.

## Next experiment

Do not tune PPO rewards from this heuristic-controller failure. Inspect the
first 250 action/observation samples and the route trace against the official
track frame, then run a controlled, single-factor diagnostic (first validate
steering polarity and pose/heading against official telemetry, then evaluate a
candidate learned checkpoint). Keep all results marked invalid until a full,
collision-valid 10-lap evaluation completes.
