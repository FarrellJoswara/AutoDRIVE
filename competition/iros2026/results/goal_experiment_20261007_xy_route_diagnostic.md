# Official X/Y Route Diagnostic — 2026-10-07

## Purpose and validity

This is a diagnostic of the deterministic LiDAR-gap controller, not a learned
policy score and not a candidate for promotion. It used the official IROS 2026
API and simulator images. The run completed no warm-up lap, so it has no valid
race time and no scored race collisions. No checkpoint or champion was changed.

The initial route report incorrectly projected ROS `/ips` into X/Z. The
evaluator receives ROS IPS points directly; in the official AutoDRIVE frame X/Y
are horizontal and Z is vertical. The original report is retained but marked
superseded. This report and its metrics use X/Y as the ground plane.

## Exact experiment

- Starting checkpoint: none; fixed deterministic `lidar_gap` controller.
- Derived policy/evaluator image: `aicar-iros2026:latest`, image ID
  `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`.
- Official Devkit base image:
  `autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`, image ID
  `sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2`.
- Simulator image: `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`,
  image ID `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
- Inputs: official permitted sensor observations; no route trace/IPS data was
  passed into the controller. IPS is sampled only by the evaluator for this
  post-run diagnostic.
- Controls: bidirectional throttle, normal steering, scales/gains 1.0,
  straight-steering threshold 0.15.
- Protocol: one warm-up lap, then up to 10 scored laps; one attempt; 300-second
  wall guard, 150,000-step guard, 180-second sensor timeout. Control interval
  median 50.47 ms; observed LiDAR scan rate about 40 Hz.
- Full machine-readable report:
  [official_sensor_lidar_gap_xy_route_diag_20261007.json](official_sensor_lidar_gap_xy_route_diag_20261007.json).

## Result

The run reached 5,929 control steps in 300.04 seconds and stopped at the wall
guard. It completed zero laps. The raw simulator collision counter reached
320; these are not official scored race collisions because the warm-up lap
never finished.

The corrected horizontal trace accumulated 1,093.15 m of path but ended only
15.59 m from its initial position (path efficiency 0.0143). Maximum horizontal
displacement was 19.37 m. X spanned roughly 3.02 m, Y roughly 18.82 m, and
vertical Z only 0.0074 m (about 5.4–6.1 cm). This rules out the earlier
interpretation that the car was moving vertically. It instead shows repeated
motion within a narrow part of the track, with strong backtracking/oscillation
and repeated contacts, rather than a completed route.

The measured control loop averaged 19.76 Hz (50.47 ms median interval) while
the scan metadata reported about 40 Hz. This is an observed property of this
local controller run, not a competition-mandated action period. Because this
was a fixed-controller diagnostic with a local wall guard, its route behavior
and rate are debugging evidence only; it is not a comparable race score or
promotion result. The official process uses the official Devkit base with the
team's algorithm container; see the [2026 technical guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/).

This result diagnoses the fixed LiDAR-gap baseline as ineffective under this
setup. It does **not** establish that PPO has the same behavior, identify a
reward defect, or justify changing PPO rewards. The position trace is
restricted evaluator data and does not alter the official policy observation,
control, or scoring path.

## Cause/change/verification record

- Cause: a Unity-swizzled telemetry frame assumption was mistakenly applied to
  the evaluator's direct ROS `/ips` point.
- Change: evaluator-only route summaries now project ROS IPS into X/Y, retain a
  bounded 3D sample, and separately report vertical range.
- Verification: all five position-diagnostic unit tests passed inside the
  official API image. The corrected full official-image run produced the
  metrics above. Existing reports, checkpoints, and the prior commit remain
  preserved.

## Next step

Do not tune PPO from this controller result. Compare an existing learned
checkpoint in the same official images with the exact permitted inputs, first
using a short diagnostic evaluation that records completion, collisions,
control rate, action/observation traces, and route progress. Any speed or reward
experiment must use a fixed starting checkpoint and change one factor at a
time. Promotion still requires repeated valid 10-lap evaluations with a better
median adjusted time and no loss of completion consistency.
