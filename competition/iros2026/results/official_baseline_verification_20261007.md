# Official RoboRacer Baseline Verification — 2026-10-07

## Rules and scoring target

The official IROS 2026 format is 12 laps: one unscored warm-up lap, 10 scored
race laps, and an optional unscored cool-down lap. A collision resets the car
to its last checkpoint; the lap timer continues. Race collisions add 10, 20,
30, ... seconds, and more than 10 race collisions disqualifies the run. The
race is ranked by total time for the 10 scored laps, with best lap as a
tie-breaker. These details come from the [official IROS 2026 competition
rules](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-rules-2026/).

The current published IROS 2026 Phase 2 leader is Siga Siga Racing Team at
71.82 seconds for 10 laps (7.182 s mean lap, 7.16 s best lap, zero collisions).
The Phase 1 qualification leader is NTU DeepSpeed at 49.83 seconds (4.94 s
best lap); that is a separate phase and should not be compared as the Phase 2
target. The user's 5.5-second average-lap goal is an aspirational 55-second
10-lap time, faster than the published Phase 2 result. Standings are from the
[official IROS 2026 event page](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2026/).

## Permitted runtime data and simulator setup

The [official technical guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/)
allows LiDAR, front camera, IMU, left/right encoders, steering feedback, and
throttle feedback as runtime inputs. IPS, odometry, TF, reset, collision count,
and lap metrics are restricted to debugging/training/evaluation and must not
be used during autonomous inference. The policy transport subscribes only to
the allowed input topics and publishes only throttle and steering commands;
the evaluator has a separate restricted-metrics transport. This is also
consistent with the guide's warning that changes to the simulator, vehicle,
track, communication interface, Devkit elements, or containerization are
prohibited; the team's algorithm must be added as its own package.

Official image tags used by this workspace:

- Simulator: `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`,
  local image ID `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
- Official Devkit base: `autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`,
  local image ID `sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2`.
- Local algorithm/evaluator image is derived from that Devkit base and adds
  this repository's policy and evaluation code; it is not the unmodified
  official Devkit image. Local evaluations record the base and derived image
  identities separately where available.

## Our verified result

There is **no verified valid official 10-lap result** in the saved official
evaluation artifacts. The historical `transfer_baseline_20261007.json` and
`cadence40_20261007.json` each completed 10 laps in about 130.6–130.8 seconds,
but incurred 30–31 collisions and were disqualified. The best official PPO
partial attempts recorded in this workspace completed at most three race laps
before exceeding the collision tolerance or reaching a guard. Thus the custom
simulator's 66.98-second figure is not an official result and is excluded from
this goal's progress comparisons.

No training run was active at verification time (`/train/runs` contained only
stopped/failed runs; Docker had only the `aicar_brain` container running). No
checkpoint was changed by this verification.

## Evaluation protocol for this goal

Every candidate evaluation must retain the official simulator image, official
12-lap sequence, allowed runtime observations, official checkpoint-reset and
collision accounting, and a fixed candidate checkpoint. Each run records the
checkpoint hash/path, exact PPO and action settings, simulator and Devkit image
IDs, per-lap times, collision counts, completion/disqualification status, and
diagnostics. Compare candidates only under matched conditions. Promotion
requires repeated valid 10-lap attempts with a better median total race time
and no decline in completion consistency. Diagnostic runs that use a fixed
controller, altered evaluation mapping, or a local time guard are explicitly
not promotion evidence.

The corrected `lidar_gap` route diagnostic is only a controller/debugging
probe. It ran on the official simulator image with permitted LiDAR input, but
used the team's derived policy/evaluator image and local wall guard; it
completed no lap and is not an official result. Its IPS trace is restricted
evaluator-only data. See
[the corrected route report](goal_experiment_20261007_xy_route_diagnostic.md).
