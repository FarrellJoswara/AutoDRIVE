# Official LiDAR gap-center experiment — 2026-10-07

## Question and validity

Does steering toward the center ray of the widest safe LiDAR opening work
better than steering toward the farthest ray in that opening? This is a
deterministic controller diagnostic, not a PPO candidate. Both tests used the
official IROS simulator and permitted sensor streams. Restricted IPS was read
only by the local evaluator for post-run diagnostics. Neither result is a valid
race time; no checkpoint or champion was changed.

## Fixed setup

- Starting checkpoint: none (the fixed `lidar_gap` diagnostic controller).
- Route: Porto, one warm-up lap followed by up to 10 scored laps.
- Controls and policy parameters: bidirectional throttle; negative-throttle
  mode `allow`; steering mode `normal`; steering scale 1.0; straight throttle
  gain 1.0; threshold 0.15. Gap controller defaults: max range 10 m, minimum
  gap 0.7 m, safety radius 0.32 m, max steering angle 0.75 rad, cruise speed
  4.0 m/s, minimum speed 1.5 m/s, speed gain 0.24.
- Evaluator: deterministic, one attempt, 300 s wall guard, 150,000-step guard,
  180 s sensor timeout, 250-step action trace.
- Official simulator image:
  `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`,
  `sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`.
- Both policy/evaluator images extend the same official Devkit base
  `autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`,
  `sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2`.
- Baseline runner image `aicar-iros2026:latest`:
  `sha256:0a24abcba892faf8b51b3ebc680b26f2f80fb121719f3325992dae348da598a8`.
- Candidate runner image `aicar-iros2026:gap-center`:
  `sha256:41994d243d4ee0766aac573840b8f79f55e6d122d024ee57da1e85d04cf56359`.

## One-factor change

The candidate changes only target-ray selection: both controllers choose the
widest contiguous safe opening, but the baseline picks its farthest ray while
the candidate picks the opening's center ray. Gap thresholds, bubble, throttle,
steering mapping, runtime, simulator and evaluation protocol are fixed.

## Results

| Controller | Warm-up | Scored laps | Collisions | End condition | Valid race time |
| --- | --- | ---: | ---: | --- | --- |
| Farthest ray (baseline) | 0 laps in 300.04 s; 0 warm-up collisions | 0 | 0 scored; 319 raw total before warm-up | Wall timeout at 5,909 steps | No |
| Gap center (candidate) | 1 lap in 51.56 s; 24 warm-up collisions | 0 | 11 race collisions (35 raw total) | Disqualified at 1,341 steps / 67.78 s | No |

The baseline's evaluator verified movement with up to 19.18 m displacement
from its start, but did not complete a lap within its wall guard. The candidate
did complete warm-up, but immediately accumulated the official disqualification
threshold after race scoring began. Its post-run X/Y path length was 160.12 m,
net displacement 18.95 m, and path efficiency 0.118; these restricted
diagnostics were not supplied to the controller. Control cadence was about
19.7–19.8 Hz and reported LiDAR scan cadence about 40 Hz in both runs.

This is suggestive evidence that centering the safe gap improves route
completion versus the farthest-ray heuristic, but a single attempt per variant
is not enough to establish a reliable improvement. More importantly, neither
is remotely race-valid: the baseline fails to complete warm-up and the
candidate is disqualified. Do not promote either controller, use its score as
a leaderboard comparison, or infer that PPO will behave the same way. The
diagnostic does not justify changing PPO rewards.

## Artifacts and cleanup

- Candidate output:
  [`official_sensor_lidar_gap_center_route_diag_20261007.json`](official_sensor_lidar_gap_center_route_diag_20261007.json)
- Matched baseline output:
  [`official_sensor_lidar_gap_farthest_control_20261007.json`](official_sensor_lidar_gap_farthest_control_20261007.json)
- Candidate unit tests passed inside the official-derived policy image before
  the run. The baseline was confirmed to connect to the same official
  simulator and receive the full official sensor frame before evaluation.
- The two disposable containers and their isolated Docker networks were
  removed after both attempts. The main UI/hub and all checkpoints were left
  untouched.

## Next step

Return to preserved learned checkpoints and run a controlled official-runtime
PPO evaluation/training experiment. A fixed LiDAR heuristic is only useful for
debugging sensor/action conventions. Promotion requires repeated complete
official 10-lap attempts, better median adjusted time, and no loss of completion
consistency.
