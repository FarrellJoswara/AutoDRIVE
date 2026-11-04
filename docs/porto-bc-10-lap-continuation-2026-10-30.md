# Porto BC checkpoint continuation: 10-lap practice run — 2026-10-30

## Outcome

The saved behavior-cloning checkpoint completed one fresh, native Windows Porto practice simulator run for exactly 10 scored laps in one process. The evaluator stopped at `race_complete` after 10/10 laps. Its practice diagnostic gate passed. This remains a noncompetitive practice diagnostic and does not establish a competition score.

The checkpoint was not modified, and no PPO training or update ran.

## Run evidence

- Full evaluator result: [`student_evaluation.json`](../logs/rl/porto_bc_20261030_eval10_continue/student_evaluation.json)
- Evaluator output: [`evaluator_stdout.log`](../logs/rl/porto_bc_20261030_eval10_continue/evaluator_stdout.log)
- Launch configuration: [`launch_config.json`](../logs/rl/porto_bc_20261030_eval10_continue/launch_config.json)
- Native runtime evidence: [`runtime_preflight.json`](../logs/rl/porto_bc_20261030_eval10_continue/runtime_preflight.json)
- Neutral barrier evidence: [`neutral_preflight.json`](../logs/rl/porto_bc_20261030_eval10_continue/neutral_preflight.json)
- Camera, LiDAR, and pose preflight: [`sensor_preflight.json`](../logs/rl/porto_bc_20261030_eval10_continue/sensor_preflight.json)
- Simulator log and process record: [`simulator.log`](../logs/rl/porto_bc_20261030_eval10_continue/simulator.log), [`simulator_process.json`](../logs/rl/porto_bc_20261030_eval10_continue/simulator_process.json)

Checkpoint: `logs/rl/porto_bc_20261030_fit/behavior_cloned_policy.zip`, SHA-256 `bb815d3612cecbb8f8bdf9936ec98c8da4c689b0551a33f5b52b31051cbae2c4`.

| Scored lap | Official lap time (s) |
| ---: | ---: |
| 1 | 187.022 |
| 2 | 94.513 |
| 3 | 95.698 |
| 4 | 93.838 |
| 5 | 94.363 |
| 6 | 95.124 |
| 7 | 94.387 |
| 8 | 95.761 |
| 9 | 95.126 |
| 10 | 94.214 |

The race time was 1040.046 seconds, over 16,631 policy steps and 886.001 seconds wall time. Best lap was 93.838 seconds. The latest official lap-time topic matched the tenth stored lap time.

## Diagnostic gates and runtime

- 10/10 official laps completed with fresh positive timing; zero missing lap times.
- Zero race or raw collisions; no disqualification.
- Zero monotonic control-gap faults; zero invalid route-projection samples.
- All 16,631 camera samples were fresh; zero stale or missing samples.
- ROS header-stamp gaps: 27, reported separately from monotonic control gaps. Median monotonic control interval was 0.0522 seconds.
- Practice diagnostic passed; `valid_for_comparison` is false by design.
- Fresh official Windows practice GUI process PID 24412, verified executable SHA-256 `f5efe8c38b78a72a37d44f07ea35e184cf1c41e4dd4e33eb3c9dc4415d470e69`.
- Verified practice scene SHA-256 `5edbd1bacc15647746e6e89d9bcf1bb9e202d62d47e76ae352497be1404b3f8d`; verified Porto route SHA-256 `87d26be59ccc88ff6fc273905038906b8ed9e883b111c33de008a38f83a15bb3`.
- Native WSL ROS 2 domain/base 86; GUI camera rendering; safe neutral fallback 0.20 seconds; no Docker.
- Before evaluation, continuous zero throttle/steering was held before GUI launch and through preflight. The car remained at spawn, both counters were zero, feedback was zero, 150 camera/LiDAR frames advanced, and relay response timeouts remained zero.

The earlier three fresh 3-lap diagnostics remain documented in [`porto-bc-results-2026-10-30.md`](porto-bc-results-2026-10-30.md); this report preserves the 10-lap continuation as a separate attempt.

## Completion and cleanup

The evaluator result was saved and checked before shutdown. GUI PID 24412 exited cleanly. The student evaluator and neutral publisher exited. The task relay and ROS Devkit were stopped; ports 4567 and 4586 have no listeners. Cleanup evidence is in [`cleanup.json`](../logs/rl/porto_bc_20261030_eval10_continue/cleanup.json). The focused Porto diagnostic suite passed 9 tests after adding the 10-lap option; no other implementation tests were needed.
