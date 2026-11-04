# Porto practice behavior cloning results — 2026-10-30

## Result

The accepted behavior-cloned student completed three separate, fresh native Windows Porto practice simulator processes. Every attempt completed three full timed laps, with zero race and raw collisions, valid per-lap timing, fresh camera input on every step, valid route projection, verified motion, and no monotonic control-gap faults. These runs are **practice diagnostics only**; they do not establish a competition score or change the competition-best model.

The model was trained only from the accepted CenterlineExpert demonstrations. No PPO model was created or updated.

## Teacher gate and collection

The teacher gate passed in [teacher_start2_retry1](../logs/rl/porto_bc_aef213965b2b/teacher_start2_retry1/teacher_summary.json): three completed timed laps, zero collisions, zero expiry, two safe neutral fallbacks, and fresh lap counter baseline 0.

The accepted collection is in [porto_bc_20261030_collect](../logs/rl/porto_bc_20261030_collect/behavior_cloning_collection_acceptance.json). It contains 10 clean laps across two fresh five-lap episodes, 20,494 transitions, zero collisions, zero telemetry or control faults, valid camera rendering and dataset integrity. The whole second episode was reserved for validation.

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Demonstrations | `logs/rl/porto_bc_20261030_collect/behavior_cloning_demonstrations.npz` | `4df65ff5df4b7717e87ecf3eb424bfd67a90a713c823caa2aa57998a65a70ce9` |
| Episode boundaries | `logs/rl/porto_bc_20261030_collect/collection_boundaries.json` | `300dfd734602d764436aafb1c3e5115ecd36ee136404b746a689cbf46751feae` |
| Fitted checkpoint | `logs/rl/porto_bc_20261030_fit/behavior_cloned_policy.zip` | `bb815d3612cecbb8f8bdf9936ec98c8da4c689b0551a33f5b52b31051cbae2c4` |

Offline fitting used 8 epochs, batch size 256, learning rate 0.001, 10,203 training transitions from episode 1, and 10,291 validation transitions from episode 2. Validation loss decreased from `0.0142263354` to `0.0000844296`; mean training loss was `0.0006308416`. The checkpoint records sensor-only camera/LiDAR/state inputs and no privileged teacher state.

## Fresh student practice diagnostics

Each run used the same checkpoint, fixed practice route, official Windows practice executable and verified scene, GUI rendering, native WSL ROS 2 domain 86, and `AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S=0.20`. A persistent zero-throttle/zero-steering publisher was started before the simulator and held through connection and sensor checks. Before each policy launch, the car was at the official spawn near `(0.800, 3.158)`, lap and collision counters were both zero, actuator feedback was zero, and position changed at most 0.0001 m over the neutral settling check. The publisher was stopped before student control began.

| Run | Windows simulator PID | Timed laps (s) | Steps | Collisions | ROS stamp anomalies | Gate |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| [eval2_neutral1](../logs/rl/porto_bc_20261030_eval2_neutral1/student_evaluation.json) | 64548 | 176.933, 94.924, 94.665 | 4,907 | 0 | 8 | Passed |
| [eval3_neutral1](../logs/rl/porto_bc_20261030_eval3_neutral1/student_evaluation.json) | 53296 | 141.997, 94.439, 94.896 | 4,995 | 0 | 9 | Passed |
| [eval4_neutral1](../logs/rl/porto_bc_20261030_eval4_neutral1/student_evaluation.json) | 49680 | 139.807, 96.631, 94.627 | 4,970 | 0 | 9 | Passed |

All three reports have `practice_diagnostic_passed: true`, `race_lap_timing_valid: true`, zero missing lap times, zero monotonic control-gap faults, zero invalid route-projection samples, no stale or missing camera samples, and verified vehicle motion. The official latest lap time matches the stored third lap time in each report.

ROS sensor header timestamps showed 8–9 jumps above one second in each accepted run. They are retained as `ros_sensor_stamp_gap_count`; monotonic command timing showed zero actual control gaps. These values are kept separate in the reports.

## Runtime identity and safety

The simulator executable is the separately installed official Windows 2026-IROS practice build. Executable SHA-256: `f5efe8c38b78a72a37d44f07ea35e184cf1c41e4dd4e33eb3c9dc4415d470e69`. Official practice scene SHA-256: `5edbd1bacc15647746e6e89d9bcf1bb9e202d62d47e76ae352497be1404b3f8d`. Practice route SHA-256: `87d26be59ccc88ff6fc273905038906b8ed9e883b111c33de008a38f83a15bb3`.

Each accepted run directory contains `runtime_preflight.json`, `neutral_preflight.json`, `sensor_preflight.json`, the launch configuration, simulator log, evaluator output, and full student evaluation JSON. The WSL relay's response-timeout count did not increase during any accepted attempt.

Shutdown now publishes neutral throttle and steering before destroying the ROS transport, with the control-loop lock preventing an in-flight action from publishing after the terminal neutral command. After the final report was saved, the student simulator was closed and the WSL bridge/relay were stopped. No Docker runtime or Docker command was used.

## Rejected attempts retained

- `logs/rl/porto_bc_20261030_eval1/` is preserved unchanged. It completed three laps with zero collisions, but the old evaluator treated ROS header-stamp jumps as control gaps and its diagnostic gate failed. The run also predates freshness-verified lap-time reporting, so it is not included in the three accepted attempts.
- `logs/rl/porto_bc_20261030_eval2_retry1/` is preserved unchanged. It received policy actions without a prelaunch neutral barrier, collided after 18 steps, and is excluded. A subsequent neutral probe documented the wrong position and rising collision count; the lifecycle was corrected before the three accepted runs.
- `logs/rl/porto_bc_20261030_eval2/` contains the pre-fix attempt that was interrupted before completion and remains separate from the accepted run.

## Verification and cleanup

The final affected offline suite passed **152 tests** across control-loop safety, official race environment timing, Porto diagnostic gates, BC collection, diagnostic service, and official workflows. `git diff --check` passed.

After the three accepted reports were saved, the last GUI process was closed and the task's ROS Devkit bridge and Socket.IO relay were stopped. No simulator, policy evaluator, or neutral command publisher remains active. Docker was not used.

