# IROS 2026 RoboRacer evaluation

This is a competition-facing evaluation path built on the official IROS 2026
simulator and Devkit images. The four-layer boundaries remain explicit:

1. **Layer 1** (`src/layer1/ros2_racer.py`) maps permitted LiDAR, IMU, wheel
   encoder, and actuator-feedback topics into the established
   `TelemetrySnapshot` parser and sends throttle/steering commands. It derives
   forward speed from wheel-angle changes. Policy mode does not subscribe to
   odometry, IPS, TF, reset, lap-count, or collision-count topics.
2. **Layer 2** (`src/layer2/official_race_env.py`) builds policy observations
   from Layer 1 data and tracks the official race sequence. It does not use
   centerlines, map files, simulator ground truth, or non-ROS interfaces.
3. **Layer 3** (`src/layer3/official_policy.py`) runs the continuous
   competition controller. The separate local evaluator
   (`src/layer3/official_evaluate.py`) reads restricted lap/collision metrics
   only to report a test result; those metrics never reach policy inputs or
   choose actions.
4. **Layer 4** remains the existing Train/Watch UI and training infrastructure;
   this work does not change its runtime or the simulator compose stack.

## Official rules represented

The evaluator ignores the warm-up lap, measures the following 10 race laps,
and stops after the tenth race lap. Race collisions do not end the attempt: the
official simulator resets the car to its checkpoint. Their cumulative penalty
is 10 seconds for the first collision, 20 for the second, 30 for the third,
etc. A race with more than 10 counted collisions is marked disqualified. The
cool-down lap is not required for completion.
The competition policy never publishes `/autodrive/reset_command`. Restricted
score topics are enabled only in the local test evaluator, separately from the
continuous policy runner.

Layer 1 also runs a small Socket.IO relay in front of the official Devkit
bridge. The official simulator emits one startup `Bridge` packet before its
first LiDAR array is available, while the stock Devkit callback indexes that
array unconditionally and fails before sending a reply. The relay answers only
incomplete packets with neutral throttle/steering and forwards every complete
packet unchanged to the unmodified official bridge. The policy still reads
only the allowed ROS topics listed above.

## Build and run

Pull the official simulator image separately and start it with the official
IROS `compete` settings. Build the policy image from the repository root:

```powershell
docker build -f competition/iros2026/Dockerfile -t aicar-iros2026 .
```

The policy image is derived directly from
`autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`. Mount the model
read-only and attach this container to the same Docker network as the simulator.
The container runs the unmodified official ROS bridge on port 4567 and the
Layer 1 relay on port 4568. Point the simulator to the policy container's
network name/IP on port 4568 (for example `-ip relay -port 4568` on a shared
Docker network):

```powershell
docker run --rm --name aicar-iros-policy `
  --network aicar-iros-eval `
  --network-alias relay `
  -e AICAR_MODEL_PATH=/models/policy.zip `
  -v "${PWD}/logs/rl/optimization_minimal_updates_20261005/best_evaluated_model.zip:/models/policy.zip:ro" `
  aicar-iros2026
```

The runner starts the official Devkit launch unchanged and runs the continuous
policy by default. For a local diagnostic, set `AICAR_MODE=evaluate`; that mode
observes restricted lap/collision topics but does not feed them to the policy.
It also subscribes to restricted IPS only for an evaluator-side motion-validity
check: if the car does not move at least 0.25 m within 3 seconds, the attempt
stops as `vehicle_motion_not_verified` and cannot be included in score
comparisons. The competition `AICAR_MODE=policy` path does not subscribe to or
use IPS.
Each additional attempt requires a newly launched simulator process. The report
is written to `/tmp/aicar-iros-evaluation.json` by default.
`AICAR_RACE_WALL_TIMEOUT_S` and `AICAR_RACE_STEP_GUARD` are local safety guards,
not simulated race-time limits and do not affect the measured score.

For GUI operation, configure the simulator's Connection target as the relay
host on port 4568, press Connection, then switch to Autonomous mode and the
required Ultra graphics setting. For headless operation, use the official
simulator's documented `-ip <relay-host> -port 4568` arguments. For repeatable
local comparisons, start a fresh simulator process for every attempt and
preserve the exact image tags.

For controlled transfer diagnostics, `AICAR_NEGATIVE_THROTTLE_MODE` accepts
`allow` (the policy's signed output is passed through), `zero` (negative
throttle becomes coasting), or `positive_magnitude` (negative output is mapped
to the same positive magnitude). The default is `allow`; alternate mappings
are experiments, not competition defaults. Evaluation output includes action
and allowed-observation distributions so a mapping can be assessed from the
official sensor path without exposing restricted race state to the policy.
`AICAR_STEERING_MODE` similarly accepts `normal` or `invert`; both controls
default to the unmodified policy output for competition use.

`AICAR_THROTTLE_MODE` accepts `bidirectional` or `forward_only`. Use the mode
that matches the model's training configuration. `forward_only` maps PPO's
normalized throttle output from `[-1, 1]` to actuator throttle `[0, 1]`; it is
not interchangeable with a bidirectional model's action semantics.

## Interpretation

Only compare policies tested with these official image tags and this same
12-lap flow. Earlier 10-lap results from AiCar's custom simulator/evaluator
are useful development evidence but are not directly comparable to the
official IROS score. The model's action cadence is driven by fresh ROS 2
LiDAR frames; this evaluator does not set a simulator timestep or modify the
official simulator.

### Sensor-only gap-following diagnostic

Layer 3 also includes a deterministic LiDAR gap-following controller used to
validate sensor ordering, steering direction, and actuator response without
PPO. It consumes the same canonical LiDAR and encoder-derived speed observation
and does not use map geometry or restricted race metrics. Select it with
`AICAR_CONTROLLER=lidar_gap`; no checkpoint mount is needed. This is a
diagnostic baseline, not a learned or competition-ready policy. Compare its
official-image lap/collision output before drawing conclusions about PPO.
