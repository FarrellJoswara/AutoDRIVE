# IROS 2026 RoboRacer evaluation

This is a competition-facing evaluation path built on the official IROS 2026
simulator and Devkit images. The four-layer boundaries remain explicit:

1. **Layer 1** (`src/layer1/ros2_racer.py`) maps permitted LiDAR, IMU, wheel
   encoder, and actuator-feedback topics into the established
   `TelemetrySnapshot` parser and sends throttle/steering commands. Its
   wheel-angle speed estimate is retained for diagnostics, not used as body
   speed by the policy. Policy mode does not subscribe to odometry, IPS, TF,
   reset, lap-count, or collision-count topics.
2. **Layer 2** (`src/layer2/official_race_env.py`) builds policy observations
   from permitted Layer 1 data, estimates body motion by matching successive
   LiDAR scans with the IMU yaw change, and tracks the official race sequence.
   Policy observations never use centerlines, map files, simulator ground truth,
   or non-ROS interfaces. Training alone may use restricted official IPS
   position for the documented frontier reward and episode boundaries; that
   position is kept out of the policy observation and action path.
3. **Layer 3** (`src/layer3/official_policy.py`) runs the continuous
   competition controller. The separate local evaluator
   (`src/layer3/official_evaluate.py`) reads restricted lap/collision metrics
   only to report a test result; those metrics never reach policy inputs or
   choose actions.
4. **Layer 4** creates official simulator/API pairs for Train and Replay.
   Parallel environments use separate ROS domains and feed one PPO learner.
   Watch is scoped to a selected run; Replay preserves saved observation/action
   settings in a fresh official evaluation. See the
   [reliability audit](../../docs/reliability-audit-2026-10-08.md) for verification
   evidence and remaining limitations.

## Official rules represented

The evaluator ignores the warm-up lap, measures the following 10 race laps,
and stops after the tenth race lap. Race collisions do not end the attempt: the
official simulator resets the car to its checkpoint. Their cumulative penalty
is 10 seconds for the first collision, 20 for the second, 30 for the third,
etc. A race with more than 10 counted collisions is marked disqualified. The
cool-down lap is not required for completion.
The local evaluator stops and saves diagnostics as soon as the race collision
count exceeds 10, because the attempt is already officially disqualified.
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

Mission Control keeps CPU as the portable default. To enable NVIDIA PPO
updates, build the optional CUDA image and choose **NVIDIA GPU (CUDA)** in the
official training settings. The host must support Docker GPU passthrough (the
NVIDIA Container Toolkit on Linux, or a configured Docker Desktop WSL2 GPU).
The CUDA image uses the same official ROS/Devkit base and only changes the
PyTorch wheel; simulator/API bridge workers remain CPU containers.

```powershell
docker build -f competition/iros2026/Dockerfile `
  --build-arg AICAR_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 `
  --build-arg AICAR_TORCH_PACKAGE='torch==2.2.2+cu121' `
  -t aicar-iros2026:cuda .
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

## Train against the official simulator

Use the same `2026-iros-compete` simulator and Devkit images for PPO training
when producing a policy intended for this race. The custom Layer 2 Unity fleet
remains useful for development, but its physics and episode behavior are not a
substitute for official-image training. The official trainer uses one official
simulator instance, the same scan-paced ROS 2 action path and the shared
`OfficialObservationBuilder` used by deployed policies. It runs one complete
warm-up plus 10-lap race per episode; collisions reset to official checkpoints
and only disqualification ends an episode early. Training may use the guide's
restricted lap/collision/reset streams for reward and episode control, but those
streams are excluded from the policy observation and policy-mode ROS node.

Build the team Devkit image, start its trainer, then start the official
simulator on the same Docker network. The mounted run directory keeps PPO
checkpoints and TensorBoard data on the host:

```powershell
docker network create aicar-official-train
docker build -f competition/iros2026/Dockerfile -t aicar-iros2026 .
docker run --rm --name aicar-official-train-api `
  --network aicar-official-train --network-alias api `
  -e AICAR_MODE=train -e AICAR_TRAIN_TIMESTEPS=1000000 `
  -e HUB_URL=http://host.docker.internal:8090 `
  -e AICAR_TRAIN_OUT=/runs/official `
  -v "${PWD}/logs/rl/official_training:/runs/official" `
  aicar-iros2026
```

In a second terminal, start the official simulator and point its Devkit
connection at the trainer container's relay:

```powershell
docker run --rm --name aicar-official-train-sim `
  --network aicar-official-train `
  --entrypoint /bin/bash autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete `
  -lc './AutoDRIVE\ Simulator.x86_64 -batchmode -nographics -ip api -port 4568'
```

`AICAR_MODE=train` is a development/training mode, not a submission mode. It
enables restricted reward/reset handling only inside the trainer. `AICAR_MODE=policy`
remains the race path and cannot publish reset or subscribe to restricted
metrics. Official training publishes the simulator car, normalized LiDAR
display, lap/collision counters, PPO metrics, and rollout progress to the Watch
page. The launcher defaults `HUB_URL` to `http://host.docker.internal:8090`;
set `AICAR_HUB_URL` or `HUB_URL` for another hub address. Official PPO training
uses the demonstrated-route frontier for progress, charges simulated time, and
ends a car life on collision. A collision subtracts the fixed collision cost
plus 100% of that life’s positive frontier return; route progress is not
awarded again after reset. Lap crossings are diagnostics only and do not award
training reward. The frontier uses restricted IPS only for training reward and
episode control; policy observations remain sensor-only. Shaping weights are
recorded in each run's `config.json` and must be evaluated by completed
official races, not rollout reward alone.

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
PPO. It consumes the same canonical LiDAR and LiDAR-derived motion observation
and does not use map geometry or restricted race metrics. Select it with
`AICAR_CONTROLLER=lidar_gap`; no checkpoint mount is needed. This is a
diagnostic baseline, not a learned or competition-ready policy. Compare its
official-image lap/collision output before drawing conclusions about PPO.
