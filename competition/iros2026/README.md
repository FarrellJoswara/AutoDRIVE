# IROS 2026 RoboRacer evaluation

This is a competition-facing evaluation path built on the official IROS 2026
simulator and Devkit images. The four-layer boundaries remain explicit:

1. **Layer 1** (`src/layer1/ros2_racer.py`) subscribes to official ROS 2 data
   and sends throttle, steering, and reset commands. It translates ROS message
   fields back into Bridge V1 fields so the established `TelemetrySnapshot`
   parser remains the single owner of telemetry transformations.
2. **Layer 2** (`src/layer2/official_race_env.py`) builds policy observations
   from Layer 1 data and tracks the official race sequence. It does not use
   centerlines, map files, simulator ground truth, or non-ROS interfaces.
3. **Layer 3** (`src/layer3/official_evaluate.py`) loads the policy, runs
   deterministic attempts, and reports completion, lap times, collision
   penalties, and adjusted race time.
4. **Layer 4** remains the existing Train/Watch UI and training infrastructure;
   this work does not change its runtime or the simulator compose stack.

## Official rules represented

The evaluator ignores the warm-up lap, measures the following 10 race laps,
and stops after the tenth race lap. Race collisions do not end the attempt: the
official simulator resets the car to its checkpoint. Their cumulative penalty
is 10 seconds for the first collision, 20 for the second, 30 for the third,
etc. A race with more than 10 counted collisions is marked disqualified. The
cool-down lap is not required for completion.

## Build and run

Pull the official simulator image separately and start it with the official
IROS `compete` settings. Build the policy image from the repository root:

```powershell
docker build -f competition/iros2026/Dockerfile -t aicar-iros2026 .
```

The policy image is derived directly from
`autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`. Mount the model
read-only and attach this container to the same Docker network as the simulator:

```powershell
docker run --rm --name aicar-iros-policy `
  --network aicar-iros-eval `
  --network-alias devkit `
  -e AICAR_MODEL_PATH=/models/policy.zip `
  -v "${PWD}/logs/rl/optimization_minimal_updates_20261005/best_evaluated_model.zip:/models/policy.zip:ro" `
  aicar-iros2026
```

The runner starts the official Devkit launch unchanged, waits for sensor topics,
then evaluates one complete race by default. Set `AICAR_EVALUATION_ATTEMPTS=3`
for repeated local trials (the runner resets between attempts). The report is
written to `/tmp/aicar-iros-evaluation.json` by default. `AICAR_RACE_WALL_TIMEOUT_S`
and `AICAR_RACE_STEP_GUARD` are local safety guards, not simulated race-time
limits and do not affect the measured score.

The official simulator connection is opened from the simulator's Connection
Button after both containers are running. Use Autonomous mode and the required
Ultra graphics setting, as in the competition procedure. For repeatable local
comparisons, restart/reset to the same initial simulator state before each
attempt and preserve the exact image tags.

## Interpretation

Only compare policies tested with these official image tags and this same
12-lap flow. Earlier 10-lap results from AiCar's custom simulator/evaluator
are useful development evidence but are not directly comparable to the
official IROS score. The model's action cadence is driven by fresh ROS 2
LiDAR frames; this evaluator does not set a simulator timestep or modify the
official simulator.
