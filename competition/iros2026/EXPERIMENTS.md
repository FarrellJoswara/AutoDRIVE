# Official IROS 2026 experiments

Each result below uses the same preserved PPO checkpoint and the official
competition image tags. Only the final policy path is intended for competition
use. A valid score comparison also requires allowed sensor inputs, no restricted
reset command, a fresh simulator process, and a complete race.

## Transfer baseline: callback-rate control

- Date: 2026-10-07
- Checkpoint: `optimization_minimal_updates_20261005/best_evaluated_model.zip`
- Simulator: `autodriveecosystem/autodrive_roboracer_sim:2026-iros-compete`
  (`sha256:749fbef07942109d18497cbcf6ffe9452e06ae487f2bfc915e92440bc4b4663d`)
- Devkit base: `autodriveecosystem/autodrive_roboracer_api:2026-iros-compete`
  (`sha256:4ce4334657feb4c6760aa61f23a76f8962bf80e8e746e2547b082985a95a46a2`)
- Control: one policy action per received LaserScan callback.
- Result: 10/10 timed laps, 130.624 s total, 13.062 s mean lap,
  12.828 s best lap, 30 reported race collisions; disqualified by the 10-hit
  rule. Diagnostic only; not a valid competition result.
- Caveat: the simulator had already been running for about 18 minutes before
  this attempt, so its 1092.868 s warm-up lap time is invalid. Timed race laps
  are reported separately. This first report did not expose the raw collision
  baseline; the evaluator now records warm-up collisions, the raw counter, and
  the race baseline explicitly.
- Rules audit: this pre-audit evaluator also published the restricted reset
  topic and used odometry-derived speed in the policy observation. The official
  2026 guide marks `/odom` and `/autodrive/reset_command` restricted. Do not use
  this or any reset-based trial as an official score comparison.
- Full raw result: `results/transfer_baseline_20261007.json`.

## Hypothesis: action rate should follow sensor cadence

Official LaserScan messages advertise `scan_time=0.025 s` (40 Hz), while the
simulator/bridge can deliver callback updates faster. Layer 1 now waits at
least the advertised period and for a scan received after the action command
before returning a policy step. This tests whether reacting to repeated scan
callbacks caused unstable or excessively rapid control. Reward, checkpoint,
observation processing, and Layer 2 race rules remain fixed.

### Cadence attempt v1: pacing defect found

- Date: 2026-10-07; fresh simulator process.
- Result: 10/10 timed laps, 130.797 s total, 13.080 s mean lap, 12.745 s
  best lap, 31 race collisions, disqualified. Three warm-up collisions were
  excluded correctly (raw count 34, race baseline 3).
- The implementation intended to respect 40 Hz but measured 60.96 policy
  actions/s (median interval 16.65 ms). Therefore this is a failed control
  implementation, not a valid test of the 40 Hz hypothesis. It also did not
  improve over the callback-rate baseline.
- This run also used the pre-audit reset/odometry path and is diagnostic only.
- Full raw result: `results/cadence40_20261007.json`.

The scan pacing was corrected to enforce the interval between scan observations
actually returned to the policy. The policy path was also changed to use only
permitted LiDAR, IMU, encoder, and actuator-feedback inputs. A fresh
simulator-process evaluation is required before drawing conclusions.

### Compliance-path smoke attempt: no sensor frames

- Date: 2026-10-07; official simulator and Devkit image tags above.
- The simulator and Devkit bridge established a Socket.IO connection, and ROS
  graph inspection showed the bridge's LiDAR publisher and policy subscriber.
  However, no LiDAR messages arrived. The evaluator failed closed after its
  180-second sensor timeout with `No official LiDAR frame at the declared scan
  cadence`; it produced no lap or score result.
- The bridge logged a `KeyError` for `V1 LIDAR Range Array` on a control-only
  `Bridge` payload. The exact cause of the missing telemetry stream is not yet
  established. This is a simulator/bridge launch or handshake issue to resolve
  before interpreting policy performance; it is not evidence that the policy
  is driving poorly.
- The failed run's containers were stopped. Existing training and checkpoint
  artifacts were left intact. Do not compare this attempt as a performance
  result.

### Compliance-path smoke retry: GPU passthrough corrected, telemetry still absent

- Date: 2026-10-07; same official image tags, same preserved policy, fresh
  simulator process. The simulator was started with Docker `--gpus all`, as in
  the official container instructions; `nvidia-smi` inside the simulator
  confirmed the host RTX 3060 was visible.
- The Devkit again logged a Socket.IO connection followed by a control-only
  `Bridge` payload missing `V1 LIDAR Range Array`. No LiDAR frame reached the
  evaluator within 60 seconds, so it stopped before policy inference and
  produced no lap result.
- GPU passthrough was missing in the prior smoke setup, but correcting that
  did not resolve the stream. The telemetry/handshake failure remains
unlocalized; neither attempt is a policy performance result.

### Root cause: the official bridge deadlocks on the first startup packet

- Date: 2026-10-07; exact official simulator/API images above.
- A packet probe on a disposable network captured the simulator's actual
  startup sequence. Its first `Bridge` event contains 17 fields and omits only
  `V1 LIDAR Range Array`; the next events contain the full 18-field telemetry
  payload. The unmodified Devkit callback indexes the missing LiDAR field and
  raises before returning its actuator response. The simulator waits for that
  response, so the official bridge receives no later full frame.
- Layer 1 now replies to incomplete startup frames with neutral throttle,
  steering, and reset=false. Complete packets are forwarded unchanged to the
  unmodified official Devkit bridge, and its returned actuator command is
  forwarded back to the simulator. The policy still subscribes only to allowed
  ROS sensor and actuator-feedback topics.
- The relay's first local integration test also found that this official image
  lacks `requests`, required by its Python Socket.IO client's polling
  transport. The submission image now pins `requests==2.32.3` and the relay
  fails visibly if its upstream Devkit connection cannot be established.
- Validation: packet probe received the partial first frame followed by full
  frames; relay passed full frames and returned Devkit commands. Against the
  official Devkit image, the ROS LiDAR topic then produced 1081-beam scans at
  the advertised 40 Hz. This proves the transport and sensor path only; it is
  not a race-time result.
