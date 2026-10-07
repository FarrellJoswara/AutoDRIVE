# Official IROS 2026 experiments

Each result below uses the same preserved PPO checkpoint and the official
competition image tags. Only the final policy path is intended for competition
use. A valid score comparison also requires allowed sensor inputs, no restricted
reset command, a fresh simulator process, and a complete race.

## Benchmark target and local-training boundary

- The official IROS 2026 page now publishes the completed results. Phase 1's
  fastest result was 49.83 s (4.94 s best lap); Phase 2's fastest result was
  71.82 s (7.16 s best lap), with zero collisions. Phase 2 is the relevant
  benchmark for the unseen-track time-attack objective.
- The rules prohibit modifying the competition framework and prohibit using
  restricted ground-truth streams such as IPS, odometry, TF, reset, lap-count,
  and collision-count in the submitted policy. The technical guide lists
  LiDAR, camera, IMU, wheel encoders, and actuator feedback as policy inputs.
- The normal `docker-compose.yml` training stack uses this repository's own
  Unity player and Layer 2 reward signals derived from internal route state. It
  remains useful for development, but its lap times cannot establish an
  official-compatible result. Competition claims require the separate
  `competition/iros2026` path and exact official simulator/Devkit image tags.
- Date: 2026-10-07. A fresh PPO trial was briefly started with the official
  sensor observation profile but the local custom simulator/reward stack. It
  reached 4,096 steps / one PPO update (KL 0.00268, clip fraction 0.019,
  explained variance -0.00072, action std 0.80) and was stopped before an
  evaluation. It is preserved as `official_sensor_fresh_lr1e4_20261007_*` in
  `logs/rl`; it has no race result and is not a candidate checkpoint. This
  confirms that merely switching observations does not make the custom training
  stack a competition-faithful benchmark.
- Next experiments should target the official runtime and allowed sensor path;
  keep local multi-environment training results labeled as development-only.

### Forward-only fine-tune on the local development simulator — not promoted

- Date: 2026-10-07; run:
  `logs/rl/iros_official_forwardonly_50ms_resume_20261007`.
- Resumed the preserved `optimization_control0946_gain1025_lr2e6_20261007_20261007_063151/best_evaluated_model.zip`
  checkpoint with four local environments, official-sensor-shaped observations,
  forward-only throttle mapping, 50 ms actions, and the existing reward config.
- Deterministic local snapshots at 20,480, 40,960, and 61,440 steps all had
  zero completed laps and three collisions per three attempts. Frontier pace
  improved from about 0.082 to 0.098 m/s by 61,440, but remained noncompetitive.
- The 98,304-step snapshot's first attempt eventually completed one local lap
  in 270.34 s with no collision. Mean throttle was 0.416 and it spent no time
  at full throttle. The full three-attempt snapshot was not completed; the
  training job was stopped at 122,400 steps and all checkpoints were retained.
- This run used the custom `aicar-sim` image and Layer 2 internal route rewards.
  Its lap is development-only and cannot be compared to official IROS time.
  No checkpoint was promoted. Full metrics are in
  `results/local_forwardonly_training_20261007.json`.
- Next, evaluate the saved checkpoint through the official simulator/Devkit
  images before changing the official controller. Any training reward or
  action-transfer experiment must retain the same four-layer boundaries and
  be compared on that official permitted-sensor path.

### Official transfer check: local forward-only PPO checkpoint at 120,000 steps

- Date: 2026-10-07; checkpoint:
  `logs/rl/iros_official_forwardonly_50ms_resume_20261007/ckpt/ppo_120000_steps.zip`.
- Evaluated with the exact official IROS 2026 simulator and Devkit image tags,
  normal steering, forward-only throttle, and the allowed LiDAR/IMU/encoder/
  actuator observation path. No restricted telemetry was fed to the policy.
- The 300-second diagnostic guard ended the incomplete attempt after one race
  lap at 121.977 s. It recorded 5 warm-up collisions and 6 race collisions
  (210 s official penalty); the attempt was neither complete nor disqualified.
  This is not a valid 10-lap time and the checkpoint was not promoted.
- Raw result:
  `results/forwardonly_ppo120k_official_300s_20261007.json`.
- The local training snapshot's 270.34 s lap is not comparable: the official
  simulator completed one lap in 121.98 s under the same policy checkpoint.
  Both results remain far from the 71.82 s 10-lap benchmark, and the collision
  rate makes this transfer unsuitable. Next work should address control/action
  transfer inside the existing four layers and screen on the official runtime.

### Official action-transfer sweep on the 120,000-step checkpoint — rejected

- All variants used the same checkpoint, official simulator/Devkit images,
  policy observation path, and forward-only throttle. These are incomplete
  diagnostics, not race scores.
- Steering scale 0.92 (all else at the 0.946 baseline) completed one timed lap
  in 124.905 s with 5 race collisions. The 0.946 screen was faster at 121.977 s
  but had 6 race collisions. This one-lap difference is insufficient to promote
  0.92; neither test completed the 10-lap race.
- Straight throttle gain 1.25 (scale 0.946, other controls held fixed) was
  continued to the official disqualification endpoint. It completed laps in
  122.377 and 127.848 s, then reached 11 race collisions and was disqualified
  with 660 s collision penalty. Reject this gain; it did not yield a clean or
  competitive race. Per-run reports are the `forwardonly_ppo120k_*` files in
  `results/`.
- The run findings do not justify promoting any of these action mappings.
  Keep the checkpoint and original training stack. The next candidate needs a
  policy/control improvement that changes both sustained pace and collision
  rate, verified by complete official races.
- As a direct action-semantics check, the same 120,000-step checkpoint was run
  with bidirectional throttle (all other settings unchanged). It completed
  zero laps in 300 s, had no collisions, and moved at most 2.35 m. Its output
  remained negative on about 50% of actions. The forward-only mapping is the
  better of these two mappings for this checkpoint, but it still is not a
  viable policy; the raw comparison is in
  `results/ppo120k_bidirectional_official_300s_20261007.json`.

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

## Policy transfer: throttle-sign diagnostics

These short tests use the same preserved checkpoint and exact official image
tags above. They diagnose action semantics only; neither produced a completed
lap and neither is a score comparison.

- Signed pass-through (`allow`): the deterministic policy repeatedly commanded
  full reverse (`throttle=-1.0`); both permitted rear-wheel encoder positions
  decreased. No lap completed. This confirms that the current checkpoint's
  output is actively reversing under the official actuator mapping.
- Positive-magnitude mapping (`positive_magnitude`): the same negative output
  became full forward throttle (`throttle=+1.0`) and wheel encoder positions
  increased. The car registered 47 collisions and completed no lap before the
  diagnostic was stopped. This mapping changes direction but is not a viable
  fix by itself; steering/observation/action transfer still needs diagnosis.
- The evaluator now records policy and applied throttle/steering distributions,
  reverse-action fraction, normalized state-channel distributions, and minimum
  normalized LiDAR readings. It excludes pose, lap count, and collision count
  from policy observations. The three controlled throttle mappings live in
  Layer 2 and are exposed by the Layer 3 evaluator/runner; the Layer 1 bridge
  continues to relay official telemetry and actuator commands.

### Steering polarity trial

- Date: 2026-10-07; same checkpoint and official simulator/API images; fresh
  simulator; 35-second diagnostic cap.
- Compared with the preceding forward-throttle/normal-steering attempt, this
  changed only steering polarity to `invert`.
- Result: 0 laps, 58 collision events in 35 seconds, versus 35 collision events
  in 35 seconds with normal steering. The policy's mean steering was +0.533,
  applied mean steering was -0.533, and mean applied throttle was +0.994.
  Steering inversion made the short-run result worse; keep the default
  `normal` mapping. This is a failed diagnostic, not a score comparison.
- Raw report: `results/steering_invert_20261007.json`.

### Checkpoint transfer comparison

- Date: 2026-10-07; fresh official simulator; same 35-second cap, forward
  throttle mapping, and normal steering as the first diagnostic.
- Compared `optimization_action50ms_champion_20261006_20261007_011022` with
  `optimization_minimal_updates_20261005` while keeping all controls fixed.
- Result: both completed 0 laps and recorded 35 collisions in 35 seconds.
  The comparison checkpoint selected reverse 98.6% of the time and had mean
  policy throttle -0.973, nearly identical to the first checkpoint's 98.5%
  and -0.971. Changing checkpoints alone did not resolve transfer behavior.
- Raw report: `results/checkpoint_compare_20261007.json`.

### Pooled official-sensor policy: first official-image transfer diagnostic

- Date: 2026-10-07; training used the local four-environment Layer 2 sim with
  `official_sensors`, forward-only throttle, 25 ms action interval, and the
  pooled LiDAR CNN. The tested checkpoint was the 20,480-step recovery point.
- The local deterministic evaluator reported 0/3 completed laps; all three
  attempts collided after 0.925 simulated seconds at about 1.275 m of frontier
  distance. The deterministic action trace was nearly constant as front LiDAR
  distance fell from about 0.138 to 0.036 normalized range.
- A transfer diagnostic then ran against the unmodified IROS 2026 simulator
  and official Devkit images, using the same forward-only throttle, steering
  scale (0.946), and straight throttle gain (1.025) as training. The relay
  received full sensor packets and returned official Devkit commands.
- After roughly 2.5 wall-clock minutes the official lap counter remained 0
  and collision count had reached 244. The diagnostic was stopped before a
  lap completed; it has no official race-time score and is not comparable to
  leaderboard results. This confirms that the 20k checkpoint is not yet a
  usable competition policy. Continue training from the preserved checkpoint
  and use official-image runs for performance comparisons.
- A separate 30-second official-image diagnostic captured 592 control actions:
  measured control cadence was 19.73 Hz (median interval 50.91 ms) while the
  advertised LiDAR scan rate was 40 Hz. It completed 0 laps and observed 55 raw
  collisions. Applied commands were nearly constant (mean throttle 0.478,
  mean steering 0.0345); the normalized minimum LiDAR return averaged 0.0435.
  Policy inference itself took under 1 ms in an isolated CPU microbenchmark, so
  the lower control cadence is in the bridge/simulator exchange, not the CNN.
  Align the next Layer 2 training trial to the measured ~50 ms command cadence;
  this changes only action interval and is an explicit training experiment.

### Control-rate-matched pooled policy continuation (planned)

- Hypothesis: the pooled policy trained at 25 ms action intervals is mismatched
  to the official image path's measured ~50.9 ms command interval. Training at
  50 ms may improve closed-loop behavior under the official cadence.
- Start from the preserved `ppo_20000_steps.zip` checkpoint in
  `official_sensor_pooled_forward_20261007_20261007_112825`; keep map (Porto),
  4 environments, official-sensor observations, pooled architecture, action
  mapping, rewards, and all PPO/exploration settings fixed. Change only
  `action_interval_s` from 0.025 to 0.05. Use a new run directory.
- Compare deterministic internal evaluations and PPO diagnostics, then test a
  saved checkpoint through the unmodified official simulator + Devkit images.
  Do not call a training-only result an official score; require completed
  official laps for a valid race-time comparison.

- Outcome: the trial resumed from the 20k checkpoint and changed only the Layer 2
  action interval to 50 ms. At the 20,480-step snapshot, all three deterministic
  attempts collided after 0.95 simulated seconds, averaging 1.59 m frontier
  distance; none completed a lap. A later local checkpoint had essentially zero
  explained variance and still no lap progress.
- On the official images, the 20k checkpoint ran at 19.53 actions/s and recorded
  30 raw collisions with no lap in 30 seconds. The 30k checkpoint ran at 19.49
  actions/s and recorded 42 raw collisions in 30 seconds. A 180-second official
  run of that 30k checkpoint completed 0 laps and recorded 259 raw collisions.
  The images and policy mapping were unchanged; diagnostics are not race scores.
- Conclusion: 50 ms training did not produce an official-compatible driver. It
  did not change official control cadence relative to the 25 ms checkpoint
  (19.73 actions/s), and its later checkpoint became more action-variable while
  accumulating more collisions. Do not promote this branch. Keep all checkpoints
  for diagnosis; resume optimization from a known lap-capable legacy checkpoint
  or address the observation/action transfer with a separately controlled test.
- Raw reports: `results/pooled_50ms_checkpoint_diag_30s_20261007.json`,
  `results/pooled_50ms_30k_checkpoint_diag_30s_20261007.json`, and
  `results/pooled_50ms_30k_official_180s_20261007.json`.

### Legacy checkpoint through the current compliant sensor path

- Date: 2026-10-07; tested the preserved
  `optimization_minimal_updates_20261005/best_evaluated_model.zip` checkpoint
  with current Layer 1/2 code, the exact official IROS image tags, and its
  original bidirectional action semantics. Policy mode subscribed only to
  LiDAR, IMU, encoders, and actuator feedback; the local evaluator alone read
  lap/collision counters, and no reset command was published.
- After the 300-second guard: 0 laps, 1 raw collision, 5,915 policy steps,
  measured action cadence 19.71 Hz. A live topic check showed throttle=-1.0
  and lap_count=0; the checkpoint remained in full reverse at that point.
- This corrects the earlier apparent “lap-capable” baseline: the 10-lap,
  ~13-second result in `transfer_baseline_20261007.json` used the pre-audit
  odometry-derived speed and restricted reset path, and is not evidence that
  this checkpoint can race using allowed policy inputs. Preserve it as a
  historical diagnostic, not a valid baseline.
- Raw report: `results/legacy_compliant_baseline_20261007.json`.

### Full-resolution LiDAR CNN challenger (planned)

- Hypothesis: the pooled extractor may discard angular detail needed for
  last-second wall avoidance. Compare the default full-resolution `lidar_cnn`
  against `lidar_cnn_pooled` while holding the four-environment local simulator,
  Porto map, official-sensor observation profile, forward-only action mapping,
  50 ms action interval, reward, and PPO settings fixed.
- Start fresh because the two policy architectures are checkpoint-incompatible;
  preserve all pooled checkpoints. Keep the 20,480-step deterministic
  three-attempt evaluator and record both local and official-image diagnostics.
- A candidate is not promoted unless it completes laps on the permitted
  observation path and then passes the official-image evaluation without
  disqualification. The legacy checkpoint's pre-audit laps are excluded from
  this criterion because that path used restricted inputs/reset.
- Outcome: the fresh full-resolution run reached its 20,480-step deterministic
  snapshot with 0/3 completed laps. All attempts collided at about 1.0 simulated
  second and roughly 1.89 m of frontier distance. The 30k checkpoint was then
  tested in the unmodified official simulator and Devkit images for 30 seconds:
  it completed 0 laps and recorded 44 raw collisions across 587 actions. The
  measured control cadence was 19.54 Hz (median 51.28 ms); its policy outputs
  were effectively constant (throttle std 2.9e-6, steering std 3.2e-6).
- The full-resolution model reached slightly farther in the local snapshot than
  the pooled model's 1.59 m, but neither produced a lap-capable policy; its
  official diagnostic also showed no useful steering adaptation. Do not promote
  this architecture or infer a race-time improvement. Preserve the checkpoints
  as training diagnostics. Raw report:
  `results/fullres_50ms_30k_checkpoint_diag_30s_20261007.json`.

### Sensor-only gap-following controller

- Hypothesis: before spending more PPO compute, verify the permitted sensor and
  actuator path with an adaptive Layer 3 controller whose steering responds
  deterministically to clear LiDAR gaps. A successful lap would validate
  LiDAR orientation and steering sign; failure would isolate low-level control
  or simulator/bridge issues from policy learning.
- The controller uses only the canonical LiDAR array and encoder-derived
  forward speed already exposed in the official observation. It does not read
  map geometry, pose, lap count, or collision count. Unit tests cover obstacle
  side, steering direction, and speed response. Treat its official-image run as
  a control-path diagnostic, not a record candidate.
- Outcome: with normal steering it ran 30 seconds at 20.25 Hz, completed 0
  laps, and recorded 33 raw collisions. Outputs were adaptive (steering std
  0.201, throttle std 0.083) and mean encoder-derived speed was about 5.70
  m/s, so the constant-output PPO behavior is not simply forced by the bridge.
  The steering-inverted repeat completed 0 laps and recorded 42 collisions in
  30 seconds at 19.72 Hz. Steering inversion therefore worsened this diagnostic;
  retain the normal actuator direction. These bounded samples are diagnostics,
  not official race scores. Raw reports:
  `results/lidar_gap_diag_30s_20261007.json` and
  `results/lidar_gap_invert_diag_30s_20261007.json`.
- Code audit identified that the first controller revision imposed a 0.15
  throttle floor, so it could never coast while overspeed. Since negative
  throttle commands reverse instead of brake, this made its target-speed
  calculation ineffective. The next controlled repeat changes only this
  throttle rule: clamp to zero at/above target speed, retaining the same target
  speed curve, steering, sensor path, and official images.
- Validity correction: a bounded trace of that repeat showed the official
  simulator's restricted IPS position stayed at x=0.8001 m throughout a 10 s
  sample, IMU yaw rate stayed zero, and LiDAR sector distances were effectively
  unchanged, while encoder angles continued increasing. Thus the car was not
  traversing the track; encoder-derived wheel speed in this situation reflected
  wheel rotation, not body progress. The 0-collision/0-lap result is not evidence
  of a safe controller, and the preceding headless official-image results must
  not be treated as performance comparisons until vehicle motion is verified.
  The official technical guide describes Manual as the default mode and calls
  for a separate Autonomous toggle after connection, while the IROS 2026
  submission page says connecting should start the car. This run's stationary
  telemetry shows our headless harness did not reproduce the intended running
  state. Resolve/verify simulator mode before any more policy tuning or scoring.
Diagnostic artifacts: `results/lidar_gap_coast_diag_30s_20261007.json`,
  `results/lidar_gap_trace_30s_20261007.json`, and
  `results/lidar_gap_encoder_verify_15s_20261007.json`.
- Attempted to activate the documented GUI mode under Xvfb without changing the
  simulator binary. The official `2026-iros-compete` image's player exited with
  `No supported renderers found` (Vulkan detection 0; its Xvfb OpenGL renderer
  was rejected), so the GUI toggle could not be exercised here. Until the
  simulator can be started in Autonomous mode on a supported display, stop
  controller tuning and reject any run that lacks verified body movement.

## Evaluation validity guard

- Date: 2026-10-07.
- The stationary-run diagnosis exposed two evaluator defects: encoder rotation
  was mistaken for body motion, and a zero-lap attempt was serialized as a
  `0.0 s` adjusted score. Neither value should be mistaken for a result.
- In the local `AICAR_MODE=evaluate` path only, Layer 1 now reads restricted
  `/autodrive/roboracer_1/ips` alongside the evaluator's existing restricted
  race counters. The policy transport (`AICAR_MODE=policy`) does not subscribe
  to IPS; the pose is kept out of the observation/action path and used only to
  validate that the vehicle moved at least 0.25 m within 3 seconds.
- A stationary run now stops with `vehicle_motion_not_verified`. Incomplete
  races and disqualified races have no adjusted comparison score; a valid
  candidate must complete all 10 timed laps, pass the motion check, and remain
  within the collision limit.
- Validation: unit tests cover stationary and moving IPS traces, early abort,
  policy-observation isolation, and incomplete-score handling. No new official
  race result is claimed until this guard is exercised against the official
  runtime with the car visibly traversing the course.

## Local policy action-scale sweep (development-only)

- Date: 2026-10-07. This uses the repository's fixed-step Porto development
  simulator and the preserved `optimization_control0946_gain1025_lr2e6` policy.
  It is not an official-image evaluation or a competition score; the official
  runtime still needs a verified moving-car evaluation before promotion.
- Replay-path fix: replay had been cloning the simulator with inherited
  environment variables while Layer 2 used the selected run's environment
  settings. Commit `c82ae22` makes replay pass the same simulator and Layer 2
  configuration used by training/evaluation. Focused coverage: 4 tests pass.
- Baseline: 25 ms action interval, frame skip 1, throttle gain 1.025, steering
  scale 0.946; 10/10 laps, 0 collisions, 63.921 s.
- One-factor steering-scale probe (all other settings and policy unchanged):

  | Steering scale | 10-lap time | Completion | Collisions |
  | ---: | ---: | ---: | ---: |
  | 0.85 | 63.823 s | 10/10 | 0 |
  | 0.90 | 63.821 s (two identical runs) | 10/10 | 0 |
  | 0.91 | 63.997 s | 10/10 | 0 |
  | 0.92 | 63.696 s (two identical runs) | 10/10 | 0 |
  | 0.93 | 63.773 s | 10/10 | 0 |
  | 0.946 baseline | 63.921 s | 10/10 | 0 |
  | 1.00 | 64.548 s | 10/10 | 0 |

- The 0.92 challenger repeats exactly and is 0.225 s (0.35%) faster than the
  baseline in this deterministic local harness. Treat 0.92 as a local runtime
  candidate only; leave the competition/default setting and checkpoint
  unchanged until the same policy is tested in the unmodified official runtime
  with verified vehicle movement.
- Straight-throttle gain 1.05 was tested twice with the same policy and other
  settings. Both runs stopped after 9 laps at 70.25 simulated seconds with
  `frontier_stagnation` and 0 collisions, so reject this setting. Its artifacts
  and all sweep outputs remain under
  `logs/rl/throttle_gain_25ms_retest_20261007/`.

### Official Windows simulator transfer checks

- Date: 2026-10-07. All results below use the official IROS API and simulator
  image tags, the official Windows simulator release in batch/no-graphics mode,
  its native control cadence, the Porto competition track, and the preserved
  `optimization_control0946_gain1025_lr2e6_20261007` checkpoint. Each attempt
  uses a fresh simulator process. Local watchdog duration is not simulated race
  time. Only a complete, non-disqualified 10-lap attempt is a score comparison.
- Control-path validation: Layer 3's fixed-throttle diagnostic moved at least
  19.47 m in 30 s, confirming the official Windows simulator connects and
  accepts commands. The fixed straight controller had 61 raw collisions, as
  expected for an intentionally non-driving diagnostic; it is not a candidate.
  Report: `results/official_straight_control_path_20261007.json`.
- Unmodified bidirectional policy baseline: 120 s, 2,267 control steps,
  18.89 Hz, verified 2.05 m displacement, no completed warm-up or race laps,
  and one raw collision at finish. In this policy's observed actions, throttle
  was frequently negative; this is consistent with its near-zero net movement.
  No score comparison. Report:
  `results/official_ppo_baseline_120s_20261007.json`.
- One-factor forward-only transfer (`[-1,1]` policy throttle mapped to
  `[0,1]` actuator throttle; steering scale 0.946 and gain 1.025 unchanged):
  the 120 s pilot completed its ignored warm-up in 118.68 s, with 3 warm-up
  collisions, then timed out before a scored lap. A longer attempt completed
  scored laps in 121.805 s and 122.145 s, but had accumulated 8 race collisions
  by lap 2. It was stopped as a losing diagnostic before exhausting its
  900-second watchdog. Therefore it is not a valid or promotable race result.
  The full 120 s report and partial longer-run event log are retained at
  `results/official_forward_only_120s_20261007.json` and
  `results/official_forward_only_900s_partial_20261007.log`.
- Interpretation: forward-only mapping prevents reverse commands and proves
  that the policy can complete laps on the official build, but its current
  official-runtime pace and collision rate are far from the target. The model's
  frequent reverse outputs become zero throttle under this mapping, which
  explains the low average drive input; mapping them to forward throttle is a
  candidate hypothesis, not yet evidence of a faster or safer policy. Do not
  compare these incomplete laps with the official 10-lap leaderboard or change
  the saved champion based on them.
- Evaluator improvement: after race collisions exceed the official limit of
  10, Layer 3 now ends the attempt immediately with
  `collision_disqualification_limit`, preserving diagnostics and avoiding
  wasted watchdog time. Focused unit tests cover the official threshold and
  saved disqualification status.
- Throttle-absolute-value probe (`bidirectional` throttle with negative
  outputs remapped to their positive magnitude): aborted during warm-up. At
  the observation point, official lap count was still 0, lap timer was
  103.19 s, position remained near the initial point, and raw collision count
  had reached 104. This is an unsafe diagnostic, not a race collision count or
  score; reject `abs(throttle)` as a transfer fix. Report and raw container log:
  `results/official_abs_throttle_partial_20261007.json` and
  `results/official_abs_throttle_partial_20261007.log`.
- One-factor steering-scale transfer (`forward_only`, scale 0.92; same
  checkpoint, throttle gain, official images, and sensors): warm-up took
  183.762 s with 3 collisions. The first three scored laps took 120.788,
  121.621, and 121.871 s; the attempt reached 11 race collisions and was
  disqualified at 4 total laps (including warm-up). It therefore has no valid
  score. Compared with the 0.946 forward-only diagnostic's first two laps
  (121.805 and 122.145 s, 6 race collisions by lap 2), the 0.92 scale improved
  those two laps by about 1.54 s combined but had the same 6 race collisions by
  lap 2 and was much slower to finish warm-up. Reject it as a candidate. The official
  cadence was 18.91 Hz (median 52.31 ms) while reported LiDAR scans remained at
  40 Hz. Report and full run log:
  `results/official_forward_only_steer092_900s_20261007.json` and
  `results/official_forward_only_steer092_900s_20261007.log`.
- Across the official PPO transfers, the model emits reverse throttle on
  approximately half its actions. `forward_only` maps those outputs to zero,
  and yields an applied-throttle mean of about 0.44. Absolute-magnitude
  remapping instead causes a high-collision warm-up failure. This is evidence
  that output remapping alone cannot safely repair the model; a controlled
  continuation should train with the official forward-only action semantics
  and the official sensor observation profile, while leaving all unrelated
  reward/PPO settings fixed.
