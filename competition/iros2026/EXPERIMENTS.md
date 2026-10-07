# Official IROS 2026 experiments

Each result below uses the same preserved PPO checkpoint and the official
competition image tags. A trial is considered comparable only when its
simulator is freshly started and the full warm-up plus ten timed laps complete.

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
  rule. The run is a clear failed transfer of the custom-simulator champion.
- Caveat: the simulator had already been running for about 18 minutes before
  this attempt, so its 1092.868 s warm-up lap time is invalid. Timed race laps
  are reported separately. This first report did not expose the raw collision
  baseline; the evaluator now records warm-up collisions, the raw counter, and
  the race baseline explicitly.
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
- Full raw result: `results/cadence40_20261007.json`.

The follow-up fixes the interval to be enforced between the scan observations
actually returned to the policy, rather than between command timestamps. A
new fresh-simulator attempt is required to assess the hypothesis.
