# Porto behavior-cloned model progress log

This document records the Porto practice behavior-cloning model from data
collection through the current PPO-continuation investigation. It is a running
experiment log; new runs should be added below rather than replacing prior
evidence.

## Current status — 2026-11-05

The behavior-cloned checkpoint is the strongest currently verified Porto
student policy. It completed three fresh evaluation attempts of three laps each
with zero race or raw collisions. A later PPO continuation initialized from the
same checkpoint but became collision-heavy. A fresh one-car deterministic
BC-only diagnostic on the Linux runtime has now also completed three laps with
zero collisions.

The current PPO run is `4a394bb27503` (`1M PPO continuation from Porto BC model
(6 envs)`). Before the controlled BC-only test, preserve its logs and stop it
if it is still running so it cannot compete for simulator/GPU resources.

## Model lineage

1. A CenterlineExpert teacher was used to collect demonstrations on the Porto
   practice route.
2. The accepted collection contains 20,494 transitions and 10 clean laps over
   two fresh five-lap episodes, with zero collisions and no telemetry/control
   faults.
3. Behavior cloning fit the sensor-only policy for 8 epochs with batch size
   256 and learning rate `0.001`. The final validation loss was
   `0.0000844296`.
4. The fitted checkpoint is:

   `logs/rl/porto_bc_20261030_fit/behavior_cloned_policy.zip`

   SHA-256:

   `bb815d3612cecbb8f8bdf9936ec98c8da4c689b0551a33f5b52b31051cbae2c4`

## Accepted BC evaluations

The checkpoint was evaluated in three separate fresh native Windows official
practice simulator processes. Each completed three timed laps with zero
collisions, valid lap timing, fresh camera input, valid route projection, and
no monotonic control-gap faults.

| Evaluation | Lap times (seconds) | Steps | Collisions | Result |
| --- | ---: | ---: | ---: | --- |
| `eval2_neutral1` | 176.933, 94.924, 94.665 | 4,907 | 0 | Passed |
| `eval3_neutral1` | 141.997, 94.439, 94.896 | 4,995 | 0 | Passed |
| `eval4_neutral1` | 139.807, 96.631, 94.627 | 4,970 | 0 | Passed |

These are practice diagnostics, not a competition score. The complete source
report is [`porto-bc-results-2026-10-30.md`](porto-bc-results-2026-10-30.md).

## PPO continuation attempt

The BC checkpoint was used as the starting policy for run `4a394bb27503`:

- 6 environments in parallel
- PPO rollout size: 2,048 steps per environment, 12,288 transitions per update
- learning rate: `0.0001`
- entropy coefficient: `0.01`
- bidirectional throttle/action settings
- collision penalty: `-200`
- Linux external simulator backend

At the latest captured telemetry point, the run had completed 4 valid laps and
had a best lap of 12.161 seconds, but had also accumulated 187 collision-ended
episodes and zero clean ten-lap episodes. The recorded lap times were 12.343,
12.161, 76.717, and 18.174 seconds. PPO episode rewards remained strongly
negative and explained variance was approximately zero.

## Working diagnosis

The BC checkpoint itself has positive evidence: it can drive complete laps
slowly and cleanly. The regression is associated with the PPO continuation and
its runtime conditions, not with the existence of the checkpoint.

The most likely mechanism is that PPO immediately samples exploratory actions
and updates the policy from collision-heavy rollouts. Six parallel environments
and the large collision penalty provide a noisy early learning signal, so the
policy can move away from the stable imitation behavior before PPO has learned
a useful correction. The Linux external runtime also differs from the native
Windows runtime used for the accepted BC evaluations and should be treated as a
secondary variable to verify.

## BC-only Linux diagnostic — 2026-11-05

The BC checkpoint was loaded directly with deterministic inference. No PPO
updates, exploration, teacher actions, or policy changes were applied. One
fresh Linux practice simulator was used.

| Laps | Lap times (seconds) | Total | Collisions | Result |
| ---: | ---: | ---: | ---: | --- |
| 3/3 | 106.848, 97.418, 94.248 | 298.514 | 0 | Completed |

The evaluator recorded fresh camera samples on all 3,024 control steps, valid
route projection on all samples, and verified vehicle motion. It recorded one
control-gap diagnostic and one ROS timestamp gap, but no collision, camera
freshness, or route-projection failure. This confirms that the BC checkpoint
still drives cleanly on the current Linux runtime; the PPO continuation is the
clear source of the collision-heavy regression.

Evidence: [`evaluation.json`](../logs/rl/official_run_porto_bc_linux_diag_20261105/evaluation.json)
and [`relay.log`](../logs/rl/official_run_porto_bc_linux_diag_20261105/relay.log).

## Next controlled experiment

Run the original BC checkpoint without PPO training or policy updates:

1. Use one fresh Porto practice simulator.
2. Load `behavior_cloned_policy.zip` directly.
3. Use deterministic inference and the same neutral-start safety procedure as
   the accepted evaluations.
4. Record lap count, lap times, collisions, route projection, camera freshness,
   and termination reason.

Interpretation:

- If BC-only succeeds again, PPO continuation/exploration is the primary
  failure mechanism.
- If BC-only fails on Linux but succeeds on Windows, the runtime/platform or
  simulator setup is the primary difference.
- If BC-only fails in both environments, re-check checkpoint loading,
  observation preprocessing, action scaling, and the neutral/start lifecycle.

## Evidence index

- Accepted results: [`porto-bc-results-2026-10-30.md`](porto-bc-results-2026-10-30.md)
- BC checkpoint: `logs/rl/porto_bc_20261030_fit/behavior_cloned_policy.zip`
- Current PPO telemetry: `logs/rl/official_run_4a394bb27503/latest_telemetry.json`
- Current PPO lap/collision events: `logs/rl/official_run_4a394bb27503/race_metrics.jsonl`
- Current PPO trainer log: `logs/rl/official_run_4a394bb27503/hub_train.log`
