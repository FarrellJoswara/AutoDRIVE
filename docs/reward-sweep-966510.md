# Reward comparison from PPO step 966,510

Authorized November 6, 2026. All six candidates start from the same completed
optimizer update (15) of run `11e94f5da0ce`, not its partial-rollout final save.

Checkpoint: `logs/rl/official_run_11e94f5da0ce/checkpoints/post_optimizer_update_00000015_steps_000000966510.zip`

SHA-256: `8527be92e5e08b2d9edd323b47c0846e87cac6574f620f7681b540b5a7c215b5`.

## Screening

- Six independent policies: episode-average frontier pace strength 0, 0.5, 1.0
  crossed with time cost 0 or 0.02 reward points per simulated second.
- Centerline shaping zero in every candidate, including control.
- Collision and failure penalties 250 for every candidate. Other PPO settings
  remain as approved; seed 0, CUDA, 2048 rollout steps per environment.
- Three trainers concurrently, two isolated simulator environments per trainer.
- Each candidate receives 53,248 additional transitions, 13 optimizer updates.
  Screening total 319,488. Baseline evaluation shared by all candidates.
- Train six updates, stop all trainers in that batch and evaluate immutable
  checkpoints, then resume for seven updates and evaluate again. Evaluations
  use fresh simulators, deterministic policy actions and no exploration noise.
  Existing training-probe protocol is one warmup plus three scored laps;
  these are practice probes, not official competition records or ten-lap tests.

## Metrics and selection

Store exact configurations, checkpoint hashes, parent lineage, per-lap/episode
results, collisions, failures, sampled speed/action/reward/transport telemetry,
PPO finite-state audits, throughput, system usage and trajectory clips. Report
unavailable measurements as missing, never zero. Compare speed profiles and
lap distributions; do not compare raw reward totals across reward formulas.

Advance strongest safe configurations for longer confirmation. Continuation
under another seed measures robustness but is not an independent replication
from the common starting checkpoint. Require clean completion and held-out
improvement; if no reward beats control convincingly, retain the control.

After a documented selection, resume the best validated checkpoint with its
reward configuration and six parallel environments, total_timesteps=0,
stop_on_plateau=false, max_duration_hours=0. Save each optimizer update and
continue metrics, trajectory recording, and periodic evaluation. Automated
evaluation must use a verified cooperative pause protocol or explicit staged
checkpoint/evaluation/resume; never queue evaluations against trainers that
cannot acknowledge a pause.

Durable execution state and reports live under `logs/rl/reward_sweep_966510/`.
Wall-time estimates exclude confirmation and serialized evaluation overhead.

## Execution

Shared Porto baseline `2b978ace17ff4964` passed all nine laps with zero
collisions. Median across all laps: 30.9546 s; median flying lap: 30.7957 s.
The first three screening runs are `5294d28a5086` (pace 0, time 0),
`ac23d7c3e9ec` (pace 0, time 0.02), and `72dfacfcb4fb` (pace 0.5, time 0).
Each verified two environments, CUDA, a 4096-transition update and zero
centerline shaping. Detailed per-run sample recording is enabled.

`scripts/run_reward_sweep.py` orchestrates batches, midpoint/final practice
evaluations, independent seed-1 confirmation of the top two plus control,
and automatic unlimited six-environment continuation. A proposed reward must
improve the held-out lap median by at least 2% against both the initial model
and confirmed control; otherwise the clean validated control is retained.
This is a practical selection threshold, not a statistical confidence bound.
Invalid audits/configurations/lineage stop orchestration for investigation.

Production has no training step limit. Approximately every 98,304 new
transitions the orchestrator checkpoints/stops, evaluates in fresh practice
simulators, then resumes six environments. If a checkpoint fails the clean
gate, it resumes the last validated model. This avoids the unreliable live
pause path while preserving periodic evaluations. No manual winner approval
is required; the user authorized the complete workflow.
