# Official PPO campaign operations

This note describes the verified campaign path and its recovery rules. Campaign artifacts and checkpoints live under `logs/rl`; they are runtime evidence, not source files to include in a source commit.

## Training and checkpoint accounting

The authorized campaign policy is `unbounded_until_readiness`. The original UTC/monotonic clock and earlier events remain in the append-only campaign state; the user override is a later amendment, not a reset. Training does not freeze merely because elapsed time or a probe count reaches a threshold. Final validation requires a separate explicit readiness recommendation bound to the candidate and evidence.

Campaign PPO uses one environment and 2,048-step rollouts. Post-optimizer checkpoints and automatic probes are scheduled on global 10,000- and 50,000-transition buckets. A bucket is a trigger, not necessarily the saved model's exact timestep: PPO can publish only after finishing its rollout and optimizer update. Read the sidecar's `trigger_step` and `num_timesteps` separately, and use the full checkpoint SHA-256 and authenticated lineage when selecting or resuming a model.

The campaign configuration is recorded in each run's effective `config.json`: learning rate `3e-5`, target KL `0.01`, `bc_campaign` exploration, initial standard deviation `[0.01, 0.05]`, bounds `[0.005, 0.02]` to `[0.01, 0.05]`, and the fixed PPO settings (`n_steps=2048`, batch 64, four epochs, gamma `0.9995`, GAE lambda `0.95`, clip `0.1`, entropy coefficient `0`, value coefficient `0.5`, gradient norm `0.5`). Reward policy stays at its baseline unless a separately declared, calibrated experiment is authorized and its declaration hash is linked through every descendant.

## Evaluation pause and resume

When an evaluation is queued, the trainer pauses at the next completed rollout boundary, after the optimizer update. It emits a neutral actuator command, saves and authenticates the post-update checkpoint, then writes the pause acknowledgement. Mission Control retires the training simulator, runs the evaluation on a fresh simulator, and waits for new simulator frames and fresh ROS telemetry before sending the matching resume request. The trainer resets its vector environment and last observation before continuing, so a transition cannot join the old simulator state to the new one.

The callback allows up to 9,000 seconds for evaluation completion and receipt of the matching resume request; that timer starts after the pause acknowledgement. The separate 120-second fresh-telemetry window starts only after the resume request arrives. Both waits check operator stop. A timeout, stale sensor stream, or failed evaluation is infrastructure evidence: preserve its report and checkpoint lineage, do not count it as a driving regression, and do not weaken freshness checks. A neutral command being published is logged evidence; it is not a physical actuator acknowledgement.

For an operator stop, use the normal Mission Control stop path. Confirm the durable terminal context, neutral-release record, process cleanup, and latest authenticated post-optimizer checkpoint before starting another run. Do not send process signals to pause a supported trainer.

## Competitive route verification

The 650,000-trigger candidate is a validated benchmark, not a campaign winner. The checkpoint is `logs/rl/official_run_2777e0310250/checkpoints/automatic_evaluation_000004_steps_650000.zip` (SHA-256 `dd7f148d368dc4b0631236bbc753ace0262c01e6a85fdab069a1870097df1fde`). Three standard competitive evaluations each used one warmup and ten scored laps. All 30 scored laps were finite, with zero warmup, race, and raw collisions; route projection, official scene/route hashes, simulator process identity, and camera freshness were verified. The median across the 30 scored lap times was 18.5324506759644 seconds per lap. The create-only evidence bundle `logs/rl/official_campaign/competitive_benchmark_650k_repeat_terminal_evidence.json` (SHA-256 `86e50b2c41a28678fcae10dc04dcbb8a59848dde48a095a690d55a59ca5291df`) binds the raw reports, event journal, and the subsequent cooperative pause/resume with fresh ROS telemetry and a finite post-update checkpoint at global step 720,896. The source reports are:

- `logs/rl/official_run_d5d2b7e88852/evaluation.json` (SHA-256 `8071f7e242bbe07f3b55ce94da51a5f467ed6a622eb628ebcc9f1d1cf1169daa`)
- `logs/rl/official_run_ffd510f6351e/evaluation.json` (SHA-256 `9c06cba14e7dceb68fa1cd7ad6b22bedad134b00431c39339376030b007ec0f6`)
- `logs/rl/official_run_3a80ae01ebc6/evaluation.json` (SHA-256 `8dafc43ae6d76780bcb2d9146f823abcdeef279420a44967c62edf75fe95f55d`)

This benchmark confirms the instrumented competitive path and is one candidate measurement. Continue authenticated probes and comparisons; make a final candidate decision only through the readiness and closeout evidence gates. Do not infer visual UI acceptance from API tests: the browser-based Mission Control review remains open.
