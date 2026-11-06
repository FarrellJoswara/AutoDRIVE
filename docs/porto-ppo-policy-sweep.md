# Porto PPO policy sweep

This is a sequential research sweep from the verified Porto BC checkpoint. Each
candidate uses one fresh simulator environment and starts from the same BC
parent; candidates are not parallel and are never chained from a failed policy.

## Parent and gate

Parent checkpoint:

`logs/rl/porto_bc_20261030_fit/behavior_cloned_policy.zip`

The parent baseline completed three deterministic laps in 298.514 seconds with
zero collisions on the current Linux runtime. A candidate is retained only if
it completes the deterministic diagnostic without a collision and does not
show a clear regression against that baseline. A collision or infrastructure
fault stops that candidate and preserves its artifacts for comparison.

## Initial candidate matrix

| Candidate | Profile | PPO learning rate | Target KL | Action-noise bounds | Purpose |
| --- | --- | ---: | ---: | --- | --- |
| `bc-smoke` | `bc_smoke` | `1e-5` | `0.01` | throttle `0.01`, steering `0.05` | safest first update |
| `bc-campaign` | `bc_campaign` | `3e-5` | `0.01` | throttle `0.005–0.01`, steering `0.02–0.05` | controlled campaign schedule |
| `lr-5e-6` | `standard` | `5e-6` | `0.005` | throttle `0.005–0.01`, steering `0.02–0.05` | ultra-low update size |
| `lr-1e-5` | `standard` | `1e-5` | `0.01` | throttle `0.005–0.01`, steering `0.02–0.05` | low-rate control |
| `lr-1e-5-tight-noise` | `standard` | `1e-5` | `0.005` | throttle `0.005`, steering `0.02` | minimum exploration |
| `lr-3e-5` | `standard` | `3e-5` | `0.01` | throttle `0.005–0.01`, steering `0.02–0.05` | upper conservative rate |

The first pass uses one 2,048-transition rollout per candidate. It is a policy
stability screen, not a claim that a candidate is competition-ready. A passing
candidate can receive a second small rollout only after the first deterministic
diagnostic passes.

## Execution rules

1. Stop the active candidate before launching the next one.
2. Evaluate each candidate deterministically after training.
3. Never resume a failed candidate for another trial.
4. Keep the BC parent untouched and use it as the parent for every branch.
5. Record training config, checkpoint hash, lap times, collisions, and stop
   reason in the candidate directory.

The sweep is exploratory. The best candidate is the best passing branch under
this gate, not automatically the new production model.

## Measurement protocol

Each candidate record must include:

- parent checkpoint path and SHA-256;
- candidate configuration and effective configuration after profile defaults;
- simulator backend, runtime image identity, executable identity, route hash,
  ROS domain, and simulator port;
- PPO rollout count, transitions, optimizer updates, wall time, FPS, loss,
  approximate KL, clip fraction, entropy, action standard deviations, value
  loss, explained variance, and episode reward;
- deterministic evaluation lap times, completed laps, collisions, collision
  type, route-projection validity, camera freshness, control-gap faults,
  simulator timestamp gaps, path length, and motion verification;
- stop reason, artifact paths, checkpoint SHA-256, and the keep/reject decision.

The BC-only Linux diagnostic is the control observation: three laps in
298.514 seconds, lap times `106.848`, `97.418`, and `94.248` seconds, zero
collisions, 3,024 fresh camera samples, valid route projection on every sample,
and one control-gap diagnostic. It is not included as a PPO candidate.

## Interpretation rules

Infrastructure failures are recorded separately from policy failures. A policy
failure is a collision, repeated wall contact, invalid route tracking, or a
material loss of lap completion relative to the BC control. A candidate that
passes the first screen is not declared better until it also improves pace or
maintains clean laps across a second fresh evaluation.

No candidate is allowed to overwrite the BC checkpoint. Every candidate is an
independent branch so that the sweep can be replayed and the best passing model
can be selected after the fact.

## Results

### Candidate `bc-smoke` — completed

Training branch: `logs/rl/ppo_sweep_bc_smoke_20261105/`  
Evaluation branch: `logs/rl/official_run_ppo_sweep_bc_smoke_20261105_eval/`

Effective PPO settings were one environment, one 2,048-transition rollout,
learning rate `1e-5`, target KL `0.01`, clip range `0.1`, entropy coefficient
`0`, four epochs, and `gamma=0.9995`. PPO stopped the update at the KL guard
(`step 0`, maximum KL `0.02`) and saved the candidate checkpoint.

Deterministic evaluation completed all three laps: `95.313`, `84.971`, and
`86.020` seconds; total `266.305` seconds; zero collisions; zero control-gap
faults; zero ROS timestamp gaps; 3,432 fresh camera samples; and zero invalid
route-projection samples. Decision: **pass stability gate; retain for later
comparison**. This is promising, but it is not yet declared better than the BC
control because it has only one PPO rollout and one fresh evaluation.

### Candidate `bc-campaign` — rejected before training

The standalone trainer did not apply the documented `bc_campaign` preset. Its
effective configuration resolved to learning rate `1e-4` with no target KL,
which violates the safety requirements for this sweep. The process was
interrupted during the first rollout, its recovery checkpoint was preserved,
and no deterministic evaluation was run. Decision: **reject configuration;
do not use this branch as policy evidence**. This exposed an important
reproducibility issue: the campaign profile is enforced by the Mission Control
campaign path, not by the direct trainer invocation used for this research
runner. Future candidates must verify their effective configuration before any
rollout is accepted.

### Candidate `lr-5e-6` — completed

Training branch: `logs/rl/ppo_sweep_lr_5e6_20261105/`  
Evaluation branch: `logs/rl/official_run_ppo_sweep_lr_5e6_20261105_eval/`

The effective configuration verified as one environment, learning rate
`5e-6`, target KL `0.005`, standard PPO clip `0.2`, entropy `0.01`, and explicit
noise bounds of throttle `0.005–0.01` and steering `0.02–0.05`. The update hit
the KL guard after the first rollout and saved successfully.

Deterministic evaluation completed three laps: `100.577`, `88.514`, and
`88.794` seconds; total `277.885` seconds; zero collisions; zero control-gap
faults; zero ROS timestamp gaps; 3,518 fresh camera samples; and zero invalid
route-projection samples. Decision: **pass stability gate; retain for later
comparison**. It was slower than `bc-smoke` in this single evaluation but
remained clean.

Checkpoint hashes:

- `bc-smoke`: `91c0339c8645aadcd03d4d8d4826bf6ea201918843d0bc176938ac3bc2438863`
- `lr-5e-6`: `c809352798fb46a7e11b239e774fd20a617b0398306a88535ccb174d388f1721`

## Environment-scaling experiment

The next controlled comparison uses the same untouched BC parent and the same
explicit conservative PPO settings at `n_envs=1`, `2`, `4`, and `6`. Each run
gets one rollout per environment, then its saved policy is evaluated in fresh
deterministic practice sessions. The one-environment `bc_smoke` result remains
the reference; multi-environment branches use explicit settings so the trainer
cannot silently force the `bc_smoke` preset back to one environment.

The comparison records both per-environment and aggregate metrics. A passing
branch must preserve clean deterministic laps; higher throughput alone cannot
outweigh collisions or route/telemetry faults.

### Results: 2026-11-05

The corrected experiment used the same verified BC parent checkpoint for every
branch (`bb815d3612cecbb8b8df9936ec98c8da4c689b0551a33f5b52b31051cbae2c4`),
the explicit standard PPO profile, CPU inference, seed 0, one 2048-step
rollout per environment, learning rate `1e-5`, target KL `0.01`, four epochs,
and action-noise bounds throttle `0.005..0.010` and steering `0.020..0.050`.
Automatic evaluation was disabled during training. Each checkpoint was then
run in a fresh official Linux simulator session for one deterministic
three-lap Porto practice diagnostic.

| Environments | Transitions | Train FPS | Approx KL | Entropy loss | Eval laps (s) | Mean/best lap (s) | Collisions | Control Hz | Gap/stamp faults | Fresh camera / route-invalid | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 2,048 | 15 | 0.02503 | 6.14932 | 94.403 / 83.559 / 84.070 | 87.344 / 83.559 | 0 | 11.768 | 0 / 0 | 2,942 / 0 | `81c7c7721d0b...b98fdae` |
| 2 | 4,096 | 31 | 0.00940 | 6.14930 | 100.877 / 91.608 / 90.656 | 94.380 / 90.656 | 0 | 11.299 | 0 / 0 | 3,062 / 0 | `30afed8ea0ea...a4079d` |
| 4 | 8,192 | 59 | 0.01577 | 6.14932 | 101.475 / 95.588 / 94.772 | 97.278 / 94.772 | 0 | 8.933 | 4 / 4 | 2,499 / 0 | `a4a264a8ff45...3768c0` |
| 6 | 12,288 | 84 | 0.01015 | 6.14929 | 99.393 / 88.036 / 89.426 | 92.285 / 88.036 | 0 | 11.679 | 0 / 0 | 3,093 / 0 | `4e1d96fab6a3...45a1ff` |

Evidence folders:

- Training: `logs/rl/ppo_envscale_{1,2,4,6}env_20261105/`.
- Evaluation: `logs/rl/official_run_ppo_envscale_{1,2,4,6}env_eval_20261105/`.
- Effective PPO configurations are in each `config.json`.
- Optimizer metrics are in each `ppo_optimizer_audit.jsonl`; timing and
  throughput metrics are in each `timing_diagnostics.json`.

Interpretation: six environments gave the best throughput (84 FPS) and the
best multi-environment evaluation mean (92.285 s), while one environment gave
the fastest overall diagnostic mean (87.344 s). Six environments is therefore
the best scaling candidate, but this does not establish a policy-quality gain
after only one rollout. Four environments showed four control and four ROS
timestamp gap faults, so it should not be promoted until that transport issue
is reproduced and fixed. All four checkpoints completed all three diagnostic
laps without collisions, camera drops, or invalid route projections.

These are `porto_practice_diagnostic` runs. They are valid engineering
diagnostics, but `valid_for_comparison` is intentionally false and they are
not official competitive scores. The next research stage should repeat the
top one or two settings with multiple seeds and substantially more than one
rollout before selecting a production continuation.
