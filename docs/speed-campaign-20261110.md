# Speed campaign, 2026-11-10

The previous reward sweep halted with no safely confirmed winner. Its shared
parent PPO checkpoint at step 966,510 passed the Porto practice diagnostic
with 9/9 laps clean, zero collisions, and 30.7957-second median flying lap.
SHA-256: `8527be92e5e08b2d9edd323b47c0846e87cac6574f620f7681b540b5a7c215b5`.

## Adaptation completed

Official run `bad20be1b5d4` resumed that checkpoint with six simulator
environments, 122,880 additional transitions (ten PPO updates), learning rate
`1e-5`, frontier pace bonus strength `0.5`, pace target `6 m/s`, time cost zero,
centerline shaping zero, and fixed collision/failure penalties of 250. It saves
after every update. This bounded phase establishes a zero-centerline parent;
it does not claim that earlier centerline training has been erased. The run
completed at step 1,089,390 with ten finite optimizer audits and clean full
ten-lap training episodes. Its checkpoint passed the required fresh Porto
diagnostic before seeding the comparison.

The fresh diagnostic passed: 9/9 laps clean, zero collisions, 31.8936-second
median flying lap. The earlier 966,510-step parent measured 30.7957 seconds
on the same diagnostic. Three independent two-environment screening runs are
active; the fourth is queued by the durable campaign runner.

## Diagnosis

The prior sweep's sampled median speed was about 1.06–1.17 m/s, and median
throttle command only 0.042–0.048. In bidirectional mode, the action mapping
does not impose a low throttle cap. At 1.1 m/s and a 6 m/s pace target, a
quadratic bonus of strength 1 changes the progress reward by only about 3.4%.
The simulator's full-throttle speed ceiling remains unmeasured; low commanded
throttle alone cannot establish a simulator limit.

## Candidate comparison

After a clean adaptation gate, independently train from exactly the same
authenticated adaptation checkpoint, with zero centerline shaping and equal
PPO settings, environment count, seed, collision/failure costs and action
mapping. Compare the existing pace target 6 m/s control to lower target
1.5 m/s at bonus strengths 0.5 and 1, plus a modest simulator-time cost
candidate. Each candidate gets 13 updates (53,248 transitions) in two
environments. Do not combine an action-mapping change with a reward change.

Evaluate immutable checkpoints on three fresh Porto practice simulator starts
with three complete clean laps each. Record all lap times, collisions, route
and camera validity, command faults, speed/throttle/steering distributions,
PPO finite-state audits, trajectories and resource usage. Rank by safety then
median flying-lap time; do not compare raw training returns across formulas.

Confirm finalists and a control from the common parent with another seed and
longer training. Require repeatable clean diagnostics and clean full ten-lap
episodes before calling a checkpoint production ready. If no candidate
passes, halt and preserve the last verified parent. If one does, resume its
reward configuration with a single PPO model across six environments,
`total_timesteps=0`, no plateau or wall stop, a checkpoint after every update,
and periodic fresh evaluation with staged stop/resume. The existing hub
cannot run Porto evaluations reliably against a trainer with cooperative
pause disabled, so evaluations happen after stopping at update checkpoints.

Luna agents audit speed/action data, implement the configurable pace target,
prepare the durable runner, and independently review evaluation safety.

## Native-domain bridge continuation, 2026-11-10

After the env5 API bridge assignment fix, the native focused suites passed:

- Native readiness, lifecycle, resume, hub startup and training gate: 47 passed.
- Bridge transport, run manager, official workflow/competition, job lifecycle,
  PPO audit, speed runner and backend ownership: 154 passed.

With no active training run and no active evaluation, the hub was restarted so
it loaded the current native backend. The selected checkpoint was verified at
step 1,195,886 (SHA-256
`eb0b921a7133ff0ee20cc33564c15f5d93aa2d11ca43d473cd7043a455e6b70f`). One
isolated six-environment, 12,288-transition probe was submitted as run
`5fc8d65b8c82`; its full request and failure evidence are retained under
`logs/rl/speed_campaign_20261110/probe_domain5/` and
`logs/rl/official_run_5fc8d65b8c82/`.

The probe failed during native simulator startup. Environments 0–4 connected;
environment 5 was refused with `ROS domain 5 required by this environment is
already reserved; refusing to start its bridge on a different domain`. No
trainer sensor-readiness record, rollout progress, optimizer audit, or new
post-update checkpoint was produced. No retry was issued. Campaign state remains
`halted_production_startup`, production cycle 0, and the earlier `production:0`
retry count remains 1. The hub is available; training and evaluation are idle.

## Domain recovery and cycle 1 continuation, 2026-11-10

Hub startup recovered and archived the completed adaptation's all-gone
`api-5` manifest, eliminating its stale domain-5 reservation. The hub was
restarted as PID 2062324 after confirming no active training jobs or evaluations.
The earlier failed one-update probe remains preserved as run `5fc8d65b8c82`;
no recovery archive or checkpoint was created from it.

A fresh isolated probe, run `9fc051aa71df`, resumed the selected confirmed
checkpoint at step 1,195,886 with the production reward values unchanged:
target 1.5 m/s, pace bonus strength 1.0, and simulated-time cost 0.1. All six
sensor workers reported ready on ROS domains 0 through 5, including positive
callback counts for every required topic. The probe completed one optimizer
update at step 1,208,174; all metrics, gradients, parameters, and optimizer
state audits were finite. Its audited checkpoint and evidence are recorded in
`state.json` under `probe_evidence` and in
`logs/rl/speed_campaign_20261110/probe_domain5_recovery_01/` and
`logs/rl/official_run_9fc051aa71df/`.

Campaign state advanced atomically to phase `production`, cycle 1, resuming
from the selected 1,195,886-step checkpoint. The original `production:0`
intent remains in state with both failed attempts, retry count 1, and its retry
exhausted guard intact. Supervisor and monitor checks now treat that prior
cycle as history while continuing to guard the current cycle. The supervisor
runs in tmux session `speed-campaign-supervisor`; the monitor runs every 30
seconds in `speed-campaign-monitor`.

Production cycle 1 is run `c28b1eef5278`, display name
`speed-campaign-speed_campaign_20261110-production-1-target15-strength1-time01`.
It is running with six environments, unlimited `total_timesteps=0`, a
12,288-transition checkpoint interval, and the same three reward values. At
verification, all six sensors were ready and live telemetry had advanced to
step 1,197,902. The campaign monitor reported no alerts. The earlier
`supervisor.log` retry-exhaustion messages predate the cycle 1 launch; the
current supervisor has one live runner and the hub contains exactly one
cycle 1 run with the matching display name.

## November 11 PPO pivot

Production cycles 1, 3, and 4 failed fresh Porto evaluations after starting
from the 1,195,886-step confirmed model. Cycle 3 recorded 95 collision episodes
and two clean ten-lap episodes. Cycle 5 ended with NFS quota exhaustion during
race-metrics fsync. Its latest audited checkpoint was at step 1,527,662.

The new comparison branches twice from the same authenticated 1,195,886-step
checkpoint, whose confirmed Porto median flying lap was 24.768 seconds.
Each trial runs six environments for 24,576 transitions. Both use a learning
rate of 3e-6, two PPO epochs, target KL 0.003, and zero centerline shaping.
Trial 1 keeps the previous pace and time rewards. Trial 2 halves the pace
bonus strength to 0.5 and removes the per-second time cost. Both get a fresh
three-attempt, three-lap Porto diagnostic.

Storage recovery moved the older 4b01951505ec checkpoint set to private
host-local storage on cycle3. A hash manifest is stored at
`/var/tmp/f1tenth-storage-recovery-4b01951505ec/recovery-manifest.json`.
The checkpoint and sidecar required by the active model ancestry were restored
to their original physical paths. Other historical snapshots retain links to
the archive and can be restored to their original directory for evaluation.
The 512 MiB write and fsync probe passed after relocation.
