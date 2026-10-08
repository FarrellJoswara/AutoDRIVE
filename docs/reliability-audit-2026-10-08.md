# Reliability audit — 8 October 2026

## Diagnosis and phased plan

The repository mixed official ROS training with custom-simulator startup and replay. Lifecycle handling and UI status did not consistently match the official trainer. Independent runs did not provide parallel environments to one learner. Existing user changes were retained; no commits or resets were performed. History through HEAD `6bc1cc3` was inspected, including official Watch, SIGTERM classification, warmup reward, and replay work.

1. **Lifecycle first.** Evidence: frontend type errors, shell signal/wait behavior, and an actual resumed trainer losing ROS context during Stop. Intended result: cooperative saving and accurate terminal status. Verification: focused tests plus actual UI start, resume, stop, checkpoint, and resource cleanup.
2. **Official workflows.** Evidence: custom replay and startup remained reachable. Intended result: official Train and Replay with saved observation/action semantics. Verification: real container image/argument inspection, checkpoint loading and sensor telemetry; mocked orchestration checks are reported separately.
3. **Learning contract.** Evidence: shared observation/action builders separate restricted race metrics from policy inputs, but ROS snapshots use host monotonic time. Intended result: measured action-to-frame, reset, reward and rollout alignment. Focused checks exist; load-dependent timing validation remains outstanding.
4. **Usability and monitoring.** Evidence: duplicate global controls, confusing settings text, final progress stuck at sampled counts. Intended result: run-specific actions and understandable status. Verification: browser interactions, live telemetry and terminal state. Broader responsive/accessibility work remains.
5. **Parallel PPO.** Evidence: multiple standalone runs were not a shared learner. Intended result: one PPO model with subprocess environments and separate official simulator/ROS domains. Verification: two real environments and one rollout buffer; controlled 1/2/4-environment benchmarks remain necessary before claiming speedup.

The user approved official-only workflows, one parallel learner, and an isolated Docker-socket verification hub on localhost:8091.

## Executed path and changes

The launcher now prepares official images and starts the hub. Custom Compose simulation is opt-in. The manager snapshots settings, creates an isolated network and official API/simulator pairs. The primary API runs the learner; additional APIs run bridges only. Each environment chooses its ROS domain before initialization.

Layer 1 waits for official sensor readiness and converts ROS messages to telemetry. Layer 2 builds policy observations, maps actions, and computes reward/episode signals from separate race metrics. Layer 3 collects `n_steps * n_envs` transitions per rollout into one PPO model, verifies effective settings, and writes checkpoints. Layer 4 receives run-scoped telemetry and reconciles persisted completion records with Docker state.

Stop signals the learner while sensors remain available, allows saving, then cleans up bridges, simulators and network. ROS no longer replaces the trainer's signal handlers. Vector cleanup is bounded. Replay filters for official checkpoint provenance and uses recorded observation/action settings in a fresh official evaluation. Deterministic policy actions do not imply deterministic simulator timing.

Mission Control now has run-specific controls, a visible creation form, parallel-environment settings, clearer seed/timeout descriptions, and an official Replay result panel. Completed progress uses the saved transition count instead of the last telemetry sample. Official runs no longer inherit unrelated custom-map overlays.

## Actual simulator evidence

- Direct parallel smoke: 128 transitions from two official environments, one rollout of 64 * 2, checkpoint saved, exit 0. About 35 aggregate transitions/s was reported for that short rollout; this is not a controlled benchmark. Artifacts: `logs/reliability-audit/parallel-smoke` and `parallel-trainer.log`.
- Initial resumed Stop failed because ROS replaced signal handlers; a recovery checkpoint was saved. This motivated the signal fix. Artifacts: `resume-stop` and `resume-stop.log`.
- UI run `2314aa8ef102`: two official environments, 256 transitions, periodic/final checkpoints and `timestep_limit`. The last telemetry count was only 200, exposing the final-progress bug.
- UI resumed run `3d15af9d0205`: loaded the previous checkpoint and saved at 704 transitions after UI Stop with `operator_stop`.
- UI Replay `9245e651f0da`: loaded the saved model in a fresh official pair and displayed live sensor telemetry. Final outcome is appended below.

Final evaluation: `wall_timeout` after the five-minute limit, `score_status=incomplete`, zero completed valid attempts and no adjusted race score. The UI displayed that result after a hub restart. UI shutdown exited the isolated hub with code 0; no managed official containers remained running. Logs and checkpoints were retained.

Final automated checks from the initial audit: 83 focused Python tests passed, TypeScript checking passed, and the production frontend build passed. The timing follow-up adds separate timestamp/fallback and diagnostics coverage; see the latest checks in the task handoff. These tests do not substitute for simulator runs. A preexisting recovery-checkpoint assertion was made platform-neutral for Windows path separators. Replay restart recovery and final progress have focused regression coverage. The shorter simulator cleanup timeout is covered by code checks but has not had another real Stop timing benchmark.

The [verification screenshot](../logs/reliability-audit/mission-control-evaluation.png) records the actual incomplete evaluation panel. Real crash recovery with active learners remains untested; only completed history was verified after a live hub restart.

Verification artifacts are isolated under `logs/reliability-audit`. The image tag `aicar-iros2026:reliability-audit` leaves the normal API image tag intact. Existing user work and experiment logs were preserved.

## Remaining limits

- Elapsed intervals now use advancing ROS LaserScan timestamps, with monotonic receipt-time fallback. Two- and four-environment runs show ROS time tracking host wall time within about 1%. Only two lap-clock comparisons were captured and differed by 3–6%; more completed laps are needed to establish agreement with the official scoring timer, including reset and collision behavior.
- Focused tests cover observations, actions, warmup, collision accounting and restricted-input separation. They do not certify competition compliance or establish a valid ten-lap race.
- Short smoke models establish plumbing, not policy quality.
- Resume applies requested PPO settings with fresh optimizer state; it is not exact optimizer continuation. Replay preserves recorded semantics.
- The four-environment 4,096-transition run completed four PPO rollouts and saved successfully, but this is only a two-minute stability sample. The single-run 1/2/4 comparison indicates local throughput gains; sustained stability, hardware saturation, repeated speedup and concurrent independent runs remain unverified.
- Production Compose retains its existing GPU requirement. The audit used a cached hub image and CPU learner; first-install behavior on other machines was not proven.
- Hub recovery has mocked coverage; real crash injection, disk-full and checkpoint-corruption behavior remain unverified.

## Follow-up clock and throughput measurements

Official `LaserScan` messages in the runtime tests carry advancing ROS header timestamps. Layer 1 now uses those stamps for sensor/action intervals, including elapsed time while PPO updates; missing, zero, or regressing stamps use monotonic receipt time. The training config names the clock source. Each supervised run writes `timing_diagnostics.json` with clock-source counts, per-environment ROS-to-wall ratios, and comparisons against official lap times when laps cross.

- A one-environment 64-step probe used ROS stamps for all 64 intervals.
- A final 64-step probe against the rebuilt image verified the saved config names the clock source; its short 3-second ROS/wall ratio was 1.037, too brief for a stable rate estimate.
- A two-environment 512-step load probe used ROS stamps for all intervals. Each environment accumulated about 14.65 ROS seconds and 14.54 wall seconds (ratio about 1.008).
- The four-environment 4,096-step run completed four 1,024-transition rollouts and PPO updates in 123 seconds. All intervals used ROS stamps; each environment's ROS-to-wall ratio was 1.0010–1.0016. The final checkpoint and timestep-limit record were saved. No clean scored race lap or winning episode completed.
- Two lap comparisons in that four-environment run differed by 3–6%. They are affected by episode-start and lap-boundary accounting and are too few to claim exact official scoring-clock agreement.

Three fresh official runs each collected 512 transitions with the same 64-step PPO rollout configuration and seed, with resume explicitly disabled. PPO reported about 18, 35, and 40 transitions/s at 1, 2, and 4 environments. That is approximately 1.9x throughput at two environments and 2.2x at four on this machine. Four adds little over two. Each setting ran once; the data is a short local comparison, not a hardware-independent guarantee. See `logs/reliability-audit/parallel-benchmark.json` and `parallel-stability.json`.

Mission Control now names the elapsed-time setting the simulator-clock cost/watchdog and explains the ROS-stamp fallback. The remaining timing question is agreement with the official scoring timer across more complete laps and reset/collision cases.

## GPU and interruption follow-up

The official API trainer had been hard-coded to CPU, even though this host's
Docker Desktop WSL2 path exposes an NVIDIA GPU. Official training settings now
offer an explicit CPU/CUDA choice. CUDA selects the optional CUDA PyTorch image
and requests an NVIDIA device only for the one learner; bridge workers remain
CPU-only. The default image and setting remain CPU-compatible. The optional
image was built from the pinned official Devkit base with PyTorch 2.2.2/cu121
(12.74 GB on this host); both `torch.cuda.is_available()` and the device name
were verified inside a GPU-enabled container.

An actual one-environment official run completed two PPO rollouts on CUDA at
128 transitions and wrote both periodic and final checkpoints. A second live
run used two official simulator pairs and one CUDA learner. The hub was killed
and restarted while the run was active; Mission Control recovered the same
running run and four original containers, and training advanced from step
1,000 to 1,476 without restarting the learner. The learner was then killed.
The manager marked the run failed and retained its latest periodic checkpoint
at 1,536 steps. A fresh official run resumed those policy weights and completed
the next PPO rollout at global step 2,048, saving a new checkpoint. Resume
intentionally starts a fresh optimizer; this verifies policy-weight recovery,
not exact optimizer-state continuation. Artifacts are under
`logs/rl/official_run_3ce77d284274`, `official_run_45703e3cf013`, and
`official_run_e4ea93704235`.

A longer four-environment, one-learner CUDA run completed 16,384 transitions,
four 4,096-transition PPO rollouts, and periodic/final checkpoint writes in
278.6 seconds. All eight official API/simulator containers remained healthy
through the run, then were removed during cleanup. All 16,384 action intervals
used ROS sensor stamps. Each environment accumulated about 279.06 simulator
seconds versus 278.61 wall seconds, a 1.0016–1.0018 ratio. This extends the
stability sample to about 4.6 minutes; it is still one hardware/configuration
trial. The run completed no laps, so it provides no lap-timer comparison or
evidence of policy quality. Artifacts are under
`logs/rl/official_run_9b5a0371a200`.

The fresh official race evidence still shows the leading preserved checkpoint
can finish ten scored laps but is disqualified for roughly 30 collisions.
Other saved policies are repeatedly disqualified before a scored lap. The
latest sustained parallel training run also completed zero laps. Therefore the
remaining central product gap is a safe, valid ten-lap learned policy. Sensor
traces show some candidates keeping high throttle when front LiDAR clearance
is below half a meter, but a new hand-designed LiDAR safety shield would change
the learned control contract. This audit does not apply that design without a
matched official-race experiment. No valid ten-lap score has been established.

The additional direct official-simulator tests used `aicar-iros2026:latest`
rebuilt from the current repository Dockerfile. Before rebuilding, this
machine's cached `latest` policy image did not support the `AICAR_MODE=bridge`
entrypoint used for extra parallel environments; its exited worker correctly
failed the run. The rebuilt documented image passed the one-, two-, and
four-environment official runs. This was a stale local image issue; a clean
build from the current Dockerfile contains bridge mode.

## Primary references

- [Official competition](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2026/)
- [Official rules](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-rules-2026/): warmup plus ten scored laps, escalating collision penalties and disqualification.
- [Official guide](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2026/)
- [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.4.1/modules/ppo.html): rollout size depends on steps and environment count.
