# 08 — Reliability / Chaos (Resilience · Fingerprints · Honest Eval)

> Persona: reliability / chaos engineer.  
> Lens: **does the system tell the truth under failure?** Start/Stop races, half-written checkpoints, frozen `live_status`, contract mismatches, and dashboards that look green while the car is dead or the metric is laundered.  
> Scope from `IDEAS_MEGA.md`: Resilience/chaos (§), Metrics fingerprinting, Eval reproducibility; adjacent Start/Stop, Watch map mismatch, contracts ABI.  
> **No code** — triage only.

---

## Verdict in one line

Ship **truth-preserving control-plane + artifact integrity + sealed eval** before any prettier curves, W&B mirrors, or chaos theater that doesn’t catch a lying green light.

---

## Must-fix (control plane & truth)

These are not “nice chaos drills.” They are the failure modes that make every other metric untrustworthy.

### 1. Start / Stop ownership & races
- **Double-Start**, **Stop→Start** races, and **two UIs / processes fighting one `runs/` dir** must be impossible or loudly fatal.
- Clear **UI-owned vs external-train** ownership; Stop consequences in plain English (what dies, what is preserved).
- **Two trains same `run_id`** must be detected *before* mutual zip corruption — not after leaderboard pollution.
- Sleep/wake / orphaned PID / “Stop left a zombie that still writes status” is a control-plane bug, not an ops anecdote.

### 2. Checkpoints (atomic or don’t promote)
- **Corrupt / truncate / half-write** JSON and zips: **never promote** partial `best_model`.
- **Atomic checkpoint rename** under torture; resume prefers **last complete** artifact only.
- Disk-full mid-checkpoint must leave a recoverable state (incomplete marked incomplete), not a “best” that loads as garbage.
- Dual Best / Latest / Checkpoints theater is secondary until multi-day runs exist — integrity of *one* durable artifact comes first.

### 3. `live_status` honesty
- Heartbeat must prove **gradients / training progress exist** — not just HTTP/poll green or file mtime twitching from a stuck writer.
- Detect **frozen `live_status`**, **dead PID**, orphaned trains; status banner must match process reality.
- Subproc worker kill / straggler / Dummy fallback must **match status reality** (don’t claim N workers when you’re on DummyVecEnv of 1).
- Map mismatch banner: Watch map A vs train map B; **bridge-down ≠ gym-train-down**.
- Conspiracy checklist baseline: PID, mtime, step count, file size, wheels actually moved — enough to catch “green while dead.”

### 4. Contracts mismatch = hard refuse
- Freeze obs contract early (beams, units, FOV, frame, **version hash**); bridge remap = breaking ABI.
- **Contracts / obs-dim mismatch → hard refuse load before predict** (Watch, race, continue-train).
- Crash taxonomy must include **contract mismatch** as a first-class failure, not a silent reshape.
- Calibration gate later: bridge LiDAR stats outside gym DR envelope → refuse deploy (related integrity, not UI polish).

---

## TOP 15 (ship order for “no lying dashboards”)

| # | Idea | Why it ranks |
|---|------|----------------|
| 1 | Double-Start / Stop→Start / dual-UI / same-`run_id` collision guards | Prevents corrupted runs and false “training” |
| 2 | Atomic checkpoints; never promote partial best/latest | Stops silent weight rot |
| 3 | Frozen / dead / orphan `live_status` + PID truth | Green light without progress is the #1 lie |
| 4 | Contracts/obs-dim mismatch hard-refuse before predict | Deploy/Watch ABI bombs |
| 5 | Immutable config fingerprint (hparams, git SHA, torch/CUDA, VecEnv, map hash, contracts ver) | Makes every curve attributable |
| 6 | Separate **eval metric** from train reward; fixed checkpoint selection (no test fishing) | Kills reward-as-sport dashboards |
| 7 | Frozen held-out protocol: seeds, episode budgets, pre-registered primary metrics | Reproducibility that can’t drift quietly |
| 8 | Mean ± CI (≥5 seeds) + tails (worst-decile / survival), not only means | Medians hide crash policies |
| 9 | Crash taxonomy (wall, spin, stall, NaN, **contract mismatch**) | Explains “why bad” without vibes |
| 10 | Subproc/Dummy/straggler status must match reality | Parallelism tax vs fake n_envs bragging |
| 11 | Map mismatch / bridge-down ≠ gym-train-down banners | Wrong world, wrong diagnosis |
| 12 | Throughput + reward skew across n_envs (parallelism tax) | Detects reward laundering / dead workers |
| 13 | NaN obs isolation; quarantine empty/partial zips | One bad env poisoning the batch |
| 14 | Deterministic **and** stochastic eval both reported | Stops “looks great with noise off” |
| 15 | Stress eval tails (jitter, latency, occlusion) + null baselines | Prevents “sexy Watch on map0” claims |

**Just outside Top 15 (still KEEP soon):** leaderboard with CI bands / beat-FTG Δt; overfit detectors (scrambled LiDAR bins); one-command “reproduce Table 1” CI; artifact lineage DAG; W&B mirror of heartbeat (after local truth works).

---

## KEEP

### Resilience / chaos (core)
- Heartbeat that proves gradients exist — not poll-green alone.
- Frozen `live_status` / dead PID / orphaned trains / dual-UI same runs dir detection.
- Corrupt / truncate / half-write JSON & zips; never promote partial `best_model`.
- Atomic checkpoint rename torture; resume → last **complete** artifact.
- Contracts / obs-dim mismatch → hard refuse before predict.
- Subproc kill / straggler / Dummy fallback status must match reality.
- Map mismatch banner; bridge-down ≠ gym-train-down.
- NaN obs isolation; antivirus/quarantine empty zip detection.
- Double-Start, Stop→Start races, sleep/wake, clock skew, disk-full during checkpoint.
- Conspiracy checklist: PID, mtime, step count, file size, wheels moved.
- VecEnv reward-laundering / early-truncate participation trophies / max-range “open field forever” detectors.
- Two trains same `run_id` collision detection before mutual zip corruption.

### Metrics fingerprinting
- **Immutable config fingerprint**: hyperparams, git SHA, torch/CUDA, VecEnv type, map hash, contracts version.
- Throughput + parallelism tax: steps/sec, GPU util, reward skew across `n_envs`.
- Full PPO internals with entropy-death / KL / clip / explained-var alerts (training health, not vanity).
- Trajectory / failure-only shards; replay packs (seed + action stream) for bit-replay in Watch.
- Leaderboard artifact with **CI bands** and “beat FTG by Δt” (not raw `ep_rew_mean`).
- Model/dataset cards; **negative-result registry** so leaderboard isn’t survivor-biased.
- `live_status` heartbeat mirrored to W&B **after** local heartbeat is truthful; log ingest for OOM/spawn/CUDA fallback.

### Eval reproducibility (anti–lying dashboard)
- Frozen held-out protocol: fixed seeds, episode budgets, pre-registered primary metrics.
- Mean ± 95% CI over ≥5 seeds; distributional metrics (median, IQR, worst-decile).
- Survival / time-to-crash curves; crash taxonomy including contract mismatch.
- Cross-map transfer heatmaps before claiming “racing skill.”
- Match FTG vs PPO on wall-clock, env steps, and human engineering hours (honesty about cost).
- Null baselines: random, constant-throttle, open-loop; teleop/oracle upper bound.
- Deterministic vs stochastic eval both reported; **fixed checkpoint selection policy** (no test fishing).
- Stress eval: start jitter, opposite direction, occlusion, latency profiles; **report tails not medians**.
- Match RoboRacer-style rules when claiming competition readiness (warm-up ignored, scored laps, DQ collisions).
- Separate eval metric from train reward so shaping doesn’t become the sport.
- Pin never-changing hardware-validation / holdout map packs with hashes (week-to-week compares).
- Mid-run map YAML swap must **fail loud** or remap explicitly.

### Adjacent integrity (KEEP, not polish)
- Clear UI-owned vs external-train ownership; Stop in plain English; “What’s happening now” states including Crashed + reason.
- Session resume: Continue vs restart spelled out (ties to checkpoint integrity).
- Promote-to-race-candidate only after held-out clean evals.
- Freeze obs contract early; match gym beam count/order/FOV to real sensor.
- Overfit detector: scramble LiDAR at eval; if return stays high, you’re not reading rays.
- Partial-obs drills: freeze last scan, inject NaNs/zeros; isolate bad envs.
- Randomize delay/jitter/packet loss as first-class enemies (chaos that teaches truth).

---

## DEFER

- Artifact lineage DAG (run → parent → weights → map pack → eval report) — valuable after fingerprints + sealed eval exist.
- One-command “reproduce Table 1” CI that fails on ε drift — after primary metrics are frozen.
- Pre-register success criteria / multiplicity correction / TOST vs FTG — research-grade; after basic CI bands.
- Human preference thumbs → RLAHF-lite — not reliability-critical.
- Compute cost / CO₂ / FTG-equivalent laps per watt — nice honesty, not control-plane.
- ESTOP / LiDAR-dropout Safe Halt / shadow steward synthetic dropouts — **critical for car**, defer until sim2real path is real; keep gym integrity first.
- System-ID / chronoskepticism (warm vs cold, clock skew as DR) — stretch realism.
- Partner fantasy overnight ranked models; conference booth minimal Start/Stop — growth, not truth.
- Coach panel / grade from reward slope — only after metrics aren’t launderable.
- Ensemble of checkpoints for race selection — after single-artifact integrity is boringly solid.
- Kill Watch / live_status entirely if they don’t move race metric — revisit *after* honesty layer exists; don’t delete the sensor that tells you you’re lying yet.

---

## CUT

### Bad / fun & red-team jokes (do not build)
- Cheerful lying UI (“you’re crushing it” while spinning) — antithesis of this review.
- Emoji-only status / vibes dial / haiku run names / auto-start on page load.
- Voice narrating every timestep; 16 Watch windows “for transparency.”
- Audio telemetry of reward; per-ray saliency every update; keep every crash video forever.
- Replace TensorBoard with a wilting plant.
- NFT of best_model hash / pay-to-name checkpoint / donate to raise exploration noise.
- Full Subproc pipe traces as artifacts; kurtosis of every gradient element.
- Cyber-Physical Integrity Quiz before operator credentialing (bureaucracy as meme).
- Delete best_model if `ep_rew` is “suspiciously round.”
- Maximize Watch chaos / stochastic diversity as if entertainment = podium.
- Chase `n_envs` / batch bragging rights while eval still ranks reward.
- Overfit map0 until Watch looks sexy (temptation to refuse — treat as anti-pattern, not backlog).

### Premature / harmful “features”
- Softmax discrete steer / camera RGB on LiDAR-only contract (ABI sabotage).
- Hot-swap reward mid-run via Discord emoji (destroys fingerprint + reproducibility).
- Minimal-mode toggle / plugin bus / second advanced UI (“simplify by adding”).
- Kill dual best/latest/checkpoints *theater* is good discipline — but **CUT the theater**, not the eventual need for durable checkpoints once multi-day runs are real.
- Online overnight PPO on physical car with no rate limits; OpenCV multi-twin on same Jetson as controller.

---

## Lying-dashboard failure modes (design against these)

| Lie | Detector / antidote |
|-----|---------------------|
| UI green, train dead | Frozen status + PID + step monotonicity + gradient heartbeat |
| `ep_rew` climbing, still useless | Separate eval metric; adjusted_time / clean-lap; null baselines |
| Best model “promoted” | Atomic write + refuse partial zip; quarantine empty archives |
| n_envs = 32 “parallelism” | Reward skew / dead-worker / Dummy fallback truth in status |
| Watch looks great | Held-out maps + scramble-LiDAR overfit test + stress tails |
| Continue-train loads wrong brain | Contracts version + fingerprint gate before predict |
| Bridge down blamed on policy | Explicit bridge-down ≠ gym-train-down |
| Leaderboard heroics | CI bands, negative-result registry, fixed checkpoint policy |
| Same run_id double-writer | Collision detection before zip corruption |
| Map swap mid-run | Fail loud; hashed map pack in fingerprint |

---

## Chaos drills worth institutionalizing (KEEP as tests, not features)

Run these as **regression / soak tests**, not product knobs:

1. Kill train mid-checkpoint write → resume must ignore incomplete.
2. Double-click Start / Start while Stopping → one owner or hard refuse.
3. Truncate `live_status.json` / stall writer → UI shows Crashed/Stale, not Learning.
4. Load v1 zip into v2 contracts → refuse before first `predict`.
5. Kill one Subproc worker → status and throughput reflect reality.
6. Disk-full during save → no silent “best” promotion.
7. Watch on map A while train on map B → banner, not quiet wrong world.
8. Inject NaNs in one env → isolation, not batch poison.
9. Two processes same `run_id` → collision fail before mutual corruption.
10. Eval with fixed seeds twice → metrics within ε (repro smoke).

---

## What “done” means for this lane

1. **Start/Stop** cannot lie about ownership or leave dual writers.
2. **Checkpoints** are complete or invisible; resume never loads half-zips.
3. **`live_status`** stale/dead/orphaned is a first-class failure state.
4. **Contracts mismatch** never reaches `predict`.
5. Every reported number carries a **fingerprint** and a **sealed eval** path that can’t quietly swap metric, map, or checkpoint.

Until those five hold, extra dashboards are decoration on a falsehood.
