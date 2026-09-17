# CAMPAIGN_RESEARCH.md — Evidence-based guidance (9h)

**Status:** COMPLETE (standing researcher `f4ac2241` — tick W3 confirm, ~2026-09-17 04:09 America/Chicago).  
**Role:** Standing research brief for the orchestrator (resumed ~every 50 min with proposals).  
**Scope:** GPU/CPU, continuous retrain, fast validation, UI ROI, map curriculum, explicit non-builds, MUST Go/No-Go evidence.  
**Sources:** `PLAN.md`, `IDEAS_TRIMMED.md`, `CAMPAIGN_CRITIQUE.md`, `CAMPAIGN_LOG.md`, `CAMPAIGN_BOARD.md`, `research_notes.md`, `ideas_review/{02_speed,campaign_racer,campaign_minimalist,campaign_reliability}.md`, plus mechanisms in `train_ppo.py`, `racing_env.py`, `control_ui.py` / `ui_ops.py`, `live_status.py`, `eval_protocol.*`, `map_pack.py`.  
**Rule:** Docs only — no implementation from this file unless fixing a research-proven bug.

**Protected run (do not kill without Continue restart):** `overnight_soak_20260917_082739`  
Live snapshot (W3 confirm ~04:09): timesteps≈**295k**, phase=`validating` (map3, select_timeout=220, patience=5, warmup_evals=2, no_improve=0), lock pid=**19244**, `collision_first=false`. Sacred knobs still in `config.json`: `race_eval_every=50000`, `select_timeout_s=220`, warmup=2, min_ts=100k, patience=5. Mid-train pause is **honest validating**, not stuck. GPU idle still = CPU-bound ([`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md)).

**Overnight note:** soak keeps `collision_first: false` (Overnight preset). Collision-first MUST = UI expose (W2/W3 shipped) + **parallel A/B crash_rate** — never morph this soak mid-run.

---

## 1. GPU vs CPU bottlenecks (SB3 PPO + SubprocVecEnv)

### What actually burns wall-clock here

| Stage | Where it runs | This codebase |
| ----- | ------------- | ------------- |
| Env step (occupancy LiDAR cast, collision, Frenet progress) | **CPU**, per worker | `RacingEnv.step` / `cast_lidar` in `racing_env.py` |
| VecEnv gather / IPC | **CPU** (+ pipes for Subproc) | `train_ppo` builds `SubprocVecEnv` when `n_envs>1`, else Dummy; fallback sets `vec_env_active` truth |
| PPO forward + update | **GPU if `device=cuda`**, else CPU | `resolve_device()`; model `device=` on construct/load |
| Mid-train race eval | **CPU**, **blocks `learn()`** | `RaceBestModelCallback` → `_race_eval_maps` with `select_timeout` |

Campaign measurement already matches theory: overnight soak at **~252 steps/s**, GPU **partially idle** with tiny VRAM — classic **env-bound** PPO gym ([`CAMPAIGN_LOG.md`](CAMPAIGN_LOG.md), critique TRAP row).

### When “use the GPU” is real flex

GPU work helps **only** when the update phase is a measurable fraction of the step loop:

1. **Rollouts are already fast** (`live_status.steps_per_sec` high and stable) **and**  
2. **Profiler / nvidia-smi** shows high GPU util during `model.learn` updates (not just during idle), **and**  
3. You are not spending most wall-clock in `phase=validating` (blocking race eval).

Then (and only then) consider: larger `batch_size` within `n_steps * n_envs` clamp, slightly wider net, AMP/`torch.compile` as **ablations after env FPS is logged** (`ideas_review/02_speed.md` DEFER).

### When GPU flex is theater

| Theater | Why it fails here |
| ------- | ----------------- |
| Move `RacingEnv` / LiDAR to CUDA | Occupancy grid + raycast is Python/NumPy gym; rewrite is weeks, not 9h |
| Fat MlpPolicy “to fill the 3060” | Inflates predict + Subproc IPC; GPU stays hungry, CPU gets slower |
| AMP / compile before measuring steps/sec knee | Optimizes the short side of the critical path |
| Raise `n_envs` until GPU “looks busy” | Oversubscription → steps/sec **drops**; coach already warns (`coach_hints`: sps&lt;30 with n_envs&gt;8) |
| Watch / TB / OpenCV in the **train** process | Steals cores from workers; UI design keeps train headless |

### Actionable throughput protocol (orchestrator)

1. Log **`steps_per_sec`**, **`vec_env_active`**, **`n_envs`**, crash rate from `live_status` (already written by `LiveStatusCallback`).  
2. Sweep **n_envs ∈ {1,4,8,12,16}** at fixed map/hparams; pick the **knee** (max sps), not max workers.  
3. If Subproc fails on Windows → Dummy fallback is honest; do not pretend Subproc is live.  
4. Map raster size dominates RAM/CPU (`PLAN_PROGRESS`: scale/occupancy cells) — do **not** regen maps in the hot loop (`map_pack` / `trackgen` cache only).  
5. Treat **validating pauses** as scheduled downtime: denser `--race-eval-every` = more GPU theater and less learning.

**Verdict for 9h:** ROI is **n_envs knee + sparse mid-train eval + collision truncate**, not CUDA cosplay.

---

## 2. Continuous / auto-retrain after plateau — pitfalls

### What already exists (reuse, don’t reinvent)

- **Unlimited + early-stop:** `--unlimited-timesteps` requires patience &gt;0; auto-raises patience floor; safety ceiling `UNLIMITED_TIMESTEPS_SAFETY=50M` (`train_ppo` / `ui_ops`).  
- **Patience math:** `early_stop_patience_tick` with **warmup_evals**, **min_timesteps** grace, **meaningful** improve gate (`is_meaningful_race_improvement`) separate from tiny `best_model` promotes.  
- **Overnight preset:** budget OFF, patience=5, eval every 50k, warmup=2, min_ts=100k, `select_timeout=220` (`ui_ops.PRESETS["overnight"]`).  
- **Sacred:** do not regress `MID_TRAIN_SELECT_TIMEOUT_S≥220`, `AUTO_RACE_EVAL_EVERY_FLOOR≥50k`, warmup/min_ts ([`CAMPAIGN_CRITIQUE.md`](CAMPAIGN_CRITIQUE.md) Safety table).

### Pitfall A — Nonstationarity when chaining runs

| Trap | Mechanism | Mitigation |
| ---- | --------- | ---------- |
| Resume mid-curriculum without fingerprint check | Policy sees new reward / maps / DR under same `run_id` | Continue only after `refuse_config_contracts` / obs refuse; fingerprint in `config.json` |
| Auto-spawn **new** run_id on every EarlyStop with changed knobs | Leaderboard becomes a soup of incomparable smokes | Outer loop must **fork explicitly** (`--fork-resume`) or new stamp; never silent hparam morph |
| Mid-run Discord reward dial / lava maps | Forbidden loop in PLAN | Curriculum = **staged, fingerprinted** starts only |
| Train on validation pin / holdouts then “continue” | Poisons `RaceBestModelCallback` selection map | `assert_train_safe` / `start_guard`; UI allow-holdout voids claims loudly |

### Pitfall B — Leaderboard pollution

- Smoke / mid-train / DNF-proxy rows must stay `kind≠official` (`metrics_io.append_leaderboard`).  
- Official append only via protocol path + `assert_seals_intact`.  
- Auto-retrain that auto-appends every EarlyStop as “official” **lies**.  
- Ranking must use `race_score_key` / `row_race_score_key` (finishers ≻ DNFs) — never raw `adjusted_time` DQ proxies or `ep_rew_mean`.

### Pitfall C — Compute waste

Documented burn: unlimited + dense eval + 60s select timeout → EarlyStop at ~40–100k with permanent DNF plateau ([`PLAN_PROGRESS.md`](PLAN_PROGRESS.md) overnight autopsy). Fixed constants exist; **continuous outer loop that ignores them will re-burn nights**.

Outer-loop scaffold rules (campaign log “Keep — scaffold first”):

1. **Do not change** overnight early-stop math inside the trainer for the outer loop.  
2. On `phase=early_stopped`: archive run; optional **one** Continue from last **complete** ckpt OR fork with **declared** curriculum delta.  
3. Cap chain length (e.g. max N auto-restarts / wall-clock budget).  
4. Dual-writer refuse: `train.lock` + `_train_busy` — auto-retrain must not Start while live.  
5. Soft Stop ≠ hard kill: Continue expects lock cleared + complete zip (`find_last_complete_checkpoint`).

**Verdict:** Continuous train is useful as a **supervised outer shell** around existing EarlyStop — not as unbounded self-play against a polluted board.

---

## 3. Faster validation without lying

### Two different products (never conflate)

| Mode | Purpose | Timeout / budget | Board |
| ---- | ------- | ---------------- | ----- |
| **Official** (`official_v2`) | Beat-FTG claims | `timeout_s=400`, `episodes_per_map=5`, seeds `[0..4]`, sealed maps | `kind=official` only |
| **Mid-train select** (`RaceBestModelCallback`) | Promote `best_model`, feed early-stop | `MID_TRAIN_SELECT_TIMEOUT_S=220` (was 60 → permanent DNF) | **Never** official |
| **Progress probe** (proposed) | Cheap “is learning?” signal | Short eps / shorter timeout OK | Label `kind=smoke` or status-only; **no promote, no beat-FTG** |

`eval_protocol.yaml` states why 400 exists: laps need ~180–210 s at 6 m/s; under 60s every policy DNFs and beat-FTG is undecidable. Mid-train 220 is the compromise so finishers can appear without full official cost.

### Honest speed-ups

1. **Fewer mid-train episodes** (e.g. 1 seed × validation map only) — already the spirit of `_race_eval_maps`; keep official at 5×maps.  
2. **Sparser cadence** — floor 50k; Overnight preset uses 50k explicitly. Raising eval frequency “to use GPU” is a lie.  
3. **Progress-only probe** during early DNF regime — report `mean_progress_frac` / crash tags in `live_status`, **do not** write `adjusted_time` as if finished.  
4. **Bootstrap CI on official only** when N≥5 seeds (protocol already has 5 seeds + `eval_spawn_jitter: true` so spread is real). Report mean ± bootstrap percentile of `adjusted_time` **among finishers**, plus DNF rate separately. Do **not** bootstrap a mix of finishers and timeout proxies as one number.  
5. **Matched FTG vs PPO** same protocol_id — `compare_models --official`.

### Ways to lie (forbidden)

- Shorten **official** `timeout_s` below a finishable lap.  
- Promote / early-stop on `ep_rew_mean`.  
- Rank DNF proxy (`timeout + 10·cols`) against finishers by raw float.  
- Drop `eval_spawn_jitter` so 5 seeds are clones (fake CI).  
- Call mid-train 220s race an “official” row.  
- Retune FTG mid-campaign to manufacture a PPO win (protocol pin).

**Verdict:** Fast path = labeled probe + sparse 220s select; truth path = frozen `official_v2`. Bootstrap belongs on the truth path.

---

## 4. UI redesign that helps operators vs cosmetics

### Already shipped (do not re-skin)

Start/Stop ownership, preview, Continue, presets, map gen + thumbs, model load/delete, race-candidate via `race_score_key`, banner phases (`Idle` / `Learning` / `Validating` / `EarlyStop` / `Crashed|Stale`), coach hints, steps/sec + crash rate ([`PLAN.md`](PLAN.md) / `PLAN_PROGRESS` / critique: map gen & timeline **LOW/done**).

### Operator ROI (build only if missing / broken)

| Need | Why | Mechanism to lean on |
| ---- | --- | -------------------- |
| Soft-stop vs hard-kill **copy** | Continue soak still deferred; wrong mental model corrupts zips | One sentence + banner; critique SHOULD #7 |
| Continue-after-Stop soak proof | Overnight survival north star | `_continue_train` → `continue_train_argv` + lock clear |
| Holdout refuse loud in preview | Train-on-seal voids claims | `start_guard` / `holdout_warn` |
| Coach tied to **race KPIs** | Crash rate + sps already; avoid GPU shopping advice | `coach_hints` — keep “not buy GPU” |
| Sectioned knobs + tooltips | Campaign log: expose existing `train_ppo` argv, not new APIs | Tk/HTTP panel already; temper “drastic upgrade” |

### Cosmetics / traps (skip this 9h)

- Watch chrome, ghost polish beyond shipped KPIs (Watch stays **unofficial**).  
- Electron/web rewrite on Windows.  
- Seasons/XP/emoji status.  
- Second “minimal mode” UI / plugin bus.  
- Health traffic lights that nag GPU % without steps/sec context.  
- Auto-open Watch/TB in train critical path.

**Design test:** If the change does not reduce **wrong Starts**, **lost Continues**, or **misread Validating/EarlyStop**, it is cosmetics.

---

## 5. Map autogen + autotrain curriculum — best practices

### Infrastructure already correct

- Roles: `train_ok` / `validation` / `holdout` in `map_pack` + `eval_protocol.yaml` (`map0/map1` train; `map3` validation pin; `map2/map4` sealed).  
- Seals = `content_hash` (yaml+image+centerline+start_pose); `verify_pack` / `assert_seals_intact` before official claims.  
- `trackgen` / `map_pack generate`: cache missing ids; per-index RNG so growing pack does not shift old geometry.  
- Train diversity: `--map map0,map1` rotates across `n_envs` in `train_ppo`.  
- Spawn jitter + collision-first / speed-gate / TTC in `RacingEnv` (curriculum levers **inside** one map).

### Curriculum best practices (this stack)

1. **Gate on crash rate / finishers, not timesteps.** Unlock speed reward only under `SPEED_GATE_CRASH_RATE` window; unlock Map1 after Map0 shows clean progress / FTG-relative gate (`IDEAS_TRIMMED` P1).  
2. **Never train the validation pin or sealed holdouts** unless `--allow-holdout` (loud void). Autotrain must call `assert_train_safe`.  
3. **Generate offline, train online.** Autogen job → refresh thumbs → then Start; never `trackgen` inside `step`.  
4. **Fingerprint pack hashes** into `config.json` / `pack_fingerprint` so Continue cannot silently pick new geometry.  
5. **Raster cost:** large occupancy grids dominate RAM at high `n_envs` (`PLAN_PROGRESS`); prefer existing scaled maps over “richer” huge rasters this week.  
6. **Autotrain map cycle:** outer loop picks from `train_safe_maps()` only; after EarlyStop, optionally add next train_ok map **as a new fingerprinted run**, not mid-episode morph.  
7. **Official transfer claim:** only after sealed holdout eval under `official_v2` — generating 50 maps does not buy a podium.

### Gym → AutoDRIVE transfer (observe this 9h — do not build P3)

**W3 decision:** SHIP constraint note only; **SKIP** bridge / Jetson / latency product this window. Bridge `:4567` stays **Phase 3** (after sealed holdout beat-FTG). Gym work must not drift from that eventual bridge:

- **Freeze contracts `2.0.0`** — obs dim / action space / beam ABI stay refuse-load on mismatch (eval, resume, Watch, Continue). No “temporary” obs reshape for gym speed.
- **Mid-train DR already counts** — keep light LiDAR noise/dropout (and existing spawn/curriculum levers); do not invent a second DR stack or full actuator-lag theater this window.
- **No privileged GT in observations** for faster gym wins — shaping may use map GT / contact in reward; race obs stays LiDAR + legal proprio. Do not add IPS/pose/progress channels then plan to strip at bridge.
- **Document, don’t implement** — latency/actuator match and sim-to-sim calibration are constraints to *not violate*, not tickets to ship before FTGΔ / holdout podium.

### Anti-patterns

- 8–32× identical map0 clones (“fake skill” — PLAN).  
- Regen / lava morph mid-episode.  
- Seal break then re-pin FTG without re-baselining.  
- Curriculum that changes contracts / obs dim (frozen `2.0.0`).

---

## 6. What NOT to build in the next 9h

Hard skips aligned with critique + PLAN serialization:

| Do not build | Reason |
| ------------ | ------ |
| CUDA / AMP / torch.compile / “fill the 3060” | CPU env-bound; theater until sps knee measured |
| Bridge `:4567` / Jetson / TensorRT / camera | Phase 3 gated on holdout beat-FTG |
| SAC / Dreamer / PBT / meta-RL / constrained RL first | Algo zoo before collision + overnight |
| Residual FTG / BC warm-start **before** crash rate drops + overnight survives | Cosplay assist |
| Watch carnival / seasons / ghost / embed-in-Control epic | Unofficial; pixels ≠ policy — **W3 exception:** compact overlay + glossary OK (operator honesty, not chrome) |
| Map-regen-in-hot-loop / giant occupancy “sensors” | Slows CPU further; contracts freeze |
| Unconstrained continuous retrain without patience/warmup/floor | Replays EarlyStop burn |
| Second UI framework / Discord reward dial | Operator harm + non-repro |
| Official timeout shorten / FTG retune to “win” | Metric suicide |
| On-car overnight PPO | Safety CUT |

**Spend the hours on:** overnight regression lock, Continue soak, collision-first crash drop, honest official FTGΔ, holdout refuse — in that order ([`CAMPAIGN_CRITIQUE.md`](CAMPAIGN_CRITIQUE.md) MUST list).

---

## 7. Standing researcher — MUST queue evidence (W3 confirm)

Aligned with critique + racer + minimalist (+ reliability where it gates continuous). Full Go/No-Go narrative lives in [`CAMPAIGN_BOARD.md`](CAMPAIGN_BOARD.md).

| # | MUST | Verdict | Evidence (W3 ~04:09) | Next proof |
| - | ---- | ------- | -------------------- | ---------- |
| 1 | Overnight early-stop regression lock | **PASS / hold** | Live soak ~295k in honest `validating` @220s; sacred knobs intact; Continue-restart earlier this hour survived | Touch RaceBest/`ui_ops` → re-smoke; else observe-only |
| 2 | Continue-train soak | **PASS (CLI)** | W2: `_continue_soak_smoke` disposable run 1024→**2048**; Start/Continue no longer `_kill_train_tree` | Optional UI Stop→Continue once; never kill overnight to re-prove |
| 3 | Collision-first → crash_rate down | **PARTIAL** | UI checkboxes + argv wired (W2/W3); overnight flag still **off** by design | **Still owed:** ≤100k A/B crash_rate with flag on vs off |
| 4 | Official FTG vs PPO (FTGΔ) | **GO — owed** | FTG `official_v2` ≈198.16 s; still **no** PPO official_v2 row | Read-only official eval of best/complete zip; seals intact |
| 5 | Holdout/validation refuse | **GO — smoke** | ui_selftest Start refuse map2/map3; soak allow=false | CLI refuse + pack verify before official append |

### Soft UX W3 vs Go/No-Go (confirm)

| Decision | Verdict | Why it does / doesn’t violate |
| -------- | ------- | ----------------------------- |
| **SHIP** compact Watch + raise follow `--every` | **Go** | Design test: fewer misread KPIs; keeps lag-behind / unofficial; train stays headless; `test_watch_overlay` must stay green; ≠ carnival |
| **SHIP** glossary / tooltips drawer | **Go** | Fewer wrong Starts / misread Validating·patience·official vs smoke; extend `title=` / small drawer — not second UI |
| **SKIP** Control reskin | **Correct No-Go** | Cosmetics; risks Overnight preset / refuse copy |
| **SKIP / DEFER** embed Watch-in-Control | **Correct** | Conditional unmet (perf + MUST queue); re-vote only after #3 A/B + operators still dual-window blocked |
| **SKIP** bridge / Jetson | **Correct No-Go** | P3; transfer = observe constraints only (§5 note) |

Soft UX is **optional** behind remaining race work — do not starve MUST #3 A/B or #4 FTGΔ.

### Next-tick priorities (after MUST #2 / #3 path)

1. **Finish #3:** short collision-first A/B → publish crash_rateΔ (or honest null).  
2. **#4 FTGΔ:** official PPO row vs pinned FTG when a complete zip is ready (soak mid-validate OK to wait).  
3. **#5 holdout CLI smoke** if not logged green this hour.  
4. Soft: implement W3 SHIP compact Watch + glossary **only if free hands**; re-vote embed later.  
5. Still **NO-GO:** continuous outer loop, GPU theater, reskin, bridge.

**Reliability note:** Start/Continue kill-tree removed (W2) — good. Continuous outer loop still **NO-GO** until heartbeat-join race after `early_stopped` is closed ([`campaign_reliability.md`](ideas_review/campaign_reliability.md)).

---

## Go / No-Go checklist (orchestrator gate before shipping a feature)

Every feature must pass **all** applicable rows. Fail any → **No-Go** (cut the feature, not the guard).

### A. Sacred invariants

- [ ] **Contracts `2.0.0`:** no obs-dim / action-space change; refuse-load on mismatch (eval, resume, Watch, Continue).  
- [ ] **Overnight early-stop:** `select_timeout ≥ 220`, auto eval floor ≥ 50k, warmup ≥ 2, min_ts ≥ 100k when unlimited+patience; `phase=early_stopped` not overwritten by `LiveStatusCallback`; Overnight preset still budget-OFF + patience&gt;0.  
- [ ] **Holdouts / validation:** `assert_train_safe`; Start without allow dies on sealed/validation maps; seals verified before official append.  
- [ ] **Promotion honesty:** `best_model` / recommended = `race_score_key` on validation/official only — never `ep_rew_mean`. Atomic complete zips only.  
- [ ] **Dual-writer:** refuse Double-Start / Continue while `train.lock` / live PID.  
- [ ] **Protocol:** FTG + PPO same seeds/episodes/timeout/λ=10; no mid-campaign FTG retune for vanity wins.  
- [ ] **Protected run:** does not kill `overnight_soak_20260917_082739` without planned Continue restart on same `run_id`.

### B. Honesty & metrics

- [ ] Does **not** append `kind=official` unless `eval_protocol` path + seals intact.  
- [ ] Fast validation / probe is **labeled** (smoke/status) and cannot promote or claim beat-FTG.  
- [ ] Official claims use `timeout_s=400` and `race_score_key` (finishers ≻ DNFs).  
- [ ] CI / bootstrap (if any) separates finishers vs DNF rate; no fake precision from identical seeds.

### C. Throughput & compute

- [ ] Success criterion includes **`live_status.steps_per_sec`** (or explicit “N/A — not throughput work”).  
- [ ] GPU work justified by measured update-bound evidence — else No-Go.  
- [ ] Does not densify mid-train eval or lower select timeout “for speed.”  
- [ ] No display/mapgen in the train process hot path.

### D. Operator / continuous loop

- [ ] Continuous/auto-retrain: does **not** alter early-stop constants; has budget/chain cap; no silent hparam morph; no board pollution.  
- [ ] Continue path prefers last **complete** checkpoint; contracts fingerprint checked.  
- [ ] UI change passes the design test: fewer wrong Starts / lost Continues / misread banners — else cosmetics No-Go.

### E. Curriculum / maps

- [ ] Autogen maps are cached offline; train list ⊆ `train_safe_maps()`.  
- [ ] Fingerprint / pack hash recorded if maps change across runs.  
- [ ] No train-on-holdout unless allow flag **and** claims voided.

### F. Acceptance smoke (when touching the named surface)

- [ ] Overnight: `python -m rl._overnight_soak_smoke` (or equiv) if touching `RaceBestModelCallback` / overnight constants.  
- [ ] Continue: Stop → Continue same `run_id` → timesteps rise.  
- [ ] Holdout: `train_ppo --map map2|map3` exits ≠0 without allow.  
- [ ] UI: `ui_selftest` still green if touching `ui_ops` / Control.  
- [ ] Watch (if touched): lag-behind / unofficial labels; `test_watch_overlay` green; promotion path unchanged.

### One-line decision rule

**Go** only if the feature moves **official adjusted_time vs pinned FTG on sealed maps**, **overnight Continue integrity**, or **measured crash-rate / steps/sec honesty** — without touching sacred guards. Everything else is **No-Go** for this 9h window.

**Standing:** Resume this researcher each tick with the latest proposal; append `CAMPAIGN_BOARD.md` and refresh §7 evidence — do not drive-by feature code unless a research-proven bug.
