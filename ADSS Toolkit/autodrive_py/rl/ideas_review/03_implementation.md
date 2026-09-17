# Implementation review — next 2–4 weeks

**Lens:** highest value per engineering hour on *this* codebase (gym PPO + Control UI + Watch + contracts v2), not research novelty.

**Stack snapshot (already exists):**
| Piece | Reality today |
| ----- | ------------- |
| `contracts.py` / `contracts.md` | **v2.0.0 frozen** — LiDAR 180 + prev-action + speed + IMU×3; MultiDiscrete `[4,11]`; `adjusted_time = lap + 10·collisions` |
| `racing_env.py` | Frenet **high-water Δs** progress, stall timeout (8 s), collision terminate, near-wall penalty; **fixed `start_pose`**; single map per env |
| `train_ppo.py` | SB3 PPO, `n_envs`, Subproc/Dummy, checkpoints, `live_status`, `latest_model.zip`; **final zip named `best_model`** (not EvalCallback / not adjusted_time) |
| `control_ui.py` | Start/Stop train, map thumbs, `n_envs`, TB auto-open, Open Watch; **no resume / load-delete / coach / presets** |
| `watch.py` | Lag-behind **multi-color twins**, FTG placeholder until weights, auto-reload latest/ckpt; overlay = train_ts / ep_rew / n_envs |
| `trackgen.py` | Procedural maps → `maps/<id>/` |
| `ftg.py` / `run_ftg.py` / `bridge_ftg.py` | Classical baseline + Bridge path (v2 obs helpers) |
| `eval_cli.py` / `compare_models.py` / `metrics_io.py` | Gym eval + CSV leaderboard sorted by `adjusted_time` — **often null** when no lap; thin protocol |

**North-star for 2–4 weeks:** one clean metric (`adjusted_time` / clean-lap survival) that beats a fixed FTG baseline on map0 + ≥1 held-out map — then stop feature addiction.

---

## KEEP — cheap & high leverage

Do these. Most are thin extensions of code you already own.

### Metric truth & selection
- **Rank / promote by `adjusted_time` (and collision rate), not `ep_rew_mean`.** Contracts + `metrics_io` already define it; train UI/TB still center `ep_rew_mean`. Change what “best” means.
- **Checkpoint race / EvalCallback-style selection:** keep periodics you already write; *pick* lowest-collision / best adjusted_time member for `best_model.zip` instead of “last learn() weights.”
- **Separate eval metric from train reward** (already in contracts) — enforce in UI copy + `compare_models` / Watch overlay so shaping never becomes the sport.
- **Frozen held-out protocol:** fixed seeds, episode budget, primary metrics pre-registered; bake FTG + PPO through `eval_cli` the same way. Depends on: `eval_cli`, `trackgen`, sealed map list.
- **Null / classical baselines on every board:** random / constant-throttle optional; **FTG required** (`run_ftg --save` already exists).

### Env / reward (gym-only, no new algos)
- **Collision-first curriculum:** big terminal collision cost early; unlock speed term only under a rolling collision budget (or “cowardly qualifier → time-attack” as two short runs). Depends on: `racing_env` reward weights + thin train CLI flags.
- **Spawn diversity:** jitter pose / heading / lateral offset on reset (Watch twins already do this; train envs do not). Highest-ROI anti-overfit change.
- **Parallel envs ≠ map clones forever:** diversify maps/starts across `n_envs` (or rotate map packs). Depends on: `trackgen` outputs + `_make_env` in `train_ppo`.
- **Dense Δs clamp / anti-orbit** — largely done (high-water Frenet + stall). Keep; maybe add heading–tangent mismatch tax only if circling returns after spawn jitter.
- **TTC / frontal-range soft terminate** (imminent crash) — cheap LiDAR-only add; reduces wall-bounce farming. Soft version first (no MPC).
- **Truncate doomed episodes early** — collision + stall already; ensure timeouts don’t award participation trophies in VecEnv logging.

### Obs / contracts hygiene
- **Keep contracts freeze;** refuse load on obs-dim / contracts major mismatch (Watch already has a PPO obs check — push into train resume + eval load).
- **Config fingerprint:** hyperparams, torch/CUDA, vec type, map id/hash, `contracts_version`, seed — extend existing `config.json` write in `train_ppo` (git SHA if cheap).
- **Light domain randomization only:** beam dropout / range noise / small obs delay — first-class enemies without camera. Defer full latency/actuator drama.
- **LiDAR frame stack (k=2–4)** for closing rate — only if single-frame plateaus *and* you accept a **contracts bump** (or side-channel features that don’t break zip ABI). Prefer engineered L/R/F + Δmin_range *without* width change first.

### Throughput & ops
- **Steps/sec + collision estimate in `live_status`** (partially there) — make throughput + crash rate first-class so n_envs / Dummy vs Subproc decisions are evidence-based.
- **Keep train headless** (already); never put OpenCV in train. Vectorize for steps/sec; don’t chase GPU util theater.
- **Atomic / complete artifacts:** never promote half-written zips; resume prefers last complete checkpoint (pairs with resume work).
- **Default shorter serious runs until proven** (50k–100k with real eval) — README’s 500k default is cosplay until metric truth exists.

### UI / Watch (polish on existing surface)
- **Status banner:** Idle / Learning / Saving / Stopping / Crashed + reason — Control UI already polls; make state machine language.
- **Presets:** Quick try / Overnight / Debug (1 env) — thin wrappers over knobs you have.
- **“What will Start do?” preview** (map, steps, n_envs, run path) before commit.
- **Watch overlay: adjusted_time / collisions / crash tags**, not just `ep_rew` — extends follow overlay; still lag-behind honest.
- **Session resume: Continue last unfinished run** — depends on checkpoint dir + `PPO.load` + timestep bookkeeping (on radar; high overnight value).
- **First-launch / three-jobs copy:** Control trains · Watch watches · TB curves — docs already say this; one modal or README strip beats a coach panel.

### Classical / hybrid (small, not research)
- **Warm-start from FTG rollouts (BC or action cloning few epochs)** before PPO — burns fewer thrash steps on the 3060. Depends on: `ftg` + short demo collector.
- **FTG hard veto envelope at deploy / bridge only** when you actually drive AutoDRIVE — gym can stay pure; don’t build veto into every train step yet.
- **Bayesian / manual tune of FTG gains first** if PPO can’t beat a tuned FTG — cheap classical frontier check.

### Maps
- **Cache / pin held-out eval tracks** (hashed list) + one never-changing “validation” map. `trackgen` exists; **don’t build map-generator UI** this month (Control UI already has thumbnails).
- **Curriculum unlock Map1 after Map0 beats FTG gate** — once eval protocol exists.

---

## DEFER — good ideas, wrong horizon

Worth remembering; don’t schedule in the next 2–4 weeks unless KEEP items are done.

| Idea | Why defer |
| ---- | --------- |
| SAC / TD3 / TQC / DroQ / REDQ vs PPO | Matched-protocol ablations need the eval harness first; algo swap ≠ metric |
| LSTM-PPO / recurrent | Try short stack or Δrange features first; recurrent = slower debug |
| Residual RL / tube-MPC / dual-rate classical inner loop | High design cost; FTG veto at bridge is enough until policy is competent |
| Hierarchical modes / safety filter projection every step | Same — complexity tax before clean laps |
| Constrained RL (PPO-Lag / CPO / CVaR) | Collision curriculum is 10× cheaper |
| Offline RL (IQL/CQL), Dreamer / TD-MPC / world models | No bag pipeline; research track |
| PBT / meta-RL / self-play / Go-Explore | Overnight compute toys without held-out truth |
| Full domain randomize µ / friction / actuator lag / voltage sag | After light LiDAR DR proves value |
| Camera / optical flow / segmentation FTG | Contracts MAJOR; mega explicitly warns against gadgets before clean eval |
| TensorRT / Jetson FLOPs / thermal | No physical race loop this sprint; gym→`:4567` sim2sim later |
| System-ID Pacejka / model-based residual dynamics | Needs logs + hardware |
| W&B lineage DAG / CO₂ / preference thumbs | Leaderboard + TB enough |
| Aesthetic Watch (reward weather, photo-finish, Grad-CAM) | Doesn’t move adjusted_time |
| Coach panel / glossary / a11y suite | After status banner + presets |
| Load/delete model timeline theater | Resume + “recommended race candidate” after selection exists; avoid dual best/latest product surface |
| Map generator UI / partner PNG upload | CLI `trackgen` + thumb picker sufficient |
| RoboRacer 10-lap / DQ rules full compliance | Adopt scoring bits into eval protocol; full steward later |
| Kill bridge / Watch entirely | Bridge + Watch already earn their keep for demos; don’t delete — just don’t expand |

---

## CUT — don’t build (or actively resist)

- Entire **`[bad/fun]`** bucket (voice, NFT, perfume, hive-mind, lava maps, wilting plant TB, etc.).
- **LLM planner + RL tracker**, diffusion/ACT/flow policies, Rainbow-Atari on continuous car.
- **Camera + IMU gadget creep before one clean 10-lap held-out eval** (IMU already in v2 — don’t add more channels “because we can”).
- **Cheerful lying UI**, emoji-only status, auto-start on page load.
- **Second advanced UI / plugin bus / “minimal mode” that adds surface area.**
- **Maximize Watch chaos / n_envs bragging** while eval still ranks reward.
- **Softmax-only steer / YOLO action spaces / train the wall.**
- **Replace TensorBoard**; **16 Watch windows**; OpenCV on Jetson control core.
- **Online overnight PPO on physical car** with no rate limits.
- **Privilege GT pose in obs** then strip at race (leak path).
- Research-notes novel / paper-quote tooltips as product.

**Simplify that *is* KEEP:** one primary metric, one eval path, one success criterion (beat fixed FTG on map0 + holdout once). Optional cut of *new* checkpoint theater — keep `latest_model` for Watch lag-behind only.

---

## TOP 15 by ROI (ordered)

Effort ≈ person-days of focused work on *this* tree. Value = moves race metric or prevents wasted GPU/overnight runs.

| # | Item | Effort | Depends on / extends | Why now |
| - | ---- | ------ | -------------------- | ------- |
| 1 | **Held-out eval protocol** (fixed maps/seeds/episodes; primary = `adjusted_time` + collisions; FTG + PPO same CLI) | S–M | `eval_cli`, `metrics_io`, `trackgen`, sealed map list | Without this every other idea is unfalsifiable |
| 2 | **Select `best_model` by eval adjusted_time / collision**, not final weights / ep_rew | S | `train_ppo` checkpoints + eval helper | Leaderboard today often null; “best” is a lie |
| 3 | **Train spawn jitter** (pose/heading/lateral) | S | `racing_env.reset` (Watch already has twin jitter) | Stops map0 start-pose memorization cheaply |
| 4 | **Multi-map / diversified `n_envs`** | M | `trackgen`, `_make_env`, Control UI map list | 8 clones of map0 fake generalization |
| 5 | **Collision-first / speed-gate curriculum** | S–M | `racing_env` reward terms + train flags | Matches competition priority; uses existing collision penalty |
| 6 | **Resume from last complete checkpoint** (+ UI “Continue?”) | M | `checkpoints/`, `PPO.load`, Control UI Start path | Overnight runs already checkpoint; UI can’t continue |
| 7 | **Config fingerprint + hard refuse obs/contracts mismatch** | S | `config.json`, Watch’s `_check_ppo_obs` | Prevents silent v1/v2 and zip poison |
| 8 | **live_status: steps/sec + crash/collision rate** surfaced in UI | S | `LiveStatusCallback`, Control UI status panel | Guides n_envs / Dummy vs Subproc with evidence |
| 9 | **Watch overlay: collisions / projected adjusted score / stall-crash tags** | S | `watch.py` follow overlay | Aligns eyes with the real KPI; still lag-behind |
| 10 | **UI presets + Start preview + plain status banner** | S | `control_ui.py` | Cheap operator leverage; no new backend |
| 11 | **Always-register fair FTG baseline** on leaderboard under protocol | S | `run_ftg --save`, `compare_models` | Gives a permanent “beat this” line |
| 12 | **Light LiDAR DR** (noise / dropout / max-range clip) | S–M | `observation.py` / env step | First sim2real inch without Bridge work |
| 13 | **Warm-start from FTG demos** (short BC or biased init) | M | `ftg`, demo dump, `train_ppo` | Saves thrash hours on 3060 once eval exists |
| 14 | **TTC / frontal collapse truncate** (no MPC) | S | `racing_env` + raw scan | Cheap anti-crash shaping on existing LiDAR |
| 15 | **Race-candidate gate:** promote only after held-out clean eval (ensemble = optional lowest-collision ckpt) | S | #1 + #2 | Closes the loop: train → select → ship candidate |

**Explicitly not in Top 15 this month:** SAC, residual RL, LSTM, camera, Jetson deploy, W&B, coach panel, map-gen UI, aesthetic Watch, constrained RL.

---

## Suggested 2–4 week sequence

1. **Week 1 — Truth:** #1, #2, #7, #11, #8 (eval + honest best + fingerprint + FTG line + throughput).
2. **Week 1–2 — Generalization & survival:** #3, #4, #5, #14 (jitter, multi-map, collision curriculum, TTC truncate).
3. **Week 2–3 — Operator loop:** #6, #9, #10, #15 (resume, Watch KPI overlay, UI presets, promote gate).
4. **Week 3–4 — Only if plateaus:** #12 then #13 (light DR, then FTG warm-start). Stop if map0+holdout already beats FTG on adjusted_time.

**Success criterion to freeze:** beat a pinned FTG baseline on map0 **and** one held-out map on `adjusted_time` (mean over ≥5 seeds or ≥N episodes with collision count reported). Then cut feature work and only raise timesteps / polish.
