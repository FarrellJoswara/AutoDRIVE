# AutoDRIVE F1TENTH RL — Implementation Plan

Actionable plan derived from 12 personality reviews → `IDEAS_TRIMMED.md`. **Plan only — no code in this document.**

Companion: [`IDEAS_TRIMMED.md`](IDEAS_TRIMMED.md) · contracts: [`contracts.md`](contracts.md) · backlog source: [`IDEAS_MEGA.md`](IDEAS_MEGA.md)

---

## Goal / non-goals

### Goal
Ship a **trustworthy gym loop** that (1) trains PPO without destroying overnight runs, (2) **promotes models by `adjusted_time = lap_time + 10·collisions`**, and (3) **beats a pinned FTG baseline** on map0 **and** ≥1 sealed held-out map under a frozen protocol — then stop feature addiction.

### Non-goals (until P1 exit)
- AutoDRIVE `:4567` / Jetson productization as a training path
- Camera / vision / Grad-CAM
- Algorithm zoo (SAC, Dreamer, meta-RL, PBT, constrained RL as first upgrade)
- Aesthetic Watch, seasons/XP, conference booth chrome
- Overnight unrestricted on-car PPO

### Dual north stars
| Track | North star |
| ----- | ---------- |
| **Training / race** | Held-out beat-FTG on adjusted_time with collision budget |
| **Operator** | One honest panel: start, stop, continue, load/delete, generate maps, coach — Watch that admits lag and shows race KPIs |

---

## Current state (what already works)

Grounded in `rl/` as of review synthesis (Sep 2026):

| Piece | Status |
| ----- | ------ |
| `contracts.py` / `contracts.md` | **v2.0.0 frozen** — LiDAR 180 + prev-action + speed + IMU×3; MultiDiscrete `[4,11]`; λ=10 adjusted_time |
| `racing_env.py` | Frenet **high-water Δs**, stall timeout, collision terminate, near-wall penalty; **fixed `start_pose`**; one map per env |
| `observation.py` | v2 obs builder |
| `train_ppo.py` | SB3 PPO, Subproc/Dummy, checkpoints, `live_status`, `latest_model.zip`; **final zip named `best_model` (not eval-selected)** |
| `control_ui.py` | Start/Stop, map thumbs/picker, `n_envs`, TB auto-open, Open Watch; **no resume / load-delete / coach / presets / map gen** |
| `watch.py` | Lag-behind multi-color twins, FTG placeholder, auto-reload; overlay ≈ train_ts / ep_rew / n_envs |
| `trackgen.py` | CLI procedural maps → `maps/<id>/` |
| `ftg.py` / `run_ftg.py` / `bridge_ftg.py` | Classical baseline + bridge check/live FTG |
| `eval_cli.py` | Prints JSON; **does not append honest official leaderboard rows** |
| `metrics_io.py` / `compare_models.py` / `models/leaderboard.csv` | Schema exists; **train smoke often writes null laps**; v1 rows still on disk |
| `start_train.ps1` / `start_ui.ps1` | Launchers; **defaults disagree** (ps1 subproc / CLI dummy / UI forces subproc when n_envs>1); default **500k** timesteps |
| Resume / soft-stop / holdout seals / spawn jitter in train | **Missing** |

---

## Success metrics

Pre-register and refuse to “win” on anything else:

| Metric | Target / rule |
| ------ | ------------- |
| **Primary:** `adjusted_time` | Mean over ≥5 seeds (or ≥N episodes) on sealed maps; report collisions + DQ flag |
| **Beat FTG** | Δt & collision count vs **pinned** FTG row on same protocol (map0 + ≥1 holdout) |
| **Crash rate** | Rolling train collision rate → ~0 before unlocking speed reward; eval under DQ budget (e.g. >10 collisions = DQ) |
| **Throughput** | Log **steps/sec**, GPU util, reward skew across `n_envs`; tune workers to the knee |
| **Honesty** | No promote on `ep_rew_mean`; no partial `best_model`; contracts mismatch never predicts |
| **Operator** | Continue-train recovers overnight; Start/Stop cannot dual-write one `run_id` |

Secondary (log, don’t rank): train `ep_rew`, shaping terms, TB losses.

---

## Phased workstreams

### Phase 0 — Truth (week 1 focus)

**Exit:** Official eval path writes non-null adjusted_time; `best_model` means held-out race score; fingerprint + refuse-load; FTG line on board; steps/sec visible.

#### Training track (ordered)
1. **Held-out eval protocol** — sealed map list + fixed seeds/episodes; `eval_cli` runs FTG + PPO identically.  
   *Touch:* `eval_cli.py`, `trackgen` outputs / map pack manifest, `metrics_io.py`
2. **Leaderboard honesty** — only append `kind=official` from protocol (or retag train smoke); hide null adjusted_time; filter `contracts_version`.  
   *Touch:* `metrics_io.py`, `train_ppo.py` post-eval, `compare_models.py`
3. **Select `best_model` by eval adjusted_time / collisions** (periodic ckpt race), not final `learn()` weights.  
   *Touch:* `train_ppo.py`
4. **Config fingerprint** in `config.json` (vec type, map hash(es), contracts, seed, device, git SHA if cheap).  
   *Touch:* `train_ppo.py`, `metrics_io.write_run_artifacts`
5. **Hard refuse contracts/obs-dim mismatch** on load (eval, future resume, Watch already partial). Quarantine/relabel v1 zips.  
   *Touch:* `eval_cli.py`, `watch.py` (harden), shared helper near `contracts.py`
6. **Pin FTG baseline** via `run_ftg --save` under protocol; permanent beat-this row.  
   *Touch:* `run_ftg.py`, `compare_models.py`
7. **live_status: steps/sec + collision/crash estimate**; Dummy fallback must set `vec_env_active` truth.  
   *Touch:* `live_status.py`, `train_ppo.py`, `control_ui.py` status panel

#### Operator track (parallel after #5 gate for resume safety)
8. **Start/Stop ownership** — refuse Double-Start / same-`run_id` dual writers; plain English Stop (hard kill vs soft later).  
   *Touch:* `control_ui.py`
9. **Status banner** Idle / Learning / Saving / Stopping / Crashed|Stale + reason (PID + step monotonicity).  
   *Touch:* `control_ui.py`, `live_status.py`
10. **Start preview card** (map, steps, n_envs, run path).  
    *Touch:* `control_ui.py`
11. **Watch overlay** adjusted_time / collisions / stall-crash tags; lag-behind copy.  
    *Touch:* `watch.py`

**Phase 0 exit criteria**
- [x] Official eval JSON + leaderboard row for FTG and ≥1 PPO zip with non-null adjusted_time *(FTG pinned under official_v2; PPO official row may still be DNF — ranking still honest)*
- [x] `best_model.zip` chosen by that metric (documented in config/metrics)
- [x] Loading a v1 zip fails closed with a clear message
- [x] UI shows steps/sec and refuses dual Start on one run

---

### Phase 1 — Survival & generalization (week 1–2)

**Exit:** Train envs don’t memorize one pose/map0; collision curriculum exists; doomed eps don’t farm; multi-map diversity optional but wired.

#### Training track
1. **Spawn jitter** on `RacingEnv.reset` (pose/heading/lateral).  
   *Touch:* `racing_env.py` (mirror Watch twin jitter)
2. **Diversified `n_envs`** — map list / rotate packs in `_make_env`.  
   *Touch:* `train_ppo.py`, Control UI map multi-select later
3. **Collision-first / speed-gate** reward flags (big terminal collision early; unlock speed under rolling crash budget; pace floor).  
   *Touch:* `racing_env.py`, `contracts.py` constants if needed, `train_ppo` CLI flags
4. **TTC / frontal collapse truncate** (LiDAR-only, no MPC).  
   *Touch:* `racing_env.py`
5. **Anti-circling suite** only if jitter reveals orbiting (heading–tangent / yaw-while-s-flat).  
   *Touch:* `racing_env.py`
6. **Default shorter runs** (50k–100k) in README / `start_train.ps1` until P1 win; unify Subproc default story.  
   *Touch:* `start_train.ps1`, `README.md`, `train_ppo.py` defaults

#### Operator track
7. **`--resume` / Continue last run** — same run_id vs fork; `reset_num_timesteps` explicit; prefer last **complete** checkpoint; refuse contracts mismatch.  
   *Touch:* `train_ppo.py`, `control_ui.py`, `live_status` / artifact layout
8. **Atomic checkpoint rename** discipline (incomplete never promoted).  
   *Touch:* `train_ppo.py`
9. **Presets:** Debug (1 env) / Quick / Overnight.  
   *Touch:* `control_ui.py`

**Phase 1 exit criteria**
- [x] Jittered starts; at least one multi-map train smoke
- [x] Continue-train recovers after UI Stop from a complete ckpt *(wired; overnight soak still P1)*
- [ ] Collision rate visibly drops under collision-first settings

---

### Phase 2 — Beat FTG in gym (week 2–4)

**Exit (hard P1 product gate):** Beat pinned FTG on **map0 + one sealed holdout** on adjusted_time (CI / tails reported). Then freeze features; only raise timesteps / polish.

#### Training track
1. **Seal holdouts** (hash + never-train flag); Start refuses unless `--allow-holdout`.  
   *Touch:* map pack manifest, `control_ui.py`, `train_ppo.py`
2. **Light LiDAR DR** (noise/dropout/max-range).  
   *Touch:* `observation.py` / env step
3. **Engineered clearances / TTC features** (avoid obs-dim bump if possible).  
   *Touch:* `observation.py`, `racing_env.py` — contracts MINOR only if width changes
4. **Warm-start FTG/BC** short demos → PPO.  
   *Touch:* `ftg.py`, demo dump helper, `train_ppo.py`
5. **Residual RL / FTG corridor filter (gym)** if pure PPO plateaus — bounded Δu.  
   *Touch:* new thin wrapper or env action filter; keep classical `ftg.py`
6. **Race-candidate promote** action: held-out official eval → pin “recommended”.  
   *Touch:* `compare_models.py`, `control_ui.py`, `metrics_io.py`
7. **Ghost/compare FTG vs PPO** in Watch or CLI using race KPIs.  
   *Touch:* `watch.py`, `compare_models.py`
8. **Crash taxonomy + overfit scramble-LiDAR smoke**.  
   *Touch:* `eval_cli.py`, `racing_env` info dict

#### Operator track
9. **Map generator UI** — Generate N / seed / tags → cache + refresh thumbs (CLI `trackgen` already exists).  
   *Touch:* `control_ui.py`, `trackgen.py`
10. **Model load / delete + timeline** (Best / Latest / Checkpoints).  
    *Touch:* `control_ui.py`, models dir conventions
11. **Coach + health panel** (reward slope, crash rate, Watch opened?, too many workers).  
    *Touch:* `control_ui.py`
12. **Post-run summary + Open folder**.  
    *Touch:* `control_ui.py`
13. **Map mismatch banner**; mid-run map change fails loud (Stop required).  
    *Touch:* `watch.py`, `control_ui.py`

**Phase 2 exit criteria (THE GATE)**
- [ ] Official rows: PPO adjusted_time **better than** pinned FTG on map0 **and** holdout H
- [ ] Collisions within budget; worst-decile / survival not catastrophic
- [x] Recommended race candidate ≠ “highest ep_rew” *(Integration: `race_score_key` + protocol filter in UI + compare)*
- [x] Operator can generate map, continue train, load/delete without shell

**After gate:** stop new algorithms/sensors; only longer runs, DR strength, residual polish.

---

### Phase 3 — Bridge / Jetson (only after Phase 2 gate)

Serialize per architecture review — **do not parallelize with Phase 0–2 science**.

1. Sim-to-sim identical maps gym → `:4567` (`bridge_ftg` → PPO path)
2. Latency p99 budget + calibration gate (LiDAR stats ∈ gym DR)
3. Shadow mode + classical hard veto + intervention budget
4. ESTOP / watchdog / Safe Halt on dropout
5. Deploy-readiness one-pager filled
6. Then Jetson FLOPs / TensorRT / optional rate-limited fine-tune
7. Camera = contracts **MAJOR** last

*Touch:* `bridge_ftg.py`, future `bridge_ppo`, `eval_cli --backend autodrive`, deploy docs

**Phase 3 exit criteria**
- [ ] Shadow intervention budget acceptable on validation track
- [ ] p99 latency ≤ trained control step under soak
- [ ] GT-ablation pass; no privileged topics in race binary

---

## Operator track vs training track (parallelization)

```text
SAFE IN PARALLEL (after their mini-gates):
  Operator copy/UX  ║  Training reward/env work
  Watch overlays    ║  (Watch never inside train process)
  Docs / three-jobs ║  Eval protocol implementation
  Map gen UI        ║  after trackgen cache exists
  Coach panel       ║  after live_status exposes crash rate + steps/sec

SERIAL (do not parallelize):
  contracts freeze  →  long PPO / bridge sweeps / algo zoo
  eval protocol     →  “best_model” / promote / beat-FTG claims
  atomic ckpts      →  continue-train / recommended-for-racing
  P1 beat-FTG gate  →  bridge product / Jetson / camera
  gym DR envelope   →  bridge calibration refuse-deploy
  shadow + veto     →  closed-loop PPO authority on bridge/car
```

---

## Hard serialization rules (from architecture)

1. Obs/contracts freeze before long runs, bridge remap, or adding camera/IMU gadgets.
2. Primary eval metric + holdouts before reward-shaping sweeps and algo bake-offs.
3. FTG gym baseline numbers before declaring PPO wins or tuning residual/veto quality.
4. Beat-FTG on holdouts before bridge-as-train-path, Jetson optimize, or camera.
5. Gym DR envelope before bridge calibration refuse-deploy.
6. Shadow + classical veto before PPO owns actuators.
7. Latency p99 proof before raising Hz, viz on control box, or physical overnight train.
8. Atomic promote rules before multi-day continue and “recommended” pins.
9. Sealed holdout hashes before map-pack curriculum unlock / transfer claims.
10. LiDAR stack proven under stress before vision MAJOR.

**Forbidden loops:** bridge latency ↔ gym reward redesign without frozen metric; camera “pad zeros” into LiDAR ABI; judge training by sexy Watch twins; algo swap mid-metric definition; on-car PPO without ESTOP/veto; map regen in train hot loop.

---

## Risks / anti-patterns

| Anti-pattern | Antidote |
| ------------ | -------- |
| Lying `best_model` (final weights / ep_rew) | EvalCallback-style pick on adjusted_time |
| Ranking / early-stop on `ep_rew_mean` | UI + TB + Watch show race KPIs; promote only official |
| Camera-first / sensor creep | Contracts forbid; gate after LiDAR win |
| Overnight on-car PPO unrestricted | CUT; shadow + rate limit only after P3 gates |
| 32× map0 clone farm | Diversify maps/starts; seal holdouts |
| Dual UI / dual train same `run_id` | Collision refuse before zip corruption |
| Partial zip as best | Atomic rename; quarantine empty |
| Green `live_status` while dead | PID + step heartbeat + Stale/Crashed |
| 500k cosplay before metric truth | Default shorter serious runs |
| Residual + hierarchical + constrained RL all at once | One classical assist at a time |
| Mid-run Discord reward dial | Fingerprinted staged curriculum only |
| “Fine-tune 5 min on car” after ideal gym | Not a substitute for calibration/shadow |

---

## Suggested first week checklist

**Day 1–2 — Truth**
- [x] Define sealed eval YAML/list (map0 + holdout H hashes, seeds, N episodes)
- [x] Wire `eval_cli` → official leaderboard append; stop smoking null rows as race scores
- [x] Pin FTG baseline under that protocol
- [x] Change `best_model` selection to held-out adjusted_time / collisions
- [x] Fingerprint + refuse v1/obs mismatch on eval load

**Day 3–4 — Env generalization**
- [x] Spawn jitter in train reset
- [x] steps/sec + crash rate in `live_status` / UI
- [x] Collision-first flag + TTC truncate sketch
- [x] Unify Subproc/Dummy default + status truth on Windows

**Day 5–7 — Operator loop**
- [x] Start preview + ownership/dual-Start guards + status banner
- [x] `--resume` / Continue last complete ckpt
- [x] Watch overlay: collisions / adjusted score / crash tags
- [x] Presets Debug / Quick / Overnight; shorten default timesteps in launcher docs
- [ ] Smoke: overnight continue after Stop; official compare FTG vs current PPO (even if PPO loses) *(partial: compare works; overnight soak deferred)*

**Week 1 success:** you can trust a number, continue a run, and know whether you beat FTG — even if you don’t beat it yet.

---

## File touchpoint map (quick)

| Concern | Primary files |
| ------- | ------------- |
| Metric / artifacts | `metrics_io.py`, `train_ppo.py`, `eval_cli.py`, `compare_models.py` |
| Env / reward | `racing_env.py`, `contracts.py` |
| Obs / DR | `observation.py`, `contracts.md` |
| Throughput / status | `live_status.py`, `train_ppo.py`, `start_train.ps1` |
| Operator UI | `control_ui.py`, `start_ui.ps1` |
| Watch | `watch.py`, `viewer.py` |
| Maps | `trackgen.py`, `maps/` |
| Classical / bridge | `ftg.py`, `run_ftg.py`, `bridge_ftg.py` |

---

## Sequencing one-liner

**Freeze contract → measure adjusted_time vs FTG → win that in gym on holdouts (with honest Start/Stop/continue) → only then bridge/sim2sim with veto & latency → only then Jetson/car → camera last as a new contract.**
