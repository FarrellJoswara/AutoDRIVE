# 12 — Architecture: phased sequencing

Systems view of `IDEAS_MEGA.md` against the current `rl/` stack (`contracts` → gym `racing_env` / `observation` → `trackgen` → FTG → `train_ppo` → `eval_cli` / leaderboard → `watch` / Control UI → `bridge_ftg`).

**North star:** one clean gym win on a frozen held-out protocol before bridge, Jetson, or camera. Camera is a contracts MAJOR; bridge is a backend, not a training substitute.

---

## PHASE order (P0–P3) with idea groups

### P0 — Gym foundation (freeze the ABI, measure truth)

**Goal:** A single, reproducible gym loop: map0 (or small pack), FTG baseline, PPO trainable headless, eval that ranks something other than `ep_rew_mean`.

| Idea group (from mega) | Why this phase |
| ---------------------- | -------------- |
| **Obs contract freeze** (beams/FOV/units/version hash; no deploy-time interpolation) | Every later weight, bridge remap, and ablation hangs on ABI stability. Already `2.0.0` LiDAR+proprio; do not reopen for gadgets. |
| **Eval & primary metric** (adjusted race score `time + λ·collisions`; held-out seeds; FTG-matched protocol; null baselines) | Unlocks “did we improve?” Without this, reward shaping and UI polish fake progress. |
| **Reward core** (Frenet-s progress, stall / reverse abort, anti-circling, collision terminal early, separate train reward vs eval metric) | Makes PPO optimizable toward the race metric instead of wall-bounce farming. |
| **FTG gym baseline + compare path** (`run_ftg`, `compare_models`, leaderboard skeleton) | Fixed bar every later gate must beat. |
| **Maps: pin + cache** (hardware-validation / week-to-week track; cache offline; no regen in hot loop) | Repro and CI need hashed tracks before curriculum packs. |
| **Train throughput basics** (vectorize `n_envs`, headless train, truncate doomed eps, match control Hz intent) | Steps/sec is the sample budget; Watch stays off the train core. |
| **Artifact integrity** (atomic checkpoint rename; refuse obs-dim mismatch; config fingerprint) | Prevents false “best” and silent ABI poison. |
| **Minimal Control UI / docs** (Start/Stop ownership, live_status honesty, three-jobs modal) | Operator loop without blocking science. |

**Unlocks:** P1 can trust numbers and load weights.

**Explicitly out of P0:** bridge product work, camera, Jetson, gamification, algo zoo.

---

### P1 — Gym wins (beat FTG under held-out protocol)

**Goal:** Policy that beats fixed FTG on frozen holdouts with collision budget; curriculum and DR that generalize across map pack—not map0 cosplay.

| Idea group | Why this phase |
| ---------- | -------------- |
| **Collision-then-speed curriculum** (hard-gate speed reward; cowardly qualifier → time-attack; crawl → clean laps → raise speed; sector/lap gates) | Aggression only after survival is real. |
| **Classical assist (gym-side)** (warm-start BC/FTG; residual Δu; safety filter into FTG gap; dual-rate setpoints) | Cuts early thrash on the 3060; keeps deploy story classical-veto-ready later. |
| **Observation engineering (still LiDAR contract)** (L/R/F, TTC, k-frame stack; channel ablations; beam DR / delay / dropout) | Improves signal without ABI break. |
| **Map pack + curriculum unlock** (procedural packs; sealed holdouts; MapN after Map0 gate; mid-run map swap fails loud) | Diversified `n_envs` must mean diversified maps, not 32 clones. |
| **Matched algo ablations** (MultiDiscrete vs continuous; LSTM/history; SAC/TD3 family under same steps+eval; constrained RL optional) | Only after protocol exists—otherwise you cannot pick a winner. |
| **Logging / leaderboard truth** (throughput tax, PPO internals, CI bands “beat FTG by Δt”, negative-result registry) | Makes P1 exit criteria public and anti-survivor-biased. |
| **Watch as debug surface** (twins, ghost FTG, crash tags, adjusted_time overlay)—not as training path | Visibility for failure taxonomy; must not steal train FPS. |
| **Resilience / chaos (gym)** (orphan trains, dual UI, NaN isolation, map-mismatch banner, live_status heartbeat ≠ gradients-exist) | Protects multi-day runs that P1 needs. |

**Exit gate (hard):** held-out clean (or budgeted-collision) eval beats FTG with pre-registered metric + ≥5-seed CI. Until then, stay in P1.

**Unlocks:** P2 sim2sim / bridge calibration against a policy worth transferring.

---

### P2 — Sim2sim & bridge (AutoDRIVE `:4567` after gym wins)

**Goal:** Same maps, same obs contract, closed-loop on simulator bridge with latency and veto—before any chassis.

| Idea group | Why this phase |
| ---------- | -------------- |
| **Sim-to-sim first** (gym → AutoDRIVE identical maps; offline bridge replay scoring) | Isolates backend/adapter bugs from policy bugs. |
| **Latency & control path** (scan→act budget; p99 fail; UDP/binary preference; single-threaded bridge; viz off control core) | Real-time contract the gym abstraction hid. |
| **Calibration gate** (bridge LiDAR stats inside gym DR envelope or refuse deploy) | Prevents silent domain cliff. |
| **Shadow + hard veto** (policy infers, FTG/human drives; FTG envelope while PPO is a menace; intervention budget as failure metric) | Safe closed loop without on-car learning. |
| **Actuator / sensor match** (steer rate, ESC lag; encoder speed path per contracts; no GT pose in race obs) | Closes the last legal proprio gaps. |
| **Deploy-readiness one-pager** (sensor model, latency, net size, veto, ABI, thermal) | Gate into P3 hardware. |
| **Eval CLI dual backend** (`gym` vs bridge) under same metrics schema | Cross-eval without rewriting the sport. |

**Still out of P2:** physical overnight PPO, camera heads, competition bureaucracy stretch unless required for event.

**Unlocks:** P3 Jetson / track / (optional) vision MAJOR.

---

### P3 — Hardware, competition, and deferred stretch

**Goal:** Chassis shadow → limited fine-tune; Jetson throughput; optional camera as new contract; research stretch only if P1–P2 KPIs plateau.

| Idea group | Why this phase |
| ---------- | -------------- |
| **Jetson deploy** (FLOPs cap, TensorRT/quant student, thermal headroom, sealed image) | Needs a proven actor + veto from P2. |
| **Safety & rules hardening** (ESTOP + watchdog, GT-ablation gate, RoboRacer scoring alignment, Change Control) | Event-real, not training-real. |
| **Camera / vision** (hybrid LiDAR safety + cam apex; flow; seg free-space)—**contracts MAJOR** | Explicitly after LiDAR gym+bridge wins; never parallel with P0 obs freeze. |
| **SysID / model-based residual, world models, offline RL from bags, meta-RL, PBT** | Expensive tracks; only if free-form PPO frontier is flat. |
| **Booth / demos / aesthetic / gamification** | Growth layer on a working race metric—never a substitute for P1 exit. |

---

## HARD DEPENDENCIES

Edges are **A → B** = “A must exist (or pass gate) before B is worth building.”

```text
contracts freeze (obs/action/metrics/artifacts)
    → FTG gym baseline + eval protocol (adjusted_time, holdouts)
        → reward/curriculum that optimize that metric
            → PPO gym train (+ throughput / n_envs diversity)
                → held-out “beat FTG” gate  ★ P1 EXIT
                    → map-pack curriculum / sealed holdouts (meaningful generalization)
                    → bridge adapter + identical-map sim2sim
                        → latency budget + calibration gate
                            → shadow mode + classical veto
                                → deploy-readiness → Jetson / chassis
                                    → camera MAJOR (optional)

eval protocol ──────────────────────────────→ algo zoo / sweeps / constrained RL
eval protocol ──────────────────────────────→ leaderboard CI / negative registry
gym DR envelope ────────────────────────────→ bridge calibration refuse-deploy
classical FTG (gym) ────────────────────────→ residual RL / safety filter / bridge veto
atomic checkpoints + config fingerprint ────→ continue-train / promote-to-race
live_status + headless train ───────────────→ Watch follow twins (debug only)
P1 exit gate ───────────────────────────────→ bridge product surface (mega “kill bridge until car” is directionally right: until gym win, bridge is check-only)
P2 veto + latency ──────────────────────────→ any on-car fine-tune
LiDAR-only race stack proven ───────────────→ camera / multi-view Watch
```

**Loop to refuse (dependency cycles):**

| Forbidden loop | Why it breaks |
| -------------- | ------------- |
| Bridge latency tuning ↔ gym reward redesign without frozen metric | You never know which side moved. |
| Camera obs ↔ LiDAR contract “just pad zeros” | Creates fake ABI; mega and `contracts.md` forbid it. |
| Watch / UI polish ↔ “training works” judged by twins looking good | Overfit map0 / entertainment ≠ podium. |
| Algo swap ↔ metric definition mid-sweep | Survivor bias; no fair compare. |
| On-car PPO ↔ missing ESTOP/veto/calibration | Unsafe and scientifically noisy. |
| Map regen in train loop ↔ hashed holdouts | Repro collapses; leaderboard lies. |

---

## DO NOT PARALLELIZE (serialization required)

These tracks look independent but **must be serial** (or strictly gated). Parallelizing them creates false progress or ABI debt.

1. **Obs/contracts freeze before any long PPO run, bridge remap, or “add IMU/camera” experiment.** Parallel feature-adding into obs = permanent zip graveyard (`1.x` already obsolete vs `2.0.0`).

2. **Primary eval metric + held-out protocol before reward shaping sweeps and before algo family bake-offs.** Shaping and SAC-vs-PPO in parallel without a frozen score = optimizing different sports.

3. **FTG (or classical) gym baseline numbers before declaring PPO wins or tuning residual/safety filters.** Filter quality is undefined without a baseline envelope.

4. **P1 exit (beat FTG on holdouts) before bridge productization, Jetson optimization, or camera.** Mega’s own anti-pattern: gadgets and UI before one clean held-out eval. Bridge `--check` is fine anytime; bridge-as-train-path is not.

5. **Gym domain-randomization envelope before bridge calibration gate / refuse-deploy.** Calibration needs a target distribution that already exists.

6. **Shadow mode + classical hard veto before any closed-loop PPO authority on bridge or car.** Policy-in-the-loop without veto serializes risk into the chassis.

7. **Latency budget proof (scan→act p99) before raising control Hz, enabling viz on the control box, or overnight physical training.** Throughput theater without timing = crashes that look like “bad policy.”

8. **Atomic checkpoint / promote rules before multi-day continue-train, leaderboard promotion, or “recommended for racing” UI.** Partial zips and dual-train collisions corrupt the registry.

9. **Sealed holdout map hashes before map-pack curriculum unlock and before cross-map transfer claims.** Curriculum without sealed eval = training on the test set by drift.

10. **LiDAR stack proven under stress eval before camera MAJOR or LiDAR→cam distillation fantasies.** Vision work must not reopen P0.

**Safe to parallelize (after their gates):** Control UI copy/UX with P1 training; Watch overlays with P1 (not in-process); docs/teaching with any phase; aesthetic/demo skins after P1 exit; research stretch (Dreamer, meta-RL) as a **side lane** that must not steal the P1 exit metric or break contracts.

---

## Sequencing one-liner

**Freeze contract → measure adjusted race score vs FTG → win that in gym on holdouts → only then bridge/sim2sim with veto & latency → only then Jetson/car → camera last as a new contract.**
