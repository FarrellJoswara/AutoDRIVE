# 9h campaign critique (adversarial)

Audience: orchestrator. Grounded in `PLAN.md` / `IDEAS_TRIMMED.md` + current `control_ui` / `train_ppo` / `PLAN_PROGRESS` (Sep 2026). **Do not implement from this doc — prioritize.**

**Verdict:** Phase 0 honesty + operator loop are largely shipped. The remaining 9h win is **race skill + overnight survival**, not more UI chrome. Beat-FTG on holdouts is still unclaimed; burning hours on GPU, Watch cosmetics, or bridge is a trap.

---

## 1. High-ROI vs traps

| Request / urge | ROI | Why |
| --- | --- | --- |
| **Fix/harden overnight early-stop + Continue soak** | **HIGH** | False EarlyStop burned overnight already; warmup/grace/select_timeout=220 shipped — **do not regress**. Continue-after-Stop soak still deferred. |
| **Collision-first curriculum that visibly drops crash rate** | **HIGH** | PLAN P1 exit unchecked; without it speed unlock / longer runs farm death. |
| **Honest official compare: current PPO vs pinned FTG** (even if lose) | **HIGH** | North star is `adjusted_time`; without a fresh official row you’re flying blind. |
| **Multi-map train + sealed holdout refuse (smoke)** | **HIGH** | map_pack exists; still easy to train-on-holdout via UI checkbox or argv mistakes. |
| **Light LiDAR DR / clearance features / residual FTG** | **MED** | Only after crash rate drops and one serious overnight survives. Residual before clean-lap is cosplay. |
| **“Use the GPU / CUDA / AMP / torch.compile”** | **TRAP** | Gym step is **CPU-sim + VecEnv** bound. Log `steps/sec` + GPU util first; expect near-idle GPU. Tuning `n_envs` / Dummy vs Subproc knee beats CUDA theater. |
| **Fancy Watch / coach polish / more banners** | **TRAP** | Watch KPIs + ghost + mismatch already shipped. More pixels ≠ better policy. Watch stays **unofficial**. |
| **Map gen UI / model timeline chrome** | **LOW / done** | Already in Control UI. Don’t re-skin. |
| **Warm-start FTG/BC, residual RL, algo zoo** | **TRAP (this 9h)** | Serialization: metric + overnight + collision first. Residual only if pure PPO plateaus *after* finishing laps. |
| **Bridge / `:4567` / Jetson / camera** | **HARD SKIP** | Phase 3 gated on holdout beat-FTG. |
| **Continuous train without budget discipline** | **TRAP** | Unlimited + patience without warmup/min_ts/sparse eval → EarlyStop at ~40–100k. Dual Start / same `run_id` dual-write corrupts zips. Soft-stop ≠ hard kill — document, don’t invent Discord reward dials. |
| **500k / 5M “serious” defaults before crash rate ~0** | **TRAP** | PLAN: shorter serious runs until metric truth; crash-first before aggression. |

**User-request pattern:** Anything that *looks* productive (GPU %, prettier panel, new algo) but doesn’t move **official adjusted_time vs FTG on sealed maps** or **overnight continue integrity** is a trap for this window.

---

## 2. Ordered 9-hour backlog

### MUST (~4–5h)

1. **Regression lock overnight early-stop** — Keep: select_timeout ≥220, auto eval floor ≥50k, warmup≥2, min_ts≥100k, Overnight preset, `phase=early_stopped` preserved, cp1252-safe prints. Re-run soak smoke if touching `RaceBestModelCallback` / `ui_ops` constants.
2. **Continue-train soak** — UI Stop → Continue same `run_id` from last **complete** ckpt; lock cleared; contracts refuse on mismatch; timesteps resume without dual-writer.
3. **Collision-first default path that moves crash rate** — Short run (≤100k) with flags on; crash_rate in `live_status` down vs baseline; no reward Discord mid-run.
4. **Official FTG vs PPO row** — `eval_cli` / compare under `official_v2`; seals intact; non-null adjusted_time; lose honestly if needed.
5. **Holdout/validation refuse** — Start without `--allow-holdout` dies on map2/map3/map4; pack `verify` before official append.

### SHOULD (~3–4h)

6. Multi-map overnight smoke (`map0,map1`) + fingerprint includes pack hashes.
7. Soft-stop vs hard-kill copy in UI (one sentence + banner); no new stop machinery unless Continue soak fails.
8. If collision rate drops: light LiDAR DR **or** engineered clearances — **one** of them, not both.
9. Raise mid-train select only if finishers needed for promote; keep official timeout=400.

### SKIP (this 9h)

- Bridge, Jetson, camera, SAC/Dreamer/PBT, Watch aesthetics, seasons/XP, TensorRT, map-regen-in-hot-loop, second UI, unconstrained on-car PPO, GPU optimization before measured steps/sec knee.

---

## 3. Safety — never break

| Sacred | Rule |
| --- | --- |
| **Contracts `2.0.0`** | No obs-dim / action-space change. Refuse-load on mismatch everywhere (eval, resume, Watch, Continue). |
| **Overnight early-stop fixes** | Do not lower `MID_TRAIN_SELECT_TIMEOUT_S` back to 60; do not drop auto eval floor to 10k; do not remove warmup/min_timesteps; do not let `LiveStatusCallback` overwrite `early_stopped`. Overnight preset must stay budget-OFF + patience>0 + grace. |
| **Holdouts / validation pin** | `assert_train_safe`; map3 never train-default; sealed content_hash; official claims call `assert_seals_intact`. UI “allow sealed” voids claims — keep loud. |
| **Promotion honesty** | `best_model` / recommended = `race_score_key` on validation/official only — never `ep_rew_mean`. Atomic zip only; incomplete never promoted. |
| **Dual-writer** | `train.lock` + `_train_busy` refuse Double-Start / Continue on live run. |
| **Protocol** | FTG + PPO same seeds/episodes/timeout; λ=10; don’t retune FTG to “win” mid-campaign. |

If a feature fights any row above: **cut the feature**, not the guard.

---

## 4. Acceptance tests (major features)

| Feature | Pass iff |
| --- | --- |
| **Overnight early-stop** | `python -m rl._overnight_soak_smoke` (or equiv): ≥ warmup+patience Validating→Learning cycles; **no** `early_stopped` before grace; UI build stamp still `overnight-earlystop-*`. Manual: Overnight preset → run >100k steps without EarlyStop solely from first DNF plateau. |
| **Continue after Stop** | Start → ≥1 complete ckpt → Stop (lock cleared) → Continue same `run_id` → `live_status.timesteps` rises past prior; refuse Continue while locked/live. |
| **Dual-Start refuse** | Second Start while train alive → plain English refuse; one writer under `models/<run_id>/`. |
| **Contracts refuse** | Load/resume/eval v1 or wrong obs_dim → fail closed, clear message (`ui_selftest` / `_p0_checks`). |
| **Holdout gate** | `train_ppo --map map2` / `map3` exit ≠0 without allow flags; UI preview REFUSED without checkbox. |
| **Seal integrity** | Tamper holdout centerline → `map_pack verify` fails; official append blocked. |
| **best_model honesty** | Mid-train promote only when race key improves on validation map; `best_model_meta` cites race metrics not `ep_rew`. |
| **Collision-first** | Same seed budget: crash_rate_estimate (or crash eps) **lower** with flag than without on a short smoke. |
| **Official compare** | Leaderboard: FTG pin + ≥1 PPO `kind=official`, non-null `adjusted_time`, same `protocol_id`; `compare_models --official` prints Δt without ranking DNF proxy over finishers. |
| **Throughput honesty** | `live_status` steps/sec >0; Dummy fallback sets `vec_env_active` truth; coach may flag low steps/sec — **not** “buy GPU”. |
| **Watch (if touched)** | Overlay still labels lag-behind / unofficial; map/contracts mismatch banners; `test_watch_overlay` green. Promotion path unchanged. |

---

## Orchestrator one-liner

**Protect overnight early-stop + contracts + holdouts → prove Continue → drop collisions → publish honest FTGΔ → only then DR/residual. Skip GPU theater, Watch chrome, and bridge.**
