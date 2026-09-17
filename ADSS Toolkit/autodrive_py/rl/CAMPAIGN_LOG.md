# F1TENTH RL Continuous Improvement Campaign

**Start:** 2026-09-17 03:54 America/Chicago  
**Deadline:** ~2026-09-17 12:54 America/Chicago (9 hours) or credits exhausted  
**Branch:** `rl/phase-1-research`  
**Head note:** `5e89e7a` pushed; campaign docs commit separately as they land.

---

## Keep-alive loop (mandatory — ~03:58+)

- **Sentinel:** `AGENT_LOOP_TICK_autodrive-rl-campaign` (~50 min).
- **On EVERY tick:** parent MUST execute [`campaign_tick.md`](campaign_tick.md) — full review board (Researcher → Racer/Minimalist/Reliability votes → Orchestrator ships approved only). **Not orchestrator-only.**
- **Deadline:** after 12:54 America/Chicago 2026-09-17, STOP loop; do not re-arm.
- Prior loop PID ~46964 (orchestrator-only prompt) aborted; re-armed with full-board prompt pointing at `campaign_tick.md`. Do not kill `train_ppo` when re-arming.

## Governance (mandatory — 03:57+)

1. **Ingest before ship:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_{racer,minimalist,reliability}.md` + `CAMPAIGN_RESEARCH.md` when present.
2. **Before each non-trivial feature:** CAMPAIGN_LOG / `CAMPAIGN_BOARD.md` Go/No-Go with research checklist + ≥2 personality opinions (agree/dissent).
3. **Prefer:** overnight survival, race-score honesty, steps/sec truth, operator Start/Stop/Continue — over cosmetics and GPU theater.
4. **Fail review → SKIP** and log why. Ship reviewed work only; commit+push frequently.
5. **Sacred:** early-stop defaults (select≥220, eval≥50k, warmup≥2, min_ts≥100k), contracts `2.0.0`, holdouts, atomic promote, dual-writer refuse. Cut the feature, not the guard.
6. **Never kill** `overnight_soak_20260917_082739` without Continue restart.

**Review intake status**

| Doc | Status |
| --- | --- |
| `CAMPAIGN_CRITIQUE.md` | READ |
| `campaign_minimalist.md` | READ |
| `campaign_racer.md` | READ |
| `campaign_reliability.md` | READ |
| `CAMPAIGN_RESEARCH.md` | READY (tick 0) |
| `CAMPAIGN_BOARD.md` | OPEN (tick 0) |

MUST 1–5 gated by critique+racer+minimalist; reliability **vetoes** unattended continuous until dual-writer/heartbeat notes addressed. Board owns per-tick Go/No-Go.

---

## Protected training run

| Field | Value |
| ----- | ----- |
| **run_id** | `overnight_soak_20260917_082739` |
| **Status** | ACTIVE — DO NOT KILL without Continue restart |
| **live_status (~03:57)** | timesteps≈198k, steps/s≈269, phase=learning, n_envs=8, vec=subproc |
| **PIDs (last seen)** | 31472, 34256 |
| **GPU** | ~20–40% util — evidence for **KILL GPU theater**, not a mandate |

---

## Aligned backlog (critique + racer + minimalist)

### MUST (ship in order)

| # | Item | Critique | Racer | Minimalist | Gate |
| - | ---- | -------- | ----- | ---------- | ---- |
| 1 | Overnight early-stop regression lock | MUST | MUST #1 | KEEP #1 | **GO** — verify/smoke only; no constant lowers |
| 2 | Continue-train soak (Stop→Continue same run_id) | MUST | MUST #2 | KEEP #2–3 | **GO** when reliability not dissenting; prove with UI soak |
| 3 | Collision-first path → crash_rate down | MUST | MUST #3 | KEEP #7 | **GO** — short ≤100k smoke; no reward Discord; don't touch overnight process |
| 4 | Honest official FTG vs PPO row | MUST | MUST #4 | KEEP #8 | **GO** — read-only eval vs pinned FTG; seals intact |
| 5 | Holdout/validation refuse + pack verify | MUST | MUST #5 | KEEP #5 | **GO** — smoke argv/UI refuse |

### NO-GO / KILL (this 9h)

| Urge | Why | Sources |
| ---- | --- | ------- |
| GPU / CUDA / AMP / compile | CPU VecEnv-bound; idle GPU expected | all three |
| Watch / coach chrome | Unofficial; KPIs shipped | all three |
| Bridge / Jetson / camera | Phase 3; holdout beat-FTG gate | all three |
| Algo zoo / residual / FTG-BC warm-start | Before clean-lap + overnight survival | all three |
| Continuous-until-plateau outer product | Dual-writer + EarlyStop traps; don't touch overnight math | minimalist+racer **KILL**; critique TRAP |
| Drastic UI reskin / second UI | Operator min only; not prettier pit wall | minimalist+racer |
| Faster val that becomes race score | Official timeout stays honest | racer |
| 500k–50M defaults before crash≈0 | Farms death | all three |

### Prior user-ideas row (SUPERSEDED 03:58)

Earlier log said “keep continuous scaffold / drastic UI / GPU diagnose as Keep.” **Retracted.** Continuous outer loop and UI chrome are **NO-GO this window**. Throughput honesty (log steps/sec, tune n_envs knee) OK; CUDA theater is not.

---

## Go/No-Go — next ship candidates

### A. Overnight early-stop regression audit (MUST #1)

- **Checklist:** Do not change select/eval/warmup/min_ts defaults downward; re-run `_overnight_soak_smoke` only if touching RaceBest/ui_ops; overnight train process untouched.
- **Racer:** agree (MUST #1).
- **Minimalist:** agree (KEEP #1).
- **Verdict: GO** — audit constants + smoke if needed; no feature invent.

### B. Continuous-train CLI/UI toggle scaffolding

- **Racer:** KILL #5 (map product / continuous outer).
- **Minimalist:** refuse unlimited continuous without discipline.
- **Verdict: NO-GO** — skip; log: fails personality review.

### C. Holdout refuse smoke (MUST #5)

- **Checklist:** `train_ppo --map map2/map3` exit≠0 without allow; pack verify.
- **Racer + Minimalist:** agree.
- **Verdict: GO** — parallel to overnight; no train kill.

---

## Timeline

### 03:54–03:57 — Snapshot

- Protected `overnight_soak_20260917_082739`; CURRENT_RUN refreshed.
- Governance added after user mandate; speculative first-90m (GPU speedups, continuous scaffold, UI panels) **halted pending review**.

### 03:58 — Critique + racer + minimalist folded

- Backlog aligned to MUST 1–5 / KILL list above.
- Next gated actions: (1) constant audit + holdout refuse smokes, (2) Continue soak proof, (3) collision-first short compare — all without killing overnight.

### ~03:58+ — Keep-alive re-armed for FULL board

- Wrote `campaign_tick.md` (exact parent tick prompt).
- Old orchestrator-only loop (~46964) was aborted; new loop emits full-board `AGENT_LOOP_TICK` (~50m) until 12:54 Chicago.
- Parent on each tick: execute `campaign_tick.md` (Researcher → votes → Orchestrator). `train_ppo` overnight left running.

---

## Shipped / Tests

_(appended as reviewed work lands)_
