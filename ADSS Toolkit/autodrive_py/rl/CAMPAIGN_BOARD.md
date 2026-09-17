# CAMPAIGN_BOARD — Standing researcher ticks

**Campaign:** 9h (start ~03:54 America/Chicago, 2026-09-17)  
**Protected run:** `overnight_soak_20260917_082739` — DO NOT KILL without Continue  
**Sources of truth for MUST order:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_racer.md` (+ minimalist KEEP / reliability vetoes)  
**Companion brief:** `CAMPAIGN_RESEARCH.md` (COMPLETE)

Append one entry per resume tick. Newest first.

---

## PAUSE+SHIP — 2026-09-17 ~04:01 — Tick0 smokes only (reliability P0 in flight)

- **id:** `20260917-T0-must-smokes`
- **type:** chunk
- **proposal:** Ship Tick0 verify/smokes only: holdout refuse CLI, overnight constant audit, read-only official FTGΔ, disposable collision-first A/B. Continue soak deferred. No continuous loop. Do not kill overnight. Defer feature spam until heartbeat/kill-tree patches land.
- **overnight plan:** observe-only `overnight_soak_20260917_082739`
- **Researcher:** Go — Tick0 all five MUSTs for verify/smoke.
- **Racer:** Approve — MUST order; FTGΔ read-only.
- **Minimalist:** Approve — smokes only.
- **Reliability:** Approve smokes; **Block** any Start/Continue that could `_kill_train_tree` overnight until P0 patches land.
- **Integrator:** **SHIP** smokes (W1a/b holdout+constants, FTGΔ eval, collision A/B on disposable run_ids). **DEFER** Continue-on-soak + UI Start paths that kill-tree.

---

## W2 — FAST wave triage — 2026-09-17 ~03:59 America/Chicago
- **id:** `20260917-W2-fast-wave-triage`
- **type:** tick / wave
- **proposal:** Standing board FAST vote on 7 next-wave candidates. SHIP only if research Go + ≥2 of {Racer, Minimalist, Reliability} Approve + 0 Blocks. Overnight observe-only.
- **overnight plan:** Do not kill/morph `overnight_soak_20260917_082739`. Collision-first = parallel short smoke. Continue soak = disposable run_id or post-soak only.

### Candidates

#### 1. Collision-first UI exposure → **SHIP**
- **Researcher:** **Go** — MUST #3; `--collision-first` wired; overnight flag off so live crash_rate≠A/B; ≤100k twin smoke (RESEARCH §7#3).
- **Racer:** **Approve** — MUST #3; λ=10 → crash↓ is the race.
- **Minimalist:** **Approve** — expose existing flag; no new subsystem.
- **Reliability:** **Approve** — parallel smoke only; no mid-run morph of protected soak.
- **Next:** UI checkbox → short A/B on `crash_rate_estimate` vs baseline.

#### 2. Continue soak / resume reliability → **SHIP**
- **Researcher:** **Go** — MUST #2; `continue_train_argv` exists; PLAN_PROGRESS still open; prove Stop→Continue same `run_id` from complete ckpt (RESEARCH §7#2).
- **Racer:** **Approve** — MUST #2; dead resume = zero finishers.
- **Minimalist:** **Approve** — KEEP #2–3; dual-writer refuse already sacred.
- **Reliability:** **Approve** — north star; refuse while lock/live; soft-stop copy OK; never kill overnight to “prove.”
- **Next:** disposable-run Continue soak (or post-soak).

#### 3. Fast validation probe (honest) → **SHIP**
- **Researcher:** **Go** — RESEARCH §3 progress probe; label `kind=smoke`/status-only; no promote; do not shorten official 400s.
- **Racer:** **Approve** — early “is learning?” OK if it cannot cheat beat-FTG.
- **Minimalist:** **Approve** — thin status/`live_status` only; refuse board pollution.
- **Reliability:** **Approve** — must not densify mid-train eval or touch sacred select/floor.
- **Constraint:** smoke/status only; never promote / never `kind=official`.

#### 4. n_envs knee / steps-sec coach → **DEFER**
- **Researcher:** **Conditional** — RESEARCH §1 ROI is knee+sps honesty, but **after** Continue+collision MUST; coach already warns sps&lt;30 @ n_envs&gt;8.
- **Racer:** **Abstain** — throughput later; not podium until crash↓.
- **Minimalist:** **Approve** — measure/coach only; no CUDA.
- **Reliability:** **Approve** — OK if overnight untouched.
- **Why not SHIP:** Conditional unmet (MUST #2/#3 first). Re-vote after collision+Continue.

#### 5. Continuous outer retrain loop → **SKIP**
- **Researcher:** **No-Go** — RESEARCH §2/§6 + §7 reliability veto; P0 dual-writer / EarlyStop replay.
- **Racer:** **Block** — KILL #5; scaffold later, not mid-soak.
- **Minimalist:** **Block** — continuous cosplay; refuse.
- **Reliability:** **Block** — P0 kill overnight / dual-Start / sacred-math regression.

#### 6. Big UI visual redesign → **SKIP**
- **Researcher:** **No-Go** — RESEARCH §4 design test fail (wrong Starts / lost Continues / misread banners only).
- **Racer:** **Block** — chrome ≠ adjusted_time.
- **Minimalist:** **Block** — pixels ≠ policy; operator min already shipped.
- **Reliability:** **Block** — reskin risks dropping Overnight preset knobs.
- **Note:** soft-stop one-liner allowed under Continue (#2), not a reskin.

#### 7. GPU AMP / torch.compile → **SKIP**
- **Researcher:** **No-Go** — env-bound PPO; soak GPU 20–40% idle; theater until sps knee (RESEARCH §1/§6).
- **Racer:** **Block** — KILL #1; VRAM% ≠ podium.
- **Minimalist:** **Block** — refuse CUDA cosplay this window.
- **Reliability:** **Abstain** — not sacred, still waste.

### Integrator gate
**SHIP now (order):** (1) collision-first UI + short A/B → (2) Continue soak proof → (3) honest labeled validation probe.  
**DEFER:** n_envs knee/coach.  
**SKIP:** continuous outer loop · big UI redesign · GPU AMP/compile.  
Orchestrator may implement **SHIP items only**.

---

## Tick 0 — 2026-09-17 ~03:59 America/Chicago (first board)

**Live soak:** timesteps≈208k · steps/s≈245 · phase=`learning` · crash_rate≈0.0 · n_envs=8 subproc · lock held  
**Config sacred check:** eval_every=50k · select_timeout=220 · warmup=2 · min_ts=100k · patience=5 · unlimited · allow_holdout=false  
**Flag note:** soak `collision_first=false` (expected — curriculum A/B is separate)

### MUST Go / No-Go (orchestrator next actions)

| # | Item | Go/No-Go | Evidence | Action now | Do not |
| - | ---- | -------- | -------- | ---------- | ------ |
| 1 | **Overnight early-stop regression lock** | **GO** (verify only) | Constants in `ui_ops.py` match sacred floors; live `config.json` mirrors them; run already past min_ts grace without EarlyStop | Constant audit + leave soak alone; re-smoke `_overnight_soak_smoke` **only if** editing RaceBest/`ui_ops` | Lower select/eval/warmup/min_ts; kill overnight; “tune” early-stop for continuous |
| 2 | **Continue-train soak** | **GO** (prove — still deferred) | Path shipped (`continue_train_argv`, ui_selftest); PLAN_PROGRESS still **P1**: overnight Continue-after-Stop soak | Prove on a **short disposable** run (Stop→complete ckpt→Continue same `run_id`→timesteps↑) **or** after soak ends | Stop protected soak just to demo Continue |
| 3 | **Collision-first → crash_rate down** | **GO** (short A/B) | `--collision-first` wired; overnight flag off; live crash_rate=0 is **not** controlled evidence | ≤100k twin smokes ±flag; compare `live_status.crash_rate_estimate`; no mid-run reward Discord | Morph overnight rewards/flags; raise budget to 500k+ before crash moves |
| 4 | **Official FTGΔ (PPO vs pinned FTG)** | **GO** (read-only eval) | FTG `official_v2` finisher ≈**198.16** s on board; **zero** PPO `official_v2` rows | Official eval of current `best_model` / last complete zip under `official_v2`; publish Δ even if lose; seals intact | Shorten official timeout; retune FTG; call mid-train 220s “official” |
| 5 | **Holdout / validation refuse** | **GO** (smoke) | `assert_train_safe` + ui_selftest Start refuse map2/map3; soak allow flags false | CLI refuse smoke + pack `verify` before any official append | Quiet `--allow-holdout`; train map3 “for convenience” |

### Personality consensus (tick 0)

- **Racer:** MUST 1→5 order stands; FTGΔ without a PPO official row is flying blind.
- **Minimalist:** KEEP overnight/Continue/holdout/collision/FTGΔ; refuse continuous outer + GPU theater.
- **Reliability:** Continuous outer **NO-GO** until heartbeat-join + Start-kill-on-false-busy are addressed; don’t touch overnight math to feel “continuous.”

### Priority for next ~50 min (if orchestrator picks one ship)

1. Holdout refuse CLI smoke (cheap, parallel to overnight) **and/or** overnight constant audit (read-only).  
2. Schedule Continue soak on a **non-protected** short run.  
3. Queue collision-first A/B when a free GPU/CPU slot exists (not the overnight workers).  
4. When a complete zip is stable enough, kick **official** PPO vs FTG (read-only).

### Explicit NO-GO this tick

GPU/CUDA/AMP · Watch chrome · Bridge/Jetson · algo zoo / residual / FTG-BC · continuous-until-plateau product · densifying mid-train eval · killing `overnight_soak_20260917_082739`

---
