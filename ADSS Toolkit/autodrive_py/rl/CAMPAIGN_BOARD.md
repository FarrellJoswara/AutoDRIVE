# CAMPAIGN_BOARD — Standing researcher ticks

**Campaign:** 9h (start ~03:54 America/Chicago, 2026-09-17)  
**Protected run:** `overnight_soak_20260917_082739` — DO NOT KILL without Continue  
**Sources of truth for MUST order:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_racer.md` (+ minimalist KEEP / reliability vetoes)  
**Companion brief:** `CAMPAIGN_RESEARCH.md` (COMPLETE)

Append one entry per resume tick. Newest first.

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
