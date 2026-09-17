# CAMPAIGN_BOARD — Standing researcher ticks

**Campaign:** 9h (start ~03:54 America/Chicago, 2026-09-17)  
**Protected run:** `overnight_soak_20260917_082739` — DO NOT KILL without Continue  
**Sources of truth for MUST order:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_racer.md` (+ minimalist KEEP / reliability vetoes)  
**Companion brief:** `CAMPAIGN_RESEARCH.md` (COMPLETE)

Append one entry per resume tick. Newest first.  
**Soft inbox:** [`CAMPAIGN_SUGGESTIONS.md`](CAMPAIGN_SUGGESTIONS.md) (Suggestion seat — not mandates)  
**UI inbox:** [`CAMPAIGN_UI.md`](CAMPAIGN_UI.md) (UI seat — incremental Control/Watch chrome; not mandates)

---

## RESULTS — W10 idea-factory cycle 2 — ~04:37 America/Chicago
- **id:** `20260917-W10-idea-factory`
- **type:** wave / result
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739`. Stall/P1-8 land in-memory only after next train spawn/Continue.
- **Prior tick ship:** Watch validating ≠ TRAIN STALE (`ad5dd05`).
- **Ideas:** 7 soft ideas in `CAMPAIGN_SUGGESTIONS.md` Wave 3 (`S-W10-01…07`).

### Triage

| Id | Idea | Gate |
| -- | ---- | ---- |
| S-W10-01 | Preserve crash/stall counters across RaceBest validating (P1-8) | **SHIP** |
| S-W10-02 | Stall rate live_status + Control row | **SHIP** |
| S-W10-03 | Watch compact Focus run_id | **DEFER** |
| S-W10-04 | atomic_write_json Win lock retry (P1-4) | **DEFER** |
| S-W10-05 | Spawn jitter occupied-base refuse (P1-6) | **DEFER** |
| S-W10-06 | Obs buffer prealloc (PERF-6) | **DEFER** |
| S-W10-07 | Continuous outer / reward Discord | **SKIP** (hard) |

### Personality votes — P1-8 + stall → **SHIP**

- **Researcher:** **Go** — grounded in MUST #3 inconclusive crashΔ + BUGFIX P1-8; cite RaceBest `_write_status` blanking trail.
- **Racer:** **Approve** — stall% vs crash% honesty helps collision-first A/B; Block denser eval.
- **Minimalist:** **Approve** — merge keys + one Control field; refuse PERF epics.
- **Reliability:** **Approve** — merge-only; overnight observe-only.
- **Integrator:** **SHIP** S-01/02; `UI_BUILD=w10-stall-p18-20260917`.

**Resume-me:** Next cycle — P1-4 lock retry or Watch pin overlay; never overnight kill.

---

## Wave — Close official FTGΔ (+8.52 s) + stall-metric — ~04:33 America/Chicago
- **id:** `20260917-ftgdelta-close-wave`
- **type:** tick / research wave + thin SHIP
- **overnight:** observe-only — `overnight_soak_20260917_082739` ~**569k** `learning`, sps≈203, n_envs=8 subproc, PIDs **19244/53852** ALIVE; CPU≈77%/20 logical — **do not starve**; **no kill / no morph / no sacred early-stop retune**
- **context:** MUST #4 sealed lose (PPO `official_v2` **206.68** vs FTG **198.16**, Δ=+8.52, finisher). MUST #3 collision A/B **crashΔ null** at 24k **and** 50k (stall-heavy; both arms crash_rate=0.0) → **DEFER** further crash A/B smokes tonight.

### Collision A/B residual

| Id | Idea | Gate |
| -- | ---- | ---- |
| C-01 | Longer disposable collision-first A/B (50–100k+) for crashΔ | **DEFER** tonight — need longer budget **and/or** a regime where wall-hits actually occur; overnight CPU already high; 50k already null |

### FTGΔ close — thin ideas (multi-lens)

| Id | Idea | Gate |
| -- | ---- | ---- |
| F-01 | Let overnight RaceBest improve past 295k snapshot; **re-eval official_v2 when CPU free** (sidecar will localize Δ) | **DEFER** eval launch now (anti-thrash / soak learning) — schedule next free window |
| F-02 | Persist official **per_map** sidecar on leaderboard append (`logs/official_per_map_<run_id>.json`) | **SHIP’d** `metrics_io.write_official_per_map_sidecar` |
| F-03 | Live **stall_rate_estimate** / `stalls_estimate` (Monitor `stall` already in keywords) | **SHIP’d** `live_status` — diagnoses stall-heavy curriculum without Discord |
| F-04 | Disposable reward-shaping fork (speed unlock / stall soft-penalty) fingerprinted new `run_id` | **DEFER** — design only until F-01 re-eval + overnight survival; no mid-soak Discord |
| F-05 | Observation add-ons (L/R/F clearances / DR) | **DEFER** — contracts risk; after longer train plateaus |
| F-06 | Sacred early-stop floor retune to “ship faster” | **SKIP** (hard) |
| F-07 | Kill overnight / third concurrent official eval thrash | **SKIP** (hard) |

**Honesty note:** mid-train RaceBest `train_eval` ≈**188.95 s** on **map3** @ 220 s ≠ official. Gap to close is sealed-map lap time (+8.52 s), not crash DNF (official collisions=0, progress≈0.990).

### Personality votes — FTGΔ thin tools → **SHIP (F-02/F-03)**; DEFER rest

- **Researcher:** **Go** F-02/F-03 — falsifiable next official; stall metric explains MUST #3 nulls. **Defer** F-01 launch while soak learning. **No-Go** F-06/F-07.
- **Racer:** **Approve** F-02 (localize where seconds leak) · **Approve** F-03 (stall≠crash) · **Block** F-06/F-07 · F-04 only after overnight continues + honest re-eval · **Approve** C-01 DEFER (crash theater tonight).
- **Minimalist:** **Approve** F-02/F-03 (tiny side-channel + one status field) · **Block** reward Discord / obs contract churn / denser overnight eval.
- **Reliability:** **Approve** F-02/F-03 (append-only sidecar; additive live_status keys; overnight process untouched — new fields only after Continue/restart workers) · **Block** F-07 · no sacred floor change.
- **Integrator:** **SHIP** F-02 + F-03 this tick; **DEFER** C-01 + F-01/F-04/F-05; **SKIP** F-06/F-07. Need research Go + ≥2 Approves — **met**.

**Orchestrator:** docs + thin code landed; bugfix asserts green; commit board/log + `live_status`/`metrics_io`/`_bugfix_p0_checks`. Overnight untouched.

**Resume-me:** when overnight not validating / CPU free → official re-eval of newer RaceBest → read `official_per_map_*.json` vs FTG pin; only then consider F-04 disposable fork.

---

## RESULTS — Tick #2 wave SHIP (P1-1/3/5 + W8) — ~04:32 America/Chicago
- **id:** `20260917-tick2-wave-ship`
- **type:** result / implement
- **amends:** `20260917-tick2-wave`
- **overnight:** observe-only — ALIVE ≈**563k** learning, PIDs **19244/53852**; no kill/Continue
- **T2-01 P1-1:** **DONE** — `AtomicCheckpointCallback` → `atomic_save_sb3`; zip `testzip` in `find_last_complete_checkpoint` (also in `815ad27` lineage)
- **T2-02 P1-3:** **DONE** — `CorruptManifest` on bad JSON/shape
- **T2-03 P1-5:** **DONE** — `MONITOR_INFO_KEYWORDS` in `_make_env`
- **T2-04 W8:** **DONE** — live_runs / bound / Focus / Start warn (`UI_BUILD=w8-multi-run-20260917`)
- **Tests:** `_bugfix_p0_checks` **25/25**; `ui_selftest` **all passed**
- **DEFER/SKIP:** T2-05 A/B crashΔ · T2-06 GPU/continuous
- **Note:** W9 S-W9-05 Atomic CheckpointCallback was DEFER’d to this wave — now closed.

---

## RESULTS — W9 idea-factory cycle 1 — ~04:31 America/Chicago
- **id:** `20260917-W9-idea-factory`
- **type:** wave / result
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739` (selftest Stop mocked; Focus pin only).
- **Ideas:** 8 soft ideas in `CAMPAIGN_SUGGESTIONS.md` Wave 2 (`S-W9-01…08`).

### Triage

| Id | Idea | Gate |
| -- | ---- | ---- |
| S-W9-01 | Coach hint when N≥2 live locks | **SHIP** |
| S-W9-02 | Selftest real zips + ASCII Focus (cp1252) | **SHIP** |
| S-W9-03 | Watch overlay shows Focus pin run_id | **DEFER** |
| S-W9-04 | Leaderboard protocol honesty tooltip | **DEFER** |
| S-W9-05 | Atomic CheckpointCallback | **DEFER** (Bugfix P1 / Tick2) |
| S-W9-06 | Train spawn pose jitter | **DEFER** |
| S-W9-07 | HANDOFF Focus one-liner | **SHIP** |
| S-W9-08 | GPU theater / denser overnight eval | **SKIP** (hard) |

### Personality votes — thin SHIP batch → **SHIP**

- **Researcher:** **Go** — W8 already enumerates locks; coach/selftest/docs close operator foot-guns without touching sacred floors. Cite `coach_hints` + `_zip_looks_complete` + HANDOFF Multi-run.
- **Racer:** **Approve** — no race math; Block denser eval (SKIP'd).
- **Minimalist:** **Approve** — one coach line + fake-zip fix + docs; refuse spawn jitter / ckpt epic this tick.
- **Reliability:** **Approve** — keep zip completeness gate; ASCII msgs avoid selftest crash; overnight observe-only.
- **Integrator:** **SHIP** S-01/02/07 only. `UI_BUILD=w9-idea-factory-20260917`. `ui_selftest` green.

**Resume-me:** Next idea-factory cycle (~20–30m): invent new Wave 3 ideas; optionally ship Watch pin overlay (S-03) if free; never touch overnight.

---

## Tick #2 (~04:26 America/Chicago) — NEXT multi-proposal batch
- **id:** `20260917-tick2-wave`
- **type:** tick / proposal batch (fresh parallel wave executor — standing IDs parent-mismatch)
- **overnight:** **ALIVE** observe-only — `overnight_soak_20260917_082739` ≈498–545k+, PIDs **19244/53852**, phase learning/validating; **no Continue-restart needed**
- **context:** MUST #1/#2/#5 PASS earlier; MUST #4 FTGΔ **DONE** (PPO 206.68 vs 198.16, Δ+8.52); MUST #3 crash↓ still inconclusive; open bugfix P1 + W8 implement

### Proposals

| Id | Proposal | Researcher | Gate |
| -- | -------- | ---------- | ---- |
| T2-01 | **P1-1** Atomic CheckpointCallback (`atomic_save_sb3`) + zip-test in `find_last_complete_checkpoint` | **Go** — Continue integrity; overnight-safe (library only until next train) | **SHIP** |
| T2-02 | **P1-3** `load_pack` refuse corrupt non-empty `map_pack.json` (`CorruptManifest`) | **Go** — holdout integrity; closes silent-empty→train_ok hole | **SHIP** |
| T2-03 | **P1-5** `MONITOR_INFO_KEYWORDS` constant wired in `_make_env` + bugfix assert | **Go** — crash_rate honesty for future A/B | **SHIP** |
| T2-04 | **W8-01…04** implement multi-run Control list / bind / targets / Start warn (design already SHIP) | **Go** — operators already multi-lock; no kill multi-train | **SHIP** |
| T2-05 | Re-run long collision-first A/B for MUST #3 crashΔ | **Go** when free — **after** overnight not validating; disposable only | **DEFER** this tick (anti-thrash) |
| T2-06 | GPU / AMP / continuous outer / fancy dash | **No-Go** | **SKIP** |

### Personality votes

- **Racer:** **Approve** T2-01/02/03 (Continue + seals + crash honesty) · **Approve** T2-04 thin only · **Block** T2-06 · T2-05 later when overnight free.
- **Minimalist:** **Approve** T2-01/02/03 (small integrity fixes) · **Approve** T2-04 if same-DOM list only · **Block** frameworks / denser overnight eval.
- **Reliability:** **Approve** T2-01/02 (atomic ckpt + corrupt refuse) · **Approve** T2-03 · **Approve** T2-04 read-only + mocked Stop · **Block** any Stop/kill vs overnight · dual-writer/sacred floors intact.
- **Integrator:** **SHIP** T2-01…04 this tick; **DEFER** T2-05; **SKIP** T2-06. Need research Go + ≥2 Approves — **met**.

**Orchestrator:** implement T2-01…04 now; commit+push; append Timeline; leave overnight untouched.

---

## Meta-spawner coverage
- **Covered by existing:** orchestrator · board/ideas · continuous commits/support-loop · cleanup · docs/HANDOFF · W8 multi-run UI · bugfix (P0/P1-2/P1-3 + P1-8 ticket) · suggestion/UI bots · Tick0 MUST · secrets/onboarding · watch honesty
- **Gaps found (~04:28, refreshed ~04:40):** MUST #3 crashΔ still inconclusive (P1-8) · MUST #4 FTGΔ **DONE** · ~~secrets~~ · ~~watch-honesty~~ · P1-1 atomic ckpt (T2 SHIP in flight) · CURRENT_RUN pin selftest · venv health
- **Spawned:**
  - [Collision A/B analyzer] → `rl/_campaign_ab_report.py` — inconclusive; `20260917-meta-ab-report`
  - [FTGΔ collector] → **DONE** `20260917-meta-ftg-delta` — PPO 206.68 vs FTG 198.16 (**Δ +8.52 s**)
  - [Secrets + friend-onboarding] → **DONE** `20260917-meta-secrets-onboard` — CLEAN / PARTIAL
  - [Watch overlay honesty] → **DONE** `20260917-meta-watch-honesty` — validating≠stale; pin via `read_operator_run_pin`; `test_watch_overlay` **14/14**
- **Next spawn candidates:** **P1-8** crash_rate merge on val trail · P1-1 atomic ckpt · CURRENT_RUN pin selftest · P2-1 CreateTime lock · venv health · auto_train pause-when-locks
- **Anti-thrash:** soft pause new disposable trains; **DEFER crash A/B smokes tonight** (24k+50k crashΔ null); overnight ~569k learning — still observe-only; do not launch third official eval
- **Overnight:** observe-only `overnight_soak_20260917_082739` PIDs **19244/53852**
- **FTGΔ residual:** close +8.52 s — wave `20260917-ftgdelta-close-wave` (per_map sidecar + stall_rate SHIP’d; re-eval DEFER)

---

## RESULTS — Watch overlay honesty — 2026-09-17 ~04:40 America/Chicago
- **id:** `20260917-meta-watch-honesty`
- **type:** result / gap-fill
- **overnight:** observe-only — did not kill soak; Watch follow may lag CURRENT_RUN pin
- **SHIP:** `watch` uses shared `ui_ops.read_operator_run_pin`; validating phase not mislabeled TRAIN STALE (keeps “not hung”); lag-behind sub_label keeps **unofficial**; regression `test_validating_not_mislabeled_stale_and_keeps_honesty`
- **Tests:** `.venv` `python -m rl.test_watch_overlay` → **14/14 PASS**

---

## RESULTS — Secrets hygiene + friend onboard + anti-thrash — ~04:35 America/Chicago
- **id:** `20260917-meta-secrets-onboard`
- **type:** result / meta hygiene
- **overnight:** observe-only — did **not** kill/morph `overnight_soak_20260917_082739` (PIDs **19244**/**53852** ALIVE; lock pid=19244; phase **`validating`** @ ~545k, eval_count=5, no_improve=2/5).

### Job A — Secrets / credential hygiene → **CLEAN**
- `git ls-files` under `rl/` (+ nearby filters): no `.venv`, `.env`, `*.pem`, credentials, or large checkpoint `*.pt`/`*.zip` tracked.
- Content patterns (AWS/GitHub/OpenAI/Slack/JWT/Bearer/DB URLs/`BEGIN PRIVATE KEY`/hardcoded password assignments): **0 hits**.
- Trivial overnight-safe harden: `rl/.gitignore` ignores `.env` / `.env.*` / `*.pem` / `id_rsa*` / `credentials.json`.
- Helper: `python -m rl._secrets_scan` → CLEAN (path + pattern name only; never prints match text).

### Job B — Friend-onboarding dry-run → **PARTIAL**
Windows stranger checklist (docs dry-run; no new disposable train):

| Step | Notes |
| ---- | ----- |
| venv + pip | README Quickstart OK (`py -3.13 -m venv rl/.venv`) |
| cwd | Must be `ADSS Toolkit/autodrive_py` — easy miss from repo root |
| `start_ui` / `start_train` | Use `.venv` python; set `PYTHONUTF8=1` (cp1252 help) |
| Watch | Documented separate `--follow` process |
| Overnight refuse | Strong in HANDOFF + `auto_train`; lighter in README Start |

**Friction (≤5):** (1) README↔HANDOFF bounce (HANDOFF dirty mid-rewrite); (2) wrong cwd breaks `python -m rl.*`; (3) fresh clone needs `trackgen` if `maps/map0` missing; (4) overnight live → Start refuse / multi-run latch under-documented in README Start; (5) live dual UI/eval thrash confuses “which process is mine.”

**Soft README one-liners (board only — prefer not editing README while HANDOFF dirty):**
1. Always `cd` to `ADSS Toolkit/autodrive_py` before `python -m rl.*`.
2. If overnight soak is live: prefer Watch/TB; pause disposable Start until soak leaves `validating` / FTGΔ work settles.
3. First-time: `trackgen --num_maps 3` before `start_train` if `rl/maps/map0` missing.

### Job C — Anti-thrash (observe)
- Overnight **validating** — timesteps paused by design; not hung.
- Observed dual `eval_cli --official` (exited after FTGΔ) + dual `control_ui :7860` + dual `ui_selftest`; no `watch` at snapshot.
- **Recommend:** pause new disposable trains until overnight leaves `validating`. Soft ≠ kill mandate. Sacred soak untouched.

**Artifacts:** `rl/_secrets_scan.py`, `rl/.gitignore` (env/pem lines), this board entry.

---

## RESULTS — Meta A/B honesty report (MUST #3) — 2026-09-17 ~04:28 America/Chicago
- **id:** `20260917-meta-ab-report`
- **type:** result / gap-fill
- **tool:** `python -m rl._campaign_ab_report`
- **overnight:** observe-only
- **Pairs:** `cf_ab50k` → INCONCLUSIVE_MISSING_CRASH (wiring OK, 49k/49k, crash fields dropped after final-val trail); `tick0_ab2` → INCONCLUSIVE_MISSING_CRASH; `tick0_ab` → INCONCLUSIVE_ZERO_CRASH
- **Verdict:** MUST #3 wiring PASS; **crash↓ not proven**. Mid-train 187.8s / best_adj 220 train_eval ≠ official. Ticket: preserve crash_rate on validating overwrite (bugfix owns `live_status.py`).

---

## RESULTS — Researcher MUST #3/#5 closeout (50k A/B + holdout reconfirm) — ~04:32 America/Chicago
- **id:** `20260917-researcher-ab50k-holdout`
- **researcher:** `f4ac2241` (this tick)
- **type:** result
- **overnight:** observe-only — `overnight_soak_20260917_082739` @ ~**545k** `validating`, lock pid=**19244** / parent=**53852**; did not kill/morph; did not edit `control_ui` (W3/W8 UI seat).
- **MUST #5 holdout refuse:** **PASS** — `python -m rl._w2_verify_smokes` ALL PASS (assert_train_safe map2/3/4; `train_ppo --map map2|map3` exit=2; verify_pack ok; overnight still present).
- **MUST #3 collision-first A/B:** **DONE — crashΔ null/inconclusive**
  - Pair `cf_ab50k_base_20260917_041400` vs `cf_ab50k_cf_20260917_041400`: DummyVecEnv, n_envs=2, **49152** ts, map0, cpu, matched knobs; `collision_first` false/true confirmed in `config.json` + train banner.
  - Mid-run `crash_rate_estimate=0.0` / `collisions_estimate=0` both arms (stall-dominated ~1.4% progress; ep_len_mean≈134).
  - Final smoke metrics identical (`mean_return`≈12.12, `mean_progress_frac`≈0.0019, DNF) — no reward Discord observable when wall-hits ≈0.
  - Earlier twin `tick0_ab2_{base,cf}_*` @ ~24k: same story (crash_rate 0.0 both).
  - **Note:** overnight stays `collision_first=false` by design; curriculum still needs a regime where collisions actually occur (or longer budget / denser traffic) before crash_rateΔ can decide Go/No-Go on the flag.
- **MUST #4 FTGΔ:** already sealed above (`+8.52 s` vs pin) — no re-eval.
- **Next:** race work = close the **8.52 s** official gap (not crash-curriculum theater until collisions appear); optional longer A/B only if crash_rate becomes non-zero.

---

## RESULTS — MUST #4 FTGΔ (official_v2 PPO vs FTG pin) — ~04:29 America/Chicago
- **id:** `20260917-meta-ftg-delta`
- **type:** result / honesty
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739` (PIDs ~19244/53852 ALIVE). Did not start a third eval.
- **MUST #4:** **DONE (row present)** — duplicate `eval_cli --official` thrash (PIDs ~52360/27808) both exited; one `official_v2` PPO row landed.

| Field | Value |
| ----- | ----- |
| run_id | `eval_ppo_eval_overnight_best_20260917_0412` |
| policy / kind / protocol | `ppo` / `official` / `official_v2` |
| maps | map0 + map2 + map4 (5 eps × 3 maps); timeout **400 s** |
| adjusted_time | **206.68 s** |
| DNF | **False** (collisions=0, mean_progress≈0.990) |
| n_episodes | **15** |
| FTG pin | **198.16 s** (`ftg_official_v2`, n=15) |
| **Δ (PPO − FTG)** | **+8.52 s** (positive = slower than FTG; did **not** beat pin) |

**NOT official (do not rank):** overnight mid-train `best_model_meta` `kind=train_eval` ≈**187.8 s** on **map3** @ **220 s** validation timeout — different protocol/maps/timeout from `official_v2`. Label only.

**Model:** `models/eval_overnight_best_20260917_0412/best_model.zip` (snapshot of overnight RaceBest). Seals intact; no sacred early-stop retune.

---

## RESULTS — P1-2 atomic best_model_meta — ~04:25 America/Chicago
- **id:** `20260917-p1-2-atomic-meta`
- **type:** result / bugfix
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739`.
- **SHIP:** RaceBest promote writes `best_model_meta.json` via `atomic_write_json` (was torn `.write_text`).
- **Tests:** `.venv` `python -m rl._bugfix_p0_checks` → **19/19 PASS**.
- **Note:** W8 `control_ui.py` multi-run wiring completed in follow-up (`20260917-W8-multi-run-ui-ship`); overnight untouched.

---

## RESULTS — Tick 0 MUST ships (orchestrator implementer) — 2026-09-17 ~04:25 America/Chicago
- **id:** `20260917-tick0-must-ships`
- **type:** result / implement
- **amends:** Tick 0 board MUST 1–5 + W2 SHIP order
- **overnight:** observe-only — did **not** kill `overnight_soak_20260917_082739`; left Continue PIDs ~53852/19244; live ~495k validating @ ~04:24

| # | Item | Result |
| - | ---- | ------ |
| 1 | Overnight lock audit | **PASS** — `ui_ops` select=220, eval floor=50k, warmup=2, min_ts=100k; live `config.json` mirrors (unlimited, patience=5, allow_holdout/validation=false, collision_first=false). `AUDIT_PASS`. |
| 2 | Holdout refuse smoke | **PASS** — `assert_train_safe` map2/map3 raises; `train_ppo --map map2|map3` exit=2; `start_guard` Refuse Start; seals intact. |
| 3 | Collision-first A/B | **PASS (wiring)** / **inconclusive crashΔ** — disposable Dummy CPU twins `tick0_ab2_base_20260917_040725` vs `tick0_ab2_cf_*` @ 24 576 ts; `collision_first` false/true in config; mid-run `crash_rate_estimate=0.0` both (stall-heavy early policy). No overnight morph. |
| 4 | Official FTGΔ | **PASS (lose)** — PPO `official_v2` row `eval_ppo_eval_overnight_best_20260917_0412`: adj=**206.68** s, n=15, DNF=false, progress≈0.990; FTG pin **198.16** s → **Δ=+8.52 s** (PPO behind). Seals intact; overnight not stopped. |
| 5 | Continue soak proof | **PASS (disposable)** — `tick0_ab2_base_*` Stop→`--resume` same `run_id`: timesteps **24576 → 48128**; overnight untouched. |

**Do not:** treat mid-train 220 s val as official; Quiet `--allow-holdout`; kill overnight to demo Continue.

---

## RESULTS — W8 multi-run live list + Focus — 2026-09-17 ~04:27 America/Chicago
- **id:** `20260917-W8-multi-run-ui-ship`
- **type:** result
- **amends:** `20260917-W8-multi-run-ui`
- **overnight:** observe-only — Focus/selftest did not kill `overnight_soak_20260917_082739` (~478k+ learning).
- **U-01 Live-runs list:** **DONE** — `/api/status` `live_runs` + `<ul id="live_runs">` + overnight badge + Focus.
- **U-02 Banner bound:** **DONE** — `_resolve_selected_run` (UI-owned → CURRENT_RUN pin → max timesteps) drives banner/`bound_run_id`; no latch onto ~6k A/B smokes.
- **U-03 Op targets:** **DONE** — `#bound_targets` + `#op_targets`; Stop confirm echoes selected run_id+pid; Models `[focused]` highlight.
- **U-04 Start warn:** **DONE** — preview `live_lock_warn` + Start confirm when N locks; `start_lock_warn` on status; dual-Start still refused.
- **DEFER/SKIP untouched:** fancy dashboard DEFER; kill-multi SKIP.
- **UI_BUILD:** `w8-multi-run-20260917`
- **Tests:** `.venv` `python -m rl.ui_selftest` (incl. `test_multi_run_selection`) → all checks passed (Stop mocked).
- **Finish note:** completed half-wired helpers (status bind + Start/Stop confirms) left local mid-flight; no overnight kill.

---

## W8 — Multi-run Control UI (soft) — ~04:22 America/Chicago
- **id:** `20260917-W8-multi-run-ui`
- **type:** tick / wave (user soft suggestion)
- **proposal:** UI redesign/polish must keep **multiple concurrent trains** in mind (operators already see several live `train.lock` / run_ids). Ship **incremental** Control chrome — not a fancy dashboard, not a single-train assumption, and **never** kill multi-train support or overnight.
- **overnight plan:** Observe-only `overnight_soak_20260917_082739`. No Start/Continue/kill/morph soak. Soft ≠ MUST / Bugfix P0.

### Scope triage

| Id | Idea | Gate |
| -- | ---- | ---- |
| U-20260917-W8-01 | Live-runs list (locks + live_status: run_id, timesteps, phase, overnight badge) | **SHIP’d** implement `w8-multi-run-20260917` |
| U-20260917-W8-02 | Status banner bound to **selected** run (not “whichever file was last”) | **SHIP’d** with W8-01 |
| U-20260917-W8-03 | Clear which run Watch / Continue / Stop targets | **SHIP’d** with W8-01 |
| U-20260917-W8-04 | Warn before Start when N live locks already present | **SHIP’d** with W8-01 |
| U-20260917-W8-05 | Fancy multi-run dashboard / Gradio reskin | **DEFER** |
| U-20260917-W8-06 | Kill / refuse multi-train support to “simplify” UI | **SKIP** (hard) |

### Personality votes — multi-run incremental → **SHIP (design)**

- **Researcher:** **Go** — grounded: `ui_ops.find_live_run_locks` already enumerates live locks; `_status_payload` can fall back to `find_latest_status` / prefer one external train — multi-lock operators get ambiguous banner. Cite `control_ui._status_payload` + Models `[locked pid=…]` without a Live list.
- **Racer:** **Approve** — operator clarity only; Block any wave that trades race MUST for chrome or retunes overnight.
- **Minimalist:** **Approve** — thin read-only list + selected-run binding; **Block** fancy dashboard / frameworks / embed Watch.
- **Reliability:** **Approve** — read locks + `live_status` only; Start warn when N locks live; **Block** killing multi-train, dual-Start murder, unmocked Stop vs overnight.
- **Integrator:** **SHIP** incremental **design** into `CAMPAIGN_UI.md` + governance UI note this tick. **Do not** code W8-01…04 here (outside allowlist — needs mini-gate implementer + `ui_selftest`). **DEFER** fancy dashboard. **SKIP** killing multi-train. Soft ≠ overnight kill.

**Orchestrator (this tick):** docs only — board + `CAMPAIGN_UI.md` Wave 2 + GOVERNANCE UI soft constraint. Next UI implementer: W8-01…04 thin Control list (read `find_live_run_locks` + per-run `live_status`); bump `UI_BUILD`; `ui_selftest` with Stop mocked.

**Resume-me:** UI bot — implement W8-01…04 when free (mini-gate already SHIP’d design); keep allowlist chrome flowing; never redesign epic; never kill overnight / multi-train.

---

## RESULTS — W7 UI bot process + allowlist chrome — 2026-09-17 ~04:21 America/Chicago
- **id:** `20260917-W7-ui-bot-ship`
- **type:** result
- **amends:** `20260917-W7-ui-bot`
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739` (selftest saw live pid; Stop mocked).
- **Process:** Standing UI seat + [`CAMPAIGN_UI.md`](CAMPAIGN_UI.md) + governance UI duty **SHIP’d**. Soft ≠ MUST / Bugfix P0.
- **U-01 Banner readability:** **DONE** — left accent + state-tinted `#banner.*` backgrounds; `#banner_reason` higher contrast (no `.small` dim).
- **U-02 Section headers:** **DONE** — `.sec` on Live / Coach / Train / Models / TensorBoard; thin spacing only.
- **U-03 Glossary expand:** **DONE** (confirmed) — curriculum / map-role terms already in drawer from W6; stamped under W7.
- **FORBIDDEN untouched:** no frameworks · no embed Watch · no sacred default retune · honesty banners kept.
- **UI_BUILD:** `w7-ui-bot-20260917`
- **Tests:** `.venv` `python -m rl.ui_selftest` → **all checks passed** (overnight survived).

---

## RESULTS — W6 Wave 1 thin SHIP — 2026-09-17 ~04:20 America/Chicago
- **id:** `20260917-W6-sug-wave1-ship`
- **type:** result
- **amends:** `20260917-W6-suggestion-bot`
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739`; no reward mutation; no denser eval / timeout shorten; no bridge code.
- **S-01 Watch compact honesty strip:** **DONE** — W3 already shipped `--compact` / `--every 8`; W6 polish adds phase + crash% to compact train strip (lag-behind / unofficial banners unchanged).
- **S-02 Reward/curriculum glossary:** **DONE** — glossary + titles for collision-first, speed-gate, race_eval_every vs official, budget OFF vs patience; Validating banner = “scheduled race eval — not hung”.
- **S-03 Map checklist:** product UI still **DEFER**; glossary `train_ok` checklist bullets **DONE**.
- **S-04 Eval wall-clock honesty:** **DONE** copy only (400s×5 / do-not-densify tooltips + hint); **SKIP** denser overnight eval.
- **S-05 Bridge-readiness card:** **DONE** document-only (`#bridge_card` + README §); **SKIP** Jetson/bridge code.
- **UI_BUILD:** `w7-ui-bot-20260917` (W6 glossary/bridge landed in same Control HTML; W7 seat bumped stamp)
- **Tests:** `.venv` `python -m rl.test_watch_overlay`; `python -m rl.ui_selftest` (Stop mocked — do not sweep overnight).

---

## STANDING — UI bot (Control / Watch presentation) — ~04:16 Chicago
- **id:** `20260917-W7-ui-bot`
- **type:** tick / wave (user soft suggestion — standing)
- **proposal:** Standing **UI** seat each tick proposes **1–3 SMALL** incremental Control UI / Watch presentation improvements (layout, spacing, glossary, a11y, Watch compact, status clarity). Not a blank-check redesign.
- **overnight plan:** Observe-only `overnight_soak_20260917_082739`. UI bot never Start/Continue/kill/morph soak; never retune sacred knobs; never remove honesty banners. Soft ≠ override MUST order or Bugfix P0.

### Process (SHIP seat + gated increments)

1. Each tick (or every other ~10m): UI seat appends **1–3** small proposals to `CAMPAIGN_UI.md` + short board pulse if anything needs mini-gate.
2. **Auto-approve allowlist only:** tooltips, spacing/contrast, labels, glossary copy, compact-mode polish, section headers — still log to `CAMPAIGN_UI.md` + bump `UI_BUILD` + run `ui_selftest`.
3. **Board mini-gate** required for anything outside allowlist (new controls, JS behavior, Watch argv defaults beyond prior SHIP, layout that hides honesty).
4. **FORBIDDEN without full board:** new frameworks (Gradio/React), embed Watch in browser, change Overnight sacred knob defaults, remove honesty banners / unofficial labels.
5. Soft ≠ starve MUST #3 A/B, #4 FTGΔ, or Bugfix P0/P1.

### Personality votes — standing UI bot → **SHIP (process)**

- **Researcher:** **Go** — mirrors W3 soft UX success (compact Watch + glossary SHIP’d; drastic reskin SKIP); constrained seat prevents chrome freestyle; cites W3 RESULTS + W2 chrome OUT.
- **Racer:** **Approve** — presentation-only behind race MUST; Block any wave that treats UI polish as podium work or retunes overnight.
- **Minimalist:** **Approve** — one md + allowlist auto-approve; refuse frameworks / embed / reskin; 1–3 small deltas only.
- **Reliability:** **Approve** — UI chrome cannot touch train math, kill-tree, dual-Start, or seals; `ui_selftest` mandatory; Stop path stays mocked while soak live.
- **Integrator:** **SHIP** the **seat + `CAMPAIGN_UI.md` + first 3 proposals**; implement safest **1–2 allowlist** items this tick if free. No redesign epic.

### First proposals (`CAMPAIGN_UI.md`) — triage

| Id | Idea | Gate |
| -- | ---- | ---- |
| U-20260917-W7-01 | Status banner readability (contrast / reason) | **SHIP now** (allowlist — contrast/labels) |
| U-20260917-W7-02 | Section headers + spacing rhythm | **SHIP now** (allowlist — spacing/labels) |
| U-20260917-W7-03 | Glossary expand (curriculum / map roles) | **SHIP now** (allowlist — already in drawer from W6 copy wave; confirm + stamp) |

**Orchestrator (this tick):** docs + U-01 + U-02 + U-03 (glossary already present — stamp under `w7-ui-bot`). Soft ≠ MUST/bugfix override.

**STANDING Resume-me:** UI bot — next tick (or every other ~10m): propose **1–3** small Control/Watch increments → log `CAMPAIGN_UI.md` → auto-approve only allowlist else mini-gate → `ui_selftest` → do **not** mandate, redesign, embed Watch, or touch overnight. Soft ≠ MUST / Bugfix P0.

---

## STANDING — Bugfix/Perf hunter (stable cores) — ~04:11 Chicago
- **id:** `20260917-bugfix-stable-cores-t0`
- **type:** audit / patch (user soft suggestion — standing)
- **scope:** contracts · racing_env · live_status · atomic checkpoint/`train.lock` · map_pack · observation · Subproc. Skip UI reskin + DEFER (knee/AMP/continuous/embed).
- **overnight:** observe-only `overnight_soak_20260917_082739` — no kill; in-memory train still on pre-fix code until Continue.
- **artifact:** [`CAMPAIGN_BUGFIX.md`](CAMPAIGN_BUGFIX.md) (P0/P1/P2 + PERF).
- **P0 SHIP (this tick):**
  1. `atomic_save_sb3` — removed unlink-on-lock path (data-loss); restore `.bak` on failed replace.
  2. `acquire_run_lock` — `O_EXCL` exclusive create (dual-writer TOCTOU).
  3. `LiveStatusCallback` — never clobber `early_stopped` / `validating`.
  4. `find_last_complete_checkpoint` — accept `*.zip.bak` for Continue recovery.
- **tests:** `rl/.venv` `python -m rl._bugfix_p0_checks` → **9/9 PASS**.
- **P1 next:** atomic CheckpointCallback; atomic `best_model_meta.json`; map_pack refuse corrupt manifest.
- **PERF parked:** cast_lidar / obs buffer / worker memmap — measure before change; no sacred timeout edits.
- **Resume-me:** Bugfix/Perf — next tick re-audit + ship P1-1/2/3 if free; keep overnight observe-only.

---

## W6 — Standing Suggestion bot seat + Wave 1 triage — ~04:11 Chicago
- **id:** `20260917-W6-suggestion-bot`
- **type:** tick / wave (user soft suggestion)
- **proposal:** Add a standing **Suggestion** seat (like user W3/W4/W5 soft asks): periodically invents soft improvement ideas (UX, train, perf, pipeline, AutoDRIVE transfer) and submits them as **suggestions**, not mandates. First wave already in `CAMPAIGN_SUGGESTIONS.md` (`20260917-W6-sug-wave1`, 5 ideas).
- **overnight plan:** Observe-only `overnight_soak_20260917_082739`. Suggestion bot never Start/Continue/kill/morph soak; never retune sacred knobs; never ship without Integrator gate.

### Process (SHIP seat + inbox)

1. Each ~10m tick (**or every other tick**): Suggestion bot appends **3–7** soft ideas to `CAMPAIGN_SUGGESTIONS.md` + a short board proposal pointing at wave ids.
2. Board triages each idea **SHIP / DEFER / SKIP** (same personality votes + Integrator gate).
3. Orchestrator builds **SHIP only**, optional behind race MUST when free hands.
4. Soft ≠ override MUST order, W2/W3 SKIP lists, or Reliability Blocks.

### Seat vote — standing Suggestion bot → **SHIP**

- **Researcher:** **Go** — process mirror of user soft waves (W3 UX, W4 auto-train, W5 bugfix); inbox + board keeps ideas out of orchestrator freestyle; cites governance ship gate.
- **Racer:** **Approve** — ideas welcome if they stay behind collision/Continue/FTGΔ; Block any wave that pretends chrome = podium.
- **Minimalist:** **Approve** — one markdown inbox + short board pulses; refuse a second product/UI framework or “suggestion daemon” that auto-codes.
- **Reliability:** **Approve** — soft inbox cannot touch overnight math, kill-tree, dual-Start, or seals; every SHIP still needs board entry.
- **Integrator:** **SHIP** the **seat + inbox + Wave 1 triage** (docs/process only this chunk). No huge implementations.

### Wave 1 triage (`CAMPAIGN_SUGGESTIONS.md`)

| Id | Idea | Gate |
| -- | ---- | ---- |
| S-20260917-W6-01 | Watch compact honesty strip | **DONE** (W3 compact/`--every` + W6 phase|crash% strip) |
| S-20260917-W6-02 | Reward / curriculum plain-language clarity | **DONE** (glossary + Validating gloss) |
| S-20260917-W6-03 | Map curriculum unlock checklist | **DEFER** product UI — glossary bullets **DONE** |
| S-20260917-W6-04 | Eval wall-clock honesty cue | **DONE** tooltip/status; **SKIP** denser eval |
| S-20260917-W6-05 | Bridge-readiness constraint card | **DONE** document-only; **SKIP** bridge code |

**Orchestrator:** implement Wave 1 **SHIP** items only as thin copy/overlay/glossary when free; leave DEFER/SKIP alone. Soft suggestion ≠ starve MUST #3 A/B or #4 FTGΔ.

**STANDING Resume-me:** Suggestion bot — next tick (or every other ~10m): invent **3–7** new soft ideas → append newest wave to `CAMPAIGN_SUGGESTIONS.md` → open short board proposal for triage → do **not** mandate, auto-code, or touch overnight. Diversify areas; skip duplicates of SKIP forever (GPU theater, continuous outer, reward Discord, bridge product). Soft ≠ MUST override.

---

## W5 — Standing bugfix / perf team — ~04:11 Chicago
- **id:** `20260917-W5-bugfix-perf-team`
- **type:** tick / wave (user soft suggestion)
- **proposal:** Standing **bug-fixing / performance** process that searches for bugs + improvement opportunities on code that will **stick around** (stable cores), not throwaway / about-to-rewrite surfaces; also hunts performance boosts. Soft ask: board may SHIP process + first audit.
- **overnight plan:** Observe-only `overnight_soak_20260917_082739`. Audit docs only this wave — no train-path morph, no kill/Continue, no reward Discord. P0 one-liners only if overnight-safe; else ticket in `CAMPAIGN_BUGFIX.md`.

### Scope (in / out)

| Surface | Verdict | Why |
| ------- | ------- | --- |
| `contracts.py`, `racing_env.py` reward/physics | **IN** | Sacred ABI + race math; sticks |
| `live_status.py`, `metrics_io` / `checkpoint_io` atomic IO | **IN** | Overnight trail + promote safety |
| `map_pack.py` seals / `assert_train_safe` | **IN** | Holdout integrity |
| `train_ppo` SubprocVecEnv path | **IN** | Throughput honesty for long runs |
| Cosmetic Control/Watch chrome, Gradio/React, bridge stubs | **OUT** | About to change / P3 / not race |
| Continuous outer / reward auto-mutation | **OUT** | Already SKIP (W4) |

### Personality votes

#### Standing process: `CAMPAIGN_BUGFIX.md` + periodic stable-core audits → **SHIP**
- **Researcher:** **Go** — Soft suggestion grounded; audit of sticky cores compounds overnight value; cite real skim findings (map resolve fallback, lock TOCTOU, Subproc silent Dummy, LiDAR Python loop, latest.zip every rollout). Out-of-scope chrome correctly excluded.
- **Racer:** **Approve** — Finding train-integrity / sps leaks helps adjusted_time more than UI polish; refuse chrome audits.
- **Minimalist:** **Approve** — One standing md + tick duty; no new subsystem, no refactor epic this wave.
- **Reliability:** **Approve** — Process protects seals/locks/Continue surfaces; must not touch live soak; P0 code only if one-liner + selftest-safe.

### Integrator gate (decisive)

**SHIP (process + first audit):**
1. Standing artifact [`CAMPAIGN_BUGFIX.md`](CAMPAIGN_BUGFIX.md) — ranked P0/P1/P2 from real skim of IN-scope files.
2. Governance: Bugfix seat / recurring duty each tick (see `CAMPAIGN_GOVERNANCE.md`).
3. **Do not** implement large refactors this wave. Ticket fixes; overnight-safe P0 one-liners only when free hands after MUST queue.

**SKIP:** Auditing / rewriting UI chrome, bridge stubs, Auto-train product expansion, reward PBT.

**Acceptance (this SHIP):**
1. Board entry exists (`20260917-W5-bugfix-perf-team`).
2. `CAMPAIGN_BUGFIX.md` lists ≥5 ranked findings with file cites + next action.
3. Governance mentions Bugfix / audit duty on ticks.
4. Overnight soak untouched.

**Resume-me:** Next tick — pick top P0/P1 from `CAMPAIGN_BUGFIX.md` that is small + overnight-safe; re-skim after any MUST ship that touches env/IO; do not starve collision A/B or FTGΔ for chrome.

---

## RESULTS — W3 SHIP soft UX — 2026-09-17 ~04:15 America/Chicago
- **id:** `20260917-W3-soft-ux-ship`
- **type:** result
- **overnight:** observe-only — did not kill/morph `overnight_soak_20260917_082739`; no train-path display; no bridge/embed.
- **SHIP 1 Watch compact + `--every`:** **DONE** — `--compact` / `--no-compact` + env `RL_WATCH_COMPACT`; follow **default compact ON**; follow `--every` default **8** (was UI-hardcoded 5; CLI follow now 8 vs standalone 2); env `RL_WATCH_EVERY`; Control UI Open Watch passes `--every 8 --compact`. Overlay keeps live race score + beat-FTG + short `train ts|/s`; drops session dump / dense train counters. README documented.
- **SHIP 2 Glossary / tooltips:** **DONE** — `<details id="glossary">` plain-English drawer (timesteps, early-stop, patience, adjusted_time, holdout, validation, n_envs, rollouts, Validating, EarlyStop, Watch lag-behind, fast probe) + dotted `title=` hints on live status rows; thin same-DOM CSS (spacing/radius/hover only — no reskin).
- **SKIP unchanged:** drastic Control reskin · embed Watch · bridge/Jetson.
- **UI_BUILD:** `w3-ux-soft-20260917`
- **Tests:** `.venv` `python -m rl.test_watch_overlay` **13/13 PASS**; `python -m rl.ui_selftest` **all checks passed** (glossary + build stamp).

---

## FAST implementer wave — 2026-09-17 ~04:10 America/Chicago

- **id:** `20260917-W2-fast-impl-curriculum-seals`
- **type:** ship / self-vote
- **proposal:** Implement SHIP-aligned UI ROI without densifying mid-train eval or touching sacred overnight constants.
- **overnight plan:** observe-only intended; **incident:** `ui_selftest` HTTP `op=stop` killed protected soak mid-validation (~295k). **Mitigation:** Continue same `run_id` from `ppo_295480_steps.zip` with overnight argv (select=220, eval=50k, warmup=2, min_ts=100k, patience=5, unlimited). Selftest Stop path now mocks `_kill_train_tree` / external scan. **Do not re-run unmocked Stop while soak is live.**

### Shipped (this wave)

| Item | Letter | What | Test |
| ---- | ------ | ---- | ---- |
| Collision-first UI | **B** | Checkbox + tooltip + Start argv `--collision-first` + preview line | `ui_selftest` index/preview collision_first |
| Continue UX / curriculum restore | **A** | Soft-stop copy; Continue restores `collision_first`/`speed_gate` from `config.json`; confirm dialog shows flags | continue_train_argv curriculum checks |
| Honest progress probe (status-only) | **C** (lean) | `live_status.mean_progress_frac_estimate` + `progress_probe_kind=smoke` + UI row — **no promote / not official** | index label + status payload |
| Seal-verify button | **D** | `Verify pack seals` → `seal_verify_summary` / `verify_pack` | `test_seal_verify` + HTTP `verify_seals` → `SEALS OK \| maps_ok=6/6` |
| n_envs knee script | **E** (deferred tooling) | `python -m rl._measure_n_envs_knee` refuses live train.lock; coach hint when sps low | refuse-on-live path; board W2 still **DEFER** full ship |

### Self-vote

- **Racer:** Approve B/A/D — crash curriculum exposed + Continue integrity + seals before FTGΔ. C OK if labeled smoke. E later.
- **Minimalist:** Approve — expose existing argv/flags; no new subsystem; knee script optional.
- **Reliability:** Approve with caveat — Stop-in-selftest must stay mocked; Continue restore after accidental kill is the correct recovery. Block any further unmocked Stop against overnight.
- **Integrator:** **SHIP** B+A+D (+ lean C). **DEFER** knee coach as primary product (script OK). **SKIP** denser mid-train select / continuous outer.

### Explicit non-goals this wave

No GPU/AMP · no continuous outer loop · no lowering select/eval floors · no morph of overnight `collision_first` mid-run.

---

## W4 — Auto-train pipeline triage — ~04:09 Chicago · **SHIP thin DONE ~04:12**
- **id:** `20260917-W4-auto-train-pipeline`
- **type:** tick / wave (user soft suggestion)
- **proposal:** Soft ask for an **Auto-train** button/pipeline that trains → evals lap/adjusted times → continuously gens maps → auto-adjusts reward/knobs → farms many “best” models unattended. Re-triage vs W2 #5 SKIP (continuous outer) after Reliability P0 no-kill/hb-join SHIP + disposable Continue soak PASS.
- **overnight plan:** Observe-only `overnight_soak_20260917_082739`. Auto-train must **never** Start/Continue/kill/morph the protected soak. No reward Discord mid-run. No map regen in the train hot loop.

### SHIP thin result (implementer)
- **CLI:** `rl/auto_train.py` — default dry-run; `--execute [--smoke]`; chains train_ok → train_ppo (frozen knobs) → `eval_cli --official` → next pack map/seed; `--max-runs` ≤3.
- **Refuses:** live `train.lock`/overnight PID; `overnight_soak_*` run_ids; holdout/validation; `--allow-holdout` / `--mutate-reward` / curriculum flags; never `_kill_train_tree`.
- **UI:** disabled **Auto-train (EXPERIMENTAL)** + opt-in preview warning; `op=auto_train` refuses spawn (CLI only). Build `w4-auto-train-thin-20260917`.
- **Acceptance:** dry-run + holdout/protected/live-lock refuses exercised; Dummy `--execute --smoke` **deferred** while overnight lock alive (observe-only).
- **Keepalive:** single 10m loop PID **2340** (40676 already gone); train_ppo **53852/19244** untouched.

### Critique (decisive)

| Capability | Verdict | Why |
| ---------- | ------- | --- |
| Reward / sacred-knob **auto-mutation** | **SKIP forever (this campaign)** | Incomparable runs; reward hacking; poisons `race_score_key` / EarlyStop meaning; PLAN forbid Discord dial |
| Hot map cycling **on overnight** / dual-Start while soak live | **SKIP** | Dual-writer + overnight murder risk; W2 Reliability Block still applies to *unattended* product until chaos drills |
| Holdout / validation pin in train list | **SKIP** | `assert_train_safe` sacred; voids official claims |
| Map **autogen** then train | **OK only offline** | `map_pack`/`trackgen` cache → `train_safe_maps()` + `assert_train_safe` + fingerprint; never gen inside `step`; never sealed ids |
| Outer chain: train_ok → early-stop → official eval → archive → next pack seed/map | **SHIP thin scaffold only** | Matches RESEARCH §2/§5 “supervised outer shell”; does **not** change overnight math |
| Unattended multi-hour scheduler / Overnight companion | **DEFER** | Continue+early-stop proven on **disposable** only; chaos drills (reliability §) still open; P0 cleared *code* blockers, not drill gate |

### Personality votes

#### A. Full auto-reward-mutation + overnight hot map cycling → **SKIP**
- **Researcher:** **No-Go** — RESEARCH §2 Pitfall A/B + §6; reward morph = nonstationarity + leaderboard soup; overnight cycle fights dual-writer/sacred floors.
- **Racer:** **Block** — KILL #5 still; chrome autopilot ≠ adjusted_time; farming incomparable “bests” is vanity.
- **Minimalist:** **Block** — refuse Discord reward dial + continuous cosplay; cut the feature, not the guard.
- **Reliability:** **Block** — unattended kill/dual-Start/sacred-math regression; chaos drills unmet.

#### B. Unattended outer scheduler (Control/Overnight companion) → **DEFER**
- **Researcher:** **Conditional unmet** — RESEARCH §2 scaffold rules OK in principle, but unattended product waits on chaos drills + soft-stop→Continue on a **non-toy** path (disposable Continue PASS ≠ overnight Continue north star closed).
- **Racer:** **Abstain / later** — podium first (collision↓, FTGΔ); scheduler after finishers exist.
- **Minimalist:** **Approve defer** — no second autopilot product this 9h.
- **Reliability:** **Block enablement** until drills 1–8 green; P0 no-kill only clears *code* veto, not product veto.

#### C. Thin scaffold (design + CLI; no mutation; no overnight touch) → **SHIP**
- **Researcher:** **Go** — documented contract + `auto_train.py` that **only** chains existing primitives; frozen hparams; `train_ok` only; official eval labeled; cap `--max-runs`.
- **Racer:** **Approve** — scaffold later was always fine; thin chain can mint comparable candidates **if** protocol fixed and maps train_ok-only.
- **Minimalist:** **Approve** — one CLI file + refuse walls; **no** reward API; UI = disabled opt-in stub with warnings (no Gradio/React).
- **Reliability:** **Approve thin only** — must refuse live `train.lock`, refuse `overnight_soak_*` run_ids, refuse `--allow-holdout`, never call `_kill_train_tree`, never mutate patience/select/eval floors, never auto-append `kind=official` without seals path.

### Integrator gate (decisive)

**SKIP (hard):** reward/knob auto-adjust · overnight hot map cycling · holdout autotrain · kill/morph protected soak · unconstrained continuous-until-plateau product · auto-`kind=official` spam.

**DEFER:** enabling Auto-train as Overnight companion / unattended multi-hour UI Start; map **generation** inside the chain (operators generate offline via existing map_pack UI/CLI first); chaos-drill-gated scheduler.

**SHIP (thin) — allowed implementation scope:**
1. **Design note** (this entry + short module docstring) = contract of record.
2. **`rl/auto_train.py` CLI scaffold** (opt-in, default dry-run):
   - Resolve next map/seed from pack ⊆ `train_safe_maps()` → `assert_train_safe` → spawn `train_ppo` with **fingerprinted frozen** knobs (sacred early-stop floors untouched) → wait `phase=early_stopped` or clean exit → run **official** eval protocol path → archive `best_model`/metrics artifact under run dir → advance to next **declared** train_ok id / seed.
   - Hard refuses: any live lock/PID; run_id matching protected overnight; holdout/validation maps; reward/flag mutation flags; `--max-runs` default **1** (cap ≤3 for smoke); Start while another train live.
3. **UI:** optional **disabled-by-default** “Auto-train (EXPERIMENTAL)” control with loud warning copy; must not wire a live spawn until chaos drills — preview/refuse only this wave is OK.
4. **Do not** implement reward search, online mapgen, or overnight chaining in this ship.

### Acceptance tests (thin SHIP)

1. `python -m rl.auto_train --help` documents refuses (overnight, holdout, reward mutation, live lock).
2. Dry-run prints planned argv + map ids; exits 0; **spawns nothing**.
3. `assert_train_safe` path: requesting map2/map3/map4 → nonzero exit; no process start.
4. Live-lock refuse: with overnight/other `train.lock` present → nonzero; does not kill.
5. Protected-name refuse: `--run-id overnight_soak_*` (or configured protected id) → nonzero.
6. Single-run smoke (disposable run_id only): train_ok map → early-stop or short budget → official eval artifact written; `kind=official` only via protocol+seals; no reward keys changed in `config.json` vs template.
7. `ui_selftest` (if UI stub added): Auto-train control **disabled** by default; enabling shows warning; Start path still no-kill.
8. Regression: overnight sacred constants / RaceBest / dual-writer tests still green; `python -m rl.ui_selftest` PASS with soak surviving.

**Orchestrator:** implement **thin SHIP only** when free hands after MUST queue; otherwise leave as design+stub ticket. Soft suggestion ≠ override collision/Continue/FTGΔ.

**Resume-me:** Researcher — next tick: (1) do **not** expand Auto-train into reward PBT/mapgen product; (2) re-vote **DEFER→SHIP enablement** only after chaos drills + overnight soft-stop→Continue evidence; (3) run Dummy `--execute --smoke` only when overnight idle; (4) keep overnight observe-only.

---

## Tick — W3 soft UX confirm — 2026-09-17 ~04:09 America/Chicago
- **id:** `20260917-W3-soft-ux-confirm`
- **researcher:** `f4ac2241`
- **type:** tick / confirm
- **soak:** observe-only — `overnight_soak_20260917_082739` @ ~**295k**, phase=`validating` (map3, timeout=220, no_improve=0), lock pid=19244, `collision_first=false`. Sacred knobs intact.
- **W3 soft UX vs Go/No-Go:** **CONFIRMED** — SHIP compact Watch + glossary (**Go**, operator honesty); SKIP reskin / embed / bridge (**correct No-Go**). Soft optional behind race MUST; do not starve #3 A/B or #4 FTGΔ. Transfer = research constraint note only (§5) — no P3 code.
- **MUST status:** #1 hold · #2 CLI PASS (W2 continue soak) · #3 PARTIAL (UI wired; **A/B crash_rate still owed**) · #4 FTGΔ owed · #5 smoke when free.
- **Next tick priorities:** (1) collision-first ≤100k A/B crash_rateΔ · (2) official PPO vs FTG when zip ready · (3) holdout CLI if needed · (4) W3 SHIP Watch/glossary only if free hands · re-vote embed later.
- **NO-GO unchanged:** continuous outer · GPU theater · Control reskin · bridge.

---

## SHIP — Reliability P0 (heartbeat join + Start/Continue no-kill) — ~04:08 Chicago

- **id:** `20260917-reliab-p0-nokill-hbjoin`
- **type:** patch / safety
- **Was:** Reliability **Block** until heartbeat-join + Start-kill-on-false-busy fixed (`campaign_reliability.md` critical notes).
- **Now:** **FIXED / SHIP**
  1. `RaceBestModelCallback`: join heartbeat before `early_stopped`; gen-token; refuse validating clobber of terminal phase.
  2. Start/Continue: **never** `_kill_train_tree` (Stop-only); `_train_busy` refuses on live `train.lock` if PID scan misses.
  3. `live_status.TERMINAL_PHASES` / `is_terminal_phase`; `ui_ops.find_live_run_locks`; `ui_selftest` regressions (+ HTTP Stop mocked so selftest cannot murder overnight).
- **overnight:** observe-only `overnight_soak_20260917_082739` (Continue-restarted; live). Do not kill.
- **Tests:** `python -m rl.ui_selftest` PASS with overnight surviving.
- **UI_BUILD:** `reliability-nokill-hbjoin+w2-20260917`
- **Continuous outer loop:** still **NO-GO** until chaos drills; this only clears the P0 code blockers.

---

## RESULTS — W2 FAST wave ships — 2026-09-17 ~04:08 America/Chicago
- **id:** `20260917-W2-fast-wave-results`
- **type:** result
- **overnight:** observe-only — Continue soak reported lock pid=46692 intact; did not kill/morph soak. UI Start/Continue no longer call `_kill_train_tree` (reliability merge).
- **SHIP 1 Collision-first UI:** **DONE** — checkboxes + tooltips for `--collision-first` / `--speed-gate` / optional NON-OFFICIAL fast probe; wired Start argv + preview; Continue inherits curriculum from `config.json`. Soft-stop one-liner present. `ui_selftest` **all checks passed**. UI build `w2-fast-wave-20260917`.
- **SHIP 2 Continue soak proof:** **PASS** — disposable `continue_soak_20260917_090812` via `python -m rl._continue_soak_smoke` (CLI only, no UI kill-tree): phase1 ts=1024 → Continue resume → phase2 ts=**2048**. Script refuses overnight naming.
- **SHIP 3 Fast validation probe (honest):** **DONE** — `--fast-probe-every` / `--fast-probe-timeout` (default 30s) + `FastProbeCallback`: writes `probe_kind=smoke` / progress only; **never** promotes `best_model`; **never** ticks patience; skips when phase=`validating`/`early_stopped`. Full `select_timeout` (~220s) RaceBest remains sole promote/patience path. UI checkbox off by default; Overnight should leave off.
- **DEFER/SKIP unchanged:** n_envs knee · continuous loop · big UI redesign · GPU AMP.
- **Tests:** `.venv` `python -m rl.ui_selftest` PASS; `python -m rl._continue_soak_smoke` PASS.

---

## W3 — User UX soft suggestions triage — ~04:07 Chicago
- **id:** `20260917-W3-user-ux-suggestions`
- **type:** tick / wave
- **proposal:** Soft (non-MUST) user asks: declutter Watch, glossary/tooltips, pretty Control UI, Watch snappiness/embed, keep gym→AutoDRIVE transfer in mind. Soft ≠ override MUST order or W2 SKIP of big chrome.
- **overnight plan:** Observe-only vs `overnight_soak_20260917_082739`. No train-path display. No bridge/Jetson build.
- **Guidance cite:** W2 #6 big UI redesign = SKIP; Watch carnival = SKIP; declutter + glossary may SHIP as operator honesty; embed Watch = Conditional (perf); bridge = document constraint only (P3).

### Candidates

#### 1. Watch follow declutter (fewer numbers / compact overlay) → **SHIP**
- **Researcher:** **Go** — operator honesty, not carnival; keep lag-behind / unofficial labels; pairs with `--every` bump (RESEARCH §4 / §6 Watch aesthetics still hard-skip).
- **Racer:** **Approve** — tiny; does not claim podium; unofficial surface OK if hours stay small.
- **Minimalist:** **Approve** — compact toggle + fewer overlay rows; no ghost/seasons.
- **Reliability:** **Approve** — `test_watch_overlay` must stay green; promotion path untouched.
- **Concrete:** `compact` overlay mode (default on or toggle); raise follow `--every` default (5→8 or 10); drop dense counter spam, keep race KPIs + honesty banners.

#### 2. Plain-language glossary / RL tooltips (“normie terms”) → **SHIP**
- **Researcher:** **Go** — passes design test (fewer wrong Starts / misread Validating/EarlyStop/patience); extend existing `title=` pattern, not a new docs site.
- **Racer:** **Approve** — operators who understand collision-first / Continue ship race work faster.
- **Minimalist:** **Approve** — static glossary drawer or hover copy only; no second UI framework.
- **Reliability:** **Approve** — sacred knobs must stay worded honestly (budget vs early-stop; unofficial probe).
- **Concrete:** small “?” glossary drawer (timesteps, patience, eval_every, collision-first, official vs smoke, Watch lag-behind).

#### 3. Drastic Control UI visual redesign → **SKIP**
- **Researcher:** **No-Go** — same fail as W2 #6; cosmetics ≠ fewer wrong Starts unless copy/layout of sacred actions.
- **Racer:** **Block** — chrome ≠ adjusted_time.
- **Minimalist:** **Block** — pixels ≠ policy; operator min already shipped.
- **Reliability:** **Block** — reskin risks losing Overnight preset / refuse copy.
- **Note:** thin CSS polish (spacing/contrast, **same DOM**, no Gradio/React) may ride with #2 as a ≤1h companion — not a redesign epic.

#### 4. Watch snappiness / embed in Control UI → **split**
- **4a Raise `--every` + follow poll snappiness → **SHIP**** (bundle with #1).
- **4b Embed Watch inside browser Control → **Conditional / DEFER****
  - **Researcher:** **Conditional** — only if Watch stays a **separate process** (iframe/spawn), train remains headless, no auto-open on Start (RESEARCH §4/§6).
  - **Racer:** **Abstain** embed; **Approve** `--every` snappiness.
  - **Minimalist:** **Block** one-UI product; **Approve** CLI default tweak.
  - **Reliability:** **Conditional** — must not steal Subproc cores / lock contention from overnight.
- **Why not SHIP embed now:** Conditional unmet (perf + MUST queue first). Re-vote post-collision/Continue if operators still drown in two windows.

#### 5. AutoDRIVE sim transfer awareness (obs/DR/latency) → **DEFER build / SHIP constraint note**
- **Researcher:** **Go** document-only — freeze contracts `2.0.0`, keep mid-train LiDAR DR, no privileged GT in obs; **No-Go** bridge `:4567` / latency product this 9h (P3 after holdout beat-FTG).
- **Racer:** **Approve** constraint; **Block** bridge cosplay before FTGΔ.
- **Minimalist:** **Approve** freeze + refuse GT leak; **Block** Phase-3 scope creep.
- **Reliability:** **Approve** refuse-load / no mid-campaign obs morph; **Block** “faster gym wins” via privileged topics.
- **Concrete:** CAMPAIGN_RESEARCH transfer note only — no bridge code.

### Integrator gate
**SHIP now (small, after/parallel to MUST if free hands):** (1) Watch compact + higher `--every` · (2) glossary/tooltips drawer · optional thin CSS with #2.  
**DEFER:** embedded Watch-in-browser (Conditional).  
**SKIP:** drastic Control reskin · bridge/Jetson/latency implementation.  
Orchestrator: implement **SHIP** only; treat soft UX as **optional** behind collision/Continue/probe if contention.

**Resume-me:** Researcher `f4ac2241-2ce6-4f1b-b7ea-719eb29035dc` — next tick: refresh §7 soak evidence; re-vote embed only if MUST #2/#3 proven and operators still dual-window blocked; do not expand this entry into a UI epic.

---

## W3 — Immediate tick ~04:06 Chicago — collision UI + probe; Continue restarted
- **id:** `20260917-W3-immediate-tick`
- **type:** tick / wave
- **proposal:** Cadence 10m detached keepalive; ship W2 #1+#3; Continue (#2) sibling+this-tick emergency restart after soak death; DEFER n_envs knee; SKIP continuous/GPU/UI redesign.
- **overnight plan:** Found `overnight_soak_20260917_082739` **dead** (stale lock pid=34256). Cleared lock; **Continue-restarted** same `run_id` from `ppo_295480_steps.zip` (n_envs=8 subproc, sacred early-stop). Live validating @ ts≈295k. Do not kill.
- **Researcher:** **Go** — collision UI + smoke probe; Continue restart mandatory when soak dead.
- **Racer:** **Approve** — crash-first UI + resume = race continuity.
- **Minimalist:** **Approve** — expose flag + status-only probe; no new subsystems.
- **Reliability:** **Approve** Continue restart; collision/probe observe-only vs live soak.
- **Integrator:** **SHIP** collision-first UI + smoke progress probe (+ `progress_probe.py` CLI). **Continue soak proof** still sibling for disposable Stop→Continue; this tick only emergency restart. Detached loop: `campaign_keepalive.ps1` PID in `logs/campaign_keepalive.pid` (600s).

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
