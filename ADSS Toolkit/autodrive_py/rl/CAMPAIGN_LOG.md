# F1TENTH RL Continuous Improvement Campaign

**Start:** 2026-09-17 03:54 America/Chicago  
**Deadline:** ~2026-09-17 12:54 America/Chicago (9 hours) or credits exhausted  
**Branch:** `rl/phase-1-research`  
**Head note:** Watch overlay honesty SHIP (20260917-meta-watch-honesty); MUST #4 FTGΔ DONE (Δ+8.52); MUST #3 crashΔ DEFER; overnight soak ALIVE — do not kill.
**Human handoff:** friends start at [`HANDOFF.md`](HANDOFF.md) (runbook + sacred soak + multi-run).

---

## Keep-alive loop (mandatory — ~04:00+ AGGRESSIVE)

- **Sentinel:** `AGENT_LOOP_TICK_autodrive-rl-campaign` — **every 10 min (600s)**.
- **Cadence change (~04:00):** 50m ticks were too slow / padded the 9h window. User mandate: ideas constantly worked; lots of well-implemented small features each tick. See [`campaign_tick.md`](campaign_tick.md).
- **On EVERY tick:** parent MUST (1) resume Researcher + Racer/Minimalist/Reliability **in parallel** for NEXT proposal batch → `CAMPAIGN_BOARD.md`; (2) resume Orchestrator to ship **MULTIPLE** approved small features in parallel (spawn sibling implementers); protect overnight; (3) **`git status` + commit+push** any stable reviewed SHIP (tests green) — **after each wave**, continuously through the night, not once. Exclude pycache / logs / large zips / `.venv`. **Not orchestrator-only / not single-feature trickle.**
- **Orchestrator standing order:** push after **each** wave/chunk lands; do not batch uncommitted SHIP piles until morning.
- **Deadline:** after 12:54 America/Chicago 2026-09-17, STOP loop; do not re-arm.
- Cursor-attached loops die ~60s; use detached [`campaign_keepalive.ps1`](campaign_keepalive.ps1) (PID in `logs/campaign_keepalive.pid`; ticks in `logs/campaign_loop_ticks.log`). Parent: read latest tick → run `campaign_tick.md`. Do not kill `train_ppo` when re-arming.

## Governance (mandatory — 03:57+)

1. **Ingest before ship:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_{racer,minimalist,reliability}.md` + `CAMPAIGN_RESEARCH.md` when present.
2. **Before each non-trivial feature:** CAMPAIGN_LOG / `CAMPAIGN_BOARD.md` Go/No-Go with research checklist + ≥2 personality opinions (agree/dissent).
3. **Prefer:** overnight survival, race-score honesty, steps/sec truth, operator Start/Stop/Continue — over cosmetics and GPU theater.
4. **Fail review → SKIP** and log why. Ship reviewed work only; **commit+push after each wave** (and every ~10m tick if anything SHIP’d sits dirty).
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

MUST 1–5 gated by critique+racer+minimalist; reliability **P0 dual-writer/heartbeat blockers SHIPPED ~04:08** (Start/Continue no-kill + heartbeat join). Continuous outer loop still **NO-GO** until chaos drills pass; board owns per-tick Go/No-Go.

---

## Protected training run

| Field | Value |
| ----- | ----- |
| **run_id** | `overnight_soak_20260917_082739` |
| **Status** | ACTIVE — restored after selftest Stop; DO NOT KILL without Continue |
| **live_status (~04:20 Chicago)** | timesteps≈**428k**↑, steps/s≈201, phase=`learning`, n_envs=8, vec=subproc |
| **PIDs (lock / parent)** | **19244** / **53852** (Continue from `ppo_295480_steps.zip`) |
| **Sacred argv** | select=220, eval=50k, warmup=2, min_ts=100k, patience=5, unlimited |
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

### ~04:40 — Watch overlay honesty (20260917-meta-watch-honesty)
- **SHIP:** Watch phase=validating no longer mislabeled TRAIN STALE; lag-behind sub_label keeps **unofficial**; compact strip phase/crash%/ts bind via CURRENT_RUN pin to overnight (not short A/B).
- **Tests:** python -m rl.test_watch_overlay → **14/14**. Code: d5dd05 / 851b0dd / 2bd61c7.
- Overnight observe-only PIDs **19244/53852**.


### ~04:33 — FTGΔ close wave + stall-metric thin SHIP (`20260917-ftgdelta-close-wave`)
- **Overnight:** observe-only ~**569k** learning, PIDs **19244/53852** — no kill/morph; no sacred early-stop retune; no third official eval.
- **Collision A/B:** **DEFER** tonight — 24k + 50k both crashΔ null (stall-heavy); need longer budget / collision regime; do not starve soak.
- **FTGΔ residual:** PPO still **+8.52 s** vs FTG pin (finisher). Mid-train map3 ≈188.95 s = **NOT** official.
- **SHIP’d:** `live_status` `stall_rate_estimate`/`stalls_estimate`; `metrics_io.write_official_per_map_sidecar` on official append (next re-eval localizes map Δ).
- **DEFER:** official re-eval of newer RaceBest until CPU free; reward-shaping fork / obs churn.
- **SKIP:** early-stop retune; overnight kill.
- **Tests:** `_bugfix_p0_checks` (stall keys + sidecar asserts).

### ~04:32 — Tick #2 wave executor (fresh parallel; standing resume parent-mismatch)
- **Overnight:** `overnight_soak_20260917_082739` **ALIVE** ≈563k learning, PIDs **19244/53852** — no Continue-restart.
- **Board:** `20260917-tick2-wave` Go + ≥2 Approves on T2-01…04; DEFER A/B; SKIP GPU.
- **SHIP’d:** P1-1 AtomicCheckpointCallback · P1-3 CorruptManifest · P1-5 MONITOR_INFO_KEYWORDS · W8 multi-run Control (live_runs/Focus/Start warn).
- **Tests:** `_bugfix_p0_checks` 25/25; `ui_selftest` all passed (overnight untouched).
- **Blocked:** MUST #3 crashΔ still inconclusive (anti-thrash — no new disposable A/B while soak busy); standing agent resume IDs unusable from nested wake.

### ~04:32 — Researcher closeout: 50k collision A/B + holdout reconfirm
- **Holdout:** `_w2_verify_smokes` ALL PASS (map2/3 CLI refuse exit=2; seals OK).
- **Collision A/B 50k:** `cf_ab50k_{base,cf}_20260917_041400` finished 49152 ts; crash_rate **0.0 both** (stall-heavy) → **crashΔ null**; board `20260917-researcher-ab50k-holdout` (+ meta `20260917-meta-ab-report`).
- Overnight ~545k validating untouched; no `control_ui` edits this tick.

### ~04:30 — Meta-spawner coverage (gap net)
- Standing `## Meta-spawner coverage` on board; A/B honesty report `20260917-meta-ab-report` (MUST #3 still inconclusive).
- Children: FTGΔ collector (DONE), secrets/onboarding + watch honesty (in flight).
- Anti-thrash: pause new disposable trains; overnight ~545k validating untouched.
- Commit: board + log + `leaderboard.csv` official PPO row (if not already pushed).

### ~04:29 — MUST #4 FTGΔ collected (`20260917-meta-ftg-delta`)
- Official PPO `official_v2` row: **206.68 s** (n=15, DNF=False, maps map0/2/4 @ 400 s) vs FTG pin **198.16 s** → **Δ = +8.52 s** (lose).
- Mid-train overnight `train_eval` ≈187.8 s map3 @ 220 s = **NOT official**.
- Overnight soak untouched; eval PIDs ~52360/27808 exited (no third eval). Board: `CAMPAIGN_BOARD.md` RESULTS id `20260917-meta-ftg-delta`.

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

### ~04:00 — Cadence cut 50m → 10m (aggressive)

- **Rationale:** 50-minute ticks were too slow; remaining window was being padded instead of shipping. User wants ideas constantly worked — many well-implemented small features per tick, parallel board + parallel implementers.
- Stopped 50m full-board loop PID ~52836 (46964 already dead). Did **not** kill `train_ppo` (31472/34256).
- Re-armed `AGENT_LOOP_TICK_autodrive-rl-campaign` at **600s** until 12:54 America/Chicago; `campaign_tick.md` updated for parallel Researcher/personalities + Orchestrator multi-ship with sibling implementers + frequent push.
- **New loop PID:** `40676` (first sentinel after ~10m sleep; train_ppo 31472/34256 left running).
- **~04:05:** Loop PID `40676` died externally (`exit_code=4294967295`, ~4m in, no tick emitted). Re-armed same 10m sentinel — **new loop PID `12004`**. Overnight `31472/34256` no longer present at re-arm — Continue-restart same `run_id` is required on next tick.

### ~04:16 — First 10m `AGENT_LOOP_TICK` (PID 12004)

- Sentinel fired; loop still running; deadline not reached.
- Overnight already Continue-restored and learning (~376k @ ~04:15; lock ~19244 / parent ~53852) — no new Continue this tick.
- Nested tick handler cannot `resume` standing Researcher/personalities (parent mismatch); **parent** must resume board + Orchestrator for this wake (Orchestrator may already be mid-run).

### ~04:26 — Tick #2 (PID 12004)

- Sentinel fired again; loop alive; overnight still **53852/19244**.
- Spawned fresh board+orchestrator wave agent (standing resume blocked by parent mismatch from nested handler).

### ~04:22 — Continuous commit+push mandate (orchestrator note)

- User: **commit+push continuously through the night when stable** — not only once.
- Updated `campaign_tick.md` + `CAMPAIGN_GOVERNANCE.md`: every ~10m tick MUST `git status`; if board SHIP + tests green → **commit+push**; exclude pycache/logs/zips/.venv; no large uncommitted piles.
- **Orchestrator (`4a5b7bd3-…`):** push **after each wave/chunk**, not only at end of window / morning dump. Log push hashes here when practical.
- Pre-mandate remote HEAD already had W7 chrome: **`319b117`** (`Ship W7 UI bot chrome…`) on `origin/rl/phase-1-research`. Untracked only: `__pycache__`, `_ftg_pin*.log` (left uncommitted).
- Overnight soak **untouched** (do not kill).
- **Pushed this mandate:** **`ae7ef0e`** (continuous commit+push tick rules) → **`283417c`** (W8 multi-run UI design SHIP + `safe_run_id`). Local WIP only: partial `control_ui.py` W8 helpers (not wired / not green yet — leave uncommitted).

### ~04:27 — W8 multi-run live list + Focus SHIP
- **Board:** `20260917-W8-multi-run-ui-ship` amends design `20260917-W8-multi-run-ui`.
- **Code:** `/api/status` `live_runs`/`bound_run_id`; Control `<ul id="live_runs">` + Focus → `CURRENT_RUN.txt`; Start preview multi-lock warn; `UI_BUILD=w8-multi-run-20260917`.
- **Tests:** `.venv` `ui_selftest` all passed (Stop mocked; Focus overnight pin OK).
- **Overnight:** ALIVE ~478k+ learning; never killed.

### ~04:25 — Hours-away support loop + P1-2

- Support loop: every ~18m `git status` → commit+push safe SHIP; never touch overnight PIDs; never force-push/amend remote; exclude pycache / `_ftg_pin*.log`.
- **SHIP P1-2:** `train_ppo` RaceBest `best_model_meta.json` → `atomic_write_json`; `_bugfix_p0_checks` **19/19**.
- **Pushed:** **`8af8a25`** (P1-2) → **`bec5752`** (log) → **`1eece1c`** (A/B report + Start-warn dedupe). W8 Control SHIP’d as **`891864f`**.
- Overnight soak ALIVE (~495k validating @ smoke) — untouched.

### ~04:30 — Support-loop resume + P1-3

- Loop PID **52868** still armed to ~08:28 Chicago; overnight **19244/53852** untouched.
- **SHIP P1-3:** `CorruptManifest` on bad/non-dict `map_pack.json`; checkpoint finder skips incomplete zips (test uses real zip >1KiB); `MONITOR_INFO_KEYWORDS` named (P1-5); W8 `ui_selftest` selection/warn coverage.
- **Tests:** `_bugfix_p0_checks` **22/22**; `ui_selftest` all passed (Stop mocked).
- **Pushed:** **`815ad27`**. Skipped `models/leaderboard.csv` runtime row.

---

### ~04:04–04:06 — W3 tick + overnight Continue restart
- Shipped collision-first UI + smoke progress probe (see Shipped).
- Cursor-attached 600s loops die ~60s — use detached `campaign_keepalive.ps1` (PID file `logs/campaign_keepalive.pid`).
- **Incident:** overnight soak found dead (stale lock 34256). Continue-restarted same `run_id` from ckpt `ppo_295480_steps.zip`; sacred floors intact. PIDs ~49144/38752.

---

## Shipped / Tests

### ~04:34 — Repo cleanup batch 4 (standing cleanup agent)
- **gitignore:** ignore `*.__tmp_save__` and `_tmp_*/` so atomic-save / scratch leftovers stay out of git.
- **Re-scan:** no orphan locks; no OS junk; overnight **19244/53852** ALIVE; cleared regen pycache again.
- Did not stage other agents' WIP.

### ~04:33 — Repo cleanup batch 3 (standing cleanup agent)
- **Safe deletes:** regenerated `rl/__pycache__`; confirmed junk `models/_tmp_save_test` (foo.__tmp_save__ / bar.zip atomic-save leftovers).
- **Soft fix:** `logs/CURRENT_PID.txt` stale dead PID 50444 → **19244** (matches overnight `train.lock`).
- **Sacred untouched:** overnight PID **19244/53852** ALIVE; research model dirs left alone; other agents' WIP not staged.

### ~04:30 — Repo cleanup batch 2 (standing cleanup agent)
- **Safe deletes:** regenerated `rl/__pycache__`; zero-byte `logs/*.err.log` / `*.log.err` (non-overnight).
- **gitignore:** add parent `autodrive_py/.gitignore` (`__pycache__/`, `*.pyc`, `*.pyo`, `.pytest_cache/`).
- **Soft quarantine note:** `rl/_trash_candidates/README.md` (gitignored) — smoke `runs/` + old watch debug left in place; underscore `_*.py` helpers NOT trash.
- **Sacred untouched:** overnight `overnight_soak_20260917_082739` PID **19244** ALIVE; no orphan locks; did not stage other agents' WIP.

### ~04:25 — Repo cleanup batch 1 (standing cleanup agent)
- **Safe deletes:** `rl/__pycache__`, parent `autodrive_py/__pycache__`, `_ftg_pin.log` / `_ftg_pin5.log`, empty `runs/` dirs, orphan `train.lock` for `tick0_ab2_base_20260917_040725` (PID 42732 dead; no matching process).
- **gitignore:** add `*.log`, `_ftg_pin*.log`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `_trash_candidates/`, `*.pyo`.
- **Sacred untouched:** overnight `overnight_soak_20260917_082739` PIDs **19244/53852** ALIVE; live locks for active trains left alone; did **not** stage WIP `control_ui.py` / other agents' dirty files.
- **Loop:** cleanup re-scan armed ~35m until ~4h or stop.

### ~04:25 — Tick 0 MUST ships (audit / refuse / A/B / Continue)
- **Overnight lock audit:** code+preset+live config sacred floors PASS (select≥220, eval≥50k, warmup≥2, min_ts≥100k; allow flags false). Soak left running (~495k validating).
- **Holdout refuse:** `assert_train_safe` + CLI `train_ppo` map2/map3 exit=2 + `start_guard` Refuse Start; seals OK.
- **Collision-first A/B:** disposable `tick0_ab2_{base,cf}_20260917_040725` Dummy/cpu 24 576; flag wired; crash_rate mid-run 0.0 both → no crashΔ yet at this budget.
- **Continue proof:** disposable base resume same `run_id` 24576→48128; overnight not used for demo.
- **FTGΔ:** official_v2 PPO `eval_ppo_eval_overnight_best_20260917_0412` adj=**206.68** s (n=15, finisher) vs FTG **198.16** → **Δ=+8.52 s** (lose). Leaderboard row appended; overnight untouched.

### ~04:20 — W7 UI bot allowlist + B0.1 wrong-map refuse
- **Board:** `20260917-W7-ui-bot` behind MUST/bugfix P0.
- **B0.1 SHIP:** `resolve_map_yaml(strict=True)` + `assert_resolved_map_id` on train_ppo / eval_cli / eval_protocol / run_ftg / progress_probe; Watch soft fallback kept; `assert_train_safe` missing-yaml refuse.
- **B0.2 already SHIP:** `acquire_run_lock` O_EXCL (prior tick).
- **UI allowlist:** banner state tint + left accent; `.sec` headers + spacing; `UI_BUILD=w7-ui-bot-20260917`; glossary U-03 verified present.
- **Tests:** `.venv` `python -m rl._bugfix_p0_checks` → **16/16 PASS**; overnight observe-only.
- **Overnight (~04:20):** `overnight_soak_20260917_082739` ALIVE ≈**428k** learning, sps≈201, n_envs=8 subproc; PIDs 19244/53852 untouched.

### ~04:18 — “Reset to ~6k” false alarm (UI multi-train latch)
- **Not a resume/SB3 bug.** Protected overnight `overnight_soak_20260917_082739` still live: Continue from `ppo_295480_steps.zip`, `reset_num_timesteps=False`, live_status **~395k** `phase=validating` (map3), ckpts through `ppo_395480_steps.zip`. Learning **not** lost.
- **Cause:** campaign short A/B smokes (`cf_ab50k_*` ~12–16k, earlier `tick0_ab_*` ~5.6k) running beside overnight. `_find_external_train` picked first leaf `train_ppo` (process order) → UI banner showed ~6k as if soak reset; `_train_busy` then stuck `_train_run_id` on that smoke.
- **Fix:** rank external trains by `CURRENT_RUN.txt` pin → highest live timesteps → `--resume`/`--unlimited` → leaf; status re-syncs preferred run when UI does not own the proc. Verified pick → overnight pid **19244**.
- **Action:** left overnight + disposable A/B trains running; no Continue needed. Refresh Control UI to see ~395k.

### ~04:20 — W6 Wave 1 thin SHIP (Suggestion bot)
- **Board:** `20260917-W6-sug-wave1-ship` amends `20260917-W6-suggestion-bot`.
- **S-01–05:** DONE as docs/tooltips/glossary/compact strip polish; S-03 product checklist still DEFER; no overnight touch; no reward mutation; no bridge code.
- **Code:** `watch` compact honesty strip (phase|crash%); Control glossary + `#bridge_card`; Validating banner gloss; README bridge note; `UI_BUILD=w7-ui-bot-20260917` (W6 copy + parallel UI seat stamp).
- **Tests:** `.venv` `test_watch_overlay` + `ui_selftest` (Stop mocked).

### ~04:15 — Overnight confirm: still learning after selftest Stop incident
- **Incident:** `ui_selftest` HTTP `op=stop` (unmocked) killed protected soak mid-validation (~295k).
- **Mock fix:** Stop path in `ui_selftest` now mocks `_kill_train_tree` / external scan so retests cannot sweep live `train_ppo`.
- **Restore:** Continue same `run_id` from `ppo_295480_steps.zip` with sacred overnight argv (select=220, eval=50k, warmup=2, min_ts=100k, patience=5, unlimited). Lock pid=**19244**, parent=**53852**.
- **Confirm (~04:15):** live_status timesteps **376k**↑, phase=`learning`, sps≈199, `ep_rew_mean`↑, dual-writer refuse intact. No new Continue needed; do not start competing overnight trains.

### ~04:08 — Reliability P0: heartbeat join + Start/Continue no-kill (**SHIP**)
- **Bug 1:** validation heartbeat could overwrite `phase=early_stopped` with `validating` (Event set, thread not joined). Fixed: join+gen-token; refuse non-terminal writes after `early_stopped`; stop heartbeat before terminal write.
- **Bug 2:** Start/Continue called `_kill_train_tree()` after busy check — busy false-negative murdered overnight then spawned. Fixed: **never kill** from Start/Continue (Stop-only); busy also scans live `train.lock`.
- **Regressions:** `ui_selftest` — heartbeat vs early_stopped; Start/Continue no-kill on false-busy; live-lock refuse; HTTP Stop mocked (prior unmocked Stop could sweep live `train_ppo`).
- **Verify:** `python -m rl.ui_selftest` → all passed; overnight survived retest (live Continue PIDs ~53852/19244 @ ~295k).
- **Note:** First selftest run (before Stop mock) raced overnight death mid-validation @ 300k; W3 Continue-restarted from `ppo_295480_steps.zip`.

### ~04:04 — W3 collision-first UI + smoke progress probe
- Control UI checkbox → `--collision-first` on Start; Overnight preset clears checkbox; preview shows collision line.
- `live_status`: `mean_progress_frac_estimate` + `progress_probe_kind=smoke` (Monitor `progress_frac`); UI status row labeled smoke/not official.
- Seal verify helper `seal_verify_summary` (maps_ok=6/6 smoke). Continue reliability still sibling-owned.
- Import smoke OK; overnight `train_ppo` left running.

_(further entries appended as reviewed work lands)_
