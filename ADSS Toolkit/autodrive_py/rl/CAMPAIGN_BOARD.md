# CAMPAIGN_BOARD — Standing researcher ticks

**Campaign:** 9h (start ~03:54 America/Chicago, 2026-09-17)  
**Protected run:** `overnight_soak_20260917_082739` — DO NOT KILL without Continue  
**Sources of truth for MUST order:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_racer.md` (+ minimalist KEEP / reliability vetoes)  
**Companion brief:** `CAMPAIGN_RESEARCH.md` (COMPLETE)

Append one entry per resume tick. Newest first.  
**Soft inbox:** [`CAMPAIGN_SUGGESTIONS.md`](CAMPAIGN_SUGGESTIONS.md) (Suggestion seat — not mandates)  
**UI inbox:** [`CAMPAIGN_UI.md`](CAMPAIGN_UI.md) (UI seat — incremental Control/Watch chrome; not mandates)

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
| U-20260917-W7-03 | Glossary expand (curriculum / map roles) | **DEFER code** this tick if time-box; **SHIP copy** next free hands (allowlist) |

**Orchestrator (this tick):** docs + U-01 + U-02; leave U-03 for next UI tick if not free. Soft ≠ MUST/bugfix override.

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
| S-20260917-W6-01 | Watch compact honesty strip | **SHIP** (bundle with W3 Watch compact / `--every`; free hands only) |
| S-20260917-W6-02 | Reward / curriculum plain-language clarity | **SHIP** (bundle with W3 glossary; no reward API) |
| S-20260917-W6-03 | Map curriculum unlock checklist | **DEFER** product UI — **SHIP** checklist copy as glossary bullets OK; no auto-Start |
| S-20260917-W6-04 | Eval wall-clock honesty cue | **SHIP** tooltip/status copy only; **SKIP** timeout shorten / denser overnight eval |
| S-20260917-W6-05 | Bridge-readiness constraint card | **SHIP** document-only (RESEARCH §5); **SKIP** bridge/Jetson code |

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
