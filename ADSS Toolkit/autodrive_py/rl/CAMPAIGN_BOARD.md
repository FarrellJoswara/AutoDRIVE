# CAMPAIGN_BOARD — Standing researcher ticks

**Campaign:** 9h (start ~03:54 America/Chicago, 2026-09-17)  
**Protected run:** `overnight_soak_20260917_082739` — DO NOT KILL without Continue  
**Sources of truth for MUST order:** `CAMPAIGN_CRITIQUE.md` + `ideas_review/campaign_racer.md` (+ minimalist KEEP / reliability vetoes)  
**Companion brief:** `CAMPAIGN_RESEARCH.md` (COMPLETE)

Append one entry per resume tick. Newest first.

---

## W4 — Auto-train pipeline triage — ~04:09 Chicago
- **id:** `20260917-W4-auto-train-pipeline`
- **type:** tick / wave (user soft suggestion)
- **proposal:** Soft ask for an **Auto-train** button/pipeline that trains → evals lap/adjusted times → continuously gens maps → auto-adjusts reward/knobs → farms many “best” models unattended. Re-triage vs W2 #5 SKIP (continuous outer) after Reliability P0 no-kill/hb-join SHIP + disposable Continue soak PASS.
- **overnight plan:** Observe-only `overnight_soak_20260917_082739`. Auto-train must **never** Start/Continue/kill/morph the protected soak. No reward Discord mid-run. No map regen in the train hot loop.

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

**Resume-me:** Researcher — next tick: (1) do **not** expand Auto-train into reward PBT/mapgen product; (2) re-vote **DEFER→SHIP enablement** only after chaos drills + overnight soft-stop→Continue evidence; (3) if thin CLI lands, verify acceptance #3–#6 on disposable ids only; (4) keep overnight observe-only.

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
