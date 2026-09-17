# CAMPAIGN_SUGGESTIONS — Soft idea inbox

**Seat:** Suggestion bot (standing)  
**Tone:** Soft improvement ideas only — UX, train, perf, pipeline, AutoDRIVE transfer.  
**Not mandates.** Soft ≠ MUST override. Soft ≠ overnight kill / sacred-knob retune / bridge product this 9h.

**Companion:** [`CAMPAIGN_BOARD.md`](CAMPAIGN_BOARD.md) (triage SHIP / DEFER / SKIP)  
**Process:** [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md) § Suggestion seat

---

## How this file works

1. Suggestion bot (each ~10m tick, or every other tick) appends a **wave** of **3–7** soft ideas below (newest wave first).
2. Bot also opens a short board proposal pointing at this wave’s ids.
3. Standing board triages each idea: **SHIP** / **DEFER** / **SKIP**.
4. Orchestrator implements **SHIP only** — and only behind race MUST when hands are free.
5. Never delete past waves; mark triage inline or via board entry `amends`.

### Soft idea template (copy per idea)

```
#### S-YYYYMMDD-WN-## — <one-line title>
- **soft:** yes
- **area:** UX | train | perf | pipeline | transfer
- **why (≤2 lines):** …
- **concrete (≤3 bullets):** …
- **risks / refuse:** …
- **triage:** PENDING | SHIP | DEFER | SKIP  (board fills)
- **board:** <id or —>
```

---

## Wave 3 — 2026-09-17 ~04:36 Chicago (idea-factory cycle 2)

**wave-id:** `20260917-W10-sug-wave3`  
**board:** `20260917-W10-idea-factory`  
**count:** 7 · all **soft**

#### S-20260917-W10-01 — Preserve crash_rate across validating writes (P1-8)
- **soft:** yes
- **area:** reliability
- **why:** RaceBest `_write_status` overwrote trail and blanked `crash_rate_estimate` → MUST #3 A/B `INCONCLUSIVE_MISSING_CRASH`.
- **concrete:**
  - Merge prior crash/stall/progress counters into RaceBest validating/learning status writes.
- **risks / refuse:** No overnight morph; library-only until next train process.
- **triage:** SHIP
- **board:** `20260917-W10-idea-factory`

#### S-20260917-W10-02 — Stall rate in live_status + Control
- **soft:** yes
- **area:** train / UX
- **why:** Tick0 A/B was stall-heavy with 0 crash%; operators need stall% separate from crash%.
- **concrete:**
  - Count Monitor `stall` eps; expose `stall_rate_estimate`; thin Control row.
- **risks / refuse:** In-memory overnight still old code until Continue.
- **triage:** SHIP
- **board:** `20260917-W10-idea-factory`

#### S-20260917-W10-03 — Watch compact shows Focus pin run_id
- **soft:** yes
- **area:** UX
- **why:** Carry-over W9-03; CURRENT_RUN helpers already shared.
- **concrete:** Compact strip `run=<id>` from pin / status.
- **risks / refuse:** No embed.
- **triage:** DEFER
- **board:** `20260917-W10-idea-factory`

#### S-20260917-W10-04 — atomic_write_json Windows lock retry (P1-4)
- **soft:** yes
- **area:** reliability
- **why:** UI can raise mid-read of config while train writes.
- **concrete:** Same retry/fallback as `write_live_status`.
- **risks / refuse:** Overnight observe-only.
- **triage:** DEFER (next bugfix tick)
- **board:** `20260917-W10-idea-factory`

#### S-20260917-W10-05 — Spawn jitter refuse occupied base (P1-6)
- **soft:** yes
- **area:** train
- **why:** After 24 rejects, base pose may still collide.
- **concrete:** Raise / resample from centerline if base occupied.
- **risks / refuse:** Disposable A/B only; no overnight morph.
- **triage:** DEFER
- **board:** `20260917-W10-idea-factory`

#### S-20260917-W10-06 — Obs buffer prealloc (PERF-6)
- **soft:** yes
- **area:** perf
- **why:** Tiny alloc win vs LiDAR cast; measure before cast rewrite.
- **concrete:** Preallocate obs buffer on env reset.
- **risks / refuse:** Contracts freeze; measure sps first.
- **triage:** DEFER
- **board:** `20260917-W10-idea-factory`

#### S-20260917-W10-07 — Continuous outer / reward Discord
- **soft:** yes
- **area:** train
- **why:** SKIP-forever repeats.
- **concrete:** —
- **risks / refuse:** Overnight sacred.
- **triage:** SKIP (hard)
- **board:** `20260917-W10-idea-factory`

---

## Wave 2 — 2026-09-17 ~04:30 Chicago (idea-factory)

**wave-id:** `20260917-W9-sug-wave2`  
**board:** `20260917-W9-idea-factory`  
**count:** 8 · all **soft** · avoid sacred early-stop retunes

#### S-20260917-W9-01 — Coach hint when N live locks
- **soft:** yes
- **area:** UX
- **why:** W8 list/warn helps Start path; Coach still silent when overnight + A/B coexist — operators miss Focus.
- **concrete:**
  - Pass `live_run_count` into `coach_hints`; one line when N≥2 pointing at Live runs Focus.
- **risks / refuse:** No dual-Start enablement; no kill.
- **triage:** SHIP (allowlist copy)
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-02 — Selftest fake zips must pass zipfile.testzip
- **soft:** yes
- **area:** tooling
- **why:** P1 zip completeness gate rejects `PK`+zeros stubs; Continue/model selftests falsely fail.
- **concrete:**
  - `_fake_run` writes a real minimal zip; ASCII-safe Focus msgs for cp1252 consoles.
- **risks / refuse:** Do not weaken `_zip_looks_complete` for production.
- **triage:** SHIP
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-03 — Watch overlay shows Focus pin run_id
- **soft:** yes
- **area:** UX
- **why:** Compact Watch still anonymous when several trains live; CURRENT_RUN pin already exists.
- **concrete:**
  - Compact strip include `run_id` from env/`CURRENT_RUN.txt` when following.
- **risks / refuse:** No embed Watch; no argv sacred retune.
- **triage:** DEFER (next UI tick)
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-04 — Leaderboard protocol column honesty
- **soft:** yes
- **area:** eval
- **why:** Mid-train validation rows can look like official 400s×5 if skimmed fast.
- **concrete:**
  - Docs + optional `protocol_id` tooltip in Control race_candidate line (copy only).
- **risks / refuse:** No protocol shorten; no denser overnight eval.
- **triage:** DEFER
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-05 — Atomic CheckpointCallback wrapper
- **soft:** yes
- **area:** reliability
- **why:** CAMPAIGN_BUGFIX P1-1 — SB3 ckpt writes not routed through `atomic_save_sb3`.
- **concrete:**
  - Wrap CheckpointCallback save path; keep overnight observe-only (lands on Continue).
- **risks / refuse:** No overnight morph; measure before large rewrite.
- **triage:** DEFER (Bugfix P1 queue)
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-06 — Train spawn pose jitter
- **soft:** yes
- **area:** train
- **why:** IDEAS_TRIMMED: Watch has jitter; train resets still fixed spawn → fake skill on map0.
- **concrete:**
  - Small lateral/heading jitter on reset for train_ok maps only.
- **risks / refuse:** No overnight morph; disposable A/B only; no holdout.
- **triage:** DEFER (behind race MUST / disposable A/B)
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-07 — HANDOFF multi-run Focus one-liner
- **soft:** yes
- **area:** docs
- **why:** Returning operators still Task-Manager-stop when banner looks stuck on short A/B.
- **concrete:**
  - `HANDOFF.md` bullet: Focus pin + Start refuse semantics + overnight never killed.
- **risks / refuse:** Docs only.
- **triage:** SHIP
- **board:** `20260917-W9-idea-factory`

#### S-20260917-W9-08 — GPU theater / denser overnight eval
- **soft:** yes
- **area:** perf
- **why:** Tempting to densify race_eval_every "to use GPU" while validating freezes learning.
- **concrete:** — refuse forever this campaign
- **risks / refuse:** Sacred floors; overnight soak.
- **triage:** SKIP (hard)
- **board:** `20260917-W9-idea-factory`

---

## Wave 1 — 2026-09-17 ~04:11 Chicago (first batch)

**wave-id:** `20260917-W6-sug-wave1`  
**board:** `20260917-W6-suggestion-bot`  
**count:** 5 · all **soft**

#### S-20260917-W6-01 — Watch compact honesty strip
- **soft:** yes
- **area:** UX
- **why:** Operators drown in counter spam while mid-train Watch stays unofficial; W3 already SHIPed compact + higher `--every` — a thinner “honesty strip” (phase, lag-behind, crash_rate, unofficial banner) cuts misreads without carnival chrome.
- **concrete:**
  - Default `compact` overlay to race KPIs + honesty banners only; hide dense timestep/FPS rows behind expand.
  - Keep train headless; Watch stays separate process; `test_watch_overlay` must stay green.
- **risks / refuse:** No embed-in-Control; no ghost/seasons; no auto-open on Start.
- **triage:** DONE (W3 compact/`--every` + W6 polish: compact train line = phase|ts|/s|crash%)
- **board:** `20260917-W6-suggestion-bot`

#### S-20260917-W6-02 — Reward / curriculum plain-language clarity
- **soft:** yes
- **area:** train
- **why:** Collision-first, speed-gate, patience, and “Validating ≠ stuck” are easy to misread; wrong Starts and mid-run Discord dial temptation follow from opaque copy, not from missing math.
- **concrete:**
  - Extend Control `title=` / glossary drawer: collision-first, speed-gate, race_eval_every vs official, patience vs budget-OFF.
  - One-line status gloss when `phase=validating` (“scheduled race eval — not hung”).
- **risks / refuse:** No reward mutation API; no mid-overnight flag morph; sacred floors stay worded honestly.
- **triage:** DONE (W3 glossary base + W6 glossary/titles + Validating banner gloss)
- **board:** `20260917-W6-suggestion-bot`

#### S-20260917-W6-03 — Map curriculum unlock checklist (train_ok only)
- **soft:** yes
- **area:** pipeline
- **why:** RESEARCH §5 wants crash/finisher gates before Map1 / speed unlock; operators still think “more maps = podium.” A read-only checklist next to Start reduces holdout foot-guns without autogen-in-hot-loop.
- **concrete:**
  - UI/CLI checklist: map role (`train_ok` / validation / holdout), fingerprint present, `assert_train_safe` would pass.
  - Optional: print next suggested train_ok id from pack — never auto-Start overnight.
- **risks / refuse:** No mapgen inside `step`; no sealed-id train; no dual-Start while soak live.
- **triage:** DEFER product UI — DONE glossary bullets (`train_ok` checklist copy); no auto-Start
- **board:** `20260917-W6-suggestion-bot`

#### S-20260917-W6-04 — Eval wall-clock honesty (sparse mid-train reminder)
- **soft:** yes
- **area:** perf
- **why:** Densifying mid-train eval “to use the GPU” lengthens `phase=validating` and starves learning; a soft operator cue + default copy that 50k floor / fewer mid-train episodes is the win keeps eval speed without protocol lies.
- **concrete:**
  - Status/tooltip: last validating duration + “official stays 400s / 5 seeds — do not densify overnight.”
  - Document-only or tiny preview: mid-train = 1 seed × validation map spirit (already in RaceBest) — no timeout shorten.
- **risks / refuse:** Never lower `select_timeout` / official 400; never drop `eval_spawn_jitter`; no “faster = official.”
- **triage:** DONE tooltip/status/glossary only — SKIP denser overnight eval / timeout shorten
- **board:** `20260917-W6-suggestion-bot`

#### S-20260917-W6-05 — Bridge-readiness constraint card (transfer observe)
- **soft:** yes
- **area:** transfer
- **why:** Gym→AutoDRIVE transfer is a standing constraint (contracts `2.0.0`, LiDAR DR, no privileged GT obs) but easy to forget under speed pressure; a visible card prevents bridge-incompatible obs/reward shortcuts this 9h.
- **concrete:**
  - Short Control/docs card: freeze ABI; shaping may use GT, race obs may not; bridge `:4567` = P3 after sealed holdout beat-FTG.
  - Cite RESEARCH §5 transfer note — **no** Jetson/latency/bridge code.
- **risks / refuse:** Document-only this window; Block any obs-dim morph or IPS/pose-in-obs “for gym wins.”
- **triage:** DONE document-only (Control `#bridge_card` + README note) — SKIP bridge/Jetson code
- **board:** `20260917-W6-suggestion-bot`
