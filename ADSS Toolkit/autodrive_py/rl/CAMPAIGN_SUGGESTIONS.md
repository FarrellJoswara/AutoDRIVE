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
