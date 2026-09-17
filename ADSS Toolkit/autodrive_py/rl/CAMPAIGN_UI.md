# CAMPAIGN_UI — Standing UI bot (Control / Watch presentation)

**Seat:** UI bot (standing)  
**Tone:** Small incremental presentation improvements only — layout, spacing, glossary, a11y, Watch compact, status clarity.  
**Not a blank-check redesign.** Soft ≠ MUST override. Soft ≠ overnight kill / sacred-knob retune / embed Watch / new frameworks.

**Companion:** [`CAMPAIGN_BOARD.md`](CAMPAIGN_BOARD.md) (mini-gate / SHIP) · [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md) § UI seat  
**Prior:** W3 SHIP’d compact Watch + glossary; drastic reskin SKIP; thin CSS ok.

---

## How this seat works

1. Each tick (or every other ~10m): propose **1–3 SMALL** increments below (newest wave first).
2. **Allowlist auto-approve** (still log here + bump `UI_BUILD` + `python -m rl.ui_selftest`):
   - tooltips / `title=` hints
   - spacing / contrast / focus rings
   - labels / section headers
   - glossary copy bullets
   - compact-mode polish (no argv sacred retune)
3. **Board mini-gate** for anything outside allowlist.
4. **FORBIDDEN without full board:** Gradio/React/new frameworks · embed Watch in browser · change Overnight sacred defaults · remove honesty / unofficial banners.
5. Never Start/Continue/kill overnight. Soft ≠ starve collision A/B, FTGΔ, or Bugfix P0.
6. **Multi-run soft constraint:** Prefer selected-run clarity when several `train.lock`s are live. Do not design chrome that assumes one global trainer or kills multi-train support (board `20260917-W8-multi-run-ui`).

### Proposal template

```
#### U-YYYYMMDD-WN-## — <one-line title>
- **soft:** yes
- **allowlist:** yes | no (needs mini-gate)
- **surface:** Control | Watch | both
- **why (≤2 lines):** …
- **concrete (≤3 bullets):** …
- **risks / refuse:** …
- **triage:** PENDING | SHIP | DEFER | SKIP
- **board:** <id or —>
- **shipped:** — | DONE <UI_BUILD> + ui_selftest
```

---

## Wave 2 — 20260917-W8 multi-run (soft) — ~04:22 Chicago

**wave-id:** `20260917-W8-ui-multi-run`  
**board:** `20260917-W8-multi-run-ui`  
**count:** 4 incremental + DEFER fancy + SKIP kill-multi · all **soft**  
**Implement this tick?** **Yes** — W8-01…04 code SHIP’d (`UI_BUILD=w8-multi-run-20260917`).

### Design intent (keep multi-train)

Operators already run **several concurrent trains** and see multiple live locks. Control must not pretend there is one global “the run.” Banner / Watch / Continue / Stop must name a **selected** `run_id`. Do **not** remove multi-train support to simplify chrome. Do **not** kill overnight.

**Backend already there:** `ui_ops.find_live_run_locks(models_dir)` → `{run_id,pid,alive,…}`; per-run `runs/<run_id>/live_status.json` via `read_live_status`. Gap is presentation + selection binding in `control_ui` (`_status_payload` / refresh JS).

#### U-20260917-W8-01 — Live-runs list (locks + live_status)
- **soft:** yes
- **allowlist:** no (needs mini-gate — **SHIP’d** on board `20260917-W8-multi-run-ui`)
- **surface:** Control
- **why:** Models table shows `[locked pid=…]` but Live status is one banner; with N locks, operators cannot see who is validating vs learning.
- **concrete:**
  - `/api/status` include `live_runs: [{run_id, pid, timesteps, phase, overnight: bool, selected: bool}, …]` from `find_live_run_locks` + each run’s `live_status`.
  - Thin `<ul id="live_runs">` under Live train status (same-DOM; no Gradio). Overnight badge when `run_id` matches `overnight_soak_*` (or protected id).
  - Click row = select run (sets client + optional query/`selected_run` for status).
- **risks / refuse:** Read-only list only; no auto-Stop; no kill-tree; Stop path stays mocked in selftest while soak live.
- **triage:** SHIP
- **board:** `20260917-W8-multi-run-ui`
- **shipped:** DONE `w8-multi-run-20260917`

#### U-20260917-W8-02 — Banner bound to selected run
- **soft:** yes
- **allowlist:** no (mini-gate with W8-01)
- **surface:** Control
- **why:** `_status_payload` can adopt “preferred” external train / `find_latest_status` — last-writer / highest-ts wins and hides sibling locks.
- **concrete:**
  - Banner + Live rows read `live_status` for **selected** `run_id` only.
  - Default select: UI-owned `_train_run_id` if alive → else overnight/protected if among live locks → else first live lock → else latest status.
  - Never silently flip selection mid-poll unless selected run dies.
- **risks / refuse:** No sacred retune; keep honesty / Validating gloss.
- **triage:** SHIP
- **board:** `20260917-W8-multi-run-ui`
- **shipped:** DONE `w8-multi-run-20260917` (with W8-01)

#### U-20260917-W8-03 — Watch / Continue / Stop target clarity
- **soft:** yes
- **allowlist:** no (mini-gate with W8-01)
- **surface:** Control
- **why:** Buttons imply “the” trainer; with multi-lock, wrong Stop/Watch is costly (overnight incident history).
- **concrete:**
  - Label / hint near Train toggle + Watch: `targets: <selected_run_id>` (update on select).
  - Continue from Models table already names `run_id` — keep; optional highlight when that row is selected.
  - Stop confirm (if any) must echo selected run_id + pid.
- **risks / refuse:** Do not broaden Stop to “kill all locks”; Stop remains selected/UI-owned tree only.
- **triage:** SHIP
- **board:** `20260917-W8-multi-run-ui`
- **shipped:** DONE `w8-multi-run-20260917` (with W8-01)

#### U-20260917-W8-04 — Warn before Start when N locks live
- **soft:** yes
- **allowlist:** no (mini-gate with W8-01)
- **surface:** Control
- **why:** `_train_busy_reason` already refuses Start when any live lock exists — operators still need a visible N-locks cue before they fight the refuse.
- **concrete:**
  - Preview / Start path: if `len(live_locks) >= 1`, show warn listing run_ids (“N trains locked — Start refused until Stop; multi-train is supported via separate CLI/UI ownership, not dual-Start from this panel”).
  - Do **not** change refuse semantics this wave (still refuse dual-Start from Control while any lock live) unless a later board explicitly SHIPs true multi-Start from one UI.
- **risks / refuse:** Warn ≠ enable dual-writer; never `_kill_train_tree` to “make room.”
- **triage:** SHIP
- **board:** `20260917-W8-multi-run-ui`
- **shipped:** DONE `w8-multi-run-20260917` (with W8-01)

#### U-20260917-W8-05 — Fancy multi-run dashboard
- **soft:** yes
- **allowlist:** no
- **surface:** Control
- **why:** Nice-to-have charts/cards across runs; not needed for operator survival this 9h.
- **concrete:** — (park)
- **risks / refuse:** Frameworks / reskin / embed Watch.
- **triage:** DEFER
- **board:** `20260917-W8-multi-run-ui`
- **shipped:** —

#### U-20260917-W8-06 — Kill multi-train support to simplify UI
- **soft:** yes
- **allowlist:** n/a
- **surface:** Control / train
- **why:** Tempting “one run only” product — conflicts with real operator use (several locked run_ids) and overnight + disposable A/B coexistence.
- **concrete:** — refuse
- **risks / refuse:** Would fight dual-writer honesty and campaign parallel smokes.
- **triage:** SKIP (hard)
- **board:** `20260917-W8-multi-run-ui`
- **shipped:** —

---

## Wave 1 — 2026-09-17 ~04:16 Chicago (first batch)

**wave-id:** `20260917-W7-ui-wave1`  
**board:** `20260917-W7-ui-bot`  
**count:** 3 · all **soft** · process SHIP’d

#### U-20260917-W7-01 — Status banner readability
- **soft:** yes
- **allowlist:** yes (contrast / labels)
- **surface:** Control
- **why:** Phase banner + reason are easy to miss on dark `#181818`; Validating vs Stale vs Learning need stronger visual separation without a reskin.
- **concrete:**
  - Left accent bar + state-tinted background on `#banner`.
  - Slightly larger / higher-contrast `#banner_reason`.
- **risks / refuse:** No new frameworks; keep honesty tooltips; no JS phase logic change.
- **triage:** SHIP now
- **board:** `20260917-W7-ui-bot`
- **shipped:** DONE `w7-ui-bot-20260917` (this tick)

#### U-20260917-W7-02 — Section headers + spacing rhythm
- **soft:** yes
- **allowlist:** yes (spacing / labels)
- **surface:** Control
- **why:** Live status / Coach / Train / Models / TensorBoard boxes blend together; a light `.sec` header + margin rhythm improves scan without Gradio/React.
- **concrete:**
  - Shared `.sec` class on box titles (Live train status, Coach, Train controls, Models, TensorBoard).
  - Thin same-DOM CSS only (no color theme flip).
- **risks / refuse:** No card redesign carnival; no remove of soft-stop / overnight warn copy.
- **triage:** SHIP now
- **board:** `20260917-W7-ui-bot`
- **shipped:** DONE `w7-ui-bot-20260917` (this tick)

#### U-20260917-W7-03 — Glossary expand (curriculum + map roles)
- **soft:** yes
- **allowlist:** yes (glossary copy)
- **surface:** Control
- **why:** W6 SHIP’d reward/curriculum clarity + map checklist as glossary bullets; drawer still thin on collision-first / speed-gate / train_ok vs holdout.
- **concrete:**
  - Add glossary `dt`/`dd` for collision-first, speed-gate, train_ok / validation / holdout roles.
  - Optional one-line “Validating ≠ stuck” already present — keep; do not auto-Start.
- **risks / refuse:** No product checklist widget; no auto-Start; no reward API.
- **triage:** SHIP now
- **board:** `20260917-W7-ui-bot`
- **shipped:** DONE `w7-ui-bot-20260917` (glossary already had collision-first / speed-gate / train_ok / holdout / validation; verified this tick)

---

## STANDING Resume-me

Next tick (or every other): invent **1–3** small increments → append wave → allowlist auto-approve **or** mini-gate → implement only SHIP → `ui_selftest` → never kill overnight / never redesign. Soft ≠ MUST / Bugfix P0.

**Multi-run soft constraint (W8):** Prefer chrome that names a **selected** `run_id` when multiple `train.lock`s are live. Implement W8-01…04 when free (board already SHIP’d design). **DEFER** fancy dashboard. **SKIP** killing multi-train support.
