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
