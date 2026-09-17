# Campaign board (append-only)

**Process:** [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md)  
**Rule:** No ship without an entry below for that chunk. Newest entries first under each date.

---

## How to append (≤5 min)

1. Copy **Template** below.
2. Fill proposal → research → votes → Integrator gate.
3. Paste under today’s `## YYYY-MM-DD` heading (create heading if missing). Keep **newest first**.
4. Never edit past entries; use `type: amend` + `amends: <id>` if needed.

---

## Template (copy this block)

```markdown
### YYYY-MM-DD HH:MM <America/Chicago> — `<short-id>`
- **type:** chunk | tick | amend
- **amends:** (none | prior-id)
- **proposal:** <one sentence; or NONE for tick pulse>
- **why now:** <FTG / overnight / crash / honesty — one line>
- **files/area:** <paths or n/a>

**Researcher** — Go | No-Go | Conditional
- notes:
  - …
  - …
- cites: `<path or protocol fact>`

**Votes**
- Racer: Approve | Abstain | Block — …
- Minimalist: Approve | Abstain | Block — …
- Reliability: Approve | Abstain | Block — …

**Integrator:** SHIP | SKIP | DEFER — <need ≥2 Approve + research Go; any Block → SKIP>
- **conditions (if Conditional):** …
- **next:** <implement | wait tick | pick other candidate>
```

---

## 2026-09-17

### 2026-09-17 — `board-init`
- **type:** tick
- **amends:** (none)
- **proposal:** NONE — standing board opened; no ship yet
- **why now:** Establish continuous research + multi-personality gate for remaining 9h campaign
- **files/area:** `CAMPAIGN_GOVERNANCE.md`, `CAMPAIGN_BOARD.md`

**Researcher** — Go
- notes:
  - Governance docs only; no product change in this entry
  - Subsequent chunks/ticks must use this board before ship
- cites: `CAMPAIGN_GOVERNANCE.md`

**Votes**
- Racer: Abstain — no race delta in docs-only init
- Minimalist: Approve — process without new subsystems
- Reliability: Approve — no overnight/train touch

**Integrator:** SKIP — init only; first real candidate needs a new `chunk` entry before any product ship
- **conditions (if Conditional):** n/a
- **next:** On next `AGENT_LOOP_TICK` or before first implementation chunk, append a filled template entry
