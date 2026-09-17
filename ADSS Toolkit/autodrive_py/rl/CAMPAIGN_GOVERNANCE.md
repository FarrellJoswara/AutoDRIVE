# Campaign governance — standing board (9h AutoDRIVE RL)

**Purpose:** Research + multi-personality review run **continuously** — every feature wave and every ~50 min `AGENT_LOOP_TICK` — not as a one-shot kickoff.

**Scope:** Process only. This file does not implement product features.

**Companion artifacts:** [`CAMPAIGN_BOARD.md`](CAMPAIGN_BOARD.md) (append-only ship permission); [`CAMPAIGN_SUGGESTIONS.md`](CAMPAIGN_SUGGESTIONS.md) (soft idea inbox); [`CAMPAIGN_BUGFIX.md`](CAMPAIGN_BUGFIX.md) (stable-core bugfix/perf).  
**Hard rule:** **No ship without a board entry** for that chunk (proposal → research → votes → ship/skip). Soft suggestions are **not** ship permission.

---

## Roles (fixed lenses)

| Seat | Job in ≤90s | Approve when… | Refuse when… |
| ---- | ----------- | ------------- | ------------ |
| **Researcher** | Evidence from codebase + domain docs; emit **Go / No-Go / Conditional** | Claim is grounded (file/contract/protocol cited); risk known | Speculation, missing sacred-guard check, no FTG/overnight angle |
| **Racer** | Does it help beat **pinned FTG** / `adjusted_time` on sealed maps? | Moves crash↓, finish↑, honest official Δt, or overnight that enables racing | GPU theater, Watch chrome, vanity metrics, cheating timeout/protocol |
| **Minimalist** | Complexity trap? Default refuse. | Small, reversible, one job; deletes more than it adds | New subsystem, dual paths, “while we’re here”, continuous-train cosplay |
| **Reliability** | Overnight / train safety? | Protects Continue, early-stop defaults, dual-writer refuse, seals | Touches live overnight without Continue plan; lowers sacred constants |
| **Bugfix** (recurring duty) | Bugs + perf on **stable cores** that stick | Ranked finding in [`CAMPAIGN_BUGFIX.md`](CAMPAIGN_BUGFIX.md); small overnight-safe fix | Auditing throwaway UI chrome / bridge stubs; large refactors mid-soak |
| **Suggestion** | Invent **soft** ideas (UX / train / perf / pipeline / transfer); same tone as user W3/W4/W5 soft asks | Idea is optional, reversible, diverse, and filed to `CAMPAIGN_SUGGESTIONS.md` + a short board proposal | Mandates, auto-code, MUST override, overnight morph, sacred retune, SKIP-forever repeats (GPU theater, continuous outer, reward Discord, bridge product) |
| **Integrator / Orchestrator** | Ship gate only | **Researcher = Go** (or Conditional with conditions met) **and ≥2 of {Racer, Minimalist, Reliability} Approve** | Any seat **Block**; research **No-Go**; no board entry; shipping from suggestions inbox alone |

Personality deep-dives (optional refresh, not required every tick):

- `ideas_review/campaign_racer.md`
- `ideas_review/campaign_minimalist.md`
- `ideas_review/campaign_reliability.md` (when present)
- Plus: `CAMPAIGN_CRITIQUE.md`, `CAMPAIGN_RESEARCH.md`, `CAMPAIGN_LOG.md`

Sacred (never cut to ship a feature): early-stop defaults (select≥220, eval≥50k, warmup≥2, min_ts≥100k), contracts `2.0.0`, holdouts/seals, atomic promote, dual-writer refuse. Never kill `overnight_soak_*` without Continue restart.

---

## Cadence (mandatory)

Run the board in **both** of these situations:

1. **Before each implementation chunk** (any non-trivial code/config change aimed at the campaign).
2. **On every `AGENT_LOOP_TICK`** (~10m aggressive / legacy ~50m): even if “nothing to ship,” open a board entry — reaffirm backlog, skip, or propose the next chunk.

Ticks with no candidate: still append a short **tick pulse** (see template) so continuity is visible.

### Bugfix duty in the board loop

- **When:** each tick (or when a MUST ship touches env/IO/seals/locks).
- **Does:** skim IN-scope stables (`contracts`, `racing_env` reward/physics, `live_status`, atomic checkpoint IO, `map_pack` seals, SubprocVecEnv path) → append/update ranked P0/P1/P2 in `CAMPAIGN_BUGFIX.md` → board a small fix chunk if free hands.
- **Does not:** audit cosmetic chrome / bridge stubs; large refactors; overnight morph; starve collision A/B or FTGΔ.
- **Gate:** same Integrator rules; Reliability Approve required if touching locks/seals/Continue.

### Suggestion seat in the board loop

- **When:** each ~10m tick, **or every other tick** (avoid inbox spam).
- **Does:** invent **3–7** soft ideas → append a wave to `CAMPAIGN_SUGGESTIONS.md` → open a short `CAMPAIGN_BOARD.md` proposal for triage.
- **Does not:** implement code, override MUST order, Start/Continue/kill overnight, or treat soft ideas as mandates.
- **Then:** Researcher + Racer + Minimalist + Reliability vote; Integrator marks **SHIP / DEFER / SKIP**; Orchestrator builds **SHIP only**.

---

## 5-minute tick procedure (agents)

Budget: **~5 minutes**. Do not write essays.

1. **Propose (30s)** — One candidate (or `NONE`). One sentence + intended files if known. Soft waves may propose a **batch** of suggestion ids for triage.  
2. **Research (90s)** — Skim relevant code/docs; checklist 3–5 bullets; verdict **Go / No-Go / Conditional**. Cite 1–2 paths or protocol facts.  
3. **Votes (2 min)** — Each of Racer / Minimalist / Reliability: **Approve / Abstain / Block** + ≤1 line why.  
4. **Gate (30s)** — Integrator: **SHIP** only if research Go/Conditional-met **and** ≥2 Approves **and** zero Blocks. Else **SKIP** (log why). Soft suggestions default to triage, not auto-SHIP.  
5. **Append** — Paste the filled template at the **top** of the dated section in `CAMPAIGN_BOARD.md` (newest first under today’s date).  
6. **Act** — If SHIP: implement that chunk only. If SKIP / NONE: do not code product changes; optionally update `CAMPAIGN_LOG.md` pointer. Suggestion seat: stop after inbox + board proposal unless also acting as another seat.

Fail review → **SKIP**. Partial votes without Integrator gate → **do not ship**.

---

## Ship gate (checklist)

- [ ] Board entry exists for this exact proposal (same title/id)
- [ ] Researcher ≠ No-Go
- [ ] If Conditional: listed conditions satisfied or still open items are non-blocking and logged
- [ ] ≥2 Approves among Racer / Minimalist / Reliability
- [ ] Zero Blocks
- [ ] Sacred guards untouched (or change is verify/smoke-only with explicit Approve from Reliability)
- [ ] Overnight run plan stated if train process could be affected

Integrator records: `SHIP` | `SKIP` | `DEFER` + one line.

---

## Artifacts

| File | Role |
| ---- | ---- |
| `CAMPAIGN_GOVERNANCE.md` | This standing process (stable) |
| `CAMPAIGN_BOARD.md` | Append-only dated entries; **source of truth for ship permission** |
| `CAMPAIGN_SUGGESTIONS.md` | Soft idea inbox (Suggestion seat); **not** ship permission until board triage |
| `CAMPAIGN_BUGFIX.md` | Standing bugfix / perf findings on stable cores (Bugfix duty); fixes still need board entries |
| `CAMPAIGN_LOG.md` | Operator narrative / protected run / backlog summary — may **point to** board ids, not replace them |
| `CAMPAIGN_RESEARCH.md` / critique / `ideas_review/campaign_*.md` | Standing evidence; Researcher cites, does not rewrite every tick |

**Append-only:** Never delete or rewrite past board entries. Correct mistakes with a new entry (`amends: <prior-id>`). Same for suggestion waves (mark triage; do not erase).

---

## Entry types

- **chunk** — Feature/fix/smoke candidate before implementation  
- **tick** — Periodic pulse (~10m / ~50 min); may be `NONE` / reaffirm / escalate  
- **wave** — Soft-suggestion batch triage (Suggestion seat + personality votes)  
- **amend** — Corrects a prior entry’s gate without erasing history  

---

## Vote shorthand (copy-paste)

```
Researcher: Go | No-Go | Conditional — <one line + cite>
Racer:      Approve | Abstain | Block — <one line>
Minimalist: Approve | Abstain | Block — <one line>
Reliability:Approve | Abstain | Block — <one line>
Integrator: SHIP | SKIP | DEFER — <one line> (need ≥2 Approve + research Go; any Block = SKIP)
```

---

## Anti-patterns

- Shipping from chat consensus without a board entry  
- One-shot kickoff review then silence until deadline  
- Rubber-stamp Approves with no Researcher cite  
- “Temporary” protocol/timeout/FTG retune to manufacture a win  
- Touching overnight early-stop math to feel productive  
- Board essays >5 min; if over budget, **SKIP** and note time-box

---

## Quick link for agents

On `AGENT_LOOP_TICK` or before coding: open `CAMPAIGN_BOARD.md` → copy template → fill → gate → only then implement.  
Suggestion seat: append soft wave to `CAMPAIGN_SUGGESTIONS.md` first, then board triage — never ship from the inbox alone.
