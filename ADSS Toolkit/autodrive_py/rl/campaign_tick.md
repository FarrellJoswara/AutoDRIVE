# Campaign keep-alive tick (parent prompt) — AGGRESSIVE CADENCE

**Sentinel:** `AGENT_LOOP_TICK_autodrive-rl-campaign`  
**Interval:** **10 minutes (600s)** — not 50m. User mandate: ideas constantly worked; lots of well-implemented small features; no padding the remaining window.  
**Deadline:** 2026-09-17 12:54 America/Chicago — after that, STOP the loop (kill loop shell PID; do not re-arm).  
**Repo:** `C:\Users\farre\Documents\Github\AutoDRIVE`  
**RL dir:** `ADSS Toolkit/autodrive_py/rl/`

On each tick until the deadline, the parent MUST run a **FULL review board + parallel ship wave** (not orchestrator-only, not single-feature trickle):

---

## Parent actions (exact order)

1. **Researcher + Racer + Minimalist + Reliability (PARALLEL)** — Resume standing agents for **NEXT** proposals (not one-at-a-time serial votes if avoidable):
   - Researcher: `f4ac2241-2ce6-4f1b-b7ea-719eb29035dc` (or spawn if dead) — propose / review **multiple** next small ships; append dated entry(ies) to `CAMPAIGN_BOARD.md` (proposal → evidence → Go/No-Go).
   - Racer: `b0ab82af-c52e-400e-aa7c-0481311ab3cb`
   - Minimalist: `6e80ad0d-65b8-4b3b-9760-da999439843c`
   - Reliability: `a9ade664-e7ec-4df9-a918-9ddb0acf9a9c`  
   Launch personality resumes **in parallel** on the next proposal batch. Append agree/dissent to the same board entries. Need research Go + ≥2 personality Approves to ship each item. Prefer real resume over self-simulating votes.

2. **Orchestrator — MULTIPLE approved small features in parallel** — Resume `4a5b7bd3-a4ff-42b9-ae21-c6461b5e8931` to:
   - Ship **all** currently board-approved small items that fit this tick (not just one).
   - **Spawn sibling implementer subagents** for independent approved chunks so work lands in parallel.
   - Protect overnight training (`overnight_soak_20260917_082739`); never kill live `train_ppo`.
   - **Commit + push often** as each chunk lands (do not batch-push only at end of window).
   - No freestyle / GPU theater / UI chrome / continuous-train product.

3. **Training** — If `train_ppo` for the protected run is dead, Continue-restart that same `run_id`. Never kill a live overnight process without an immediate Continue restart.

4. **Stop** — If local Chicago time ≥ 12:54 on 2026-09-17: kill the keep-alive loop shell, do not re-arm, finalize any push, end campaign.

---

## Throughput expectation

- Each 10m tick should advance **multiple** reviewed ships or clear board blockers — not idle “check-in” padding.
- Prefer small, verifiable, pushable features over one large unfinished spike.
- Sacred defaults / contracts / holdouts / dual-writer refuse stay intact; cut the feature, not the guard.

---

## Copy-paste wake prompt (for loop JSON `prompt` field)

```
CAMPAIGN KEEP-ALIVE TICK (10m aggressive): Execute ADSS Toolkit/autodrive_py/rl/campaign_tick.md in full. (1) PARALLEL: Resume Researcher f4ac2241-2ce6-4f1b-b7ea-719eb29035dc + Racer b0ab82af-c52e-400e-aa7c-0481311ab3cb + Minimalist 6e80ad0d-65b8-4b3b-9760-da999439843c + Reliability a9ade664-e7ec-4df9-a918-9ddb0acf9a9c for NEXT proposal batch; append CAMPAIGN_BOARD.md Go/No-Go. (2) Resume Orchestrator 4a5b7bd3-a4ff-42b9-ae21-c6461b5e8931 to implement MULTIPLE board-approved small features in parallel (spawn sibling implementers); protect overnight_soak_20260917_082739; commit+push often. (3) If training dead, Continue restart same run_id. (4) If past 12:54 America/Chicago 2026-09-17, STOP loop (kill loop shell, do not re-arm). Throughput: ship real features every tick — no padding.
```

---

## Rules

- No ship without a `CAMPAIGN_BOARD.md` entry for that chunk.
- Sacred defaults / contracts / holdouts / dual-writer refuse stay intact.
- Do not disrupt `train_ppo` unless it is already dead and needs Continue.
- Interval is **600s**; do not re-arm at 50m.
