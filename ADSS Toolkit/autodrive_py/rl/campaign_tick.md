# Campaign keep-alive tick (parent prompt)

**Sentinel:** `AGENT_LOOP_TICK_autodrive-rl-campaign`  
**Deadline:** 2026-09-17 12:54 America/Chicago — after that, STOP the loop (kill loop shell PID; do not re-arm).  
**Repo:** `C:\Users\farre\Documents\Github\AutoDRIVE`  
**RL dir:** `ADSS Toolkit/autodrive_py/rl/`

On each tick until the deadline, the parent MUST run a **FULL review board** (not orchestrator-only):

---

## Parent actions (exact order)

1. **Researcher** — Resume standing researcher `f4ac2241-2ce6-4f1b-b7ea-719eb29035dc` (or spawn if dead). Ask them to review what the orchestrator proposes next; append a dated entry to `CAMPAIGN_BOARD.md` (proposal → evidence → Go/No-Go). Prefer real resume over self-simulating votes.

2. **Racer + Minimalist + Reliability** — Resume/spawn quick votes on that proposal:
   - Racer: `b0ab82af-c52e-400e-aa7c-0481311ab3cb`
   - Minimalist: `6e80ad0d-65b8-4b3b-9760-da999439843c`
   - Reliability: `a9ade664-e7ec-4df9-a918-9ddb0acf9a9c`  
   Append agree/dissent to the same `CAMPAIGN_BOARD.md` entry. Need research Go + ≥2 personality Approves to ship.

3. **Orchestrator** — Resume `4a5b7bd3-a4ff-42b9-ae21-c6461b5e8931` to ship **ONLY** board-approved items; protect overnight training (`overnight_soak_20260917_082739`); commit+push reviewed work. No freestyle / GPU theater / UI chrome / continuous-train product.

4. **Training** — If `train_ppo` for the protected run is dead, Continue-restart that same `run_id`. Never kill a live overnight process without an immediate Continue restart.

5. **Stop** — If local Chicago time ≥ 12:54 on 2026-09-17: kill the keep-alive loop shell, do not re-arm, finalize any push, end campaign.

---

## Copy-paste wake prompt (for loop JSON `prompt` field)

```
CAMPAIGN KEEP-ALIVE TICK: Execute ADSS Toolkit/autodrive_py/rl/campaign_tick.md in full. (1) Resume/spawn Researcher f4ac2241-2ce6-4f1b-b7ea-719eb29035dc to review next orchestrator proposal and append CAMPAIGN_BOARD.md. (2) Resume/spawn Racer b0ab82af-c52e-400e-aa7c-0481311ab3cb + Minimalist 6e80ad0d-65b8-4b3b-9760-da999439843c + Reliability a9ade664-e7ec-4df9-a918-9ddb0acf9a9c for quick votes on that proposal. (3) Resume Orchestrator 4a5b7bd3-a4ff-42b9-ae21-c6461b5e8931 to ship ONLY approved items; protect overnight_soak_20260917_082739; push. (4) If training dead, Continue restart same run_id. (5) If past 12:54 America/Chicago 2026-09-17, STOP loop (kill loop shell, do not re-arm).
```

---

## Rules

- No ship without a `CAMPAIGN_BOARD.md` entry for that chunk.
- Sacred defaults / contracts / holdouts / dual-writer refuse stay intact.
- Do not disrupt `train_ppo` unless it is already dead and needs Continue.
