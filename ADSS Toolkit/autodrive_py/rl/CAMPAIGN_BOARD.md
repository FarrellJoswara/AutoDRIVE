# Campaign board (append-only)

Process: [`CAMPAIGN_GOVERNANCE.md`](CAMPAIGN_GOVERNANCE.md). Newest entries first under each date.

---

## 2026-09-17

### W1 — Overnight constant audit + holdout refuse smoke
- **id:** `20260917-W1-overnight-holdout-verify`
- **type:** chunk
- **time:** 03:59 America/Chicago
- **proposal:** Verify-only: assert `ui_ops` sacred overnight constants still ≥ floors; smoke `train_ppo --map map2` and `--map map3` exit ≠0 without allow flags; `map_pack verify`. **No constant changes. Do not touch overnight soak process.**
- **files:** read `ui_ops.py`, `train_ppo.py`, `map_pack.py`; run smokes only
- **research checklist (CAMPAIGN_RESEARCH §Go/No-Go):**
  - [x] A sacred: contracts untouched; overnight constants verify≥floors; holdout refuse exercised; dual-writer N/A; protected run untouched
  - [x] B honesty: no official append
  - [x] C throughput: N/A — not throughput work
  - [x] D continuous: not proposing continuous
  - [x] E maps: refuse sealed/validation train
  - [x] F acceptance: holdout smoke; overnight soak smoke **only if** constants changed (they won't)
- **votes:**
  - Researcher: **Go** — verify/smoke only; cite `ui_ops` MID_TRAIN=220, FLOOR=50k, WARMUP=2, MIN_TS=100k; `assert_train_safe` in train_ppo (~767). Aligns RESEARCH spend list.
  - Racer: **Approve** — MUST #1+#5; honesty gates for sealed beat-FTG.
  - Minimalist: **Approve** — no new subsystem; refuse path only.
  - Reliability: **Approve** — does not lower sacred constants; overnight soak left alone.
- **gate:** **SHIP** — research Go + 3 Approves + 0 Blocks. Overnight plan: observe only.
- **result:** _(pending execute)_

### SKIP logged (no board freestyle)
- Continuous-train outer scaffold — **No-Go** (research §2/§6, racer KILL#5, minimalist refuse, reliability P0).
- GPU theater / n_envs knee sweep — **DEFER** until after Continue+collision MUST; research: theater until knee measured; not this chunk.
- Fancy UI reskin — **No-Go** (design test fail).

---
