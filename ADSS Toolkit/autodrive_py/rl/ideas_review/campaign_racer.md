# Campaign racer — 9h beat-FTG triage

**Lens:** Ruthless racing product owner. Clock is `adjusted_time = lap + 10·collisions` vs a **pinned** FTG under a frozen protocol. Comfort, GPU %, and chrome do not podium.  
**Sources:** `CAMPAIGN_CRITIQUE.md`, `CAMPAIGN_LOG.md`, `IDEAS_TRIMMED.md` / `PLAN.md` Phase 0–2.  
**Window:** ~9h on `rl/phase-1-research` with protected overnight soak running. **No code in this doc.**

**Verdict:** Operator honesty is mostly shipped. The remaining hours buy **survival → clean finish → honest Δt**, not a prettier pit wall. If a feature doesn’t lower crash rate, protect overnight continue, or publish a real FTG compare on sealed maps — **cut it**.

---

## What HELPS beat FTG / adjusted_time

| Feature (9h backlog / trimmed ideas) | Why it moves the clock |
| --- | --- |
| **Collision-first curriculum that visibly drops crash rate** | λ=10: every hit is ten seconds. Speed unlock before crash≈0 farms death and loses the sport. PLAN P1 exit still open — this is the race. |
| **Official FTG vs PPO row (`official_v2`)** | Without a non-null adjusted_time Δ you are tuning vibes. Lose honestly; then iterate. Blind “improvement” is cosplay. |
| **Holdout refuse + seal verify before official append** | Beat-FTG on map0 alone is barnacle racing. Sealed holdout is the product gate; train-on-holdout voids the claim. |
| **Overnight early-stop regression lock** | Dead overnight = zero samples toward finish rate. Warmup / min_ts / select_timeout≥220 already paid for — don’t regress. |
| **Continue-train soak (Stop → same `run_id` from complete ckpt)** | Long runs are how finishers appear. Dual-writer / broken resume burns the night and the metric. |
| **Multi-map train (`map0,map1`) after crash rate moves** | Diversified envs kill map0 memorization that fails holdouts. Only after survival — otherwise you farm crashes on more walls. |
| **TTC / frontal truncate + spawn jitter (if not already biting)** | Doomed eps and fixed pose inflate fake reward; they don’t finish races. |
| **Promote / select by race_score_key only** | Already sacred. Touch only to defend — never reintroduce `ep_rew_mean` as “best”. |

**Conditional (after crash rate drops + one serious overnight survives):** light LiDAR DR **or** engineered L/R/F clearances — **one**, not both. Residual / FTG warm-start only if pure PPO plateaus *after* finishing laps.

---

## Distractions / traps (look busy, don’t podium)

| Urge | Why it’s theater this window |
| --- | --- |
| **GPU / CUDA / AMP / `torch.compile`** | Log already says CPU-bound VecEnv (~252 steps/s, GPU half-idle). More VRAM % ≠ better adjusted_time. Tune `n_envs` / Dummy–Subproc knee only if measured. |
| **Watch coach polish / banners / ghost carnival** | Watch is unofficial. KPIs + mismatch banners shipped. Extra pixels don’t change which zip wins. |
| **Map gen UI / model timeline chrome / second UI** | Done or low-ROI. Don’t re-skin the pit wall while the car still DNFs. |
| **Warm-start FTG/BC, residual RL, SAC/Dreamer/PBT zoo** | Serialization: metric + overnight + collision first. Residual before clean-lap is cosplay racing. |
| **Bridge / `:4567` / Jetson / camera / TensorRT** | Phase 3 gated on holdout beat-FTG. Hardware now is ego. |
| **500k–50M “serious” defaults / unlimited without crash discipline** | Longer death farming. Short collision-first smokes until crash rate moves; then raise budget. |
| **Faster validating that quietly becomes race score** | Mid-train progress probes are fine if labeled. Official timeout stays honest (400). Don’t cheat the scoreboard to feel fast. |
| **Seasons / XP / Discord reward dials / algo fashion** | Gamification and paper chic. Cut table in `IDEAS_TRIMMED` exists for a reason. |

**User-request pattern:** Anything that *looks* productive (GPU util, prettier panel, new algo) but doesn’t move **official adjusted_time vs FTG on sealed maps** or **overnight continue integrity** is a trap for this window.

---

## Top 5 MUST-BUILD (this 9h)

Ordered. Ship in this order; skip glam until these pass acceptance in the critique.

1. **Regression-lock overnight early-stop** — Preserve select_timeout≥220, eval floor≥50k, warmup≥2, min_ts≥100k, Overnight preset, `phase=early_stopped` integrity. Re-smoke if touching RaceBest / ui_ops constants. *Dead nights don’t beat FTG.*
2. **Continue-train soak** — UI Stop → Continue same `run_id` from last **complete** ckpt; lock cleared; timesteps rise; refuse while live. *The overnight is the experiment.*
3. **Collision-first path that moves crash_rate** — Short ≤100k with flags on; crash_rate in `live_status` down vs baseline; no mid-run reward Discord. *Hits cost ten seconds — fix the DQ economics first.*
4. **Official FTG vs current PPO row** — Same protocol, seals intact, non-null adjusted_time; publish Δt even if you lose. *North star without a number is marketing.*
5. **Holdout / validation refuse + pack verify** — Start without allow-holdout dies on sealed maps; official append blocked on tamper. *Holdout beat is the gate; map0 cosplay is not.*

---

## Top 5 KILL (this 9h)

Do not spend hours here. Cut the feature, not the guard.

1. **GPU optimization theater** — Before measured steps/sec knee and after crash curriculum. Expect idle GPU; don’t buy CUDA to feel fast.
2. **Watch / coach aesthetic polish** — Unofficial surface; already has race KPIs + honesty labels. More chrome ≠ podium.
3. **Bridge / Jetson / camera / ESTOP product** — Hard skip until holdout beat-FTG. Phase 3 gate is not optional.
4. **Residual RL / FTG-BC warm-start / algo zoo** — Before clean-lap + one surviving overnight + honest FTG row. Pure PPO collision-first first.
5. **Map-regen / timeline / continuous-until-plateau outer product** — Autogen+autotrain outer loop is fine as a *later* scaffold; inventing it mid-soak while dual-Start still matters more is distraction. Don’t touch overnight early-stop math to “feel continuous.”

---

## Protected run note

`overnight_soak_20260917_082739` is ACTIVE — do not kill casually. Improve code around it; if a fix needs stop, **Continue** same `run_id`. GPU at ~20–40% is evidence for the kill list, not a mandate to “use the GPU.”

---

## One-liner

**Protect overnight + continue → crash rate down → honest FTG Δ on sealed maps. Kill GPU theater, Watch chrome, and bridge cosplay until the clock says you won.**
