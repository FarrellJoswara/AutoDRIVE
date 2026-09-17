# 09 — RoboRacer competition / safety compliance

**Lens:** Legal sensors, no-GT at race time, `adjusted_time` scoring, crash / DQ limits. Paperwork that doesn’t change a DQ or a lap clock is theater.

**House score (already):** `adjusted_time = lap_time + 10 * collisions`. Align eval/promotion to that (and to warm-up ignored → **10 scored laps**, **DQ if collisions > 10**), not `ep_rew_mean`.

---

## KEEP (must ship for legality / scoring honesty)

### Legal sensors & no-GT
- **No GT pose / velocity / lap progress / mocap / timing-board OCR as sensing** — race binary must not subscribe to privileged topics; shaping GT in gym is fine, obs GT is a DQ.
- **GT-ablation gate: same binary without privileged topics must still behave** — one CI/eval flip that strips IPS/map pose; if score collapses, you were cheating yourself.
- **Freeze obs contract early (beams, units, FOV, frame, version hash)** — bridge remap without a contract bump is how “legal LiDAR” becomes a silent ABI cheat.
- **Match gym beam count/order/FOV to AutoDRIVE/real sensor; no deploy-time interpolation** — inventing rays at the gate is not a legal sensor.
- **LiDAR stack / engineered TTC–clearance features only from rays + legal proprio** — closing-rate without GT pose is the compliant substitute for privileged velocity.
- **Ban CAD-width wall-following / memorized lines after track reconfiguration** — holdout maps + start jitter; map0 barnacles are not racecraft.
- **Wheel odom / slip carefully—perfect odom is a free oracle you’ll lose outdoors** — treat ego-speed as noisy proprio, not IPS cosplay.
- **Calibration gate: bridge LiDAR stats inside gym DR envelope or refuse deploy** — out-of-envelope scans are a silent rules violation waiting to happen.

### Scoring = competition primary metric
- **Rank / promote by adjusted race score (`time + λ·collisions`), not `ep_rew_mean`** — λ already = 10 in-house; train reward is not the sport.
- **Separate eval metric from train reward** — shaping can use GT `s` / contact; leaderboard must report lap time + collisions (+ DQ flag).
- **Match RoboRacer rules in eval protocol: warm-up ignored, 10 scored laps, DQ at >10 collisions** — single-lap “looks good in Watch” is not a race result.
- **Hard-gate speed reward until rolling collision rate ≈ 0; unlock aggression under a frozen collision budget** — mirrors DQ economics: crashes first, then time.
- **Collision-first: contact = terminal −R_big early; later recovery curriculum** — teaches the DQ cliff before time-attack greed.
- **Near-miss shaping from min LiDAR clearance; progressive λ in `time + λ·collisions`** — soft margin so you don’t live at exactly 10 hits.
- **Ensemble / promote lowest-collision member that still finishes, not highest train reward** — podium is adjusted_time and stay under DQ, not Monitor return.
- **Frozen held-out protocol + fixed checkpoint selection (no test fishing)** — fishing for a pretty seed is the academic form of illegal timing.

### Crash limits & safety that actually stops the car
- **ESTOP hardware + software watchdog (heartbeat loss, LiDAR dropout → Safe Halt)** — legal race entry and chassis survival; not optional UX.
- **Collision-avoidance priority over lap time (ops intent, not a binder)** — when in doubt, brake/veto; DQ > heroics.
- **FTG (or classical) hard veto envelope while PPO is still a menace** — soft “intent” without an action clip does not respect a collision budget.
- **Safety filter / short-horizon TTC brake when frontal range collapses** — converts “>10 collisions = DQ” into a runtime governor.
- **Progressive cone/boundary sanctions + minimum clearance soft-stop** — train the margin that keeps you under the DQ line.
- **Deploy-readiness one-pager: sensor model, latency, net size, veto, ABI, thermal—or no track** — checklist that gates rolling out, not a compliance binder.
- **Jetson shadow mode: policy infers, FTG/human drives; log divergence** — proves no-GT policy before it owns the ESC.
- **Latency budget (scan→act); fail if p99 exceeds sim step** — late acts become illegal “ghost reactions” and crash magnets.

---

## DEFER (real later; not today’s DQ risk)

- **Approved sensor manifest / sealed compute / air-gap heats / no pit coaching** — event-day logistics; until you enter a sealed heat, don’t build tooling for it.
- **Version-pinned competition image + Change Control for mid-event patches** — useful at a real event; premature for gym/bridge iteration.
- **Shadow steward: inject synthetic LiDAR dropouts; failure to halt = DQ** — good red-team once watchdog exists; not a training KPI.
- **Camera later: hybrid LiDAR safety + cam apex** — legal only if rules allow; keep LiDAR as the safety authority first.
- **Constrained RL (PPO-Lag / CPO / CVaR) with hard crash-rate constraint** — principled DQ mirror; ship λ-scoring + veto first.
- **Cowardly qualifier (zero-collision) then time-attack fine-tune** — good staging once eval already enforces 10-lap / DQ.
- **Domain-randomize LiDAR noise/dropout/latency as first-class enemies** — sim2real insurance; secondary to GT-ablation + contract freeze.
- **Offline bridge replay / intervention-budget as failure metric** — bridge maturity metric, not gym adjusted_time.
- **Opponent-aware / traffic curriculum** — solo time-attack first; multi-agent is a different rulebook.
- **TensorRT / quantized actor for deploy** — after a legal, vetoed policy exists at target Hz.

---

## CUT (bureaucratic theater / scoring lies / illegal cosplay)

- **Compliance binder + officer of record** — theater; a GT-ablation test and ESTOP beat stationery.
- **Control Intent Statement as paperwork product** — keep the priority in code (veto/halt); don’t invent a forms workflow.
- **Cyber-Physical Integrity Quiz before operator credentialing** — red-team joke; not a race gate.
- **Privileged IPS/speed in train then strip at race time** — classic leak; ablation will expose it too late.
- **LiDAR→camera distillation then throw the laser away** — illegal sensor swap fantasy.
- **Event cam / thermal / NeRF / 8×4K GoPros** — not on the legal sensor path; dilutes LiDAR contract.
- **Timing-board OCR / mocap / map GT in the observation** — explicit DQ class; do not “just for debug” leave wired in.
- **Promote on `ep_rew_mean` / Watch vibes / photo-finish collage** — rewards lie; collisions don’t care.
- **Overfit map0 until Watch looks sexy** — memorized line fails reconfiguration ban and holdouts.
- **Online overnight PPO on physical car with no rate limits** — safety anti-pattern; burns the collision budget and the chassis.
- **Suicide-crash-to-reset as a race strategy** — in competition, contact costs λ and can DQ; don’t train “crash is cheap.”
- **Cheerful lying UI / vibes dial / emoji status as “safety culture”** — obscures crash rate and DQ proximity.
- **Season ladder / Elo / crash bingo / NFT of best_model** — gamification ≠ compliance.
- **Hive-mind / LLM planner / Atari Rainbow on the race car** — novelty that doesn’t touch legal obs or adjusted_time.
- **Kill bridge/FTG until a real car exists** — opposite of compliance; classical veto is the on-car safety layer.

---

## TOP 10 (competition / safety)

1. **No-GT race sensing + GT-ablation gate** — only way to know the binary is legal.
2. **Eval = warm-up ignored, 10 scored laps, DQ if collisions > 10** — match the actual rulebook.
3. **Promote on `adjusted_time = time + 10·collisions`**, never `ep_rew_mean`.
4. **Frozen obs/sensor contract matched to real LiDAR** — no deploy-time beam invention.
5. **ESTOP + software watchdog (heartbeat / LiDAR dropout → Safe Halt)**.
6. **Classical hard veto / TTC brake over the policy** — enforce the collision budget at runtime.
7. **Collision-first train → speed unlock under a frozen crash budget** — learn DQ economics before time-attack.
8. **Ban map memorization: holdouts + reconfiguration / start jitter**.
9. **Shadow mode on Jetson before policy owns actuators**.
10. **Deploy-readiness gate (sensors, latency, veto, ABI)—or stay off the track**.

**Compliance one-liner:** Legal rays in, no privileged pose, score the 10-lap adjusted clock, stay under 10 hits—or halt. Everything else is theater.
