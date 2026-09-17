# Reward / Training / Obs — Scientist Review

**Scope:** `IDEAS_MEGA.md` sections **Training & algorithms**, **Reward & curriculum**, **Observation & sensors**.  
**Lens:** F1TENTH time-attack. Forward Frenet-`s` progress + stall timeout already exist.  
**Priorities:** (1) anti-hacking, (2) eval ≠ train reward, (3) curriculum that doesn’t fake skill.  
**No code.**

---

## Verdict in one paragraph

Keep a thin, hard-to-hack train reward (signed Δ`s` + stall + anti-circling + collision severity), promote only on a frozen **adjusted race score**, and use curriculum that unlocks aggression only after clean progress—not after dense shaping peaks. Most algorithm chic (Dreamer, PBT, self-play, diffusion) and most sensor/UI theater should wait until held-out clean laps beat FTG under a pre-registered eval protocol.

---

## KEEP

Ideas that directly reduce hacking, separate sport from shaping, or teach real racecraft.

### Reward & anti-hack

| Idea | Why keep |
|------|----------|
| Progress only along Frenet **s** (signed); never Euclidean start-distance | Already on radar; remains the anti-farming spine |
| Stall timeout if Δ`s` < ε; reverse-travel abort past −Δ`s` budget | Already on radar; blocks parade / reverse farming |
| Anti-circling suite (heading–tangent mismatch, yaw tax while `s` flat, loop terminate, angular/path ceiling) | Circling is the classic dense-progress hack; suite > single tax |
| Dense Δ`s` with deadband/clamp | Wall-bounce / spin near gates must not mint progress |
| Separate **eval metric** from train reward | Non-negotiable: shaping must not become the leaderboard sport |
| Collision-first early (contact = terminal −R_big); later recovery curriculum | Early terminal contact kills contact-as-reset and wall-scraping scrap farming |
| Near-miss shaping from min clearance; progressive λ in `time + λ·collisions` | Aligns train pressure with how you should **rank** policies |
| Collision-free binary gate before speed reward; pace floor vs parade laps | Aggression unlock only under frozen collision budget |
| Smoothness tax (Δsteer² / Δthrottle² / jerk); throttle pays only when aimed down-track | Cuts thrash-for-Δ`s` and reverse-throttle jitter hacks |
| Lap / sector bonuses once per lap with heading gate; incomplete-lap ∝ remaining `s` | Sparse finish signal; sector tags expose “fake skill” on easy straights |
| Suicide-crash cost > timeout cost | Stops collision-as-episode-reset optimization |
| Centerline attractor **early only**, then anneal | OK if annealed; permanent centerline = barnacle, not racing line |
| Terminate on imminent collision (TTC), not only contact | Shrinks contact-farming and late-crash “almost finished” loopholes |

### Training / selection (not algorithm cosplay)

| Idea | Why keep |
|------|----------|
| Rank / promote by **adjusted race score** (`time + λ·collisions`), not `ep_rew_mean` | Primary anti-self-delusion rule |
| Hard-gate speed reward until rolling collision ≈ 0; unlock aggression under frozen collision budget | Curriculum that doesn’t fake skill |
| Separate “cowardly qualifier” then time-attack fine-tune | Two sports; don’t mix into one soggy reward |
| Ensemble / checkpoint race: lowest-collision (or best adjusted) member, not highest train reward | Stops “lucky shaping peak” promotion |
| Truncate doomed episodes early (crash/stall); lengthen horizon only when success rate justifies | Sample efficiency without participation trophies |
| Parallel envs must diversify maps/starts—32× map0 fakes generalization | Curriculum integrity |
| Warm-start PPO from FTG / BC | Burns less wall-clock on random thrash; not a substitute for eval |
| Residual RL or classical safety filter / FTG corridor project | Keeps early PPO from inventing wall-hug progress hacks |
| Match train control Hz to real servo/ESC; vectorize hard; kill display during train | Honest steps/sec; fewer sim cheats via wrong rates |
| Domain-randomize track width, µ, LiDAR noise/dropout, latency, spawn | Skill that survives reconfiguration, not memorized map0 |

### Observation / contracts

| Idea | Why keep |
|------|----------|
| Freeze obs contract early (beams, units, FOV, frame, version hash) | ABI drift masquerades as skill regressions |
| Engineered features: L/R/F clearance, TTC, relative wall rates | Closing-rate without privileged GT |
| Stack last k LiDAR frames | Closing rate without pose oracle |
| Domain-randomize beam dropout, max-range clip, angular bias, FOV crop | Anti-sim-cheat on sensor quirks |
| Randomize obs/action delay + jitter + packet loss | Latency is a first-class enemy; ignoring it fakes skill |
| Match gym beam count/order/FOV to AutoDRIVE/real; no deploy-time interpolation | Interpolation = silent distribution shift |
| Overfit detector: scramble LiDAR at eval; if return stays high → not reading rays | Direct anti-hacking / anti-oracle check |
| Calibration gate: bridge LiDAR stats inside gym DR envelope or refuse deploy | Deploy honesty |
| Ablate channels one-at-a-time under matched seeds | Know what the policy actually uses |
| Actuator dynamics in train (steer rate, ESC lag, voltage sag) | Reduces open-loop thrash policies that only work in ideal gym |
| Wheel odom / slip carefully—perfect odom is a free oracle you’ll lose outdoors | Explicit anti-privilege |

---

## DEFER

Useful later; premature now, or needs a clean baseline first.

### Algorithms / systems (after adjusted_time beats FTG)

- Discrete MultiDiscrete vs continuous action ablations (keep whichever wins **adjusted_time**)
- Explicit “brake hard” / short-horizon TTC brake limiter over policy
- Recurrent / LSTM-PPO or deeper history once stack-of-k is proven insufficient
- Constrained RL (PPO-Lag / CPO / CVaR) once collision shaping plateaus
- Hybrid classical raceline + RL speed/aggression scalar
- Dual-rate classical inner / RL outer; hierarchical entry/mid/exit/recovery modes
- `torch.compile` / AMP / bigger batches; Jetson FLOP caps; TensorRT / quantized actor
- Bayesian opt / CMA-ES over classical gains first (do this *before* more RL if FTG frontier is soft)
- Offline / batch RL (IQL/CQL) from FTG + failed bags—only if on-policy sample is the bottleneck
- Go-Explore / failure-state archive—after you have a crash taxonomy and sector tags
- Frame-skip / action-repeat ablations (measure once; don’t default)

### Reward / curriculum stretch

- Empty → static obstacles → traffic (solo time-attack can wait)
- Opponent-aware overtake / aggressor blame
- Full-circuit finish gates already implied by crawl→clean→speed; don’t over-stage into micro-missions that inflate “success rate”
- Camera / optical flow / segmentation free-space (after LiDAR-only clean held-out)

### Obs stretch

- Partial-obs drills (freeze scan, NaNs) and bad-env isolation—after basic DR works
- Shrink to LiDAR-only if proprio ablations say IMU/speed don’t help
- Camera later hybrids

### Heavy research (park until product metric is boringly good)

- SAC / TD3 / TQC / DroQ / REDQ vs PPO under **matched** steps + eval protocol
- World-model / Dreamer / TD-MPC
- PBT overnight; Meta-RL across trackgen; self-play / ghost opponents
- Model-based residual Pacejka dynamics
- Distill to tiny student for Jetson (after a real race candidate exists)

---

## CUT

Hacks, fake skill, or cost that doesn’t move adjusted race score.

### Deliberately bad / high-risk (do not implement as features)

- Hot-swap reward mid-run via UI “chaos dial” / Discord emoji — destroys eval≠train and reproducibility
- Battery-as-budget / LiDAR-entropy vibes / Mercury-in-retrograde — noise as sport
- Privileged IPS/speed in train then strip at race — **leak**; will invent untransferable policies
- Gravity flip / wet-cheese / lava-lamp morphing mid-episode as “curriculum” — chaos ≠ skill
- Hive-mind n_envs vote; Rainbow DQN @ 5 Hz “like Atari”; Diffusion / ACT / flow-matching for 2-DoF; LLM planner + RL tracker — paper chic, wrong problem
- Event cam / thermal / NeRF / 8×4K GoPros; LiDAR→camera then throw laser away — sensor theater
- Permanent centerline attractor (without anneal) — wall-barnacle / parade skill
- Ranking or early-stop on `ep_rew_mean` — trains you to believe shaping
- Softmax discrete steer + camera RGB on a LiDAR-only contract — contract suicide
- Lower physics fidelity / unlock timestep / int8 obs rings to “go faster” — sim cheat
- Suicide-crash *rewarded* or collision cheaper than timeout — invert the anti-hack
- Maximize Watch chaos / n_envs bragging while eval still ranks reward — entertainment ≠ podium

### Soft cuts (kill unless a measured need appears)

- Self-play for solo time-attack before single-car clean laps exist
- Opponent traffic curriculum for time-attack primary metric
- Mid-run reward surgery without fingerprint + held-out re-eval
- Any “participation trophy” curriculum that counts truncated near-misses as success

---

## TOP 15 (do these)

Ordered for anti-hack → honest ranking → real skill. Assume Frenet-`s` + stall already ship.

1. **Freeze a primary eval metric ≠ train reward**  
   Pre-register: adjusted race score (`time + λ·collisions`), clean-lap rate, worst-decile time. Never promote on `ep_rew_mean`.

2. **Anti-circling suite on train reward**  
   Heading–tangent mismatch + yaw tax while `s` flat + loop terminate + angular/path progress ceiling. One tax alone is farmable.

3. **Clamp / deadband dense Δ`s`**  
   Kill wall-bounce and gate-spin progress scraps.

4. **Collision economics that can’t be gamed**  
   Early: contact terminal + large −R; collision cost always > timeout; later allow recovery only under a frozen collision budget.

5. **Speed / aggression unlock gate**  
   No speed reward (or throttle payoff) until rolling collision ≈ 0; pace floor so parade laps lose.

6. **Promote lowest-collision / best-adjusted checkpoint, not best train reward**  
   Ensemble or timeline pick under held-out seeds.

7. **Sector / lap bonuses once per lap with heading gate**  
   Incomplete lap ∝ remaining `s`; tag hard corners so “skill” isn’t straight-line Δ`s`.

8. **Near-miss clearance shaping with progressive λ matching eval**  
   Train pressure should rhyme with `time + λ·collisions`, not invent a third sport.

9. **Diversified parallel starts/maps (never 32× map0)**  
   Plus domain-randomize width / µ / spawn; else curriculum fakes generalization.

10. **Obs contract freeze + LiDAR DR + latency/jitter DR**  
    Match beam ABI to AutoDRIVE; no deploy interpolation; delay as first-class enemy.

11. **Engineered clearance / TTC / wall rates + optional k-frame stack**  
    Closing rate without GT pose; ablate channels under matched seeds.

12. **Overfit / oracle detectors**  
    Scramble LiDAR bins at eval; GT-ablation / refuse privileged topics; refuse deploy if bridge stats leave gym DR envelope.

13. **Two-stage training: cowardly qualifier → time-attack fine-tune**  
    Separate collision-zero skill from aggression; don’t mush into one reward.

14. **Truncate doomed episodes; warm-start from FTG/BC; classical residual or FTG safety filter early**  
    Burn less compute inventing thrash hacks; keep early policy inside a corridor.

15. **Smoothness + “throttle only when aimed down-track”**  
    After anti-circling; cuts thrash-as-progress without becoming the primary metric.

---

## Curriculum that doesn’t fake skill (rules of thumb)

- **Unlock by eval gates, not train-reward thresholds.** Example: unlock Map1 / higher speed only after held-out adjusted_score and collision rate beat a frozen FTG bar—not after `ep_rew` climbs.
- **Widen then shrink corridors; anneal centerline.** Permanent easy geometry produces barnacles.
- **Lengthen horizon only when finish rate rises.** Short “clear one corner” micro-wins are fine as drills, not as the published success criterion.
- **Never let shaping terms enter the promotion leaderboard.** Log them; don’t rank them.
- **Diversify what “success” is measured on** (maps, seeds, start poses, latency profiles). Homogeneous success is memorization.

---

## Explicit “already on radar” (don’t re-invent)

- Forward Frenet-`s` progress + stall timeout  
- Anti-circling / signed progress (extend into the suite above)  
- IMU/encoders in obs (ablate; don’t privilege perfect odom)  
- Parallel `n_envs`, map packs / holdouts (diversify or you’re lying)

---

## What this review deliberately ignored

Maps, Watch, UI, W&B, sim2real hardware, safety bureaucracy, demos, aesthetics—except where they leak into reward hacking (e.g. chaos dial, privileged topics, ranking on train reward). Those belong in other review slices.
