# 11 — Devil’s advocate

Defend the unserious. Kill the respectable. Same backlog, opposite knives.

---

## RESCUE (bad idea → serious kernel)

Each item is tagged as it appears in `IDEAS_MEGA.md`. The surface joke is real; the kernel is what you’d ship if you stripped the costume.

### 1. Residual RL inside “hybrid” `[practical]` / neighbors `[stretch]`
**Surface:** Classical raceline + PP/Stanley forever; RL only “helps.” Sounds like you never trusted deep RL.  
**Kernel:** Bounded Δsteer / Δthrottle on top of a working FTG/PP baseline is the only architecture that keeps early PPO from burning the 3060 *and* gives you a deployable veto. Dual-rate (classical inner loop, RL setpoints every N) and hierarchical mode gates are the same idea with a clock and a state machine. Do residual first; rename it “hybrid” for the paper later.

### 2. Shadow mode inside “Jetson” `[practical]`
**Surface:** Policy on the car that doesn’t drive—booth theater.  
**Kernel:** Infer-only + log action divergence vs FTG/human is the cheapest sim2real truth serum you have. Intervention budget and offline bridge replay scoring ride on the same pipe. If you skip shadow, every “5-minute fine-tune” is a crash lottery with better branding.

### 3. Ghost race inside “demo” `[practical]` / `[stretch]`
**Surface:** Translucent twins, time-to-beat-FTG wallboard, conference eye candy.  
**Kernel:** Self-play against archived policies and “compare / ghost: FTG vs current brain” are *eval protocol*: promote-to-race only after held-out clean passes. Ghost rivals force adjusted_time + collisions onto the screen instead of ep_rew cosplay. Demo is the packaging; the kernel is non-cheating checkpoint selection.

### 4. Rainbow DQN @ 5 Hz “like Atari” `[bad/fun]`
**Surface:** Discrete steer on a continuous car—meme baseline.  
**Kernel:** MultiDiscrete throttle/steer bins vs continuous under matched steps is a real ablation. Coarse bins + explicit “brake hard” when TTC collapses often beat float32 thrash for early collision budgets. Keep the ablation; throw away the Atari cosplay and the 5 Hz joke once continuous wins (or doesn’t).

### 5. Hive-mind: n_envs cars vote `[bad/fun]`
**Surface:** Democracy for motors.  
**Kernel:** Ensemble of checkpoints racing the *lowest-collision* member, not the highest train reward. Vote/median at deploy is a poor man’s safety filter when one worker is reward-hacking wall scrapes. Parallel envs already are a committee—use disagreement as a health signal, not a meme.

### 6. Gravity flip / wet-cheese µ / lava-lamp morphing maps `[bad/fun]`
**Surface:** Chaos mid-episode for vibes.  
**Kernel:** Domain-randomize track width, µ, LiDAR dropout, latency, spawn pose—*without* regenerating mesh in the hot loop. Morphing mid-episode is just aggressive DR + curriculum unlock gates with worse marketing. Pin hashed holdouts so lava never becomes the eval set.

### 7. Hot-swap reward mid-run (“chaos dial”) `[bad/fun]`
**Surface:** Discord emoji trains your car.  
**Kernel:** Staged curricula (cowardly qualifier → time-attack; collision-first then speed unlock; anneal centerline attractor) need a *controlled* mid-run reward schedule, not a restart every λ tweak. UI dial = annealed `time + λ·collisions` with fingerprints logged. Emoji is optional; schedule versioning is not.

### 8. Battery-as-budget / vibes LiDAR-entropy `[bad/fun]`
**Surface:** Astrology for throttle.  
**Kernel:** Soft resource constraints—throttle only when aimed down-track, smoothness/jerk tax, thermal/FLOP caps for Jetson 50–100 Hz—are “budgets” with physics. Entropy of the scan as a *diagnostic* (frozen obs / open-field forever detectors) beats entropy-as-reward. Keep the budget metaphor; kill vibes-as-objective.

### 9. Privileged IPS/speed in train then strip `[bad/fun]` *(labeled leak—don’t)*
**Surface:** Cheat then amnesia.  
**Kernel:** Privileged learning / teacher features for shaping or a value bootstrap, then distill to the race ABI (LiDAR + proprio contract). Same family as warm-start from FTG/BC and “strip GT at race time” gates. The crime is *shipping* privileged topics; the method is train-with-oracle, deploy-without.

### 10. LiDAR→camera distillation, throw the laser away `[bad/fun]`
**Surface:** Delete your best sensor.  
**Kernel:** Cross-modal robustness and “camera later: hybrid LiDAR safety + cam apex” need a teacher–student story. Distill for *backup* free-space / optical-flow cues while LiDAR remains the veto. Throwing the laser away is the bad punchline; dual-stack with asymmetric trust is the kernel.

### 11. Wilting plant instead of TensorBoard `[bad/fun]`
**Surface:** Houseplant CI.  
**Kernel:** One metric that matters (adjusted_time or clean lap) plus a single health indicator people actually look at. Throughput, crash rate, and “gradients exist” heartbeats condensed to a traffic light beat a dashboard that nobody opens overnight. Plant = ruthless metric compression.

### 12. Crash bingo / mutation roulette `[bad/fun]`
**Surface:** Gamified failure.  
**Kernel:** Crash taxonomy (wall, spin, stall, NaN, contract mismatch) and failure-only shards are how you stop training the same hairpin forever. Bingo cards force *naming* failure modes; Go-Explore / archive hard corners is the adult version. Keep the taxonomy board; skip the champagne.

### 13. Compliance binder + officer of record `[bad/fun]`
**Surface:** Bureaucracy as feature.  
**Kernel:** Deploy-readiness one-pager (sensor model, latency, net size, veto, ABI, thermal), version-pinned image, Change Control, ESTOP + watchdog. Competition rules are a checklist whether you mock them or not. Binder = refuse-to-deploy gate with a signature.

### 14. “Fine-tune 5 minutes on the real car” `[bad/fun]`
**Surface:** Copium after ideal gym.  
**Kernel:** Short, rate-limited, shadow-preceded adaptation with classical hard veto and intervention budget as the *primary* failure metric—not unrestricted on-policy thrash. Pair with offline/batch RL from FTG + failed bags *before* any brave on-car PPO. The five minutes are a budget, not a dare.

### 15. Transfer from Mario Kart ghosts / rubber-duck throttle `[bad/fun]` *(deliberately bad section)*
**Surface:** IP theft and toys.  
**Kernel:** Any replayable action stream (seed + actions) is BC/offline data; ghost rivals and bit-replay in Watch are the same substrate. External ghosts are a joke; *your* past policies and FTG bags are not. Quack-volume throttle is a bad joke about mapping a scalar monitor to aggression—which is exactly what an RL speed/aggression residual should be.

---

## KILL (sounds good, wastes months)

Popular, citeable, and hungry. They eat the quarter you needed to beat FTG once on map0.

1. **World-model / Dreamer / TD-MPC “imagine crashes offline”** — Paper-chic sample efficiency before you have a frozen obs contract, held-out maps, and a residual baseline. Imagine crashes after real crash taxonomy exists.

2. **Meta-RL across trackgen for few-shot adaptation** — You don’t have a distribution of tasks yet; you have map0 and a dream. Multi-map curriculum + holdouts first; meta later if ever.

3. **Population-based training / PBT overnight** — Hyperparam theater while eval still ranks `ep_rew_mean`. Fix the race score, collision gate, and seed CIs before evolving a zoo.

4. **Constrained RL (PPO-Lag / CPO / CVaR) as the first “serious” upgrade** — Hard constraints on a broken reward and leaky obs ABI. Collision-first shaping + residual + safety filter get you 80% without the Lagrange soap opera.

5. **Camera / event-cam / NeRF / 8×4K GoPro transformers before LiDAR clean 10-laps** — Sensor maximalism. Freeze LiDAR ABI, match beam FOV to hardware, win adjusted_time; vision is a stretch *after* veto-grade ranging.

6. **Diffusion / ACT / flow-matching policies for 2-DoF** — Generative policy stack for steer and throttle. PPO/SAC under matched steps already answers “does fancy help?”; diffusion answers “can we cite ICRA?”

7. **LLM planner + RL tracker (“turn left at cone”)** — Language in the loop before cones are even a reliable feature. Hierarchical *classical* modes beat token latencies on a Jetson.

8. **Full RLAHF / Watch thumbs preference pipeline** — Human-in-the-loop dataset ops for a solo time-attack. Preference labels don’t fix Frenet-s farming; they add annotator weekends.

9. **Season ladder / Elo vs FTG / timestep XP / combo meter** — Gamification platform before one sealed leaderboard with CI bands and “beat FTG by Δt.” XP is a second reward you’ll overfit.

10. **Partner fantasy: upload map PNG → ranked model overnight + TensorRT-first deploy** — Product and compiler work while sim-to-sim (`gym → AutoDRIVE`) and shadow divergence are unproven. Rank overnight only after one-command eval doesn’t lie; quantize after fp32 shadow is boringly safe.

---

**Rule of thumb:** Rescue anything that secretly implements residual control, shadow eval, crash taxonomy, DR, or metric honesty. Kill anything that adds a second research stack before a single held-out clean race against FTG.
