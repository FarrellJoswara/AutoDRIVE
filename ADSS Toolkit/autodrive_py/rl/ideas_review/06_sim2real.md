# 06 — Sim2real / bridge / Jetson (skeptic review)

**Lens:** Gym-first. Hardware theater must not block a clean lap on map0. Bridge and Jetson are *gated* workstreams: earn them with contracts, latency, and held-out gym skill—not by shipping UDP glue early.

**Verdict in one line:** Train and evaluate like the car will eventually exist; do not act like it already does.

---

## KEEP now (gym)

These improve gym racing *and* quietly raise the floor for a later bridge. Ship them without AutoDRIVE, Jetson, or chassis.

| Idea | Why keep now |
|------|----------------|
| Freeze obs contract early (beams, units, FOV, frame, version hash) | Bridge remap = ABI break. Gym wins on a frozen contract transfer; gym wins on a mushy one don’t. |
| Match gym beam count/order/FOV to the *intended* real/AutoDRIVE sensor | No deploy-time interpolation. Decide once; train forever on that. |
| Domain-randomize LiDAR dropout, max-range clip, angular bias, FOV crop | Cheap robustness; not “sim2real complete,” but kills brittle ray-readers. |
| Randomize obs/action delay + jitter + packet loss in gym | First-class enemy of racing policies. Practice here so p99 latency isn’t a surprise. |
| Actuator dynamics in train: steer rate limit, ESC lag, friction patches | Gym-only physics honesty; still no chassis required. |
| Engineered features (L/R/F clearance, TTC, wall rates) + short LiDAR stack | Deploy-friendly features; ablate under matched seeds. |
| Cap policy FLOPs for a 50–100 Hz target with thermal headroom | Gym architecture discipline. Tiny actor now ≫ TensorRT cosplay later. |
| Match train control Hz to a plausible servo/ESC rate | Don’t let PPO batch size invent a control frequency you’ll never run. |
| Residual RL / FTG safety filter / classical veto envelope (in gym) | Teach bounded Δu and “classical always can kill the action.” Same story on car later. |
| Collision-first curriculum, TTC terminate, adjusted race score | Policies that don’t smash walls are the only ones worth bridging. |
| Procedural maps + sealed hashed holdouts; pin a fixed “validation track” | Week-to-week compares need one immutable track *even before* Jetson. |
| Parallel envs diversify maps/starts (not 32× map0) | Fake generalization will fail sim-to-sim first; catch it in gym. |
| Overfit detector: scramble LiDAR bins at eval | Proves the policy reads rays, not pose leaks / reward hacks. |
| GT-ablation gate: same binary without privileged topics | No mocap/IPS/progress oracle in the race path—ever. |
| Contracts/obs-dim mismatch → hard refuse load | Gym hygiene that becomes bridge survival. |
| Immutable config fingerprint (maps, contracts version, git SHA) | Lineage for “what almost transferred.” |
| Stress eval: start jitter, occlusion, latency profiles; report tails | Bridge fails in the tails, not the median. |
| Truncate doomed episodes; vectorize hard; freeze display during train | Throughput and sample quality—unrelated to hardware theater. |
| Warm-start / residual from FTG; Bayesian opt classical gains first | If classical frontier isn’t flat, RL-on-Jetson is vanity. |
| Kill dual best/latest/checkpoints theater until multi-day runs are real | Same for “deploy readiness dashboard” until there is something to deploy. |

**Explicit non-goals for now:** AutoDRIVE `:4567` product surface, Jetson TensorRT pipeline, OpenCV viz on the control box, overnight on-car PPO.

---

## KEEP before bridge

Do **not** open a real (or AutoDRIVE closed-loop) control path until these are true. Fail loud; refuse deploy.

### Must be true (gates)

1. **Gym skill gate** — Beat fixed FTG (or classical baseline) on sealed holdouts under pre-registered metrics (`adjusted_time` / clean laps / collision budget). Median is not enough; survival curves and worst-decile matter.
2. **Obs ABI frozen + hashed** — Gym beam count/order/FOV/units/frame match the bridge sensor model *exactly*. No interpolation at the fence.
3. **Sim-to-sim before chassis** — Gym → AutoDRIVE `:4567` on *identical* maps with the same contract. If this fails, physical car will fail louder.
4. **Calibration gate** — Bridge LiDAR stats (range histogram, dropout rate, angular bias) sit inside the gym domain-randomization envelope, or refuse deploy.
5. **End-to-end latency budget** — Measured scan→act p50/p99; fail if p99 exceeds the sim step / control Hz you trained for. Latency soak under load (CPU thermal, logging on).
6. **Classical hard veto** — FTG / tube / Safe Halt can always override PPO. Shadow mode first: policy infers, classical/human drives; log action divergence and intervention budget.
7. **ESTOP + software watchdog** — Heartbeat loss, LiDAR dropout, NaN obs → Safe Halt. No “hope the policy recovers.”
8. **Deploy-readiness one-pager** — Sensor model, latency numbers, net size/FLOPs, veto path, ABI hash, thermal headroom—or no track time.
9. **Single-threaded control core** — Prefer UDP/binary; viz and UI off the control path. Bridge-down ≠ gym-train-down (status must say so).
10. **No privileged sensing in the race binary** — GT pose/velocity/lap progress banned; GT-ablation already passed in gym.
11. **Pin hardware validation track** — One never-changing map for week-to-week Jetson/AutoDRIVE compares (hashes sealed).
12. **Offline bridge replay scoring** — Closed-loop intervention-budget as primary failure metric before trusting live PPO.

### Worth doing *then* (not now, but not cut)

| Idea | When |
|------|------|
| Jetson shadow mode + divergence logs | After sim-to-sim pass |
| TensorRT / quantized actor | After fp32 actor meets Hz + thermal on device |
| Distill best PPO into tiny student | Only if FLOPs gate fails at target Hz |
| System-ID bicycle/Pacejka from logs → mix into train | After enough clean bridge logs exist |
| Chronoskepticism (warm/cold start, clock skew as DR) | After first honest latency soak |
| Camera later (hybrid LiDAR safety + cam) | After LiDAR-only bridge is boringly safe |
| Match RoboRacer scoring rules in eval | When competition is a real deadline |

---

## DEFER

Useful someday; not on the critical path to gym wins or a minimal safe bridge.

- Dual-rate classical inner loop + low-rate RL setpoints
- Hierarchical mode machines (entry/mid/exit/recovery) with RL as gate
- Short-horizon emergency MPC / tube-MPC as production veto (classical FTG veto first)
- Offline / batch RL (IQL/CQL) from FTG + failed bags before on-car on-policy
- World-model / Dreamer / TD-MPC “imagine crashes”
- Meta-RL across trackgen; PBT overnight; self-play ghosts
- Partner fantasy: upload map PNG → ranked model overnight
- Artifact lineage DAG / CO₂ / RLAHF thumbs from Watch
- Approved sensor manifest, sealed compute, air-gap heats, Change Control binder
- Shadow steward injecting synthetic dropouts as DQ theater (keep the *mechanism*; defer the bureaucracy)
- Event cam / thermal / NeRF / multi-GoPro stacks
- “Fine-tune 5 minutes on the real car” as a *plan* (see CUT for doing it naively)

---

## CUT

Hardware theater, premature bridge product, or ideas that actively sabotage transfer.

| Idea | Why cut |
|------|---------|
| Kill bridge/FTG product surface until a real car exists *(as absolute gym purity)* | Too absolute—**keep classical FTG in gym** as baseline/veto teacher. Cut only the *AutoDRIVE/Jetson product chrome* until gates pass. |
| Online overnight PPO on the physical car with no rate limits | Destroys cars and learns noise. |
| “Fine-tune 5 minutes on the real car” after ideal gym only | Ideal gym ≠ calibrated bridge. Fine-tune is not a substitute for gates 1–8. |
| OpenCV multi-twin viz on the same Jetson as the controller | Steals cycles from the control core; fails thermal/Hz. |
| Camera + IMU gadgets + UI polish before one clean 10-lap held-out eval | Classic distraction ordering. |
| LiDAR→camera distillation then throw the laser away | Competition/safety fantasy; wrong stack. |
| Privileged IPS/speed in train then strip at race time | Leakage cosplay; GT-ablation exists so you don’t need this trick. |
| Diffusion / ACT / flow-matching / LLM planner for 2-DoF car | Paper chic; zero bridge value. |
| Rainbow DQN at 5 Hz “like Atari” | Wrong timescale for a continuous car. |
| Hive-mind n_envs vote / gravity flip / wet-cheese / lava maps mid-episode | Entertainment ≠ podium. |
| Undervolt/OC / thermal-paste theater for +3% | Not a policy problem. |
| Lower physics fidelity in train + unlock timestep + int8 obs rings | Anti-sim2real. |
| Softmax discrete steer + RGB on a LiDAR-only contract | Contract suicide. |
| Maximize Watch chaos / n_envs bragging while eval still ranks reward | Optimizes demos, not transfer. |
| NFT / TikTok crash farm / pay-to-name checkpoint | Not racing. |

---

## Skeptic checklist (print before any `:4567` or Jetson session)

```
[ ] Held-out gym: beat FTG on sealed maps (tails OK, not just mean)
[ ] Obs contract hash identical gym ↔ bridge
[ ] Beam count/order/FOV: no interpolation
[ ] Latency p99 ≤ trained control step (soak under load)
[ ] LiDAR stats inside gym DR envelope
[ ] Classical veto + ESTOP + dropout→Safe Halt proven
[ ] Shadow mode intervention budget logged
[ ] Control path single-threaded; viz elsewhere
[ ] No GT/privileged topics in race binary
[ ] Deploy one-pager filled (or stay in gym)
```

If any box is unchecked: **stay in gym.** That is not delay—that is the strategy.
