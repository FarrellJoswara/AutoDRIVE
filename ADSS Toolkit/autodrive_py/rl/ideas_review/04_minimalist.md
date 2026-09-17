# Minimalist monk review — `IDEAS_MEGA.md`

**Lens:** Default is CUT. KEEP only if it clearly moves **adjusted_time**, **clean laps**, or **beat-FTG**.  
**Rule:** If it improves training comfort, demos, docs, aesthetics, or research chic without a direct race-score path → cut.

---

## KEEP (25 max)

1. **Select / promote by adjusted race score** (`time + λ·collisions`), never `ep_rew_mean`.
2. **One success criterion:** beat a fixed FTG baseline on a sealed held-out eval (start on map0; stop feature addiction until that lands).
3. **Collision-free gate before speed/aggression reward** — clean laps unlock pace; parade-forever loses via pace floor.
4. **Frenet-s progress only** + stall timeout + reverse-travel abort (no Euclidean start farming).
5. **Anti-circling suite** (heading–tangent tax, yaw-while-s-flat, loop terminate) so reward cannot buy spin laps.
6. **Collision cost > timeout cost** early (terminal big −R); suicide-to-reset is cheaper than wall-barnacle farming.
7. **Near-miss / min-clearance shaping** that feeds the same λ used in adjusted score.
8. **Warm-start from FTG (or residual Δsteer/Δthrottle on FTG)** — shortest path to “beat FTG,” not random thrash.
9. **Hard safety envelope:** project unsafe actions into FTG gap / veto while PPO is still a menace.
10. **TTC / frontal min-range hard brake limiter** over the policy when range collapses.
11. **Curriculum:** crawl → clean laps → raise speed; ¼→½→full lap finish gates before full aggression.
12. **Truncate doomed episodes early**; diversify maps/starts across parallel envs (clones of map0 fake skill).
13. **Frozen held-out protocol:** fixed seeds, episode budgets, pre-registered primary metrics; fixed checkpoint pick rule (no test fishing).
14. **Eval reports mean ± CI (≥5 seeds)** plus worst-decile / survival — medians alone hide dirty tails.
15. **Match competition scoring in eval** (e.g. scored laps + collision DQ), so “win” means race rules, not shaped reward.
16. **Leaderboard artifact keyed to beat-FTG Δt** on sealed map hashes (with bands, not single lucky runs).
17. **Train reward ≠ eval metric** — shaping is a tool; adjusted_time/clean laps are the sport.
18. **Engineered LiDAR features:** L/R/F clearance, TTC, relative wall rates (only if ablations show they help the race metric).
19. **Freeze obs contract early** (beams, FOV, units, Hz, version hash); gym matches deploy — no silent remap.
20. **Light domain randomize** of LiDAR noise / dropout / latency / spawn — enough that walls stay walls, not gadget theater.
21. **Tune classical FTG/gains first** (Bayes/CMA); RL only when that frontier is flat on adjusted_time.
22. **Throughput that serves the metric:** vectorize hard, kill display consumers in train, match control Hz to real — more honest samples per hour.
23. **Sim-to-sim on identical maps** before chasing new algorithms or hardware polish.
24. **Crash taxonomy + time-to-crash curves** so clean-lap failures get fixed causes, not more UI.
25. **Atomic complete checkpoints only**; refuse obs-dim / contract mismatch loads — never promote a partial “best.”

---

## CUT — everything else (summary)

| Bucket | Verdict |
|--------|---------|
| **Most of Training & algorithms** | SAC/TD3/TQC/DroQ/REDQ bake-offs, LSTM stacks “just in case,” constrained RL (CPO/CVaR), offline IQL/CQL, Dreamer/TD-MPC world models, Go-Explore, PBT, meta-RL, self-play ghosts, distill-to-flowchart, diffusion/ACT/LLM planners, Discrete-vs-continuous as a project — all CUT until one FTG-beating protocol exists. Keep residual/FTG-warm and veto only (above). |
| **Reward extras** | Centerline attractor theater, opponent/overtake bonuses, hot-swap chaos dials, sector gamification beyond simple finish gates — CUT. |
| **Obs / sensors stretch** | Camera, optical flow, event/thermal/NeRF, LiDAR→cam distillation, privileged IPS then strip — CUT. Ablate proprio later; don’t add sensors to invent progress. |
| **Maps / trackgen product** | Thumbnail pickers, partner PNG upload, fantasy generators, loot-box maps — CUT. Keep sealed holdouts + hashed packs only as needed for honest beat-FTG. “Train map0 until boringly good” is the monk path — not a map-factory. |
| **Control UI / UX** | Presets, coach panels, glossary, garage aesthetics, ETA stories, health traffic lights — CUT for race score. Keep only whatever is required to start/stop a run and load a checkpoint without lying. |
| **Watch & visualization** | Rainbow twins art, reward weather, photo-finish collages, Grad-CAM, speech bubbles — CUT. Overlay adjusted_time/collisions only if it changes which checkpoint you promote. |
| **Metrics / W&B sprawl** | Lineage DAGs, CO₂, RLAHF thumbs, audio telemetry, plant dashboards — CUT. Fingerprint + race leaderboard + crash shards are enough. |
| **Eval bureaucracy** | TOST multiplicity theater, full “reproduce Table 1” CI science project — CUT until FTG is beaten once under a frozen protocol. |
| **Sim2real / hardware early** | System-ID Pacejka stacks, chronoskepticism axes, Jetson viz on control core, overnight on-car PPO — CUT until gym adjusted_time wins. Latency budget + shadow/veto stay only if you’re actually deploying. |
| **Safety / competition paper** | Compliance binders, sealed images, steward DQ drills — CUT for now; intent “collision priority” is already in the KEEP gates. |
| **Resilience / chaos suite** | Full torture matrix (sleep/wake, antivirus empty zips, conspiracy checklists) — CUT to “don’t corrupt best_model / detect dual-train collision.” |
| **Docs / teaching / demos / aesthetic / gamification** | Entire sections CUT for the race metric. One short “how to run eval vs FTG” beats a novel. |
| **Simplify-by-adding** | “Minimal mode” toggles, plugin buses, second advanced UI — CUT (they’re complexity cosplay). |
| **Deliberately bad / overkill list** | CUT as work items; keep only as red-team reminders of what not to do. |
| **Already-on-radar product surface** | Map generator UI, coach panel, Watch twins polish, compare-models chrome — CUT unless they enforce KEEP #1–2 (promote by adjusted_time / beat-FTG). |

**Net:** ~320 ideas → **25 KEEP**. Everything else is distraction, tooling comfort, or research inventory.

---

## DANGEROUS complexity traps

These look “serious” and will eat weeks without moving adjusted_time / clean laps / beat-FTG:

1. **Algorithm tourism** — swapping PPO for SAC/TQC/Dreamer/CQL before a frozen FTG-beating eval exists.
2. **Stacking hybrids** — residual + hierarchical modes + dual-rate + constrained RL + safety filter all at once; you can’t tell what worked.
3. **Sensor creep** — camera / IMU / odom “oracles” that boost train reward and vanish (or lie) at race time.
4. **World models & offline imagination** — crash dreaming as a substitute for honest on-policy clean-lap pressure.
5. **UI / Watch as progress** — prettier twins and coach copy while leaderboard still ranks `ep_rew`.
6. **Map factory before one map** — procedural packs, partner uploads, emotional generators while map0 isn’t cleanly faster than FTG.
7. **Eval fishing** — many seeds/maps/checkpoints, pick the pretty one; no pre-registered primary metric.
8. **Smoothness / centerline PhD** — tax jerk until the car is polite and slow; reward spin “stability” kills lap time.
9. **Throughput cosplay** — max `n_envs` / batch bragging while selection still isn’t adjusted_time.
10. **Simplify-by-adding** — plugin bus / minimal-mode toggle / second UI that doubles surface area.
11. **Deploy theater early** — TensorRT, thermal paste, Jetson viz on the control path before sim-to-sim beat-FTG.
12. **Privileged leak** — GT pose/speed/lap in train then “strip at race” (you trained a cheater).

**Monk close:** Beat fixed FTG once on sealed eval under adjusted_time. Until then, every new subsystem is a cut.
