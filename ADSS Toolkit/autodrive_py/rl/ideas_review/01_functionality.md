# Functionality review — race results only

Ruthless product pass on `IDEAS_MEGA.md`. Criterion: **finish clean laps** and **beat FTG on adjusted race score** (`time + λ·collisions`). UI polish, aesthetics, growth, and research chic that do not move that metric are CUT or DEFER.

---

## KEEP (must ship for race results)

- **Rank / promote by adjusted race score** — train reward lies; promotion must use time + collisions or you ship wall-bouncers.
- **Hard-gate speed until collision rate ≈ 0** — aggression before survival never finishes 10 scored laps.
- **Warm-start / residual RL from FTG** — classical baseline already races; learn Δ only, don’t relearn “don’t hit walls.”
- **Safety filter / FTG veto envelope** — project or veto RL into gap/corridor so early policies don’t DQ the run.
- **Progress along Frenet s + stall/reverse abort** — Euclidean farming and circling look like learning and aren’t.
- **Anti-circling + dense Δs deadband** — kills the main false-progress failure mode before speed work.
- **Collision-first terminal cost; near-miss / TTC terminate** — contact-only feedback is too late for clean laps.
- **Curriculum: crawl → clean lap → unlock speed** — and collision-free gate before any pace reward.
- **Freeze obs contract; match gym LiDAR to deploy** — ABI/FOV/beam mismatch is silent race death.
- **Engineered clearances / TTC / short LiDAR stack** — closing-rate without GT pose is what beats static FTG in corners.
- **Held-out eval protocol + beat-FTG leaderboard** — fixed seeds, CI, primary metric, no test-fishing promotion.
- **Match RoboRacer scoring** (warm-up ignored, N laps, collision DQ) — optimize the actual sport or “wins” don’t transfer.
- **Truncate doomed episodes; diversify parallel maps/starts** — sample efficiency and fake-generalization killers.
- **Match train control Hz + latency/dropout DR** — policies that assume perfect, free actuators lose on bridge.
- **Sim-to-sim gym→AutoDRIVE before chassis; deploy veto + latency budget** — hardware without matched contracts wastes car time.

---

## DEFER

- Separate cowardly-qualifier then time-attack stages — useful once one-stage clean-lap gate is proven.
- Dual-rate / hierarchical mode machines — complexity tax until residual + curriculum plateaus.
- Discrete MultiDiscrete action ablations — measure later; continuous + brake-hard is enough first.
- Explicit brake-hard / short-horizon TTC MPC limiter — keep as bolt-on after policy exists, not a research track.
- Ensemble race-lowest-collision checkpoint — after multi-checkpoint discipline exists.
- SAC/TD3/TQC/DroQ/REDQ vs PPO — algorithm tourism after reward/eval/FTG residual are correct.
- LSTM / history beyond short stack — only if stack ablations fail closing-rate.
- Constrained RL (PPO-Lag/CPO/CVaR) — try after shaping + gates fail to hold collision budget.
- Offline IQL/CQL, world models, Go-Explore, PBT, meta-RL, self-play — wall-clock sinks vs FTG residual.
- torch.compile / AMP / TensorRT / FLOP caps — after a policy that finishes; not before.
- Bayesian opt on classical gains first — fair, but residual RL already assumes FTG exists; do if residual stalls.
- Camera / vision hybrids — LiDAR-only must clean-lap first.
- Opponent-aware rewards / traffic curriculum — solo time-attack is the product now.
- Map picker UX / fantasy overnight upload — training needs hashed packs; fancy picker can wait.
- Most Control UI niceties (glossary, aesthetic Start, coach grades) — don’t finish laps.
- Watch heatmaps / ghost rival / FTG underdrawing — debug aids after metric overlays exist.
- Artifact DAG / model cards / W&B CO₂ / preference thumbs — process debt after Table-1 reproducibility.
- Pre-register TOST / one-command CI reproduce — stretch once protocol is manual and trusted.
- System-ID Pacejka / chronoskepticism DR — after sim-to-sim works.
- Competition bureaucracy (sealed compute, steward dropouts, change control) — when entering a real event.
- Chaos diagnostics beyond corrupt-zip / obs-mismatch / dual-train — ship the killers first; expand later.
- Docs teaching layers / progressive unlock copy — after first-run path actually trains a lap-capable zip.
- Demo wallboard / booth Start-Stop — after twins + beat-FTG metric are honest.
- Aesthetic emotion beats / reward weather — zero race function.
- Kill bridge surface until real car — defer the *product* kill; keep a thin FTG veto path for sim2sim.
- Distill tiny student / frame-skip ablations — deploy polish after win.

---

## CUT

- Diffusion / ACT / flow-matching / LLM planner / Rainbow-at-5Hz / hive-mind vote — 2-DoF car cosplay, not laps.
- Gravity flip / wet-cheese / lava morphing / vibes/Mercury rewards / Discord chaos dial — anti-curriculum.
- Privileged IPS then strip / LiDAR→cam throw-away laser / event-cam/NeRF/8×GoPro — contract lies or toys.
- Spotify/coffee-stain/emotional/loot-box maps — noise generators dressed as trackgen.
- Auto-start / haiku names / emoji status / voice every timestep / 16 Watch windows / lying “crushing it” UI — operator harm.
- Twins speech bubbles / karaoke / perfume / garage-floor projection / fake camera auto-open — entertainment ≠ podium.
- Audio telemetry / wilting plant TB / keep every crash video forever / per-ray saliency spam — storage and vanity.
- Overfit map0 until sexy Watch / fine-tune 5 min on car / overnight PPO on hardware unrestricted / OpenCV twins on Jetson controller — known ways to brick or fake progress.
- Compliance binder theater / NFT checkpoints / pay-to-name / TikTok growth / champagne bingo / season XP Elo — not racing.
- Soundtrack cello / risograph / clay sculpture / UI that gets uglier — art budget ≠ lap time.
- Minimal-mode plugin bus / second advanced UI — “simplify by adding.”
- Entire “deliberately bad / overkill” red-team list as product backlog — keep as jokes, never schedule.
- Softmax discrete steer + RGB on LiDAR contract, wall-barnacle smoothness cult, maximize Watch chaos as KPI — anti-goals.
- Kill Watch/live_status entirely *before* you have any other debug surface — cut the *feature addiction*, not the eyes you need to see circling.
- Centerline attractor forever (without anneal plan) — if kept poorly, blocks racing line; prefer anneal or skip.
- Battery-as-budget / suicide-crash-as-feature framing beyond “collision cost > timeout” — the cost inequality KEEP; the meme CUT.

---

## TOP 15 ranked for functionality

1. **Adjusted race score as the only promotion metric** — stops shipping high-`ep_rew` crashers.
2. **Frenet-s progress + stall/reverse/anti-circling suite** — makes “learning” mean forward racing.
3. **Collision-first + TTC/near-miss terminate; speed gated on ~0 crashes** — clean laps before pace.
4. **Crawl → clean lap → speed curriculum (+ pace floor)** — ordered unlocks beat parallel wishlists.
5. **Residual RL / warm-start on FTG + hard veto/safety filter** — shortest path past classical baseline.
6. **Frozen held-out eval + RoboRacer-like lap/collision rules + beat-FTG Δt leaderboard** — honest wins.
7. **Freeze obs ABI; match beams/FOV/Hz; engineered L/R/F + TTC (+ short stack)** — see walls closing.
8. **Latency / dropout / actuator lag DR in train** — policies that survive the bridge.
9. **Truncate doomed eps; diversify maps/starts; steps/sec without display tax** — more real learning per night.
10. **Hashed map packs + sealed holdouts + hardware validation track** — generalization you can trust week-to-week.
11. **Sim-to-sim identical maps before physical; shadow mode + intervention budget** — don’t burn the car debugging reward.
12. **Separate train shaping from eval metric** — shaping can help; it must not define “won.”
13. **Checkpoint selection = lowest-collision / best adjusted_time on holdout** — not best train reward.
14. **Atomic complete checkpoints + obs-dim refuse-load + no partial best_model** — broken zips don’t race.
15. **Domain-randomize track width/µ/LiDAR noise enough to kill wall-barnacle memorization** — then stop adding gadgets.

---

### Section scorecard (harsh)

| Section | Verdict | Why |
|--------|---------|-----|
| Training & algorithms | Keep ~6, defer rest, cut chic | Residual + gates + sample efficiency win; algo zoo loses. |
| Reward & curriculum | Keep core suite | This *is* the product for finishing laps. |
| Observation & sensors | Keep contract + features + DR | Camera/gadgets cut until LiDAR cleans. |
| Maps & trackgen | Keep packs/holdouts/cache | Fantasy generators cut. |
| Control UI & UX | Keep almost nothing for race | Labels/Start-Stop honesty only if they prevent bad runs; rest defer/cut. |
| Watch & visualization | Keep metric overlay + crash tags | Pretty twins defer; art cut. |
| Metrics / logging / W&B | Keep fingerprint + leaderboard + PPO health | Lineage/CO₂/thumbs defer; audio cut. |
| Eval & reproducibility | Keep protocol hard | TOST/CI reproduce defer; sexy overfit cut. |
| Sim2real / bridge | Keep sim2sim, latency, veto | Overnight on-car PPO cut. |
| Safety & rules | Keep no-GT + ESTOP + collision priority | Bureaucracy defer; binder cut. |
| Resilience / chaos | Keep corrupt/mismatch/dual-train | Full conspiracy suite defer. |
| Docs & teaching | Defer almost all | Doesn’t finish laps; one “three jobs” line max. |
| Demos / growth | Cut / hard defer | After FTG is beaten on holdout. |
| Aesthetic / art | Cut | Zero race function. |
| Simplify / cut | Keep the *discipline* | One metric, beat FTG once; kill feature addiction. |
| Deliberately bad | Cut from backlog | Red-team jokes only. |
