# AutoDRIVE F1TENTH RL — Trimmed Idea List

> **Provenance:** Synthesized from 12 personality reviews of `IDEAS_MEGA.md` (~320 ideas → this list).  
> **Consensus method:** KEEP = appeared in KEEP/TOP across multiple of: functionality, implementation, minimalist, reward/training, architecture (plus reliability / sim2real / competition where they reinforce the same kernel). Operator features elevated by UX + gaps auditor are kept on a separate **operator track** even when race-minimalists deferred them. Devil RESCUE items appear only as their serious kernels. CUT = killed by ≥3 reviewers and/or devil KILL list (unless rescued as a tiny kernel).

**Two tracks (parallelize carefully):**
- **Race-skill** — finish clean laps; beat FTG on `adjusted_time = lap + 10·collisions`.
- **Operator honesty** — Start/Stop truth, continue-train, maps, model load/delete, coach — so overnight work survives and numbers don’t lie.

---

## P0 Must ship (race + operator honesty)

### Race truth & selection
- **Promote by `adjusted_time` / collisions, never `ep_rew_mean`** — train reward lies; promotion must use the sport. *(consensus: func, impl, monk, reward, arch, competition)*
- **Frozen held-out eval protocol** — fixed maps/seeds/episode budgets; FTG + PPO same CLI; no test-fishing. *(consensus: func, impl, monk, reward, arch, reliability)*
- **Select `best_model` by held-out adjusted_time / lowest collision** — today’s final-zip “best” is a lie. *(consensus: func, impl, reward, gaps)*
- **Separate train shaping from eval metric** — log shaping; rank only race score. *(consensus: all core)*
- **Always-register fair FTG baseline** on the board under the same protocol. *(consensus: impl, monk, arch)*
- **Match RoboRacer-style scoring bits in eval** — warm-up ignored, scored laps, collision DQ / λ=10. *(consensus: func, competition, monk)*
- **Null baselines optional; FTG required** — random/constant as honesty checks. *(consensus: reliability, impl)*

### Env / reward (gym)
- **Keep Frenet-s high-water + stall; extend anti-circling suite** if spawn jitter reintroduces orbiting. *(consensus: func, monk, reward; partially shipped)*
- **Collision-first economics** — terminal −R_big early; collision cost > timeout. *(consensus: func, reward, competition)*
- **Speed/aggression unlock only under ~0 rolling collision** (+ pace floor vs parade laps). *(consensus: func, monk, reward)*
- **TTC / frontal min-range truncate** (soft, no MPC yet). *(consensus: func, impl, reward)*
- **Train spawn jitter** (pose/heading/lateral) — Watch already has it; train does not. *(consensus: impl, gaps, competition)*
- **Diversify `n_envs` maps/starts** — 8–32× map0 is fake skill. *(consensus: func, speed, impl, reward, arch)*
- **Truncate doomed episodes; shorter horizons until finish rate rises**. *(consensus: func, speed, reward)*

### Contracts / artifacts / throughput
- **Keep contracts `2.0.0` frozen**; hard-refuse obs-dim / major mismatch everywhere (eval, resume, bridge). *(consensus: all)*
- **Immutable config fingerprint** (hparams, torch/CUDA, VecEnv, map hash, contracts, git SHA if cheap). *(consensus: impl, reliability, sim2real)*
- **Atomic complete checkpoints only**; resume prefers last complete artifact; never promote half-written `best_model`. *(consensus: speed, impl, reliability, gaps)*
- **Steps/sec + crash rate first-class in `live_status` / UI** — evidence for `n_envs` / Dummy vs Subproc. *(consensus: speed, impl, reliability)*
- **Headless train; kill display consumers in the train process**. *(consensus: speed, arch)*
- **Windows Subproc/Dummy story one default + status truth** when fallback happens. *(consensus: gaps, speed, reliability)*
- **Quarantine / relabel on-disk v1 artifacts**; leaderboard hide null `adjusted_time`. *(consensus: gaps, reliability)*

### Operator track (P0 honesty — user-elevated)
- **Start/Stop ownership honesty** — UI vs external train; Stop consequences; dual-UI / same-`run_id` collision refuse. *(consensus: UX, reliability, gaps, arch)*
- **“What’s happening now” banner** — Idle / Learning / Saving / Stopping / Crashed + reason; frozen/`live_status` ≠ green. *(consensus: UX, reliability, impl)*
- **“What will Start do?” preview** (map, steps, workers, paths). *(consensus: UX, impl)*
- **Continue last unfinished run** (`--resume` / reuse vs fork; last complete ckpt). *(consensus: UX, gaps, impl, reliability)*
- **Soft stop vs hard kill documented** (finish rollout + ckpt vs kill-tree). *(consensus: gaps)*
- **Model timeline + load / delete** — Best / Latest / Checkpoints; delete bad zips without shell. *(consensus: UX, gaps; user wants)*
- **Map picker + Generate N maps UI** (seed/tags → cache thumbs); never regen in hot loop. *(consensus: UX, gaps; user wants)*
- **Three-jobs first-launch copy** — Control trains · Watch watches · TB curves. *(consensus: UX, impl, docs)*
- **Watch overlay: projected adjusted_time + collisions + crash tags** (not just `ep_rew`). *(consensus: func, UX, impl)*
- **Lag-behind honesty label** for Watch. *(consensus: UX, speed)*

---

## P1 Next (gym beat-FTG + operator UX)

### Race skill
- **Crawl → clean lap → unlock speed** curriculum (+ ¼→½→full lap gates as needed). *(consensus: func, monk, reward)*
- **Sealed hashed holdouts + pin validation track**; train Start refuses holdouts unless explicit. *(consensus: func, monk, sim2real, arch, gaps)*
- **Curriculum unlock Map1 after Map0 beats FTG gate**. *(consensus: func, impl, arch)*
- **Light LiDAR DR** — noise / dropout / max-range clip (then latency jitter). *(consensus: func, impl, reward, sim2real)*
- **Engineered L/R/F clearance + TTC / wall rates** (prefer features over contracts bump; short stack only if needed). *(consensus: func, reward, monk)*
- **Warm-start from FTG/BC demos** — burn fewer thrash hours on the 3060. *(consensus: func, speed, impl, monk)*
- **Residual RL kernel** — bounded Δsteer/Δthrottle on FTG (or gym safety filter) once pure PPO plateaus. *(consensus: func, monk, reward, devil rescue, sim2real)*
- **Race-candidate promote gate** — only after held-out clean/budgeted eval. *(consensus: impl, UX defer→promote, reliability)*
- **Ghost / compare as eval** — FTG vs current brain on adjusted_time (not carnival). *(consensus: UX, devil rescue “ghost”, competition)*
- **Crash taxonomy** (wall, spin, stall, NaN, contract mismatch) + survival / time-to-crash. *(consensus: monk, reliability, devil rescue)*
- **Overfit detector** — scramble LiDAR bins at eval; GT-ablation gate. *(consensus: reward, sim2real, competition)*
- **Mean ± CI (≥5 seeds) + worst-decile / tails** on official rows. *(consensus: monk, reliability)*
- **Leaderboard CI bands + beat-FTG Δt**; tag `kind=smoke|official`. *(consensus: reliability, gaps, arch)*
- **Channel ablations under matched seeds** (IMU/speed/stack). *(consensus: reward, speed)*
- **Default shorter serious runs** (50k–100k) until metric truth exists — not 500k cosplay. *(consensus: speed, simplify)*
- **Lock sensible `n_envs` + tiny-ish net defaults**; presets Debug / Quick / Overnight. *(consensus: speed, UX, impl)*

### Operator track (P1 polish)
- **Coach panel** — reward slope, crash rate, “did you open Watch?”, too-many-workers. *(consensus: UX, gaps; user wants)*
- **Health traffic lights** (CPU/GPU/RAM) + workers coach. *(consensus: UX, speed)*
- **Post-run summary + Open folder / TB handoff**. *(consensus: UX)*
- **Map mismatch banner** (Watch map ≠ train map). *(consensus: UX, reliability)*
- **Presets chips** wrapping existing knobs. *(consensus: UX, speed, impl)*
- **FTG placeholder underdrawing until PPO weights load** (already partial). *(consensus: UX)*
- **Common-mistake library** (camera expectations, Watch ≠ curves). *(consensus: UX)*

---

## P2 Stretch after held-out win

- **Cowardly qualifier → time-attack** as two short stages (once one-stage gate works). *(consensus: reward; deferred by func/impl until P1)*
- **Smoothness / aimed-throttle tax** after anti-circling is solid. *(consensus: reward)*
- **Sector/lap bonuses once per lap with heading gate**. *(consensus: reward)*
- **Centerline attractor early only, then anneal**. *(consensus: reward; never permanent)*
- **MultiDiscrete vs continuous ablation** (Rainbow kernel — bins + brake-hard; matched steps). *(consensus: devil rescue; defer until protocol)*
- **Ensemble: race lowest-collision checkpoint member**. *(consensus: devil rescue hive-mind kernel; after artifact integrity)*
- **Actuator dynamics in gym** (steer rate, ESC lag) + richer delay/packet-loss DR. *(consensus: sim2real, reward)*
- **Cross-map transfer heatmaps** before claiming “racing skill”. *(consensus: reliability)*
- **Stress eval tails** (jitter, occlusion, latency profiles). *(consensus: sim2real, reliability)*
- **Matched algo bake-offs** (SAC/TD3 family) **only** under frozen protocol — side lane. *(consensus: arch serialization)*
- **LSTM / deeper history** only if k-stack + features fail. *(consensus: func, reward defer)*
- **`torch.compile` / AMP / frame-skip ablations** after env FPS is measured. *(consensus: speed)*
- **Bayesian/manual FTG gain tune** if residual/PPO can’t beat tuned classical. *(consensus: monk, sim2real)*
- **Failure-only shards / bit-replay packs** for Watch debug. *(consensus: reliability, devil rescue)*
- **Negative-result registry** so leaderboard isn’t survivor-biased. *(consensus: reliability)*
- **Watch ghost rival** (yesterday’s best translucent) as compare, not confetti. *(consensus: UX, devil)*
- **Glossary / a11y** after core flows are boring. *(consensus: UX defer)*
- **Constrained RL (PPO-Lag/CPO)** only if shaping+gates fail collision budget. *(consensus: devil KILL-as-first; keep as late option)*

---

## P3 Bridge / Jetson only after gates

**Hard gate into this phase:** beat pinned FTG on map0 **and** ≥1 sealed holdout on `adjusted_time` (CI / tails OK), contracts hash frozen, classical veto story defined.

- **Sim-to-sim gym → AutoDRIVE `:4567` on identical maps** before chassis. *(consensus: func, sim2real, arch)*
- **Calibration gate** — bridge LiDAR stats inside gym DR envelope or refuse. *(consensus: sim2real, competition)*
- **Latency budget** scan→act p50/p99; fail if p99 > trained step; soak under load. *(consensus: sim2real, competition)*
- **Shadow mode kernel** — policy infers, FTG/human drives; log divergence + intervention budget. *(consensus: devil rescue, sim2real, competition)*
- **Classical hard veto / Safe Halt** (heartbeat loss, LiDAR dropout → halt). *(consensus: competition, sim2real)*
- **Single-threaded control core**; viz off the path; prefer UDP/binary. *(consensus: sim2real)*
- **Deploy-readiness one-pager** (sensor, latency, net, veto, ABI, thermal)—or no track. *(consensus: devil binder kernel, sim2real)*
- **ESTOP hardware + software watchdog**. *(consensus: competition)*
- **No-GT race binary + GT-ablation gate**. *(consensus: competition, sim2real)*
- **Pin hardware validation track** for week-to-week compares. *(consensus: sim2real)*
- **Offline bridge replay scoring** before live PPO authority. *(consensus: sim2real)*
- **Tiny actor FLOPs / TensorRT / distill student** only after fp32 shadow meets Hz. *(consensus: speed defer, sim2real)*
- **Rate-limited on-car adaptation** only after shadow+veto+calibration (never overnight unrestricted PPO). *(consensus: devil fine-tune kernel; CUT unrestricted)*
- **Camera / vision MAJOR** only after LiDAR gym+bridge is boringly safe. *(consensus: arch, devil KILL-as-early)*
- **Competition bureaucracy** (sealed image, Change Control) when entering a real event. *(consensus: competition defer)*

---

## Explicitly cut / later-never (short)

| Cut | Why |
| --- | --- |
| World models / Dreamer / TD-MPC | Devil KILL + multi-review; before crash taxonomy & residual |
| Meta-RL / PBT overnight | No task distribution; eval still wrong |
| Diffusion / ACT / flow / LLM planner / Rainbow-Atari cosplay | Paper chic; wrong problem |
| Camera / event / NeRF / multi-GoPro before LiDAR 10-lap win | Sensor maximalism |
| Full RLAHF / Watch thumbs | Solo time-attack doesn’t need annotator weekends |
| Season ladder / Elo / XP / NFT / TikTok / champagne bingo | Gamification ≠ podium |
| Partner PNG→overnight ranked model | Skips eval honesty |
| Lying / emoji / haiku / vibes / auto-start UI | Operator harm |
| 16 Watch windows / speech bubbles / perfume / garage-floor projection | Entertainment |
| Overnight unrestricted on-car PPO; OpenCV twins on Jetson control core | Safety + thermal anti-patterns |
| Privileged IPS in obs then strip; GT OCR/mocap in race path | Leak / DQ |
| Softmax+RGB on LiDAR contract; lower physics fidelity “for speed” | Contract / transfer suicide |
| Hot-swap reward via Discord; mid-episode lava morph as product | Repro death |
| Second advanced UI / plugin bus “minimal mode” | Simplify-by-adding |
| Compliance binder theater / Cyber quiz | Checklist yes; stationery no |
| Kill Watch/`live_status` before another debug surface exists | Blind operator |
| Entire `[bad/fun]` mega bucket as scheduled work | Red-team appendix only |

---

### Count check

~70 KEEP items across P0–P3 (+ cut table). Race P0 ≈ metric/env/contracts truth; Operator P0 ≈ continue/Start-Stop/maps/models; P1 ≈ beat-FTG + coach; P2 ≈ polish after win; P3 ≈ bridge/Jetson behind gates.
