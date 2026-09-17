# Ideas review — speed / throughput

> **Lens:** RTX 3060 + CPU-bound gym + SubprocVecEnv + Watch lag-behind.  
> **KPI:** steps/sec and wall-clock learning. Train-time overhead is guilt until proven useful.  
> **Reality check:** Env physics/LiDAR own the timeline; GPU update is rarely the bottleneck. Bigger nets / more logging / camera / live viz steal CPU and IPC without buying samples.

---

## KEEP (must ship for throughput)

- **Vectorize hard; kill display consumers during train** — steps/sec is the KPI; OpenCV/Watch/camera must not share the train process.
- **Log throughput + parallelism tax** — steps/sec, GPU util, reward skew across `n_envs`; without this you chase theater.
- **Truncate doomed episodes early (crash/stall); shorten max horizon** — dead steps are free latency; already aligned with stall timeout.
- **Cache generated tracks offline; never regenerate in the hot loop** — procedural maps inside step() are a silent FPS killer.
- **Warm-start PPO from FTG / BC** — fewer wasted early env-steps on random thrash (wall-clock sample efficiency, not FLOPs).
- **`torch.compile` / AMP / bigger batches / fewer PPO updates** — cheap wins on the 3060 *after* env FPS is measured; don’t pretend they fix CPU gym.
- **Cap policy FLOPs / keep tiny MLP** — CPU-bound stack: huge nets inflate Subproc IPC + predict time without filling the GPU.
- **Frame-skip / action-repeat ablations (measure only)** — if skip>1 raises effective race-Hz learning without wrecking control, keep the winner.
- **Default shorter runs until longer is proven (e.g. 50k not 500k cosplay)** — long runs without throughput discipline waste nights.
- **Lock sensible `n_envs` + net arch defaults** — stop retuning workers every session; one good Subproc width beats bragging.
- **Diversify maps/starts across workers** — 32 clones of map0 inflate steps/sec of *fake* learning; diversity is free if maps are cached.
- **Shrink obs / kill unused stacking** — LiDAR-only or short stack if proprio doesn’t help; fewer floats per pipe round-trip.
- **Atomic / complete checkpoints only; refuse partial loads** — corrupt resume burns wall-clock re-running.
- **Subproc straggler / Dummy fallback must match status** — hung workers look like “slow train”; detect, don’t guess.
- **Lag-behind honesty for Watch** — separate process, lag by design; never couple render to train step.
- **Health: CPU/GPU/RAM + “too many workers” coach** — oversubscription collapses steps/sec; UI should warn, not encourage.
- **Preset chips: Quick / Overnight / Debug (1 env)** — Debug path for iteration speed; Overnight for sample volume without knob fiddling.
- **One metric that matters for promote** — stop logging theater that forces extra eval/render during train.

---

## DEFER (good later; slows train or is deploy-only now)

- **TensorRT / quantized actor** — deploy Jetson win; train stays fp32; don’t bake into overnight gym loop.
- **Distill tiny student for Jetson** — post-train; distillation jobs compete for the same machine.
- **Recurrent / LSTM-PPO or deep history stack** — more FLOPs + state IPC; try short LiDAR stack first.
- **SAC / TD3 / TQC / DroQ / REDQ bake-offs** — algorithm churn burns wall-clock; only after PPO throughput baseline is solid.
- **PBT / Bayesian / CMA-ES sweeps** — multiply env-hours; defer until single-config FPS is maxed.
- **World-model / Dreamer / TD-MPC** — GPU-heavy imagination; wrong bottleneck for this CPU gym.
- **Offline IQL/CQL bags** — useful pre-train later; collecting + training bags is another pipeline.
- **Domain-randomize latency/dropout/µ mid-train at high rate** — extra RNG + obs work per step; enable after baseline FPS.
- **Camera / optical flow / segmentation / Grad-CAM** — obs bloat and decode cost; LiDAR contract stays.
- **Heavy Watch overlays** (heatmaps, ghosts, LiDAR beams, split-screen, photo-finish) — OK in a *viewer* process at low FPS; never in train.
- **Trajectory / failure shards / keep-every-crash-video** — disk and serialize tax; sample on demand.
- **Full PPO internals every step + per-ray saliency** — log at update cadence, not env cadence.
- **W&B live heartbeat + artifact DAGs** — fine out-of-band; don’t sync on the hot path.
- **Multi-map curriculum UI polish / map picker chrome** — maps matter; UI chrome during train does not.
- **Coach panel / glossary / garage aesthetic / conference booth** — operator UX; zero steps/sec.
- **Dual-rate classical + RL / hierarchical modes / short-horizon MPC veto** — extra classical compute *inside* each env step unless proven to cut crash-waste enough to pay back.
- **Match train Hz to servo** — correctness for sim2real; may *lower* sim steps/sec intentionally—do after gym learning works.
- **Ensemble race-time pick** — eval cost; not train throughput.
- **Go-Explore / archive resets / meta-RL / self-play ghosts** — research sample efficiency with setup overhead; defer.
- **One-command reproduce-Table-1 CI** — valuable; runs offline, not during train.

---

## CUT (slows train, fake speed, or pure theater)

- **Camera + IMU gadget stacking “for richness” before clean held-out eval** — obs and decode bloat; camera already out of contract.
- **n_envs / batch bragging rights while eval still ranks reward** — parallelism theater; steps/sec of clones ≠ learning.
- **Huge nets “to use the 3060”** — GPU idle is normal with CPU gym; fat MlpPolicy taxes predict + IPC.
- **16 Watch windows / auto-open fake camera / OpenCV on train host during PPO** — display consumers steal cores from Subproc workers.
- **Voice narrating every timestep / audio telemetry of reward** — joke load, real overhead.
- **Keep every crash video forever / risograph every 10k / clay sculpture** — I/O and artist time ≠ samples.
- **Hive-mind: n_envs vote on next action** — sync barrier destroys Subproc speedup.
- **Full Subproc pipe traces as artifacts / kurtosis of every gradient** — logging DoS.
- **Hot-swap reward mid-run via Discord emoji / vibes dial** — process churn and non-repro.
- **Event cam / thermal / NeRF / 8×4K GoPros transformer** — instant FPS death.
- **Rainbow DQN @ 5 Hz “like Atari” / diffusion-ACT for 2-DoF** — fashion algorithms, worse wall-clock learning here.
- **LLM planner + RL tracker** — latency and cost with no env-FPS upside.
- **Gravity flip / lava-lamp morphing maps mid-episode** — regenerates geometry; thrash cache.
- **Loot-box / Spotify-seeded / emotional maps in the train loop** — generator tax.
- **AFK gaze webcam tax / streaks that guilt overnight stops** — interrupts long runs.
- **Undervolt/OC / thermal-paste theater for +3%** — noise vs Subproc tuning.
- **Lower physics fidelity + int8 obs rings to “unlock timestep”** — fake speed that breaks transfer and contracts.
- **Softmax discrete + camera RGB on LiDAR-only contract** — obs ABI and compute regress.
- **Online overnight PPO on physical car / OpenCV multi-twin on Jetson controller** — wrong machine, rate-limit hell.
- **Maximize Watch chaos as if entertainment = podium** — opposite of lag-behind discipline.
- **Second advanced UI / plugin bus “minimal mode”** — more processes polling `live_status`.
- **Cheerful lying UI / haiku names / emoji-only status** — zero throughput, hides hung workers.
- **Transfer from Mario Kart / rubber-duck quack throttle** — noise.
- **Meta-learner that enables a random terrible idea each epoch** — guaranteed wall-clock waste.
- **Dual best/latest/checkpoints theater before multi-day runs are real** — extra zip I/O and compare jobs during train nights.

---

## TOP 15 ranked for speed

1. **Kill display consumers in the train process** — Watch/UI/OpenCV off the Subproc parent; lag-behind only in a separate viewer.
2. **Instrument steps/sec + worker CPU + GPU util every run** — you can’t tune what you don’t plot; catch oversubscribed `n_envs`.
3. **Right-size `n_envs` to cores (not ego)** — find the knee where steps/sec plateaus; warn past it.
4. **Tiny policy + modest batch** — keep predict cheap; let CPU envs feed the 3060 without IPC bloat.
5. **Early-truncate crash/stall + short horizons until success** — maximize *useful* steps/sec.
6. **Cache maps; never trackgen in `step`/`reset` hot path** — procedural packs must be precomputed.
7. **Warm-start from FTG/BC** — cut the long random-thrash tax on wall-clock.
8. **Obs diet: freeze contract; ablate stack/IMU; fewer floats per transition** — Subproc pipes scale with obs size.
9. **AMP / `torch.compile` / fewer update epochs** — secondary wins once env FPS is stable.
10. **Frame-skip ablation with fixed wall-clock budget** — keep only if adjusted_time improves per hour.
11. **Default Debug (1 env) vs Overnight presets** — fast iteration loops vs high-throughput nights; don’t mix.
12. **Don’t grow the net to “feed the GPU”** — idle GPU with high env FPS beats busy GPU with fat MLP.
13. **Sparse logging** (TB/W&B at update cadence; no per-step video/saliency) — I/O is a silent FPS tax.
14. **Diversified cached map pack across workers** — throughput that generalizes > clone farm FPS.
15. **Refuse camera / heavy Watch overlays / dual-UI fighting one `runs/` during train** — overhead with negative learning ROI.

---

## Speed red flags (flag anything that slows train)

| Idea class | Why it hurts |
| ---------- | ------------ |
| Extra UI polish / coach / glossary during overnight | Polling, threads, accidental Start races; operator time ≠ samples |
| Camera / multi-view / Grad-CAM / “fake camera” auto-open | Decode + RAM + sometimes GPU contention with PPO |
| Heavy Watch (beams, heatmaps, 16 windows, speech bubbles) | Steals CPU from Subproc workers if co-located; fine only as lag-behind sibling |
| Huge nets / transformers / recurrent by default | Predict + serialize cost; 3060 still waits on gym |
| `n_envs` theater (32× map0 clones) | High steps/sec of non-generalizing experience |
| Live map morph / regenerate / mid-run YAML chaos | Breaks cache; spikes reset latency |
| Per-step video, audio, saliency, pipe traces | Disk and GIL/IPC death |
| Algorithm zoo before FPS baseline | Multiplies wall-clock with no throughput plan |
| Classical MPC/safety *inside every env step* without budget | Multiplies CPU work on the already-bound path |
| Dual train processes / two UIs one run_id | Corruption + restart waste |

**Rule of thumb:** If it doesn’t raise **useful env-steps per wall-clock hour** (or cut wasted doomed steps), it is DEFER or CUT for speed—even if it’s a good racing idea later.
