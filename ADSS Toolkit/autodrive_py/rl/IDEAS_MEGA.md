# AutoDRIVE F1TENTH RL — Compiled Idea Backlog

> **Provenance:** Sourced from 20 personality brainstorms (~846 raw bullets → ~320 unique ideas after dedupe). Tags: `[practical]` `[stretch]` `[bad/fun]`.

Merged all **20** brainstorm personalities (~846 raw bullets) into one themed backlog of **~320 unique ideas**. Near-duplicates collapsed; volume kept high, including stretch and deliberately bad ideas.

---

## Training & algorithms

- Rank / promote policies by **adjusted race score** (`time + λ·collisions`), not `ep_rew_mean` `[practical]`
- Hard-gate speed reward until rolling collision rate ≈ 0; unlock aggression only under a frozen collision budget `[practical]`
- Separate “cowardly qualifier” (zero-collision) then “time-attack” fine-tune stages `[practical]`
- Warm-start PPO from FTG / BC demos so early random thrash doesn’t burn the 3060 `[practical]`
- Residual RL: classical FTG/PP always runs; policy outputs bounded Δsteer / Δthrottle `[practical]`
- Hybrid: classical raceline + PP/Stanley for path; RL only for speed, recovery, or aggression scalar `[practical]`
- Safety filter: project RL actions into largest FTG gap / tube-MPC corridor `[practical]`
- Dual-rate: high-rate classical inner loop; low-rate RL setpoints every N steps `[stretch]`
- Hierarchical modes (entry/mid/exit/recovery) with classical controllers per mode; RL as gate `[stretch]`
- Discrete MultiDiscrete throttle/steer bins vs continuous; keep whichever wins adjusted_time `[practical]`
- Explicit “brake hard” action when frontal range / TTC collapses `[practical]`
- Short-horizon emergency MPC / TTC brake limiter over the policy when min_range collapses `[stretch]`
- Ensemble of checkpoints; race the lowest-collision member, not highest train reward `[practical]`
- Try SAC / TD3 / TQC / DroQ / REDQ vs PPO under matched steps and eval protocol `[stretch]`
- Recurrent / LSTM-PPO or short LiDAR history stack for closing-rate to walls `[practical]`
- Constrained RL (PPO-Lag / CPO / CVaR): hard wall-clearance or crash-rate constraint `[stretch]`
- Offline / batch RL (IQL/CQL) from FTG + failed PPO bags before any on-car on-policy steps `[stretch]`
- World-model / Dreamer / TD-MPC “imagine crashes offline” track `[stretch]`
- Go-Explore / archive hard corners and reset curriculum into failure states `[stretch]`
- Population-based training / PBT over PPO hyperparams overnight on CPU-tiny policies `[stretch]`
- Meta-RL across trackgen maps for few-shot adaptation `[stretch]`
- Self-play / ghost opponents from past policies even for solo time-attack `[stretch]`
- Distill best PPO into tiny student (or 3-rule flowchart) for Jetson throughput `[stretch]`
- Frame-skip / action-repeat ablations; dubious for racing but worth measuring `[stretch]`
- `torch.compile` / AMP / bigger batches / fewer PPO updates for wall-clock sample efficiency `[practical]`
- Vectorize hard: steps/sec is the KPI; kill display consumers during train `[practical]`
- Truncate doomed episodes early (crash/stall); shorten max horizon until success rate justifies length `[practical]`
- Parallel envs must diversify maps/starts—32 clones of map0 fake generalization `[practical]`
- Match train control Hz to real servo/ESC rate, not “whatever PPO batch likes” `[practical]`
- Cap policy FLOPs for Jetson 50–100 Hz with thermal headroom `[practical]`
- TensorRT / quantized actor for deploy; train fp32 on desktop `[stretch]`
- Bayesian opt / CMA-ES over classical gains first; RL only if the frontier is flat `[practical]`
- Model-based residual dynamics on bicycle/Pacejka, not model-free from scratch `[stretch]`
- Diffusion / ACT / flow-matching policies for 2-DoF car (paper chic) `[bad/fun]`
- LLM planner + RL tracker (“turn left at cone” over LiDAR) `[bad/fun]`
- Rainbow DQN on discretized steer at 5 Hz “like Atari” on a continuous car `[bad/fun]`
- Hive-mind: n_envs cars vote on next action `[bad/fun]`
- Gravity flip / wet-cheese friction / lava-lamp morphing maps mid-episode `[bad/fun]`

## Reward & curriculum

- Progress only along Frenet **s** (signed); never Euclidean distance-from-start farming `[practical]` *(already on radar: forward-progress)*
- Stall timeout if Δs < ε for N seconds; reverse-travel abort past −Δs budget `[practical]` *(already on radar)*
- Anti-circling suite: heading–tangent mismatch tax, yaw tax while s flat, loop-detector terminate, angular/path progress ceiling `[practical]`
- Dense Δs with deadband/clamp so wall-bounce / spin near gates can’t farm `[practical]`
- Lap bonus with direction/heading gate; incomplete-lap penalty ∝ remaining s; decaying time bonus for early finish `[practical]`
- Collision-first: contact = terminal −R_big early; later allow recovery curriculum `[practical]`
- Near-miss shaping from min LiDAR clearance; progressive λ in `time + λ·collisions` `[practical]`
- Terminate on imminent collision (TTC), not only contact `[practical]`
- Smoothness tax: Δsteer² / Δthrottle² / jerk; throttle only pays when aimed down-track `[practical]`
- Centerline attractor early; anneal so a racing line can emerge `[practical]`
- Curriculum: crawl → clean laps → raise speed; widen corridor → shrink; ¼→½→full lap finish gates `[practical]`
- Curriculum: empty → static obstacles → traffic; short “clear one corner” horizons before full circuit `[stretch]`
- Domain-randomize track width, µ, LiDAR noise/dropout, latency, spawn pose `[practical]`
- Collision-free binary gate before any speed reward; pace floor so “parade lap forever” loses `[practical]`
- Checkpoint/sector bonuses once per lap; sector-based eval tags for hard corners `[practical]`
- Opponent-aware: overtake +s-rank bonus; aggressor blame via closing speed `[stretch]`
- Separate eval metric from train reward so shaping doesn’t become the sport `[practical]`
- Hot-swap reward mid-run via UI “chaos dial” / Discord emoji reactions `[bad/fun]`
- Battery-as-budget / vibes-only LiDAR-entropy reward / Mercury-in-retrograde unclip `[bad/fun]`
- Suicide-crash to reset: make collision cost > timeout cost `[practical]`

## Observation & sensors

- Freeze obs contract early (beams, units, FOV, frame, version hash); bridge remap = breaking ABI `[practical]` *(IMU/encoders already on radar)*
- Engineered features on top of scan: L/R/F clearance, TTC, relative wall rates `[practical]`
- Stack last k LiDAR frames for closing rate without GT pose `[practical]`
- Ablate channels one-at-a-time (LiDAR / prev-action / speed / IMU) under matched seeds `[practical]`
- Shrink to LiDAR-only if proprio doesn’t help; kill unused stacking `[practical]`
- Domain-randomize beam dropout, max-range clip, angular bias, FOV crop, beam permutation `[practical]`
- Randomize obs/action delay + jitter + packet loss as first-class enemies `[practical]`
- Partial-obs drills: freeze last scan, inject NaNs/zeros; isolate bad envs so one doesn’t poison the batch `[practical]`
- Match gym beam count/order/FOV to AutoDRIVE/real sensor; no deploy-time interpolation `[practical]`
- Add actuator dynamics in train: steer rate limit, ESC lag, voltage sag, friction patches `[practical]`
- Wheel odom / slip carefully—perfect odom becomes a free oracle you’ll lose outdoors `[practical]`
- Overfit detector: scramble LiDAR bins at eval; if return stays high, you’re not reading rays `[practical]`
- Calibration gate: bridge LiDAR stats must sit inside gym DR envelope or refuse deploy `[practical]`
- Camera later: hybrid LiDAR safety + cam apex; optical flow; segmentation free-space FTG `[stretch]`
- Event cam / thermal / NeRF track / 8×4K GoPros transformer `[bad/fun]`
- LiDAR→camera distillation then throw the laser away at deploy `[bad/fun]`
- Privileged IPS/speed in train then strip at race time (leak—don’t) `[bad/fun]`

## Maps & trackgen

- Procedural / warped map packs; freeze hashed held-out eval tracks that mimic competition release `[practical]` *(map generator UI already on radar)*
- Map picker with thumbnails + easy/twisty/long tags (not filename lottery) `[practical]`
- Cache generated tracks offline; never regenerate inside the hot loop `[practical]`
- Pin a never-changing “hardware validation track” for week-to-week Jetson compares `[practical]`
- Curriculum unlock: Map1 only after Map0 beats FTG/score gate `[practical]`
- Map pack versioning: seeds, mesh hashes, difficulty labels, sealed holdouts `[practical]`
- Mid-run map YAML swap must fail loud or remap explicitly—not “drive through walls as skill” `[practical]`
- Partner fantasy: upload map PNG → ranked model overnight `[stretch]`
- Spotify-seeded / coffee-stain / “emotional maps” (anxiety hairpin…) generators `[bad/fun]`
- Loot-box maps / rarer tracks unlock liveries `[bad/fun]`
- Train map0 only until boringly good (minimalist counter-theme) `[practical]`

## Control UI & UX

- Plain-language labels: “how many cars practice,” “average score,” “session name” `[practical]`
- “What’s happening now” banner: Idle / Warming / Learning / Saving / Stopping / Crashed + reason `[practical]`
- Progress story + ETA in clock time; outcome-framed timesteps copy `[practical]`
- Preset chips: Quick try / Overnight / Debug (1 car); Advanced unlock for workers/batch/net `[practical]`
- “What will Start do?” preview card (map, steps, workers, paths) before commit `[practical]`
- Clear UI-owned vs external-train ownership; Stop consequences in plain English `[practical]` *(Start/Stop already on radar)*
- First-run checklist: venv, maps, GPU; empty states with next action `[practical]`
- Session resume: “Continue last unfinished run?” with reuse vs restart spelled out `[practical]` *(continue training already on radar)*
- Model timeline: Best / Latest / Checkpoints + “recommended for racing” `[practical]` *(load/delete already on radar)*
- Compare / ghost: FTG vs current brain; promote-to-race-candidate only after held-out clean evals `[practical]` *(compare models already on radar)*
- Health panel: CPU/GPU/RAM traffic lights + “too many workers” coach `[practical]`
- Glossary drawer; accessibility: large type, colorblind-safe twins, keyboard Start/Stop `[practical]`
- Post-run summary + next steps; copyable log path / Open folder `[practical]`
- Coach panel / grade from reward slope, crash rate, whether you opened Watch `[practical]` *(coach panel already on radar)*
- Safe-defaults badge; TensorBoard one-click / auto-open handoff `[practical]` *(TB auto-open already on radar)*
- Lag-behind honesty: label Watch delay so it feels intentional `[practical]`
- Garage-night aesthetic: one honest big Start like a starter button `[stretch]`
- Auto-start on page load / haiku run names / vibes-only dial / emoji-only status `[bad/fun]`
- Voice narrating every timestep; 16 Watch windows “for transparency” `[bad/fun]`
- Cheerful lying UI (“you’re crushing it” while spinning) `[bad/fun]`

## Watch & visualization

- Twin rainbow / multi-color cars with named legend (“Worker A… same brain demos”) `[practical]` *(already on radar)*
- Follow twins lag-behind; pause, reset, freeze policy update, reload newest brain now `[practical]` *(already on radar)*
- Overlay projected adjusted_time + collisions, not just ep_rew `[practical]`
- Heat map of wall-contact shame; crash stamps / reason tags `[practical]`
- Ghost rival: yesterday’s best translucent twin to “pass” `[practical]`
- FTG placeholder as faint underdrawing until PPO weights go opaque `[practical]`
- LiDAR as sparse “cathedral light”; optional beams toggle for FPS `[stretch]`
- Reward weather: golden haze vs fog; plateau cooler blue; breakthrough dawn `[stretch]`
- Photo-finish collage on new PB; highlight reel of top episodes `[stretch]`
- Split-screen: numbers | twins | momentum arrow `[stretch]`
- Multi-view camera tiles / Grad-CAM curb attention (when vision exists) `[stretch]`
- Twins argue in speech bubbles; karaoke overlay; rubber-duck throttle meter `[bad/fun]`
- Project Watch 1:1 on garage floor / origami crane per crash / perfume on new best `[bad/fun]`
- Auto-open stock “fake camera” of training `[bad/fun]`

## Metrics / logging / W&B

- Immutable config fingerprint: hyperparams, git SHA, torch/CUDA, VecEnv type, map hash, contracts version `[practical]`
- Throughput + parallelism tax: steps/sec, GPU util, reward skew across n_envs `[practical]`
- Full PPO internals: loss, KL, clip frac, explained var, grad norms, entropy death alerts `[practical]`
- Trajectory / failure-only shards; replay packs (seed + action stream) for bit-replay in Watch `[practical]`
- Leaderboard artifact with CI bands and “beat FTG by Δt” `[practical]` *(already on radar)*
- Artifact lineage DAG: run → parent → weights → map pack → eval report `[stretch]`
- Model/dataset cards; negative-result registry so leaderboard isn’t survivor-biased `[practical]`
- live_status heartbeat mirrored to W&B; log ingest for OOM/spawn/CUDA fallback `[practical]`
- Human preference thumbs from Watch → RLAHF-lite dataset `[stretch]`
- Compute cost / CO₂ / “FTG-equivalent laps per watt” `[stretch]`
- Audio telemetry of reward; per-ray saliency every update; keep every crash video forever `[bad/fun]`
- Replace TensorBoard with a wilting plant `[bad/fun]`

## Eval & reproducibility

- Frozen held-out protocol: fixed seeds, episode budgets, pre-registered primary metrics `[practical]`
- Mean ± 95% CI over ≥5 seeds; distributional metrics (median, IQR, worst-decile) `[practical]`
- Survival / time-to-crash curves; crash taxonomy (wall, spin, stall, NaN, contract mismatch) `[practical]`
- Cross-map transfer heatmaps before claiming “racing skill” `[practical]`
- Match FTG vs PPO on wall-clock, env steps, and human engineering hours `[practical]`
- Null baselines: random, constant-throttle, open-loop; teleop/oracle upper bound `[practical]`
- Deterministic vs stochastic eval both reported; fixed checkpoint selection policy (no test fishing) `[practical]`
- Pre-register success criteria; multiplicity correction on sweeps; TOST for “as good as FTG” `[stretch]`
- One-command “reproduce Table 1” CI that fails if metrics drift beyond ε `[stretch]`
- Stress eval: start jitter, opposite direction, occlusion, latency profiles; report tails not medians `[practical]`
- Match RoboRacer rules: warm-up ignored, 10 scored laps, DQ at >10 collisions `[practical]`
- Overfit map0 until Watch looks sexy (temptation to refuse) `[bad/fun]`

## Sim2real / bridge / hardware

- Sim-to-sim first: gym → AutoDRIVE `:4567` identical maps before physical chassis `[practical]`
- End-to-end latency budget (scan→act); fail if p99 exceeds sim step; latency soak under load `[practical]`
- Jetson shadow mode: policy infers, FTG/human drives; log action divergence `[practical]`
- FTG (or classical) hard veto envelope while PPO is still a menace `[practical]`
- Prefer UDP/binary control path; keep bridge single-threaded; viz off the control core `[practical]`
- Deploy-readiness one-pager: sensor model, latency, net size, veto, ABI, thermal—or no track `[practical]`
- Offline bridge replay scoring; closed-loop intervention-budget as primary failure metric `[practical]`
- System-ID cheap bicycle/Pacejka from logs mixed into training `[stretch]`
- Chronoskepticism: warm vs cold start, clock skew as DR axis `[stretch]`
- “Fine-tune 5 minutes on the real car” after ideal gym only `[bad/fun]`
- Online overnight PPO on physical car with no rate limits `[bad/fun]`
- OpenCV multi-twin viz on the same Jetson as the controller `[bad/fun]`

## Safety & competition rules

- No GT pose/velocity/lap progress / mocap / timing-board OCR as sensing `[practical]`
- GT-ablation gate: same binary without privileged topics must still behave `[practical]`
- ESTOP hardware + software watchdog (heartbeat loss, LiDAR dropout → Safe Halt) `[practical]`
- Approved sensor manifest; sealed compute; air-gap heats; no pit coaching `[stretch]`
- Collision-avoidance priority over lap time in Control Intent Statement `[practical]`
- Progressive cone/boundary sanctions; minimum clearance soft-stop `[practical]`
- Shadow steward: inject synthetic LiDAR dropouts; failure to halt = DQ `[stretch]`
- Version-pinned competition image; Change Control for mid-event patches `[stretch]`
- Ban CAD-width wall-following / memorized lines after track reconfiguration `[practical]`
- Compliance binder + officer of record (bureaucracy as feature) `[bad/fun]`

## Resilience / chaos / diagnostics

- Heartbeat that proves gradients exist—not just HTTP/poll green `[practical]`
- Detect frozen live_status / dead PID / orphaned trains / dual UI fighting one runs dir `[practical]`
- Corrupt / truncate / half-write JSON and zips; never promote partial best_model `[practical]`
- Atomic checkpoint rename torture; resume prefers last complete artifact `[practical]`
- Contracts/obs-dim mismatch → hard refuse load before predict `[practical]`
- Subproc worker kill / straggler / Dummy fallback must match status reality `[practical]`
- Map mismatch banner (Watch map A vs train map B); bridge-down ≠ gym-train-down `[practical]`
- NaN obs isolation; antivirus/quarantine empty zip detection `[practical]`
- Double-Start, Stop→Start races, sleep/wake, clock skew, disk-full during checkpoint `[practical]`
- Conspiracy checklist: PID, mtime, step count, file size, wheels actually moved `[practical]`
- VecEnv reward laundering / early-truncate participation trophies / max-range “open field forever” detectors `[practical]`
- Two trains same run_id collision detection before mutual zip corruption `[practical]`

## Docs & teaching

- First-launch modal: Control UI trains; Watch watches; TB shows curves—three jobs `[practical]`
- Beginner/Intermediate/Advanced that hide knobs; progressive map unlock `[practical]`
- “What good looks like” / “Don’t panic” when reward dips; FTG toast until first zip `[practical]`
- Common-mistake library: Watch before train; expecting camera; Watch ≠ loss curves `[practical]`
- “What is PPO/FTG?” expandables; contracts version tip for v1→v2 zips `[practical]`
- Progressive docs: 30s start → 5min understand → deep dive `[practical]`
- End-of-session recap: where logs/models/runs live `[practical]`
- Kill research-notes novel → 20 lines max (minimalist) `[practical]`
- Tooltips that quote papers instead of next click `[bad/fun]`

## Demos / growth / gamification

- One-click “Watch twins learn”; ghost PPO vs FTG; time-to-beat-FTG wallboard `[practical]`
- “From zero to lap” cold start on tiny map under 3 minutes `[stretch]`
- Conference booth: big twins + iPad with only Start/Stop/Watch `[practical]`
- Leaderboard TV mode; office-hours spare monitor forever-follow `[practical]`
- Human vs agent night; clip farm of LiDAR near-misses `[stretch]`
- Season ladder, daily pit contract, timestep XP, combo meter, Elo vs FTG `[stretch]`
- Crash bingo, mutation roulette, champagne confetti on PB `[bad/fun]`
- NFT of best_model hash / pay-to-name checkpoint / donate to raise exploration noise `[bad/fun]`
- TikTok crash compilation growth hack; “guaranteed podium or compute back” `[bad/fun]`

## Aesthetic / art

- Darkroom/charcoal map; twins as ghost choir; reward-as-weather lighting `[stretch]`
- Early policy as toddler handwriting → late racecraft calligraphy `[stretch]`
- Emotion beats: first clean lap, plateau coolness, breakthrough dawn—no fireworks spam `[stretch]`
- Museum-label numbers; atelier Watch title; eavesdropping on rehearsal `[stretch]`
- Soundtrack from steering→cello; risograph every 10k steps; clay racing-line sculpture `[bad/fun]`
- UI that gets uglier the longer training runs (motivational horror) `[bad/fun]`

## Simplify / cut

- One entrypoint, fixed defaults; lock n_envs and net arch; one eval path, one map `[practical]`
- One metric that matters (adjusted_time or clean lap); delete dashboard sprawl `[practical]`
- One success criterion: beat fixed FTG once on map0—then stop feature addiction `[practical]`
- Kill dual best/latest/checkpoints theater until multi-day runs are real `[practical]`
- Kill bridge/FTG product surface until a real car exists (gym-only purity) `[stretch]`
- Kill Watch twins / OpenCV / live_status if they don’t move the race metric `[stretch]`
- Default 50k until longer is proven—not 500k cosplay `[practical]`
- “Minimal mode” toggle / plugin bus / second advanced UI (simplify-by-adding traps) `[bad/fun]`

## Deliberately bad / overkill (keep for laughs & red-team)

- Maximize Watch chaos / stochastic diversity as if entertainment = podium
- Chase n_envs / batch bragging rights while eval still ranks reward
- Add camera + IMU gadgets + UI polish before one clean 10-lap held-out eval
- Massive throttle bias / wall-bouncing for progress scraps
- Reward spin “smoothness” and kill lap time; wall-barnacle PhD policies
- Softmax discrete steer; camera RGB on LiDAR-only contract
- Lower physics fidelity in train, unlock timestep, int8 obs rings
- Undervolt/OC / thermal-paste theater for +3%
- Delete best_model if ep_rew is “suspiciously round”
- PPO that only steers left on odd timesteps; action space YOLO/BRAKE/APOLOGIZE
- Train the wall to move; opponent that exists only in LiDAR noise
- Transfer from Mario Kart ghosts; rubber-duck quack volume throttle
- Streaks that guilt overnight stops; AFK gaze webcam tax
- Full Subproc pipe traces as artifacts; kurtosis of every gradient element
- Cyber-Physical Integrity Quiz before operator credentialing
- Meta-learner that enables a random terrible idea each epoch

---

### Already on your radar

These showed up across brainstorms and match features you already planned—treat as confirmed backlog, not new:

- **Map generator UI** + multi-map curriculum / holdouts
- **Load / delete models**, **continue training**, checkpoint timeline
- **Coach panel** + plain-language Start/Stop + TB auto-open
- **Forward-progress reward + stall timeout**, anti-circling / Frenet-s
- **Parallel `n_envs`**, Watch **follow twins**, **multi-color cars**
- **IMU/encoders in obs**, **leaderboard**, **compare models** / ghost vs FTG
