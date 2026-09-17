# Gaps the mega list under-emphasizes

Skim baseline (Sep 2026): gym env + FTG + PPO train, `control_ui` Start/Stop + map picker/preview + `n_envs`, `watch --follow` twins, checkpoints/`latest_model`/`live_status`, `trackgen` CLI, `eval_cli` (print-only), `leaderboard.csv` append-on-train, contracts **2.0.0** in code with **v1 zips/rows still on disk**.

Mega treats several of these as “already on radar” one-liners. Radar ≠ shipped plumbing. Below: what the backlog should **add**, **promote**, and **demote**.

---

## MISSING ideas to ADD (new bullets)

### Continue-train plumbing (radar item, zero path)

- `--resume` / `--load` from `latest_model.zip` or a named checkpoint; choose **same `run_id`** (append TB + checkpoints) vs **fork** into a new run with parent lineage
- SB3 knobs explicit: `reset_num_timesteps`, optimizer state keep/drop, “extra timesteps” vs absolute budget
- Control UI: **Continue last unfinished run?** with reuse vs restart spelled out (paths, remaining steps, contracts version of the zip)
- Refuse continue when `config.json` major ≠ runtime contracts; surface “retrain required” not a cryptic SB3 shape error
- After Ctrl+C / UI Stop: prefer **last complete** checkpoint over half-written `latest_model.zip` (atomic rename rule)

### Pause vs Stop (only hard-kill exists)

- **Pause** = freeze `learn` loop, keep VecEnv/GPU process warm, Watch still readable
- **Soft stop** = finish current rollout, write checkpoint + status, then exit
- **Hard stop** = today’s kill-tree (document irreversible: no resume without a prior complete zip)
- UI copy: Stop does **not** pause; quitting the UI does **not** stop an adopted external train (and vice versa)
- Watch: pause / reset twins / freeze weight reload / “reload newest brain now” as separate verbs from train Stop

### Map gen UX (picker ≠ generator)

- Control UI: **Generate N maps** (seed, prefix, difficulty tags) → refresh dropdown + thumbs; not a separate terminal ritual
- Seal **holdout** maps (hash + never-train flag); train Start refuses holdouts unless `--allow-holdout`
- Cache/regenerate policy: never regenerate inside the train hot loop; “Rebuild thumbs” action
- Curriculum unlock UX: Map1 gated on Map0 score — needs generate + eval honesty first
- Mid-run map change: fail loud in UI (Stop required); no silent YAML swap mid-Subproc

### Windows Subproc reality (defaults disagree today)

- Single story for Windows: `start_train.ps1` defaults **subproc**, `train_ppo` CLI defaults **dummy**, UI Start forces **subproc** when `n_envs>1` — pick one default + document fallback
- On Subproc spawn/pickle failure: write `vec_env_active=dummy` into `live_status.json` and UI banner (today: train may fall back; UI still claims subproc intent)
- Freeze spawn checklist: `if __name__`, picklable `RacingEnv`, worker count vs RAM, antivirus locking zips
- Straggler/orphan worker detection after Stop; Dummy vs Subproc must match Watch `n_envs` legend honesty
- Throughput KPI on Windows: steps/sec + “effective parallel” (Dummy `n_envs` ≠ Subproc FPS)

### `eval_cli` vs leaderboard honesty

- **Train post-eval** (`--eval_episodes` default 2, often **null** `mean_lap_time`) currently **writes** `leaderboard.csv` — smoke rows look like race scores
- **`eval_cli` prints JSON and never appends** — the honest cross-eval path is invisible to `compare_models`
- Gate: only append leaderboard from a frozen eval protocol (fixed seeds, ≥N eps, held-out map, contracts version) — or tag rows `kind=smoke|official`
- Filter/display: hide null `adjusted_time`, filter by `contracts_version`, show train-map leakage badge when eval map ∈ train set
- One “promote to race candidate” action: held-out eval → leaderboard row → optional “recommended” pin (not `ep_rew_mean` from TB)

### Contracts migration (v1→v2 is a footgun, not a footnote)

- Load gate everywhere (`eval_cli`, future resume, bridge PPO): read `config.json` / zip side-car; **refuse** major mismatch before `PPO.load` predict (Watch already checks obs dim; eval does not)
- Quarantine or relabel on-disk `contracts_version: 1.0.0` artifacts; leaderboard rows still mix **1.0.0** with empty scores
- Docs/UI tip when selecting a model: “v1 LiDAR-only obsolete — retrain”
- Artifact side-car: embed `contracts_version` + `obs_dim` in a tiny json next to every zip so orphan checkpoints are self-describing
- Bridge↔gym obs parity test as a CI/smoke that fails closed on remap drift

### Ownership / process races (lightly mentioned, not designed)

- Double-Start, Stop→Start, UI-owned vs external `start_train.ps1`, two trains same `run_id` zip corruption
- Heartbeat that proves **gradients** (or at least timestep mtime advancing), not just HTTP green
- Disk-full / half-write during checkpoint; never promote partial `best_model`
- “Restart train after reward/env edits” as a first-class status warning (workers keep old code until Stop+Start)

### Eval protocol gaps mega buries under research cosplay

- Train clones **one map × n_envs** — diversification of maps/starts is missing as a concrete UI/train flag, not only a slogan
- Deterministic vs stochastic eval both reported; fixed checkpoint selection (best vs latest vs lowest-collision ensemble member)
- Null baselines next to FTG on the same protocol before claiming PPO wins

---

## UNDER-RANKED that should rise

These are already in the mega list (or “on radar”) but sit too low relative to what’s **blocking** a trustworthy loop:

1. **Continue training / session resume** — radar checkbox; no `--resume`, no UI continue; overnight work dies on Stop
2. **Separate eval metric from train reward** + **frozen held-out protocol** — leaderboard is currently a train-smoke diary
3. **Contracts/obs-dim mismatch → hard refuse load** — partial in Watch; must be universal before resume/bridge/eval
4. **Subproc worker kill / Dummy fallback must match status reality** — Windows default mismatch is an active footgun
5. **Atomic checkpoint rename; resume prefers last complete artifact** — needed the moment continue-train exists
6. **Clear UI-owned vs external-train ownership; Stop consequences in plain English** — Start/Stop exists; semantics don’t
7. **Parallel envs must diversify maps/starts** — `n_envs` slider ships; all workers on map0 fake generalization
8. **Immutable config fingerprint** (git SHA, vec type, map hash, contracts) — without it leaderboard rows are gossip
9. **Cache generated tracks / map pack versioning / holdouts** — `trackgen` CLI exists; UX + seal do not
10. **Rank by adjusted race score, not `ep_rew_mean`** — score formula exists; UI/TB still teach the wrong KPI
11. **Kill dual best/latest/checkpoints theater until multi-day runs are real** — theater already shipped; continue-train did not (simplify this *up*, not down)
12. **Vectorize hard / steps/sec KPI** — tied to honest Subproc vs Dummy; more important than SAC/Dreamer
13. **Mid-run map YAML swap must fail loud** — UI can change dropdown while a run is alive; need a lock
14. **Detect frozen live_status / dead PID / dual UI fighting one runs dir** — resilience section is correct but buried under aesthetics

---

## OVER-RANKED noise

High volume in the mega list that **competes with** the gaps above for attention. Keep as stretch/fun later; do not schedule ahead of continue-train / honest eval / contracts / Windows Subproc.

- **Aesthetic / art** (reward weather, calligraphy policies, soundtrack→cello, museum labels)
- **Gamification / growth** (season ladder, Elo, NFT hashes, TikTok crash comps, champagne confetti)
- **Research cosplay before one clean held-out beat-FTG:** Dreamer/world models, Meta-RL, PBT overnight, diffusion/ACT/flow, LLM planner, RLAHF thumbs, lineage DAGs, CO₂ accounting
- **Sensor/product sprawl:** camera Grad-CAM, event/thermal/NeRF, TensorRT/Jetson thermal theater, OpenCV on the same Jetson as control
- **Watch entertainment:** speech-bubble twins, karaoke throttle, 16 Watch windows, perfume-on-PB
- **UI vibes ahead of verbs:** garage-night starter button, haiku run names, emoji-only status, cheerful lying UI, voice narrating timesteps
- **Hybrid classical stacks** (tube-MPC, hierarchical modes, residual RL) *before* resume + honest leaderboard — good ideas, wrong priority vs “Stop deletes your overnight run”
- **Deliberately bad / overkill** block as backlog noise — fine as red-team appendix; do not interleave with practical P0s
- **“Partner fantasy: upload map PNG → ranked model overnight”** — skips map-gen UX, eval honesty, and contracts migration that make overnight ranking meaningful

---

### Priority nudge (non-code)

If the mega list is re-sorted once: **continue-train + pause/soft-stop → leaderboard/eval honesty → contracts refuse/quarantine → Windows Subproc/status truth → map generate+holdout UX**. Everything rainbow/Dreamer/camera after a frozen held-out FTG comparison that `eval_cli` can write without lying.
