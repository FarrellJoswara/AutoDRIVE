"""
F1TENTH RL Stack

Multi-phase reinforcement-learning stack for solo F1TENTH / RoboRacer time-attack.
Frozen interface: [contracts.md](contracts.md). Research: [research_notes.md](research_notes.md).

**Branch:** `rl/phase-1-research`

## Control UI (minimal)

Ugly single-page panel (stdlib HTTP — no Gradio). Starts/stops train as a **separate** headless process; never runs OpenCV inside train.

```powershell
cd "ADSS Toolkit/autodrive_py"
.\rl\start_ui.ps1
# or: python -m rl.control_ui
# open http://127.0.0.1:7860/
```

- Shows `live_status.json` (timesteps, `ep_rew_mean`, `run_id`, `n_envs`) when present
- Buttons: **Start training**, **Stop training** (kills `train_ppo` tree), **Open watch --follow**, TensorBoard command/link
- Knobs: map / timesteps / **Parallel sims / CPU workers** slider (`n_envs` 1–32). Start uses Subproc when >1 (Dummy fallback). Watch opens that many colored twins.

## Start training

Easiest (no chat / no activate needed) — defaults: **map0**, **500k** steps, **8 SubprocVecEnv**, **512×512** net, batch 1024, **device auto**, TB under `rl/runs`:

```powershell
cd "ADSS Toolkit/autodrive_py"
.\rl\start_train.ps1
# harder: .\rl\start_train.ps1 -NEnvs 16 -Timesteps 1000000 -NetArch "512,512" -BatchSize 2048
# if SubprocVecEnv spawn fails on Windows: -VecEnv dummy
```

- Log: `rl/logs/<run_id>.log`
- Models: `rl/models/<run_id>/best_model.zip` (+ `latest_model.zip` / `checkpoints/` while training)
- Live trail: `rl/runs/<run_id>/live_status.json` (timesteps, ep reward, etc. — no frames)
- TensorBoard: `& .\rl\.venv\Scripts\python.exe -m tensorboard --logdir rl/runs` → http://localhost:6006
- Stop: `Ctrl+C` in the train window, or **Stop** in `rl.control_ui`
- Follow while training (second terminal): `python -m rl.watch --follow`

## Quickstart (Windows)

```powershell
cd "ADSS Toolkit/autodrive_py"
py -3.13 -m venv rl/.venv
.\rl\.venv\Scripts\Activate.ps1
pip install -r rl/requirements.txt

# Phase 2 — maps
python -m rl.trackgen --num_maps 3

# Phase 3 — Follow-the-Gap baseline
python -m rl.run_ftg --map map0 --episodes 1 --save

# Phase 4 — PPO (prefer .\rl\start_train.ps1; auto GPU when CUDA torch installed)
python -m rl.train_ppo --map map0 --timesteps 100000 --device auto
python -m rl.compare_models

# Phase 5 — Bridge API check / live FTG (needs AutoDRIVE Simulator on :4567)
python -m rl.bridge_ftg --check
python -m rl.bridge_ftg

# Phase 6 — eval
python -m rl.eval_cli --backend gym --policy ftg --map map0
python -m rl.eval_cli --backend gym --policy ppo --map map0
```

## Watching (minimal) vs TensorBoard

**No camera in v2** — observation is LiDAR + proprio/IMU; viewer is still LiDAR / map only (see `contracts.md`).

Training stays **headless** (never calls OpenCV). Use `python -m rl.control_ui` for start/stop knobs; watching is a separate process that **lags behind** training on purpose.

| Need | Tool | Notes |
| ---- | ---- | ----- |
| Lag-behind twins + train stats | `python -m rl.watch --follow [--n-envs N]` | Reads `live_status.json`. Runs **N colored twin cars** on one map with `latest_model.zip` (not live Subproc poses). Overlay: timesteps, ep_rew, n_envs, color legend. |
| Fixed policy on map | `python -m rl.watch --policy ftg\|ppo` | One OpenCV window: map + car + numbers. |
| Training curves | TensorBoard | Reward / loss over time. Prefer this during training (no FPS hit). |

### How follow mode works (honest)

We **cannot** cheaply stream every train-env frame without slowing `train_ppo`. Shared-memory frame streaming is deliberately out of scope.

Instead:

1. **Train** writes a tiny JSON trail every N PPO rollouts (`LiveStatusCallback` — no render) and optionally `latest_model.zip` (includes `n_envs`).
2. **`watch --follow`** polls that JSON and drives **N colored twin cars** on one map with the latest saved weights (viewer twins — not live Subproc poses).
3. Twin episodes lag real train episodes; overlay train_ts / ep_rew / n_envs reflect the status file.

```powershell
# Terminal A — train (headless) or use control UI
.\rl\start_train.ps1
# .\rl\start_ui.ps1  → http://127.0.0.1:7860/

# Terminal B — lag-behind multi-color twins
# Follow defaults: --every 8 (lighter redraw on weak machines) + --compact overlay.
python -m rl.watch --follow --map map0 --n-envs 8 --no-beams
# denser overlay / smoother redraw if you want them:
# python -m rl.watch --follow --map map0 --every 2 --no-compact
# or pin a run / let n_envs come from live_status (omit --n-envs or pass 0):
python -m rl.watch --follow --run_id 20260101_120000_ppo_gym_map0

# One-shot / baseline view (not following train; default --every 2, compact off)
python -m rl.watch --policy ftg --map map0
python -m rl.watch --policy ppo --map map0 --n-envs 4 --every 5

# Env overrides (when flags omitted): RL_WATCH_EVERY=10  RL_WATCH_COMPACT=0|1

# Training curves only
tensorboard --logdir rl/runs
# then open http://localhost:6006
```

`train_ppo` stays fast: no OpenCV in the train loop. Disable the trail with `--no-live-status` if you want; use `--no-tb` only to disable TensorBoard logs.

## Observation / sensors (what the policy sees)

**contracts_version `2.0.0`** — fixed layout so encoders/IMU do not force a later reshape. Camera stays out (future MAJOR bump / separate head — no huge zero pads).

```text
obs[0 : N)          LiDAR (N=180 default), ranges / 10 → [0, 1]
obs[N : N+2)        prev_throttle, prev_steering          → [-1, 1]
obs[N+2]            speed_norm = clip(v / 6.0, 0, 1)      → [0, 1]
obs[N+3 : N+6)      IMU: yaw_rate, ax, ay (each / 10)     → [-1, 1]
────────────────────────────────────────────────────────────
obs_dim = N + 6 = 186 (default)
```

| In the net | Not in the net |
| ---------- | -------------- |
| LiDAR **180** (normalized) | **Camera** (deferred) |
| Previous **throttle + steering** | Raw wall-clock / episode time |
| **Speed** (gym `v` / Bridge encoders) | IPS / pose ground truth |
| **IMU×3** (`yaw_rate`, `ax`, `ay`) | `az` / `wx` / `wy` (not in v2) |

**v1 LiDAR-only (+prev action) zips are obsolete** — do not load them into a v2 env. Reward shaping may still use map GT / collisions **during training only**; that does not change the observation vector.

## Phase index

| Phase | Goal | Pass criteria |
| ----- | ---- | ------------- |
| **1 — Research & contracts** | Freeze obs/action/reward/metrics | Docs exist |
| **2 — Track generation** | Autogenerate race maps | `trackgen --num_maps 3` writes 3 map sets |
| **3 — Gym FTG** | Classical Follow-the-Gap | One episode prints metrics |
| **4 — Gym PPO + registry** | Train LiDAR PPO; save `best_model.zip` + leaderboard | Smoke train + `compare_models` ≥1 row |
| **5 — AutoDRIVE Bridge** | Live FTG over Bridge | Offline: Reset in API; live: sim on :4567 |
| **6 — Cross-eval** | One CLI for FTG/PPO × gym/autodrive | Clear errors when deps missing |

## Status

| Phase | Status | Notes |
| ----- | ------ | ----- |
| **1** | **Complete** | `research_notes.md` + `contracts.md` |
| **2** | **Implemented** | `python -m rl.trackgen` |
| **3** | **Implemented** | Gymnasium `RacingEnv` (f1tenth map format; Windows-friendly). Official `f1tenth_gym` kept optional (legacy Gym 0.19). |
| **4** | **Implemented** | SB3 PPO → `rl/models/<run_id>/` |
| **5** | **Implemented** | `reset_command` on `F1TENTH` + `rl.bridge_ftg` |
| **6** | **Implemented** | `python -m rl.eval_cli` |

## Training notes

- Observation: **LiDAR 180 + prev action + speed + IMU×3** (`obs_dim=186`; see diagram above). Camera out. v1 zips obsolete.
- Action: `MultiDiscrete([4, 11])`.
- Reward progress: **forward centerline Δs only** (high-water Frenet `s`); reverse/orbit → `0`. No spin/yaw penalty.
- Termination: collision; 60 s timeout; **stall** if no meaningful forward progress for **`stall_timeout_s=8` s** (`info["stall"]`).
- **Restart train after reward/env code changes** — Stop + Start so workers reload `RacingEnv` (an old run keeps old reward until restart).
- Score: `adjusted_time = lap_time + 10 * collisions`.
- **GPU:** `--device auto` uses the RTX when CUDA torch is installed. Env stepping stays on CPU (normal for this stack); GPU runs the PPO update. Force CPU with `--device cpu` if you want. GPU util may still stay well below 100% because physics/LiDAR are CPU-bound — parallel envs (`--n-envs` / `-NEnvs`) + bigger net/batch still train harder and raise throughput (FPS / steps/sec).
- **UI:** `rl.control_ui` / `start_ui.ps1` for start/stop; TensorBoard for curves; `rl.watch --follow` for a lag-behind twin.
- For a serious train, use `--timesteps 100000` (or more) and multiple maps.
