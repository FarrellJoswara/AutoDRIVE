"""
F1TENTH RL Stack

Multi-phase reinforcement-learning stack for solo F1TENTH / RoboRacer time-attack.
Frozen interface: [contracts.md](contracts.md). Research: [research_notes.md](research_notes.md).

**Branch:** `rl/phase-1-research`

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
- Stop: `Ctrl+C` in the train window
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

**No camera in v1** — observation and viewer are LiDAR / map only (see `contracts.md`).

There is **no** separate dashboard app. Training stays **headless** (never calls OpenCV). Watching is a separate process that **lags behind** training on purpose.

| Need | Tool | Notes |
| ---- | ---- | ----- |
| Lag-behind twin + train stats | `python -m rl.watch --follow` | Reads `rl/runs/<run_id>/live_status.json` for overlay (timesteps, ep_rew_mean, collision estimate). Loads `latest_model.zip` / newest checkpoint and rolls out in its **own** env. Shows ~latest brain, not every train frame. |
| Fixed policy on map | `python -m rl.watch --policy ftg\|ppo` | One OpenCV window: map + car + numbers. |
| Training curves | TensorBoard | Reward / loss over time. Prefer this during training (no FPS hit). |

### How follow mode works (honest)

We **cannot** cheaply stream every train-env frame without slowing `train_ppo`. Shared-memory frame streaming is deliberately out of scope.

Instead:

1. **Train** writes a tiny JSON trail every N PPO rollouts (`LiveStatusCallback` — no render) and optionally `latest_model.zip`.
2. **`watch --follow`** polls that JSON for overlay metrics and drives a **twin** car with the latest saved weights.
3. Twin episodes lag real train episodes; overlay train_ts / ep_rew reflect the status file (closest available progress).

```powershell
# Terminal A — train (headless)
.\rl\start_train.ps1

# Terminal B — lag-behind viewer
python -m rl.watch --follow --map map0 --every 5
# or pin a run:
python -m rl.watch --follow --run_id 20260101_120000_ppo_gym_map0

# One-shot / baseline view (not following train)
python -m rl.watch --policy ftg --map map0 --every 2
python -m rl.watch --policy ppo --map map0 --every 5
python -m rl.watch --policy ftg --map map0 --every 2 --no-beams

# Training curves only
tensorboard --logdir rl/runs
# then open http://localhost:6006
```

`train_ppo` stays fast: no OpenCV in the train loop. Disable the trail with `--no-live-status` if you want; use `--no-tb` only to disable TensorBoard logs.

## Observation / sensors (what the policy sees)

Training observation today (**neural net inputs**):

| In the net | Not in the net |
| ---------- | -------------- |
| LiDAR **180** beams (normalized ranges) | Camera |
| Previous **throttle + steering** | IMU |
| | Raw time / wall-clock |
| | IPS / pose ground truth |
| | Wheel encoders |

Episode **time/step** and collision counts can appear on the **viewer overlay** as metrics, but they are **not** policy inputs unless contracts change (`obs_include_speed` etc.). Reward shaping may use map GT / collisions **during training only** — that does not add those signals to the observation vector.

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

- Observation: **LiDAR 180 + prev throttle/steering only** (see table above). Camera / IMU / time / IPS / encoders are **not** policy inputs.
- Action: `MultiDiscrete([4, 11])`.
- Score: `adjusted_time = lap_time + 10 * collisions`.
- **GPU:** `--device auto` uses the RTX when CUDA torch is installed. Env stepping stays on CPU (normal for this stack); GPU runs the PPO update. Force CPU with `--device cpu` if you want. GPU util may still stay well below 100% because physics/LiDAR are CPU-bound — parallel envs (`--n-envs` / `-NEnvs`) + bigger net/batch still train harder and raise throughput (FPS / steps/sec).
- **UI:** TensorBoard for curves; `rl.watch --follow` for a lag-behind twin + train stats; plain `rl.watch` for a fixed policy. No full dashboard.
- For a serious train, use `--timesteps 100000` (or more) and multiple maps.
