"""
F1TENTH RL Stack

Multi-phase reinforcement-learning stack for solo F1TENTH / RoboRacer time-attack.
Frozen interface: [contracts.md](contracts.md). Research: [research_notes.md](research_notes.md).

**Branch:** `rl/phase-1-research`

## Start training

Easiest (no chat / no activate needed) — defaults: **map0**, **300k** steps, **device auto**, TB under `rl/runs`:

```powershell
cd "ADSS Toolkit/autodrive_py"
.\rl\start_train.ps1
# optional: .\rl\start_train.ps1 -Timesteps 100000 -Map map1
```

- Log: `rl/logs/<run_id>.log`
- Models: `rl/models/<run_id>/best_model.zip`
- TensorBoard: `& .\rl\.venv\Scripts\python.exe -m tensorboard --logdir rl/runs` → http://localhost:6006
- Stop: `Ctrl+C` in the train window

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

There is **no** separate dashboard app. Two small tools cover watching and curves:

| Need | Tool | Notes |
| ---- | ---- | ----- |
| Live car on map | `python -m rl.watch` | One OpenCV window: top-down map + car triangle + a few numbers (speed, step, episode return, collisions). **Separate process** — not hooked into `train_ppo`. |
| Training curves | TensorBoard | Reward / loss over time. Prefer this during training (no FPS hit). |

```powershell
# Live view (second terminal; raise --every if laggy; --no-beams = map+car+numbers only)
python -m rl.watch --policy ftg --map map0 --every 2
python -m rl.watch --policy ppo --map map0 --every 5
python -m rl.watch --policy ftg --map map0 --every 2 --no-beams

# Training curves only (after / during train_ppo; default TB logging is on)
tensorboard --logdir rl/runs
# then open http://localhost:6006
```

`train_ppo` stays fast: no OpenCV window in the train loop. Use `--no-tb` only if you want to disable TensorBoard logs.

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

- Observation: **LiDAR-only** (180 beams + prev action). Camera deferred — not in v1 obs or viewer.
- Action: `MultiDiscrete([4, 11])`.
- Score: `adjusted_time = lap_time + 10 * collisions`.
- **GPU:** `--device auto` uses the RTX when CUDA torch is installed. Env stepping stays on CPU (normal for this stack); GPU runs the PPO update. Force CPU with `--device cpu` if you want.
- **UI:** TensorBoard for curves; `rl.watch` for a minimal map view in another process. No full dashboard.
- For a serious train, use `--timesteps 100000` (or more) and multiple maps.
