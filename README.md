# AutoDRIVE RoboRacer — AiCar

Training stack for AutoDRIVE RoboRacer / F1TENTH sim-racing time trials on Windows (RTX 3060), with a clean Layer-1 simulator driver and a roadmap toward Gymnasium + PPO.

## Current status

| Layer | Status |
| :--- | :--- |
| **Layer 1** — `RaceTrack` / `Racer` / telemetry | **Verified** (headed + headless, multi-instance, reset, kill) |
| **Layer 2** — Gymnasium `AutoDriveEnv` | **Implemented** (`src/env/`) — see below |

| **Layer 3** — PPO / 1D-CNN | Not started |
| **Mission Control UI** (`src/ui`) | Planned only |
| **Docker training stack** | Scaffold exists; UI entrypoint not built yet |

Layer 1 talks to the Windows AutoDRIVE Simulator over Socket.IO (**Engine.IO v4**), using a **gevent** WSGI server and string-valued `Bridge` command emits (matching the AutoDRIVE Devkit reply shape).

### Layer 2 design snapshot

* **1 env = 1 car** (no multi-car inside one env; parallel training = many 1-car envs later).
* Headless; full **1080** LiDAR; `frame_skip=1` (~40 Hz) but configurable (`>= 1`; **not** 0).
* Crash: **no episode end**; `collision_penalty` defaults to **0** — stagnation ends the attempt.
* `max_episode_steps=0` → no hard step cap (stagnation only); set `>0` for a safety limit.
* Waypoints / lap bonuses: **deferred**.
* Code: `src/env/` + `tests/check_gym_env.py`.

**Full Layer 2 docs (every knob, rewards, Docker, how-to-use):** [`src/env/README.md`](src/env/README.md)

#### Construct `AutoDriveEnv`

```python
from src.env import AutoDriveEnv, RewardConfig

env = AutoDriveEnv(
    # path auto-detected (Windows .exe or Docker Linux .x86_64)
    port=4567,
    headless=True,
    frame_skip=1,          # every physics tick; cannot be 0
    max_episode_steps=0,   # 0 = unlimited (stagnation only)
    reward_config=RewardConfig(forward_scale=1.0, collision_penalty=0.0),
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.5, 0.0])
env.close()
```

Compliance check (live headless sim required):

```bash
python -m tests.check_gym_env
```

Also: [PLAN.md §6](PLAN.md).

## Quick start (Layer 1)

1. Place the Windows simulator under `simulator/windows/AutoDRIVE Simulator.exe` (gitignored binary).
2. Install Python deps (use Python 3.13 or a venv):
   ```bash
   pip install -r requirements.txt
   ```
3. Run a live demo:

   ```bash
   python scripts/demo.py
   python scripts/demo.py --headless --racers 1 --duration 10
   python scripts/demo.py --racers 2 --duration 15
   ```

   - **Headed:** set each window’s port (`4567`, `4568`, …) and click **Connect**.
   - **Headless:** uses `-batchmode -nographics -ip 127.0.0.1 -port <n>` and auto-connects.

4. Mock / unit tests (no Unity required):
   ```bash
   python -m pytest tests/test_telemetry.py tests/test_driver.py tests/test_two_instances.py -v
   ```

## Repository layout (what exists now)

```text
AiCar/
├── PLAN.md                    # Full architecture & roadmap
├── README.md                  # Overview + Layer 1 quick start
├── requirements.txt
├── scripts/
│   └── demo.py                # Live Layer 1 demo (headed / headless)
├── src/racer/                 # Layer 1 driver
│   ├── racer.py               # Socket.IO server, step/reset/kill, sim launch
│   ├── track.py               # Fleet manager, Frenet / checkpoints
│   └── telemetry.py           # Snapshot + CSV logger
├── src/env/                   # Layer 2 Gymnasium env — see src/env/README.md
│   ├── autodrive_env.py
│   ├── spaces.py
│   ├── rewards.py
│   └── README.md
├── tests/                     # Telemetry math + mock Socket.IO + check_gym_env
├── simulator/                 # AutoDRIVE binaries (mostly gitignored)
└── logs/trajectories/         # CSV exports from demos (gitignored contents)
```

See [PLAN.md](PLAN.md) for the target layout (UI, Gym env, models, Docker).

---

## Architecture overview (target)

### Hardware context

* **Host**: Windows 11 with WSL2.
* **GPU**: NVIDIA GeForce RTX 3060 (12 GB), CUDA 12.x.
* **Philosophy**: Pure RL in the simulator (no sim-to-sim transfer), using telemetry + 1080-beam LiDAR.

### Two-container target design

```
┌────────────────────────────────────────────────────────┐
│               CONTAINER 1: SIMULATOR                   │
│  - AutoDRIVE Simulator (Linux .x86_64 or Windows exe) │
│  - Headless (-batchmode -nographics / xvfb)            │
│  - Socket.IO Client (ports 4567+)                      │
└──────────────────────────▲─────────────────────────────┘
                           │ Socket.IO Bridge
                           │ Telemetry ↔ Throttle / Steering / Reset
┌──────────────────────────▼─────────────────────────────┐
│               CONTAINER 2: BRAIN (API / RL)            │
│  - Layer 1 driver (done) + Gymnasium env (planned)     │
│  - PyTorch + Stable-Baselines3 PPO                     │
│  - Mission Control UI on :8080 (planned)               │
└────────────────────────────────────────────────────────┘
```

### Bridge protocol

* Unity connects as a Socket.IO client to the Python server on port `4567` (+ instance offset).
* Each physics tick, Unity emits `'Bridge'` with vehicle telemetry.
* Python replies with an emitted `'Bridge'` payload (strings):

  ```python
  {'V1 Throttle': str(throttle), 'V1 Steering': str(steering), 'V1 Reset': str(reset)}
  ```

### Planned Gymnasium design (Layer 2+)

* **Observation**: LiDAR (1080 beams) + kinematics / IMU + previous action.
* **Action**: `Box(-1, 1, shape=(2,))` → steering, throttle/brake.
* **Termination**: collision; truncation on step limit.
* **Parallelism**: multiple headless sims on `4567..4570` via `SubprocVecEnv`.

### Roadmap phases

1. **Phase 1 (next)**: End-to-end PPO with 1D-CNN on LiDAR + IMU.
2. **Phase 2**: Follow-the-Gap baseline lap time.
3. **Phase 3**: ForzaETH / TC-Driver style raceline conditioning.

---

## Implementation milestones

1. **Layer 1: Fleet Management & API — Completed & verified**
   * `Racer` / `RaceTrack` / `TelemetrySnapshot` in `src/racer`.
   * Live verification: headless 1-car drive; headed 2-car drive; `reset_all` / `reset_single`; `kill_racer` / `kill_all`; CSV trajectory export.
2. **Gymnasium Environment — Implemented (`src/env/`)**
   * `AutoDriveEnv` + `spaces` / `rewards`; 1 env = 1 car; headless; crash penalty default 0; stagnation truncation; `frame_skip=1`.
   * Includes `__init__.py`, `tests/check_gym_env.py`, README construct snippet.
3. **2D Localhost Preview — Pending**
   * Mission Control canvas on port `8080`.
4. **PPO Training Pipeline — Pending**
   * 1D-CNN feature extractor + vectorized **many 1-car envs** on the RTX 3060 (not multi-car-in-one-env).
