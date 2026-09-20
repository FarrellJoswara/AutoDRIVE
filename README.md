# AutoDRIVE RoboRacer — AiCar

Training stack for AutoDRIVE RoboRacer / F1TENTH-style sim racing: Layer 1 driver → Layer 2 Gymnasium env → Layer 3 PPO (next), plus Mission Control UI and Docker (planned).

## Current status

| Layer / piece | Status |
| :--- | :--- |
| **Layer 1** — `src/racer/` | **Verified** (headed + headless, multi-instance, reset, kill) |
| **Layer 2** — `src/env/` | **Implemented** |
| **Layer 3** — PPO / 1D-CNN | Not started |
| **Mission Control UI** — `src/ui/` | Not started |
| **Docker (A sim × N + B brain)** | Scaffold only (single compose service today) |

Layer docs: [`src/racer/README.md`](src/racer/README.md) · [`src/env/README.md`](src/env/README.md) · design detail in [`PLAN.md`](PLAN.md).

---

## TODO

Order of work from here:

1. **Layer 3 — PPO training**
   - [ ] `src/models/` feature extractor (1D-CNN on LiDAR + MLP on state)
   - [ ] `scripts/demo.py` train / play commands (or dedicated train script)
   - [ ] Checkpoints + TensorBoard under `logs/rl/`
   - [ ] Host smoke: 1 env, save/load `.zip`

2. **Docker — split A (sim) / B (brain)**
   - [ ] `Dockerfile.sim` (Unity + Xvfb) and `Dockerfile.brain` (Python + CUDA/torch)
   - [ ] Compose: one **brain** service + **N scaled sim** services (1 Unity / container)
   - [ ] Wire ports `4567+`; `AICAR_SIMULATOR_PATH`; GPU passthrough
   - [ ] Verify `python scripts/demo.py layer2` and PPO train inside B talking to A’s

3. **Mission Control UI (`src/ui`)**
   - [ ] FastAPI app on `:8080`
   - [ ] Launch / stop / reset / kill fleet controls
   - [ ] Live 2D canvas (cars + LiDAR) + leaderboard / telemetry WS

4. **Link UI ↔ Docker**
   - [ ] UI runs in brain container (or host) and starts/stops scaled sim services
   - [ ] Single-click LAUNCH / STOP ALL wired to compose or a small launcher API
   - [ ] Logs / checkpoints volume-mounted to host `./logs`

5. **Everything else**
   - [ ] Vectorized training (`SubprocVecEnv`, many 1-car envs)
   - [ ] Reward Phase 2 (waypoints / lap progress) when ready
   - [ ] Competition packaging (submit B image; external A)

---

## Quick start

```bash
pip install -r requirements.txt
```

Place Windows sim at `simulator/windows/AutoDRIVE Simulator.exe` (gitignored).

```bash
# Layer 1 live demo
python scripts/demo.py layer1
python scripts/demo.py layer1 --headless --racers 1 --duration 10

# Layer 2 live smoke
python scripts/demo.py layer2 --steps 40

# SB3 Gym check (needs torch + stable-baselines3)
python scripts/demo.py check-env

# Mock / unit tests (no Unity)
python -m pytest tests/test_layer1.py -v
```

**Headed Layer 1:** set each window’s port (`4567`, `4568`, …) and click **Connect**.  
**Headless:** `-batchmode -nographics -ip … -port …` auto-connects.

### Construct Layer 2

```python
from src.env import AutoDriveEnv, RewardConfig

env = AutoDriveEnv(port=4567, headless=True, frame_skip=1, max_episode_steps=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.5, 0.0])
env.close()
```

---

## Repository layout

```text
AiCar/
├── PLAN.md                 # Architecture & roadmap
├── README.md               # This file
├── requirements.txt
├── scripts/
│   └── demo.py             # layer1 | layer2 | check-env
├── src/racer/              # Layer 1 — see src/racer/README.md
├── src/env/                # Layer 2 — see src/env/README.md
├── tests/
│   └── test_layer1.py      # Telemetry + mock Socket.IO fleet tests
├── docker/                 # Current single-image scaffold
├── docker-compose.yml
├── simulator/              # Binaries (gitignored) + README
└── logs/trajectories/      # CSV exports (contents gitignored)
```

---

## Architecture (target)

```text
┌─────────────────────────────────────┐
│  A × N — Simulator containers       │
│  One Unity process per container    │
│  Socket.IO client → B ports 4567+   │
└──────────────────▲──────────────────┘
                   │ Bridge
┌──────────────────▼──────────────────┐
│  B — Brain (one container)          │
│  Layer 1 + 2 + 3 (PPO) + UI :8080   │
└─────────────────────────────────────┘
```

Today’s compose file is still a **single** combined service; splitting A/B is on the TODO list above.

**Bridge:** Unity emits `Bridge` telemetry each tick; Python emits string commands `V1 Throttle` / `V1 Steering` / `V1 Reset`.
