# AutoDRIVE RoboRacer — AiCar

Training stack for AutoDRIVE RoboRacer / F1TENTH-style sim racing: Layer 1 driver → Layer 2 Gymnasium env → Layer 3 PPO (next), plus Mission Control UI and Docker (planned).

## Current status

| Layer / piece | Status |
| :--- | :--- |
| **Layer 1** — `src/layer1/` | **Verified** (headed + headless, multi-instance, reset, kill) |
| **Layer 2** — `src/layer2/` | **Implemented** |
| **Layer 3** — PPO / 1D-CNN | **Implemented** (`src/layer3/`) — see [`LAYER3.md`](LAYER3.md) |
| **Mission Control UI** — `src/ui/` | Not started |
| **Docker (A sim × N + B brain)** | Scaffold only (single compose service today) |

Layer docs: [`src/layer1/README.md`](src/layer1/README.md) · [`src/layer2/README.md`](src/layer2/README.md) · [`src/layer3/README.md`](src/layer3/README.md) (plain English) · [`LAYER3.md`](LAYER3.md) (build plan) · [`PLAN.md`](PLAN.md).

---

## TODO

Order of work from here:

1. **Layer 3 — PPO training** — see **[`LAYER3.md`](LAYER3.md)**
   - [x] `src/layer3/` — extractors, `envs.py`, `train.py`, `play.py`
   - [x] Smoke: `train --n-envs 1` (10k) and `train --n-envs 2`
   - [x] Checkpoints + TensorBoard under `logs/rl/`
   - [x] `play` loads `.zip` (1 env)

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
   - [ ] Reward Phase 2 (waypoints / lap progress) when ready
   - [ ] Competition packaging (submit B image; external A)
   - [ ] Docker: point `train --n-envs N` at pre-started A×N sims (`auto_launch=False`)

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

# SB3 Gym check / Layer 3
python scripts/demo.py check-env
python scripts/demo.py train --n-envs 1 --timesteps 10000 --out logs/rl/smoke_10k
python scripts/demo.py train --n-envs 2 --base-port 4570 --timesteps 4096
python scripts/demo.py play --model logs/rl/smoke_10k/final_model.zip --steps 200

# Mock / unit tests (no Unity)
python -m pytest scripts/test_layer1.py scripts/test_layer3_extractor.py -v
```

**Headed Layer 1:** set each window’s port (`4567`, `4568`, …) and click **Connect**.  
**Headless:** `-batchmode -nographics -ip … -port …` auto-connects.

### Construct Layer 2

```python
from src.layer2 import AutoDriveEnv, RewardConfig

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
│   ├── demo.py             # layer1 | layer2 | check-env
│   └── test_layer1.py      # pytest: telemetry + mock Socket.IO
├── src/layer1/             # Layer 1 — see src/layer1/README.md
├── src/layer2/             # Layer 2 — see src/layer2/README.md
├── src/layer3/             # Layer 3 — PPO (extractors, envs, train, play)
├── docker/                 # Current single-image scaffold
├── docker-compose.yml
├── simulator/              # Binaries (gitignored) + README
├── logs/trajectories/      # CSV exports (contents gitignored)
└── logs/rl/                # PPO runs (contents gitignored)
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
