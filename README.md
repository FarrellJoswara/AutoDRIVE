# AutoDRIVE RoboRacer — AiCar

Training stack for AutoDRIVE RoboRacer / F1TENTH-style sim racing: Layer 1 driver → Layer 2 Gymnasium env → Layer 3 PPO, plus Mission Control UI (planned) and Docker A/B (brain + scalable sims).

## Current status

| Layer / piece | Status |
| :--- | :--- |
| **Layer 1** — `src/layer1/` | **Verified** (headed + headless, multi-instance, reset, kill) |
| **Layer 2** — `src/layer2/` | **Implemented** |
| **Layer 3** — PPO / 1D-CNN | **Implemented** (`src/layer3/`) — see [`LAYER3.md`](LAYER3.md) |
| **Mission Control UI** — `src/ui/` | Not started |
| **Docker (A sim × N + B brain)** | **Compose ready** — see [`docker/README.md`](docker/README.md) |

Layer docs: [`src/layer1/README.md`](src/layer1/README.md) · [`src/layer2/README.md`](src/layer2/README.md) · [`src/layer3/README.md`](src/layer3/README.md) (plain English) · [`LAYER3.md`](LAYER3.md) (build plan) · [`PLAN.md`](PLAN.md) · [`docker/README.md`](docker/README.md).

---

## TODO

Order of work from here:

1. **Layer 3 — PPO training** — see **[`LAYER3.md`](LAYER3.md)**
   - [x] `src/layer3/` — extractors, `envs.py`, `train.py`, `play.py`
   - [x] Smoke: `train --n-envs 1` (10k) and `train --n-envs 2`
   - [x] Checkpoints + TensorBoard under `logs/rl/`
   - [x] `play` loads `.zip` (1 env)

2. **Docker — split A (sim) / B (brain)**
   - [x] `Dockerfile.sim` (Unity + Xvfb) and `Dockerfile.brain` (Python + CUDA/torch)
   - [x] Compose: one **brain** service + **N scaled sim** services (1 Unity / container)
   - [x] Wide ports `4567-4582`; `AICAR_SIMULATOR_PATH`; GPU on brain; optional sim GPU
   - [x] Ready stub on `:8090`; train with `--no-auto-launch` against pre-started sims
   - [ ] Verify end-to-end on a machine with Docker Desktop + Linux `.x86_64` binary

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

### Simulator CLI flags

| Mode | Flags |
| :--- | :--- |
| **Headed** (local Windows) | `-ip 127.0.0.1 -port <PORT>` then click **Connect** in the UI |
| **Headless** (Docker / CI) | `-batchmode -nographics -ip <brain> -port <PORT>` (auto-connect) |

Layer 1’s `Racer.launch_simulator()` passes these for you when `auto_launch=True`. Compose sims use the headless form via `docker/entrypoint-sim.sh` (`BRAIN_HOST` + per-replica `PORT`).

### Docker A/B (brief)

Full how-to: **[`docker/README.md`](docker/README.md)**.

```bash
# 1 brain (ready stub :8090) + 2 sim containers on bridge network
docker compose up --build --scale sim=2

# Train inside brain against pre-started sims (ports 4567, 4568)
docker compose exec brain \
  python scripts/demo.py train --n-envs 2 --base-port 4567 --no-auto-launch \
  --timesteps 10000 --out logs/rl/docker_smoke
```

- Published sim/Socket.IO range: **`4567-4582`** (widen both sides of the mapping for more envs; see docker README).
- Brain GPU on by default; sim GPU via `docker-compose.sim-gpu.yml`.
- Default brain command is a ready stub on **`:8090`** (not auto-train). UI later on **`:8080`**.

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
├── docker/                 # Dockerfile.sim / Dockerfile.brain + entrypoints
├── docker-compose.yml      # brain + scalable sim
├── docker-compose.sim-gpu.yml  # optional GPU for sims
├── simulator/              # Binaries (gitignored) + README
├── logs/trajectories/      # CSV exports (contents gitignored)
└── logs/rl/                # PPO runs (contents gitignored)
```

---

## Architecture (Docker A/B)

```text
┌─────────────────────────────────────┐
│  A × N — Simulator containers       │
│  One Unity process per container    │
│  Socket.IO client → B ports 4567+   │
└──────────────────▲──────────────────┘
                   │ Bridge network `aicar`
┌──────────────────▼──────────────────┐
│  B — Brain (one container)          │
│  Layer 1 + 2 + 3 (PPO)              │
│  Ready stub :8090 · UI :8080 later  │
└─────────────────────────────────────┘
```

**Bridge protocol:** Unity emits `Bridge` telemetry each tick; Python emits string commands `V1 Throttle` / `V1 Steering` / `V1 Reset`.
