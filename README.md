# AutoDRIVE RoboRacer — AiCar

Training stack for AutoDRIVE RoboRacer / F1TENTH-style sim racing:

**Layer 1** (driver) → **Layer 2** (Gymnasium env) → **Layer 3** (PPO) → **Layer 4** (Mission Control UI) + **Docker A/B** (brain + scalable sims).

---

# START HERE — one command

```bash
python main.py
```

That is the easy start. It will:

1. Check Docker Desktop is running  
2. Download the AutoDRIVE simulator zip from GitHub Releases if `./simulator/` is missing  
3. `docker compose up --build --scale sim=2` (brain + 2 sims)  
4. Wait until Mission Control is healthy  
5. Open **http://127.0.0.1:8090** in your browser  

| Command | What it does |
| :--- | :--- |
| `python main.py` | Start everything + open UI |
| `python main.py --sims 4` | Same, but 4 sim containers |
| `python main.py --stop` | Tear the stack down |

**Requirements:** Docker Desktop running. First run builds images (can take a while).  
**UI:** http://127.0.0.1:8090 — Settings · Train · Live · Fleet  

Simulator binaries stay gitignored; they live in the
[`simulator-binaries`](https://github.com/FarrellJoswara/AutoDRIVE/releases/tag/simulator-binaries)
release (`autodrive-simulator.zip`). `main.py` fetches that automatically.

---

## Current status

| Layer / piece | Status |
| :--- | :--- |
| **Layer 1** — `src/layer1/` | **Verified** (headed + headless, multi-instance, reset, kill) |
| **Layer 2** — `src/layer2/` | **Implemented** |
| **Layer 3** — PPO / 1D-CNN | **Implemented** (`src/layer3/`) — see [`LAYER3.md`](LAYER3.md) |
| **Layer 4** — Mission Control | **Implemented** (hub + Settings/Train/Live/Fleet canvas) — see guide below + [`UI.md`](UI.md) |
| **Docker (A sim × N + B brain)** | **Working** — compose + hub on `:8090`; see [`docker/README.md`](docker/README.md) |

Layer docs: [`src/layer1/README.md`](src/layer1/README.md) · [`src/layer2/README.md`](src/layer2/README.md) · [`src/layer3/README.md`](src/layer3/README.md) · [`LAYER3.md`](LAYER3.md) · [`UI.md`](UI.md) · [`PLAN.md`](PLAN.md) · [`docker/README.md`](docker/README.md).

---

## TODO

1. **Layer 3 — PPO training** — see **[`LAYER3.md`](LAYER3.md)**
   - [x] `src/layer3/` — extractors, `envs.py`, `train.py`, `play.py`
   - [x] Smoke: `train --n-envs 1` / `2`; checkpoints + TensorBoard under `logs/rl/`
   - [x] `play` loads `.zip`; `SubprocVecEnv(..., start_method="spawn")` so gevent Socket.IO binds in Docker

2. **Docker — A (sim) / B (brain)**
   - [x] `Dockerfile.sim` / `Dockerfile.brain` (CUDA torch cu124 pinned)
   - [x] Compose: brain + scaled sims; ports `4567–4582`; Mission Control on `:8090`
   - [x] End-to-end: 2 sims × 10k CUDA train via hub; fleet telemetry on canvas
   - [ ] Harden sim connect / auto-stop / image rebuild ergonomics (still rough)

3. **Mission Control — Layer 4**
   - [x] FastAPI hub + Popen TrainJob + full Settings
   - [x] Vite/React on brain B (static on `:8090`)
   - [x] Live WS + Fleet canvas (map / cars / LiDAR / collision X)
   - [ ] LiDAR angle calibration (defaults may be wrong); polish; Phase N+ hyperparams

4. **Everything else**
   - [ ] Reward Phase 2 (waypoints / lap progress)
   - [ ] Competition packaging

---

## Quick start (no UI)

```bash
pip install -r requirements.txt
```

Place Windows sim at `simulator/windows/AutoDRIVE Simulator.exe` (gitignored).

```bash
python scripts/demo.py layer1
python scripts/demo.py layer2 --steps 40
python scripts/demo.py train --n-envs 1 --timesteps 10000 --out logs/rl/smoke_10k
python scripts/demo.py play --model logs/rl/smoke_10k/final_model.zip --steps 200
python -m pytest scripts/test_layer1.py scripts/test_layer3_extractor.py -v
```

### Docker A/B (brief)

Full how-to: **[`docker/README.md`](docker/README.md)**. Docker CLI should be on PATH (Docker Desktop → `…\DockerDesktop\resources\bin`).

```bash
docker compose up --build --scale sim=2
# Browser → http://localhost:8090

docker compose exec brain \
  python scripts/demo.py train --n-envs 2 --base-port 4567 --no-auto-launch \
  --timesteps 10000 --device cuda --out logs/rl/docker_smoke
```

- Sims connect **to the brain** on ports `4567+` (Socket.IO). Brain listens; Unity is the client.
- After a Docker-mode train exits, hub **stops sim containers by default** (frees RAM) while keeping Mission Control up. Settings: `stop_sims_on_train_exit` / `stop_stack_on_train_exit`.

---

# Mission Control (Layer 4) — learn from zero

This section teaches **everything** you need to understand and run the UI if you have never seen the repo. The design plan lives in [`UI.md`](UI.md); this is the operator + learner guide.

## What problem does it solve?

Training PPO against Unity is awkward from a bare terminal:

- Long `learn()` runs block a shell.
- You want **start/stop**, live **step/reward**, and a **bird’s-eye fleet view** without writing a second driver that also calls `env.step` (that would fight PPO).

**Mission Control** is a small web app + API on the **brain** container (or local Python) that:

1. Owns **job control** (start/stop training as a subprocess).
2. Watches **telemetry** the train process publishes.
3. Never owns the RL `step` loop — **train still owns stepping**.

## Mental model (read this twice)

```text
┌──────────────┐  REST + WebSocket   ┌─────────────────────────────┐
│  Browser     │ ◄──────────────────► │  FastAPI hub (:8090)         │
│  Vite/React  │                      │  Settings / TrainJob / Bus    │
└──────────────┘                      └───────────┬─────────────────┘
                                                  │ subprocess.Popen
                                                  ▼
                                      ┌─────────────────────────────┐
                                      │  python -m src.layer3.train │
                                      │  SB3 PPO · VecEnv · Racers  │
                                      └───────────┬─────────────────┘
                                                  │ POST /telemetry
                                                  │ (non-blocking)
                                                  ▼
                                      ┌─────────────────────────────┐
                                      │  Hub TelemetryBus → /ws     │
                                      └─────────────────────────────┘
                                                  │
                      sims (Docker) ──Socket.IO──►│ ports 4567, 4568, …
```

| Who | Owns |
| :--- | :--- |
| **Train child** | `env.step`, PPO `learn()`, Socket.IO servers on `base_port…` |
| **Hub** | Start/stop child, Settings JSON, fan-out telemetry to browsers |
| **Browser** | Forms, charts-ish live numbers, Fleet **canvas** (draw only) |
| **RaceTrack** | **Not** on the train path (fleet helper for demos only) |

If two things call `step`, you get double-step bugs. The UI must **not** drive the car during train.

## Pieces on disk

```text
src/layer4/
├── settings.py          # Pydantic Settings ↔ train.py CLI argv + hub knobs
├── hub/
│   ├── app.py           # FastAPI: /health /train/* /settings /telemetry /ws + static UI
│   ├── train_job.py     # subprocess.Popen lifecycle
│   ├── telemetry.py     # in-process TelemetryBus + WS fan-out
│   └── docker_control.py # stop sim (or full stack) via Docker socket when train ends
└── web/                 # Vite + React + TypeScript
    ├── src/pages/       # Settings, Train, Live, Fleet
    └── src/fleet/       # Canvas 2D map / cars / LiDAR

src/layer3/hub_callback.py   # SB3 callback: enqueue telemetry (never block learn)
assets/maps/                 # Porto / Berlin occupancy grids for Fleet
```

Layers 1–3 = learning path. Layer 4 = infra UI. Docker files stay under `docker/` + root compose (not under `src/`).

## Concepts you must know before clicking Start

### One env = one Racer = one port

`n_envs=2` means two Gym envs → two Socket.IO ports (`4567` and `4568` if `base_port=4567`). In Docker you need **`--scale sim=2`** so two Unity containers connect to those ports.

### Docker mode vs local

| | Local | Docker |
| :--- | :--- | :--- |
| Sims | Often `auto_launch=True` (Layer 1 starts `.exe`) | Sims already running in containers; **`docker_mode=True`** forces `--no-auto-launch` |
| Train device | `cpu` / `cuda` on host torch | Prefer `cuda` inside brain (cu124 image) |
| UI | `uvicorn` on host or compose | Hub is the brain’s default command on **`:8090`** |

### Why `start_method="spawn"`?

`SubprocVecEnv` on Linux defaults to **forkserver**, which breaks **gevent** (used by Socket.IO). Then ports **4567+ never bind**, sims get connection refused, training looks “running” but cars are dead. Layer 3 forces **`spawn`** in `src/layer3/envs.py`.

### Telemetry must not stall training

The train process does **not** share memory with the hub. A callback enqueues JSON; a **daemon thread** POSTs to `/telemetry` with short timeouts, drop-oldest queue, and a circuit breaker. A hung hub must not freeze PPO.

Two cadences:

- **Metrics** — every `telemetry_every_n` steps (step, reward, episode).
- **Fleet** — time-based ~`fleet_hz` (poses, yaw, collision, min-pooled LiDAR).

LiDAR in the sim is **1080** beams; for the UI we **min-pool to ~120** so the browser isn’t crushed by `JSON.parse`.

## HTTP / WebSocket API (what the UI calls)

| Method | Path | Role |
| :--- | :--- | :--- |
| `GET` | `/health` | `{ "status": "ok" }` |
| `GET`/`PUT` | `/settings` | Full Settings model (persisted under `logs/layer4/settings.json`) |
| `POST` | `/train/start` | Body = Settings (or empty → last saved); starts `Popen` |
| `POST` | `/train/stop` | Terminate train child |
| `GET` | `/train/status` | state, pid, argv, exit_code, last_telemetry, last_fleet |
| `POST` | `/telemetry` | Internal — train callback publishes here |
| `WS` | `/ws` | Browser subscribes; hub fans out status + telemetry |
| `GET` | `/` + `/assets/*` | Built Vite SPA |
| `GET` | `/maps/*` | Static map assets for Fleet |

### Settings that become train CLI flags

Almost every field on Settings maps to `python -m src.layer3.train …`:

`n_envs`, `base_port`, `timesteps`, `out`, `seed`, `device`, `resume`, `headless`, `auto_launch`, `connect_timeout`, `frame_skip`, `max_episode_steps`, stagnation/reward knobs, …

**Hub-only** (env vars / hub behavior, not train argv):

| Field | Meaning |
| :--- | :--- |
| `run_name` | Used when `out` is empty to name `logs/rl/<run>_<stamp>/` |
| `telemetry_every_n` | Metrics publish interval (steps) |
| `fleet_hz` | Fleet sample rate |
| `lidar_display_beams` | Min-pool target (~120) |
| `telemetry_lidar_max_envs` | Cap how many envs publish LiDAR |
| `docker_mode` | Force `--no-auto-launch` |
| `stop_sims_on_train_exit` | Stop compose **sim** containers when train exits (default on) |
| `stop_stack_on_train_exit` | Stop **whole** compose project (kills UI too) |

## Browser pages

| Tab | Job |
| :--- | :--- |
| **Settings** | Edit every train flag + hub knobs; Save (PUT `/settings`) |
| **Train** | Start / Stop; shows argv, pid, state, log path |
| **Live** | Streaming metrics from `/ws` (step, reward, episode) |
| **Fleet** | Canvas 2D: map underlay, cars, optional LiDAR, collision as **X**, side panel |

### Fleet canvas (how to read it)

Layers bottom → top:

1. **Map** — Porto / Berlin occupancy (`assets/maps/`) or grid fallback. Metre scale from yaml `resolution` / `origin`.
2. **Cars** — pose on Unity **X–Z** ground (Y is up).
3. **LiDAR** — rays for selected car (angles are **calibrated defaults** in `lidarCalibration.ts` — may need a wall-tune later).
4. **Collision** — crashed car drawn as **X**.

Toggles: Map / Fleet / LiDAR / track select. Side panel: episode, steps, return, collision, speed, stale flag.

If the panel says **stale**, no fresh fleet sample recently (PPO update gaps are normal — we do **not** fake-interpolate physics).

### Frontend stack (why it looks “light”)

- **Vite + React + TS**, plain **CSS** (no Tailwind/Radix/shadcn).
- **One WebSocket**; store/refs updated without re-rendering the world every beam.
- **Canvas 2D** for the fleet (not 1080 SVG lines).
- Prod: `npm run build` → static files served by FastAPI on the **same** `:8090`.
- Dev convenience: host `npm run build` updates bind-mounted `src/layer4/web/dist`; hub **prefers** that path over a stale image bake. Hard-refresh (`Ctrl+Shift+R`) after rebuilds.

## Run Mission Control

### A) Docker (recommended)

```bash
docker compose up --build --scale sim=2
```

1. Open **http://localhost:8090**
2. **Settings**: `docker_mode=true`, `n_envs=2`, `base_port=4567`, `timesteps=50000`, `device=cuda`
3. **Train** → Start
4. **Live** / **Fleet** → watch

Train CLI equivalent inside the brain:

```bash
docker compose exec brain \
  python scripts/demo.py train --n-envs 2 --base-port 4567 --no-auto-launch \
  --timesteps 50000 --device cuda --out logs/rl/fleet_demo_50k
```

(Hub sets `HUB_URL` so the callback publishes; CLI-only runs need `HUB_URL=http://127.0.0.1:8090` in the environment to feed the UI.)

### B) Local hub (API + UI without compose)

```bash
pip install -r requirements.txt
cd src/layer4/web && npm install && npm run build && cd ../../..
python -m uvicorn src.layer4.hub.app:app --host 0.0.0.0 --port 8090
```

You still need simulators listening/connecting on the ports you configure. Local venv must have torch/gymnasium/sb3 for the train **child** to import.

### Rebuild UI without rebuilding the CUDA image

```bash
cd src/layer4/web && npm run build
docker compose restart brain   # pick up bind-mounted dist
# Browser: Ctrl+Shift+R
```

## Debugging cheat sheet

| Symptom | Likely cause |
| :--- | :--- |
| Fleet still says “Canvas placeholder” | Browser/cache or hub serving old baked static — rebuild web + restart brain + hard-refresh |
| Train “running” but no progress; sims connection refused | Ports 4567+ not listening — check `spawn` fix; ensure train actually started listeners |
| `vmmem` huge CPU/RAM | WSL2 VM holding Docker + Unity sims — stop sims / `docker compose stop` when done |
| CUDA unavailable in brain | Wrong torch CUDA tag vs host driver — image pins **cu124** |
| Hub frozen on Start/Stop | Fixed deadlock: don’t call status while holding the same lock; start/stop use `asyncio.to_thread` |
| Child exits immediately locally | Venv missing train deps — install `requirements.txt` or use brain image |

## What is still scuffed / next

- LiDAR beam **angles** not measured from AutoDRIVE yet (defaults).
- Auto-stop / docker.sock permissions / image bake vs bind-mount still need polish.
- Hyperparameter surgery (live LR / net arch) is **not** day-1 — Settings = CLI flags only.
- Redis is **not** used; in-process `TelemetryBus` is enough until multi-host fan-out.

For architecture decisions and phase checklist, read **[`UI.md`](UI.md)** next.

---

## Construct Layer 2

```python
from src.layer2 import AutoDriveEnv, RewardConfig

env = AutoDriveEnv(port=4567, headless=True, frame_skip=1, max_episode_steps=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.5, 0.0])
env.close()
```

### Simulator CLI flags

| Mode | Flags |
| :--- | :--- |
| **Headed** (local Windows) | `-ip 127.0.0.1 -port <PORT>` then click **Connect** |
| **Headless** (Docker) | `-batchmode -nographics -ip <brain> -port <PORT>` |

---

## Repository layout

```text
AiCar/
├── PLAN.md
├── LAYER3.md
├── UI.md                      # Mission Control architecture plan
├── README.md                  # This file (includes Layer 4 from-zero guide)
├── requirements.txt
├── scripts/demo.py
├── src/layer1/ … layer2/ … layer3/ … layer4/
├── assets/maps/               # Fleet map assets (BSD-2-Clause AutoDRIVE tracks)
├── docker/                    # Dockerfiles + entrypoints
├── docker-compose.yml
├── docker-compose.sim-gpu.yml
├── simulator/                 # Binaries (gitignored)
└── logs/                      # rl / layer4 / trajectories (contents mostly gitignored)
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
│  Layers 1–3 + Mission Control :8090 │
└─────────────────────────────────────┘
```

**Bridge protocol:** Unity emits `Bridge` telemetry each tick; Python emits string commands `V1 Throttle` / `V1 Steering` / `V1 Reset`.
