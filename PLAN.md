# AutoDRIVE RoboRacer Autonomous Sim-Racing: Architecture & Implementation Plan

This document synthesizes the system architecture, file layout, driver specifications, containerization strategy, and the **Mission Control UI** for the AutoDRIVE RoboRacer platform.

**Status snapshot:** Layer 1 (`src/layer1`) is implemented and live-verified. Layer 2 (`src/layer2`) is **implemented** per §6. Layer 3 plan: [`LAYER3.md`](LAYER3.md). `src/ui` planned. Prefer [README.md](README.md) for quick start; keep this file as the detailed design reference.

---

## 1. System Philosophy: The IDE & UI Separation

To combine algorithmic flexibility with operational control, the system divides responsibilities between your **IDE** and the **Mission Control Web UI**:

| In Your IDE (Code Development) | In the Mission Control UI (`http://localhost:8080`) |
| :--- | :--- |
| • Designing neural network architectures (`src/layer3/`)<br>• Writing and tuning reward formulas (`src/layer2/rewards.py`)<br>• Adjusting hyperparameters (learning rate, entropy, discount factor)<br>• Version control and unit testing (`scripts/`) | • Choosing number of racers ($N = 1, 2, 4, 16, 32\dots$ unbounded)<br>• Track selection (`IROS 2024`, `Berlin`, `Porto`)<br>• **Single-click LAUNCH / RUN** (starts Unity simulators + Python driver)<br>• **Single-click STOP ALL / KILL ALL** (terminates all background processes)<br>• **Per-car KILL RACER** (terminates a specific car process from the leaderboard)<br>• **Single-click RESET GRID**<br>• **Single-click EXPORT ALL CSVs**<br>• Live 2D top-down bird's-eye canvas (moving cars + LiDAR laser fan)<br>• Real-time leaderboard (speeds, lap times, SPS, RTF, collisions) |

---

## 2. Project Directory Structure

```text
AiCar/
├── .gitignore                                 # Git ignore (venvs, binaries, cache, trajectory CSVs)
├── .dockerignore                              # Excludes .venv, .git, and cache from Docker build context
├── docker-compose.yml                         # Docker Compose orchestration (GPU passthrough, ports, volume mounts)
├── docker/
│   ├── Dockerfile                             # Container recipe (Ubuntu 22.04, CUDA, OpenGL/Vulkan, Python)
│   └── entrypoint.sh                          # Startup entrypoint (virtual display setup & app launcher)
├── PLAN.md                                    # High-level architecture & roadmap
├── README.md                                  # Repository overview + Layer 1 quick start
├── requirements.txt                           # Core dependencies (numpy, gymnasium, socketio, gevent, torch, sb3, fastapi)
├── scripts/
│   ├── demo.py                                # Live demos: layer1 | layer2 | check-env
│   └── test_layer1.py                         # Telemetry math + mock Socket.IO (1- and 2-car)
│
├── simulator/                                 # AutoDRIVE Unity Standalone (binaries mostly gitignored)
│   ├── README.md                              # Upstream simulator usage notes
│   ├── AutoDRIVE Simulator.x86_64             # Linux binary (optional)
│   └── windows/                               # Windows build used for Layer 1 verification
│       └── AutoDRIVE Simulator.exe
│
├── src/
│   ├── __init__.py
│   │
│   ├── layer1/                                # LAYER 1 (DONE): Simulator driver & track manager
│   │   ├── __init__.py                        # Exports RaceTrack, Racer, TelemetrySnapshot, TrajectoryLogger
│   │   ├── track.py                           # RaceTrack manager (geometry, checkpoints, fleet, kill)
│   │   ├── racer.py                           # gevent Socket.IO server, lockstep step/reset/kill, sim launch
│   │   ├── telemetry.py                       # Raw telemetry model, derived dynamics, CSV logger
│   │   └── README.md
│   │
│   ├── layer2/                                # LAYER 2 (DONE): Gymnasium wrapper — 1 env = 1 car
│   │   ├── __init__.py                        # Export AutoDriveEnv
│   │   ├── spaces.py                          # Obs/action spaces + snapshot→obs
│   │   ├── rewards.py                         # RewardConfig + compute_reward
│   │   ├── autodrive_env.py                   # gym.Env reset/step/close
│   │   └── README.md
│   │
│   ├── ui/                                    # PLANNED: Mission Control web dashboard
│   └── layer3/                                # PLANNED: PPO — see LAYER3.md
│
├── logs/trajectories/                         # Demo CSV exports (contents gitignored)
└── LAYER3.md                                  # Layer 3 implementation plan (PPO / extractor / train)
```

---

## 3. Docker Containerization Architecture

The Docker setup encapsulates all complex system libraries, headless graphics drivers, and CUDA dependencies in a single reproducible environment.

```text
Host Machine (Windows / Linux)                    Inside Docker Container
─────────────────────────────                    ───────────────────────────
1. `docker compose up` ────────────────────────► Ubuntu 22.04 + CUDA 12 Runtime
                                                 Headless Graphics (OpenGL / Vulkan / Xvfb)
                                                 Mounts host `./src` to `/app/src` (Live Edit)
                                                 Mounts host `./logs` to `/app/logs`
                                                 Starts `python -m src.ui.app`
                                                            │
2. Browser: http://localhost:8080 ◄────────────── Port 8080 mapped
   (Mission Control UI)                                     │
                                                            │
3. Click [ ▶ LAUNCH ] in browser ──────────────► Backend launches Unity simulator
                                                 `./simulator/AutoDRIVE Simulator.x86_64`
                                                 Racer connects via Socket.IO (:4567+)
                                                            │
4. Real-time Telemetry & Canvas ◄─────────────── WebSocket `/ws/telemetry` (20 FPS)
```

### Key Docker Components:
1. **`docker-compose.yml`**:
   - **GPU Passthrough**: Configured with NVIDIA container runtime (`capabilities: [gpu]`) so PyTorch utilizes the host RTX 3060.
   - **Port Forwarding**:
     - `8080:8080` (Mission Control Web UI access in browser).
     - `4567-4574:4567-4574` (Socket.IO communication ports for up to 8 racers).
   - **Volume Mounts (Live Code Sync)**:
     - Mounts host `./src` into `/app/src`. Any code changes in your IDE take effect immediately inside the running container without rebuilding.
     - Mounts host `./logs` into `/app/logs` so CSV logs and training models are saved straight to the host filesystem.
2. **`docker/Dockerfile`**:
   - Base: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
   - Packages: Installs Python 3.10+, `libgl1-mesa-glx`, `libglib2.0-0`, `xvfb`, and `libvulkan1` for headless Unity execution.
   - Installs all Python dependencies from `requirements.txt`.
3. **`docker/entrypoint.sh`**:
   - Starts a lightweight virtual display (`Xvfb :99 -screen 0 640x480x24 &`) if required by the Unity binary, then executes the passed command (`python -m src.ui.app`).
4. **`.dockerignore`**:
   - Excludes `.git`, `.venv`, `__pycache__`, and temporary debug files from the build context.

---

## 4. Mission Control UI Specifications (`src/ui/`)

The UI acts as the orchestration dashboard: clicking a button in the browser drives the underlying Python code.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        YOUR BROWSER (http://localhost:8080)            │
│  [Sliders: 4 Racers]  [Dropdown: IROS 2024 Track]  [ ▶ LAUNCH RUN ]    │
│  [ 🛑 STOP ALL ]       [ 🔄 RESET GRID ]            [ 💾 EXPORT CSV ]   │
│  ────────────────────────────────────────────────────────────────────  │
│  [2D Track Canvas: Real-time top-down car coordinates & LiDAR fan]     │
│  ────────────────────────────────────────────────────────────────────  │
│  [Live Leaderboard & Fleet Controls]                                   │
│   Racer 0 | 18.2 m/s | Lap 3 (14.2s) | Collisions: 0 | [ ❌ Kill ]     │
│   Racer 1 | 16.5 m/s | Lap 2 (15.1s) | Collisions: 1 | [ ❌ Kill ]     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ REST API + WebSocket
┌───────────────────────────────────▼────────────────────────────────────┐
│                       UI BACKEND (src/ui/app.py)                       │
│  - Receives button events from browser                                 │
│  - Instantiates: track = RaceTrack(num_racers=4, auto_launch=True)     │
│  - Runs background stepping/training loop                              │
│  - Streams car coordinates and laser rays to browser                   │
│  - On STOP ALL: calls track.kill_all()                                 │
│  - On KILL RACER i: calls track.kill_racer(i)                          │
│  - On RESET: calls track.reset_all()                                   │
│  - On EXPORT: calls track.save_all_trajectories("logs/run_X")          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Calls Layer 1 & Layer 2
┌───────────────────────────────────▼────────────────────────────────────┐
│                     RaceTrack / Racer / AutoDriveEnv                   │
└────────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown:
* **`src/ui/app.py`**:
  * Lightweight ASGI web server (FastAPI) listening on port `8080`.
  * **REST Endpoints**:
    * `POST /api/launch`: Accepts `{num_racers: int, track: str, mode: str}`; instantiates `RaceTrack` and boots the simulators.
    * `POST /api/stop`: Calls `track.kill_all()` to terminate all background simulator processes and servers.
    * `POST /api/kill/{racer_id}`: Calls `track.kill_racer(racer_id)` to terminate a specific vehicle process.
    * `POST /api/reset`: Calls `track.reset_all()` to teleport all active cars back to the starting line.
    * `POST /api/export`: Calls `track.save_all_trajectories()` to export all racer CSV logs.
  * **WebSocket Endpoint (`/ws/telemetry`)**:
    * Broadcasts $(x, y, \text{yaw}, \text{speed}, \text{lidar})$ for all active racers at 20 FPS.
    * Automatically pauses when no browser clients are connected (0% CPU overhead).
* **`src/ui/static/index.html` & `app.js`**:
  * **HTML5 2D Canvas**: Renders track contour, color-coded vehicle boxes, and dynamic LiDAR laser fan hitting walls.
  * **Control Toolbar**: Sliders for racer counts, track dropdown, and global action buttons.
  * **Live Leaderboard with Kill Buttons**: Real-time table displaying for each car: Best Lap Time, Current Lap Time, Lap Count, Top Speed, G-Force, Wall Collisions, and a dedicated **[ ❌ Kill ]** button per racer.

---

## 5. Layer 1: The `layer1/` Driver & Track Manager

Layer 1 provides a clean, Pythonic interface to control and inspect the simulator with zero dependencies on Gymnasium or PyTorch.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        RaceTrack (track.py)                            │
│  - Holds Ground Truth: Centerline waypoints, total length L            │
│  - Checkpoints: Exposed num_gates, ordered gate crossing checks       │
│  - Owns & Manages: 1 or N Racer instances                             │
│  - Stepping: step_single(racer_id, th, st) and step_all(actions)      │
│  - Resets: reset_single(racer_id) and reset_all()                     │
│  - Process Control & Kill: kill_racer(racer_id) and kill_all()        │
│  - Export: save_all_trajectories(output_dir) in one single call       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Manages list of Racers
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│       Racer (Port 4567)       │       │       Racer (Port 4568)       │
│  - Socket.IO Server           │       │  - Socket.IO Server           │
│  - Lockstep step() & reset()  │       │  - Lockstep step() & reset()  │
│  - Telemetry & Profiler       │       │  - Telemetry & Profiler       │
│  - Auto-launch Sim Process    │       │  - Auto-launch Sim Process    │
│  - kill() Process Termination │       │  - kill() Process Termination │
│  - Per-racer CSV Logger       │       │  - Per-racer CSV Logger       │
└───────────────┬───────────────┘       └───────────────┬───────────────┘
                ▼ (:4567)                               ▼ (:4568)
        [Unity Sim Process 0]                   [Unity Sim Process 1]
```

### Module Breakdown:

1. **`src/layer1/track.py` (`RaceTrack` Manager)**:
   * **Parameters**:
     * `num_racers: int = 1`: Number of parallel vehicle instances.
     * `base_port: int = 4567`: Starting port (instances bind to `base_port + i`).
     * `num_gates: int = 8`: Exposed checkpoint gate count (user directly controls gate frequency).
     * `waypoints: Optional[np.ndarray] = None`: Centerline coordinates $[(x_0, y_0), (x_1, y_1), \dots]$.
     * `simulator_path: Optional[str | Path] = None`: Path to Unity executable.
     * `auto_launch: bool = False`: Automatic background process execution.
     * `headless: bool = True`: Launch with `-batchmode -nographics`.
   * **Methods**:
     * `get_frenet_progress(x, y) -> (s, d)`: Arc-length track distance ($s$) and lateral offset ($d$).
     * `update_checkpoints(racer_id, s) -> (gate_passed, lap_completed)`: Validates sequential gate crossing ($0 \to 1 \to 2 \dots$).
     * `step_single(racer_id, throttle, steering)`: Step one specific vehicle.
     * `step_all(actions)`: Step all vehicles simultaneously.
     * `reset_single(racer_id)` & `reset_all()`: Teleports cars back to the grid and zeroes stats.
     * `kill_racer(racer_id)`: Terminates a specific racer subprocess (SIGTERM/SIGKILL), closes its Socket.IO server, and removes it from the active fleet while preserving its logs.
     * `kill_all()`: Shuts down all simulator processes and stops all background servers.
     * `save_all_trajectories(output_dir)`: Exports CSV logs for **all racers** into a folder in one single call (`output_dir/racer_0.csv`, `output_dir/racer_1.csv`).

2. **`src/layer1/racer.py` (`Racer` Class)**:
   * **Parameters**: `racer_id`, `port`, `simulator_path`, `auto_launch`, `headless`, `step_timeout`.
   * **Attributes**: `telemetry` (`TelemetrySnapshot`), `logger` (`TrajectoryLogger`), profiler metrics (`step_latency_ms`, `steps_per_second`, `real_time_factor`).
   * **Transport**: `socketio.Server(async_mode="gevent")` + `gevent.pywsgi` + `WebSocketHandler`. This RoboRacer Windows build speaks **Engine.IO v4** (`python-socketio` 5.x / `python-engineio` 4.x). Commands are returned only via `sio.emit('Bridge', {string values})` (no ACK return payload). A small auto-connect shim handles Unity clients that emit `Bridge` before a formal namespace connect.
   * **Methods**:
     * `_start_server()`: Starts background gevent Socket.IO server on `0.0.0.0:port` (server object created on the server thread — required on Windows).
     * `step(throttle, steering)`: Buffers clamped commands, waits for next `'Bridge'` frame (turn-based lockstep), updates profiler, and returns `TelemetrySnapshot`.
     * `reset()`: Sets `V1 Reset` for the next Bridge reply (`"True"` / `"False"`), waits for acknowledgment, returns state.
     * `kill()`: Terminates the child OS process, stops the gevent server, and releases the port.
     * `save_trajectory(file_path)`: Exports this vehicle's trajectory history to CSV.


3. **`src/layer1/telemetry.py`**:
   * **`TelemetrySnapshot`**:
     * 100% raw data: position $(x, y, z)$, orientation quaternion $(x, y, z, w)$, linear/angular velocities, linear acceleration, wheel encoders (left/right), 1,080 LiDAR beams, scan rate, actuator states, lap stats, collisions.
     * Derived metrics: `true_speed`, `heading_yaw`, `v_long`, `v_lat`, `slip_angle`, `lateral_g`.
   * **`TrajectoryLogger`**: In-memory buffer logging step, timestamp, $(x, y, z)$, speed, slip angle, throttle, steering, encoders, and lap stats; exports to CSV.

4. **`src/layer1/__init__.py`**:
   * Exports `RaceTrack`, `Racer`, `TelemetrySnapshot`, and `TrajectoryLogger`.

---

## 6. Layer 2: Gymnasium Environment (`src/layer2/`)

Layer 2 wraps Layer 1 so a learning algorithm (e.g. PPO via Stable-Baselines3) can train without knowing Socket.IO exists. Gymnasium is only the shared `reset` / `step` interface; the env turns telemetry into **observation**, **reward**, and **episode-over** signals.

**Data flow:** Unity → Layer 1 (`TelemetrySnapshot`) → `autodrive_env.py` gathers facts → `rewards.compute_reward(...)` → score for the learner. To reward something Layer 1 does not expose yet: extend Layer 1, or derive it in the env from existing fields / external files, then pass it into `rewards.py`.

### Locked design decisions

| Topic | Decision |
| :--- | :--- |
| Env ↔ car | **Strict 1 Gym env = 1 racer.** We are **not** putting multiple cars inside one env. Parallelism later = many env *processes*, each with its own headless sim (e.g. SB3 `SubprocVecEnv`) |
| Default launch | **Headless** (watching is Mission Control UI later, not the Unity window) |
| LiDAR | **Full 1080 beams**, normalized; **no downsampling** for now |
| Waypoints / Frenet progress reward | **Deferred** — reward Phase 2; v1 does not require a centerline map |
| Crash policy | **Do not end the episode on collision.** Keep `collision_penalty` in config but **default `0.0`**. Pressure comes from lost forward progress + stagnation truncation. Wall tax is optional later |
| Episode end | **Stagnation truncation** (no meaningful forward progress for a configured idle window). Optional hard **max-steps** cap (`max_episode_steps`; **default `0` = disabled**). Not a fixed “one lap timer” |
| Control rate | Unity physics ~**40 Hz**. Default **`frame_skip=1`** (~40 Hz actions). `frame_skip` must be `>= 1` (`0` is invalid). Option e.g. `2` → ~20 Hz |

### Explicitly out of scope for Layer 2

* Multi-car / multi-agent **inside** a single `AutoDriveEnv`  
* PPO, neural nets, training loops (Layer 3)  
* Mission Control UI  
* Waypoints, Frenet \(s,d\), gates, lap bonuses (reward Phase 2)

### Modules (ship with Layer 2)

| File | Role |
| :--- | :--- |
| `src/layer2/__init__.py` | Export `AutoDriveEnv` (and maybe `RewardConfig`) |
| `src/layer2/spaces.py` | Action/obs space definitions + `TelemetrySnapshot` → obs helpers |
| `src/layer2/rewards.py` | `RewardConfig` + `compute_reward` (main file for tuning scores) |
| `src/layer2/autodrive_env.py` | `gym.Env`: `reset` / `step` / `close`; gathers facts; calls Layer 1 |
| `scripts/demo.py check-env` | SB3 / Gymnasium `check_env` live smoke |
| `src/layer2/README.md` | Layer 2 summary, layout, file/API docs, how-to-use |
| `src/layer1/README.md` | Layer 1 summary, layout, file/API docs |

### Observation / action (v1)

* **Observation** (`Dict`):
  * `"lidar"`: shape `(1080,)`, values in \([0, 1]\)
  * `"state"`: e.g. \([v_{\text{long}}, v_{\text{lat}}, \omega_z, a_{\text{long}}, a_{\text{lat}}, \beta, \text{prev\_throttle}, \text{prev\_steering}]\)
* **Action**: `Box(-1, 1, shape=(2,))` → `[throttle, steering]`

### Reward sketch (v1, no waypoints)

* **Primary signal:** forward progress proxy (e.g. \(v_{\text{long}}\) or forward displacement). Stagnation truncates the episode — main “you’re stuck” pressure (including after crashes).
* **`collision_penalty`:** present in `RewardConfig`, **default `0.0`**. Crash ≠ automatic lose.
* Optional weights (may also default to `0`): slip, steering jerk.
* Episode continues after hits unless stagnating or max steps hit.

**How to add a new reward/penalty later (edit `rewards.py`):**

1. Add a weight on `RewardConfig` (default `0.0` if unused).
2. In `compute_reward(...)`, add a term when the condition is true.
3. If you need a new **fact**, either expose it on `TelemetrySnapshot` (Layer 1), derive it in `autodrive_env.py` from existing snap fields, or load external data (e.g. waypoint file) in the env — then pass that fact into `compute_reward`.
4. Tune by changing config numbers, not rewriting the env loop.

### Termination vs truncation

* **Terminated:** unused for crashes in v1 (reserve for rare hard failures only if we add any later)
* **Truncated:** stagnation window, or optional absolute max-steps safety cap (`max_episode_steps > 0`)  
* **`reset()`:** called when a *new episode* starts (after truncate). That calls Layer 1 reset (teleport to grid). Mid-episode crashes do **not** by themselves call Unity reset.

### Still fuzzy (implementation defaults chosen)

These were open in design; v1 code uses:

1. **Stagnation:** `|v_long| < 0.15` m/s for **200** consecutive env steps (~5 s at 40 Hz with `frame_skip=1`).
2. **Forward reward:** `forward_scale * v_long` with `forward_scale=1.0`.
3. **Connect wait:** **60** s timeout on env init when launching / expecting a client.
4. **`info` dict** (diagnostics only — **not** fed to the policy): `step`, `idle_steps`, `reward`, `v_long`, `true_speed`, `collision`, `collision_event`, `collision_count`, `position`, plus `truncate_reason` when truncated.
5. **State scaling:** none beyond LiDAR → \([0,1]\); kinematic state left as raw floats (deferred).
6. **`max_episode_steps`:** default **`0`** (disabled); set positive for a hard cap.
7. **Simulator path:** auto-detect Windows `.exe` vs Docker Linux `.x86_64`, or `AICAR_SIMULATOR_PATH`.

Full teaching docs: [`src/layer2/README.md`](src/layer2/README.md).

### Deferred (later phases)

* Track waypoints, Frenet \(s,d\), checkpoint gates, lap-completion bonus  
* LiDAR downsampling options  
* Layer 3: see [`LAYER3.md`](LAYER3.md) (PPO + 1D-CNN / MultiInputPolicy)  
* Parallel vectorized training (many 1-car envs, not multi-car-in-one-env)

---

## 7. Verification Plan

### Layer 1 — Verified

1. **Unit / mock tests** (`scripts/test_layer1.py`):
   * Quaternion / slip / Frenet math; mock Socket.IO lockstep; dual mock clients.
2. **Live headless smoke** (`python scripts/demo.py layer1 --headless`):
   * Auto-connect via `-ip` / `-port`; continuous Bridge frames; speed > 0; CSV export.
3. **Live headed multi-instance** (`python scripts/demo.py layer1 --racers 2`):
   * Two GUI windows on ports `4567` / `4568`; both connect and drive in lockstep.
4. **Live control smoke** (headless, 2 cars):
   * `reset_all` / `reset_single`; `kill_racer(i)` while sibling keeps stepping; `kill_all`.

### Later layers — Pending

5. **Layer 2 Gymnasium — Implemented**:
   * `src/layer2/` + `scripts/demo.py layer2|check-env` + layer READMEs
   * Live `check_env` still requires a headless sim binary when you run the checker
6. **Mission Control UI Smoke Test**:
   * Launch `src/ui/app.py`, open `http://localhost:8080`, verify launch/stop/kill/canvas.
7. **Docker Build & Run Verification**:
   * `docker compose up --build`, GPU via `nvidia-smi`, dashboard on port 8080.
