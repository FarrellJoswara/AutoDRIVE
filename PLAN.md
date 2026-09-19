# AutoDRIVE RoboRacer Autonomous Sim-Racing: Architecture & Implementation Plan

This document synthesizes the system architecture, file layout, driver specifications, containerization strategy, and the **Mission Control UI** for the AutoDRIVE RoboRacer platform.

---

## 1. System Philosophy: The IDE & UI Separation

To combine algorithmic flexibility with operational control, the system divides responsibilities between your **IDE** and the **Mission Control Web UI**:

| In Your IDE (Code Development) | In the Mission Control UI (`http://localhost:8080`) |
| :--- | :--- |
| • Designing neural network architectures (`src/models/`)<br>• Writing and tuning reward formulas (`src/env/rewards.py`)<br>• Adjusting hyperparameters (learning rate, entropy, discount factor)<br>• Version control and unit testing (`tests/`) | • Choosing number of racers ($N = 1, 2, 4, 16, 32\dots$ unbounded)<br>• Track selection (`IROS 2024`, `Berlin`, `Porto`)<br>• **Single-click LAUNCH / RUN** (starts Unity simulators + Python driver)<br>• **Single-click STOP ALL / KILL ALL** (terminates all background processes)<br>• **Per-car KILL RACER** (terminates a specific car process from the leaderboard)<br>• **Single-click RESET GRID**<br>• **Single-click EXPORT ALL CSVs**<br>• Live 2D top-down bird's-eye canvas (moving cars + LiDAR laser fan)<br>• Real-time leaderboard (speeds, lap times, SPS, RTF, collisions) |

---

## 2. Project Directory Structure

```text
AiCar/
├── .gitignore                                 # Git ignore (virtual environments, binaries, cache)
├── .dockerignore                              # Excludes .venv, .git, and cache from Docker build context
├── docker-compose.yml                         # Docker Compose orchestration (GPU passthrough, ports, volume mounts)
├── docker/
│   ├── Dockerfile                             # Container recipe (Ubuntu 22.04, CUDA, OpenGL/Vulkan, Python)
│   └── entrypoint.sh                          # Startup entrypoint (virtual display setup & app launcher)
├── PLAN.md                                    # High-level architecture & roadmap
├── README.md                                  # Repository overview
├── requirements.txt                           # Core dependencies (numpy, gymnasium, socketio, gevent, torch, sb3, fastapi)
│
├── simulator/                                 # AutoDRIVE Unity Standalone Executable (Linux)
│   ├── AutoDRIVE Simulator.x86_64             # Unity Linux engine binary
│   ├── UnityPlayer.so                         # Unity player library
│   ├── GameAssembly.so                        # Compiled game logic
│   └── Data/                                  # Unity asset bundles (tracks, car prefabs, physics)
│
├── src/
│   ├── __init__.py
│   │
│   ├── ui/                                    # MISSION CONTROL UI (Web Dashboard)
│   │   ├── __init__.py
│   │   ├── app.py                             # Lightweight FastAPI backend & WebSocket streamer (port 8080)
│   │   └── static/
│   │       ├── index.html                     # Mission control dashboard (controls, 2D canvas, leaderboard)
│   │       ├── app.js                         # 2D canvas renderer, REST client & WebSocket listener
│   │       └── style.css                      # Modern dark-mode dashboard styling
│   │
│   ├── racer/                                 # LAYER 1: Low-Level Simulator Driver & Track Manager
│   │   ├── __init__.py                        # Exports RaceTrack, Racer, TelemetrySnapshot, TrajectoryLogger
│   │   ├── track.py                           # RaceTrack Manager (geometry, checkpoints, fleet stepping & kill controls)
│   │   ├── racer.py                           # Racer class (Socket.IO server, lockstep step/reset, kill process, profiler)
│   │   └── telemetry.py                       # Complete raw telemetry data model, derived physics metrics, and CSV logger
│   │
│   ├── env/                                   # LAYER 2: Gymnasium RL Environment
│   │   ├── __init__.py                        # Exports AutoDriveEnv
│   │   ├── autodrive_env.py                   # gym.Env implementation wrapping RaceTrack / Racer
│   │   ├── rewards.py                         # Configurable reward shaping & penalty formulas
│   │   └── spaces.py                          # Observation and action space definitions
│   │
│   └── models/                                # LAYER 3: RL Policies & Feature Extractors
│       ├── __init__.py
│       └── feature_extractor.py               # 1D-CNN LiDAR + MLP Kinematics feature extractor for SB3
│
├── logs/                                      # Exported CSV trajectories & training checkpoints
└── tests/                                     # Verification & Diagnostics
    ├── test_driver.py                         # Standalone test for Layer 1 (connectivity, lockstep, speed)
    ├── test_telemetry.py                      # Math unit tests (quaternions, slip angles, Frenet projection)
    └── check_gym_env.py                       # Gymnasium compliance checker (SB3 check_env)
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

## 5. Layer 1: The `racer/` Driver & Track Manager

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

1. **`src/racer/track.py` (`RaceTrack` Manager)**:
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

2. **`src/racer/racer.py` (`Racer` Class)**:
   * **Parameters**: `racer_id`, `port`, `simulator_path`, `auto_launch`, `headless`, `step_timeout`.
   * **Attributes**: `telemetry` (`TelemetrySnapshot`), `logger` (`TrajectoryLogger`), profiler metrics (`step_latency_ms`, `steps_per_second`, `real_time_factor`).
   * **Methods**:
     * `_start_server()`: Starts background gevent Socket.IO server on `0.0.0.0:port`.
     * `step(throttle, steering)`: Sends clamped commands, waits for next `'Bridge'` frame (turn-based lockstep), updates profiler, and returns `TelemetrySnapshot`.
     * `reset()`: Emits `V1 Reset: True`, confirms teleportation and collision counter clearing, settles, and returns initial state.
     * `kill()`: Terminates the child OS process (`subprocess.Popen.terminate()` / `.kill()`), closes the Socket.IO server socket, and releases port and memory.
     * `save_trajectory(file_path)`: Exports this vehicle's trajectory history to CSV.

3. **`src/racer/telemetry.py`**:
   * **`TelemetrySnapshot`**:
     * 100% raw data: position $(x, y, z)$, orientation quaternion $(x, y, z, w)$, linear/angular velocities, linear acceleration, wheel encoders (left/right), 1,080 LiDAR beams, scan rate, actuator states, lap stats, collisions.
     * Derived metrics: `true_speed`, `heading_yaw`, `v_long`, `v_lat`, `slip_angle`, `lateral_g`.
   * **`TrajectoryLogger`**: In-memory buffer logging step, timestamp, $(x, y, z)$, speed, slip angle, throttle, steering, encoders, and lap stats; exports to CSV.

4. **`src/racer/__init__.py`**:
   * Exports `RaceTrack`, `Racer`, `TelemetrySnapshot`, and `TrajectoryLogger`.

---

## 6. Layer 2: Gymnasium Environment (`src/env/autodrive_env.py`)

Layer 2 wraps `RaceTrack`, conforming to the standard Farama Gymnasium API (`gym.Env`).

* **Observation Space**: `Dict` space consisting of:
  * `"lidar"`: Normalized LiDAR array $[0.0, 1.0]$ with configurable downsampling (1080, 540, 270, 108).
  * `"state"`: Scaled kinematic vector $[v_{\text{long}}, v_{\text{lat}}, \omega_z, a_{\text{long}}, a_{\text{lat}}, \beta, \text{prev\_throttle}, \text{prev\_steering}]$.
* **Action Space**: Continuous `Box(low=-1.0, high=1.0, shape=(2,))` for `[throttle, steering]`.
* **Frame Skipping**: Configurable `frame_skip` (default: 2 ticks = $20\text{ Hz}$ control frequency from $40\text{ Hz}$ physics).
* **Reward Shaping**: Progress along track heading, lap bonus (verified by checkpoints), collision penalty, excessive slip angle penalty, and steering smoothness penalty.
* **Termination & Truncation**: Terminated on collision; truncated on step timeout.

---

## 7. Verification Plan

1. **Unit Tests (`tests/test_telemetry.py`)**:
   * Verify quaternion-to-yaw conversions, body-frame velocity rotations, slip angles, and Frenet $(s, d)$ coordinates.
2. **Driver Test Script (`tests/test_driver.py`)**:
   * Test Socket.IO connection, lockstep execution, single-car and fleet stepping, speed profiler metrics, `kill_racer()`, and `track.save_all_trajectories()`.
3. **Gymnasium Compliance Checker (`tests/check_gym_env.py`)**:
   * Run Stable-Baselines3 `check_env(env)` to validate observation/action spaces and reset contracts.
4. **Mission Control UI Smoke Test**:
   * Launch `src/ui/app.py`, open `http://localhost:8080`, verify that slider controls work, canvas renders track, and Start/Stop/Kill buttons orchestrate background simulator processes.
5. **Docker Build & Run Verification**:
   * Run `docker compose up --build` and verify GPU detection (`nvidia-smi` inside container) and web dashboard accessibility on port 8080.
