# AutoDRIVE RoboRacer Autonomous Sim-Racing: Architecture & Implementation Plan

This document synthesizes the architecture, hardware considerations, and step-by-step roadmap for training a competitive Reinforcement Learning (RL) model for the AutoDRIVE RoboRacer / F1TENTH sim-racing time trials.

---

## 1. System Overview & Hardware Context

* **Host Platform**: Windows 11 with WSL2 (Ubuntu distro installed).
* **Compute Hardware**: NVIDIA GeForce RTX 3060 (12 GB VRAM), CUDA Driver 12.7.
* **Prerequisite to Install**: Docker Desktop for Windows with the WSL2 backend enabled.
* **Core Philosophy**: Pure Reinforcement Learning directly in the simulator (no sim-to-sim transfer, no warm start), utilizing real-time vehicle telemetry and 1080-beam LiDAR.

---

## 2. Two-Container Architecture

The system is partitioned into two distinct containers to isolate dependencies, mirror real-world autonomous vehicle architecture, and produce a submission-ready agent:

```
┌────────────────────────────────────────────────────────┐
│               CONTAINER 1: SIMULATOR                   │
│  - Ubuntu 22.04 base                                   │
│  - AutoDRIVE Simulator Linux binary (.x86_64)          │
│  - Headless execution (-batchmode -nographics / xvfb)  │
│  - Socket.IO Client (communicates over port 4567+)     │
└──────────────────────────▲─────────────────────────────┘
                           │ Socket.IO (Local Network Bridge)
                           │ Telemetry: LiDAR, IMU, Odometry, Race Stats
                           │ Commands: Throttle, Steering, Reset
┌──────────────────────────▼─────────────────────────────┐
│               CONTAINER 2: BRAIN (API / RL)            │
│  - PyTorch with NVIDIA CUDA GPU acceleration           │
│  - Custom Gymnasium Environment (Socket.IO Server)     │
│  - 1D-CNN + PPO Policy Network (Stable-Baselines3)     │
│  - Multi-instance Vectorized Environment Worker        │
│  - Lightweight 2D Top-Down Web Monitor (Port 8080)     │
└────────────────────────────────────────────────────────┘
```

---

## 3. Gymnasium Environment Design

### A. Communication Protocol
* The AutoDRIVE Simulator connects as a Socket.IO client to the Python bridge at port `4567`.
* On every physics step, Unity emits `'Bridge'` with vehicle telemetry.
* The Gym environment replies synchronously with:
  ```python
  {'V1 Throttle': str(throttle), 'V1 Steering': str(steering), 'V1 Reset': str(reset)}
  ```

### B. Observation Space
A unified vector/tensor combining:
1. **Planar LiDAR**: 1,080 continuous range readings ($0.06\text{ m}$ to $10.0\text{ m}$ normalized to $[0, 1]$).
2. **Kinematics / Speed**: Longitudinal velocity ($v_x$), lateral slide velocity ($v_y$).
3. **IMU**: Yaw rotational rate ($\omega_z$), longitudinal & lateral acceleration ($a_x, a_y$).
4. **Action Feedback**: Previous steering angle and throttle command (prevents steering chatter/instability).

> [!NOTE]
> **Why 1D Convolution over 2D?**
> The 2D LiDAR emits a 1D sequence across angular indices ($[-135^\circ, +135^\circ]$). A 1D Convolutional layer (`Conv1d`) treats this as an angular panorama, identifying walls, gaps, and corner gradients with zero discretization loss and sub-millisecond GPU inference time.

### C. Action Space
Continuous action space: `Box(low=-1.0, high=1.0, shape=(2,), dtype=float32)`:
* `action[0]`: Steering angle $[-1.0, 1.0]$ mapped to max physical wheel angle.
* `action[1]`: Throttle / Brake $[-1.0, 1.0]$ (positive = acceleration, negative = braking).

### D. Step & Reset Mechanics
* **`reset()`**: Sends `'V1 Reset': 'True'` $\to$ Unity teleports the car to the starting grid and resets collision flags.
* **`step(action)`**: Executes commands for $k$ ticks (frame-skipping of 2 to 4 ticks recommended for stable action duration).
* **Termination Criteria**:
  * Collision: `data['V1 Collisions'] > 0` $\to$ Terminate episode with penalty.
  * Timeout / Truncation: Maximum allowed time steps exceeded.
* **Reward Shaping**:
  * Positive: Progress along the track centerline (or forward velocity along the track tangent).
  * Lap Completion: Large bonus on `lap_count` increment.
  * Penalties: Large negative reward (e.g., $-100$) on collision; minor penalty on rapid steering jerk.

---

## 4. Multi-Instance Parallel Acceleration (Vectorized Envs)

Because Unity's physics can become unstable if artificially forced to run at 10x single-thread clock speed, acceleration will be achieved via **parallel headless instances**:

* AutoDRIVE accepts `-port <PORT>`.
* Launch $N$ simulator instances simultaneously:
  * Instance 1: Port `4567`
  * Instance 2: Port `4568`
  * Instance 3: Port `4569`
  * Instance 4: Port `4570`
* Stable-Baselines3 `SubprocVecEnv` connects to all $N$ instances simultaneously.
* **Result**: $4\times$ to $8\times$ data collection speedup with zero physics degradation, fully utilizing the 12 GB VRAM on your RTX 3060.

---

## 5. Lightweight 2D Top-Down Web Monitor

To monitor training without running heavy 3D rendering:
* A tiny background thread inside Container 2 receives $(x, y, \text{yaw})$ and LiDAR scan endpoints.
* Broadcasts at 15–20 FPS over a WebSocket to an HTML5 canvas (`http://localhost:8080`).
* **Visuals**:
  * 2D top-down track contour.
  * Vehicle bounding box with heading indicator.
  * Dynamic LiDAR laser fan hitting track walls.
  * Real-time HUD: Speed, steering, lap time, episode count, reward.
* **Resource impact**: $< 1\%$ CPU, closing the browser tab results in 0% UI overhead.

---

## 6. ForzaETH & Future Roadmap Considerations

* **Phase 1 (Current Focus)**: Pure End-to-End PPO with 1D-CNN on raw LiDAR + IMU.
* **Phase 2 (Benchmark Comparison)**: Implement a classical **Follow the Gap (FTG)** controller to set an initial baseline lap time.
* **Phase 3 (ForzaETH / TC-Driver Architecture)**:
  * Map the track via SLAM / waypoints.
  * Compute minimum-time trajectory (apex optimization).
  * Condition the RL policy on following the optimal raceline under extreme tire slip angles.

---

## 7. Implementation Milestones

1. **Layer 1: Fleet Management & API (Completed)**
   * Implemented custom object-oriented `Racer` and `RaceTrack` Socket.IO servers (`src/racer`).
   * Supported multi-port asynchronous simulator communication and telemetry logging.
   * *Status:* The Python logic perfectly wraps AutoDRIVE's API. However, current execution is **BLOCKED** due to a fatal bug in the specific `AutoDRIVE Simulator.exe` (2022.3.52f1) release being used, where the simulator drops the WebSocket and halts its physics loop after sending a single frame. Until an un-bugged/ML-Agents-free build of the simulator is swapped in, visual or headless execution will time out.
2. **Gymnasium Environment Development (Pending Simulator Fix)**:
   * Wrap Layer 1 into an `AutoDriveEnv(gym.Env)` with Gymnasium interface.
   * Parse 1080-ray LiDAR and IMU observations.
3. **2D Localhost Preview (Pending Simulator Fix)**:
   * Build the lightweight HTML5 canvas monitor on port `8080`.
4. **PPO Training Pipeline (Pending Simulator Fix)**:
   * Implement 1D-CNN feature extractor in PyTorch.
   * Scale to 4-way vectorized `SubprocVecEnv` and train on RTX 3060.
