# Layer 2 — Gymnasium environment (`src/layer2/`)

## Summary

Layer 2 wraps **one** Layer 1 `Racer` as a standard [Gymnasium](https://gymnasium.farama.org/) env so a learner (Layer 3 / PPO) only calls `reset()` / `step()` / `close()`.

**Rules (v1):**

- **1 env = 1 car = 1 port** (no multi-car inside one env)  
- Headless by default; full **1081** LiDAR; `frame_skip=1` (every physics tick; cannot be `0`)
- Crashes end the episode when `terminate_on_collision` is enabled; each new collision receives the configured penalty (default `-100`)
- Episodes end by **stagnation** truncation (and optional `max_episode_steps` if `> 0`)  

No Unity / Socket.IO code lives here — only Gym spaces, rewards, and orchestration of Layer 1.

---

## Layout

```text
src/layer2/
├── __init__.py         # AutoDriveEnv, RewardConfig, compute_reward, default_simulator_path
├── autodrive_env.py    # gym.Env: reset / step / close, truncation, info
├── lap_tracker.py     # closed-centerline lap count, full-width gate, lap timing
├── spaces.py           # action/obs spaces + snapshot → obs dict
├── rewards.py          # RewardConfig + compute_reward (edit this most)
└── README.md           # This file
```

**Data flow**

```text
action [throttle, steer]
        │
        ▼
 AutoDriveEnv.step  ──frame_skip──▶  Racer.step  ──▶ Unity
        │
        ├─ snapshot_to_obs  →  obs { lidar, state }
        ├─ compute_reward   →  reward
        └─ stagnation / max steps → truncated + info
```

---

## Files and essentials

### `__init__.py`

Exports: `AutoDriveEnv`, `RewardConfig`, `compute_reward`, `default_simulator_path`.

### `spaces.py`

| Piece | Role |
|-------|------|
| `LIDAR_BEAMS = 1081`, `STATE_DIM = 9` | Full live scan and normalized sensor/control state |
| `make_action_space()` | `Box(-1, 1, shape=(2,))` → normalized `[throttle, steering]`; `throttle_mode=forward_only` maps normalized throttle onto actuator `[0, 1]` |
| `make_observation_space()` | Dict: `lidar` ∈ `[0,1]^1081`, `state` ∈ `[-5,5]^9` |
| `snapshot_to_obs(snap, prev_throttle, prev_steering)` | Build the obs dict |
| `_normalize_lidar` | Map ranges to `[0, 1]` |
| State scaling | Direct Bridge body-frame velocity/acceleration plus normalized IMU rate and actuator feedback |

**`state` vector (9 normalized floats):**
`v_long`, `v_lat`, `yaw_rate`, `a_long`, `a_lat`, measured throttle, measured steering, previous throttle command, previous steering command.

Velocity and acceleration are consumed from the Bridge's body-frame axes without a second yaw rotation. Speed is scaled by the RoboRacer maximum, acceleration by 1 g, yaw rate by 10 rad/s, and steering feedback by its 30° range. State values are clipped to ±5 after scaling; each channel keeps its own units and scale.

Live scans contain 1081 rays: both endpoints of the -135°..+135° fan at 0.25° spacing. The documented 1080-ray variant is angularly resampled. Positive infinity and sub-minimum echoes are treated as no return and map to max range. Missing, malformed, or non-finite scans fail observation construction instead of being filled with fake zero-distance obstacles.

### `rewards.py`

| Piece | Role |
|-------|------|
| `RewardConfig` | Weights for route progress, reverse motion, collision, time, slip, and steering |
| `compute_reward(...)` | Pure function: facts in → scalar PPO reward |
| `compute_reward_components(...)` | Same calculation with each term exposed for telemetry |

Default formula:

```text
r  = route_progress_scale
     * (1 + min(average_frontier_speed / pace_target, 1)^2)
     * new_frontier_metres
r -= time_penalty_per_second * simulated_seconds # default 5 per second
r -= backward_speed_penalty_scale * reverse_distance_m
r -= collision_penalty_magnitude       # default 100 on collision
r -= positive_episode_return * collision_reward_percent / 100
r -= episode_failure_penalty_magnitude # default 100 on other failed endings
r -= slip_penalty * |slip|           # default 0
r -= steer_jerk_penalty * |Δsteer|   # default 0
```

- **`v_long`** — forward speed (m/s) from Layer 1 snapshot  
- **`collision_event`** — env-built: `snap.collision` or `collision_count` increased  
- **`slip_angle`** — from snapshot  
- **Steer jerk** — `|steering − prev_steering|` × weight (not a sensor)  
- **`forward_scale`** is deprecated and ignored. Raw velocity never earns reward.

`@dataclass` on `RewardConfig` only auto-builds a simple weight bag.

On mapped tracks, Layer 2 projects each pose onto the route. Only a new
high-water frontier advance earns positive per-step reward; recovering or
repeating already-travelled route distance does not. The per-metre reward rises
quadratically with the configured frontier pace source (episode-average pace by
default, or the current action's new-frontier distance divided by acknowledged
simulated duration). Its maximum multiplier is
`1 + frontier_pace_bonus_strength` at the configurable pace target (6 m/s by
default); the default strength of 1 preserves the 2x cap. A cost of 5 reward units per
simulated second also favors faster progress and makes stalling accumulate cost.
Backward body-frame motion is penalized, and a frontier push is withheld if the
car is moving backward relative to its body beyond the reverse deadband. A
collision receives the fixed collision cost plus a percentage of positive
frontier reward accumulated during that car's life. A frontier stall ends and
resets the episode after its configured no-progress interval, but adds no
separate terminal deduction: time cost continues to accrue while progress is
stalled. Other non-collision failed endings retain the configurable
episode-failure cost. Lap completions never change reward or end an episode. No
reward depends on distance from
the centerline. The frontier remains the monotonic progress marker for lap
accounting and visualization. Builtin (`none`) runs have no route frontier, so
they receive no positive driving reward; select a validated route map for this
frontier-based objective. Reward telemetry exposes each term separately.

## Simulator telemetry vs policy input

Layer 1 parses Bridge telemetry into `TelemetrySnapshot`; Layer 2 does not pass
that entire record to PPO. The policy currently receives:

- all 1,080 LiDAR ranges, normalized to `[0, 1]`;
- body-relative forward and lateral velocity, yaw rate, body-relative forward
  and lateral acceleration, sideslip, and the previous throttle and steering
  commands.

These eight state values are derived from raw velocity, orientation, angular
velocity, acceleration, and the prior commands. The snapshot also contains
position, orientation quaternion / heading, world-frame velocity and
acceleration, wheel encoder angles and optional ticks, reported throttle and
steering, lap counters and times, and collision state/count. Those additional
values are available for route projection, logging, reset validation,
termination, or diagnostics, but are not included in PPO's observation. Route
position/frontier and lap progress are explicitly kept out of the policy input.
Encoder fields default to zero if absent from a Bridge packet, so their live
availability should be verified before using them as policy inputs.

Map centerlines are generated from occupancy data and validated by the map
pipeline. Review the generated map preview before treating a new map's lap
times as authoritative. Lap timing is enabled only when the route is closed.
Layer 2 places
the full-width finish gate at the episode's reset pose projected onto the
centerline, and adds `lap_supported`, `lap_count`, `lap_elapsed_s`,
`last_lap_time_s`, `best_lap_time_s`, and `lap_gate` to Gym `info`. An open
centerline still supports route progress but reports lap timing as unavailable.
Do not treat route metrics as representative until the route and widths have
been checked against the drivable course.

### `autodrive_env.py` — `AutoDriveEnv`

| Piece | Role |
|-------|------|
| `default_simulator_path()` | `AICAR_SIMULATOR_PATH` or Windows `.exe` / Docker `.x86_64` |
| `__init__(...)` | Spaces, own or inject `Racer`, wait for connect |
| `reset()` | Layer 1 reset + zero-action ticks → `(obs, info)` |
| `step(action)` | Frame-skip Layer 1 steps → reward, truncated, info |
| `_build_info(...)` | Diagnostics for humans/callbacks — **not** fed to the policy |
| `close()` | `kill()` if this env owns the racer |

**Important init knobs**

| Param | Default | Meaning |
|-------|--------:|---------|
| `frame_skip` | `1` | Physics ticks per env step (`≥ 1`; `0` invalid) |
| `max_episode_steps` | `0` | `0` = no hard cap; `>0` = truncate after N steps |
| `stagnation_speed_threshold` | `0.15` | m/s idle threshold |
| `stagnation_steps` | `200` | Consecutive idle → truncate |
| `map_id` | `none` | Map centerline used for route-frontier reward; frontier reward requires a validated route map |
| `frontier_stagnation_seconds` | `10.0` | Time without a frontier push before resetting a mapped episode |
| `headless` | `True` | Docker / server friendly |
| `reward_config` | defaults | See `rewards.py` |

**`info` (fuzzy):** logging only (`v_long`, `collision_event`, `truncate_reason`, …). Policy sees `obs` only.

---

## Docker notes

Compose mounts `./simulator` → `/app/simulator`. Prefer:

```text
AICAR_SIMULATOR_PATH=/app/simulator/AutoDRIVE Simulator.x86_64
```

`AutoDriveEnv` auto-detects that path when `simulator_path` is omitted. Each parallel env needs its **own** port.

---

## How to use

```python
from src.layer2 import AutoDriveEnv, RewardConfig

env = AutoDriveEnv(
    port=4567,
    headless=True,
    frame_skip=1,
    max_episode_steps=0,
    reward_config=RewardConfig(route_progress_scale=10.0, collision_penalty_magnitude=0.0),
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.5, 0.0])
env.close()
```

Live CLI:

```bash
python scripts/demo.py layer2
python scripts/demo.py check-env   # needs torch + stable-baselines3
```

---

## Out of scope (v1)

Multi-car-in-one-env · waypoints / lap bonuses · LiDAR downsampling · state normalization · PPO (Layer 3) · terminate-on-crash  

See also [PLAN.md §6](../../PLAN.md).
# Official camera observation profile

Official training and deployment may select `official_sensors_camera` to add
the permitted `/autodrive/roboracer_1/front_camera` RGB stream alongside the
existing LiDAR and state inputs. Layer 1 decodes `sensor_msgs/msg/Image`; Layer
2 samples it to a fixed 160×90 RGB tensor and fails closed if a fresh camera
frame is unavailable. Layer 3 uses the `lidar_camera_cnn` extractor. This input
shape is intentionally a separate profile: existing LiDAR-only checkpoints
remain compatible with their original profiles and require a fresh model to
learn from camera images.
