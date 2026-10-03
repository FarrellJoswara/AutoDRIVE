# Layer 2 — Gymnasium environment (`src/layer2/`)

## Summary

Layer 2 wraps **one** Layer 1 `Racer` as a standard [Gymnasium](https://gymnasium.farama.org/) env so a learner (Layer 3 / PPO) only calls `reset()` / `step()` / `close()`.

**Rules (v1):**

- **1 env = 1 car = 1 port** (no multi-car inside one env)  
- Headless by default; full **1080** LiDAR; `frame_skip=1` (every physics tick; cannot be `0`)  
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
| `LIDAR_BEAMS = 1080`, `STATE_DIM = 8` | Fixed sizes for v1 |
| `make_action_space()` | `Box(-1, 1, shape=(2,))` → `[throttle, steering]` |
| `make_observation_space()` | Dict: `lidar` ∈ `[0,1]^1080`, `state` ∈ `R^8` (±inf) |
| `snapshot_to_obs(snap, prev_throttle, prev_steering)` | Build the obs dict |
| `_normalize_lidar` | Map ranges to `[0, 1]` |
| `_body_frame_accel` | World accel → body `a_long` / `a_lat` |

**`state` vector (8 floats):**  
`v_long`, `v_lat`, `yaw_rate`, `a_long`, `a_lat`, `slip_angle`, `prev_throttle`, `prev_steering`  

LiDAR is scaled; state is **raw** (state scaling deferred).

### `rewards.py`

| Piece | Role |
|-------|------|
| `RewardConfig` | Weights for route progress, reverse motion, collision, lap, slip, and steering |
| `compute_reward(...)` | Pure function: facts in → scalar PPO reward |
| `compute_reward_components(...)` | Same calculation with each term exposed for telemetry |

Default formula:

```text
r  = route_progress_scale * signed_route_delta_m # mapped route, every step
or forward_scale * v_long             # builtin map fallback
r -= backward_speed_penalty_scale * reverse_distance_m
r += collision_penalty               # default -100 on a new collision
r -= slip_penalty * |slip|           # default 0
r -= steer_jerk_penalty * |Δsteer|   # default 0
```

- **`v_long`** — forward speed (m/s) from Layer 1 snapshot  
- **`collision_event`** — env-built: `snap.collision` or `collision_count` increased  
- **`slip_angle`** — from snapshot  
- **Steer jerk** — `|steering − prev_steering|` × weight (not a sensor)  
- **`forward_scale`** is used only when training without a mapped centerline; mapped tracks use signed route progress

`@dataclass` on `RewardConfig` only auto-builds a simple weight bag.

When `map_id` selects a map with `occupancy/centerline.csv`, Layer 2 projects
each pose onto the nearby route and rewards signed movement along it every step.
Forward route movement earns reward; backward route movement loses reward. No
reward depends on distance from the centerline. Separately, the frontier stays
as a monotonic best-progress marker and still drives the frontier-stagnation
rule, lap accounting, and frontier-speed telemetry. Watch receives both the
moving current-position bar and the stationary-unless-beaten frontier. Builtin
(`none`) training keeps the velocity fallback. Reward component telemetry
reports route progress, body-relative reverse motion, collision, lap, slip,
steering change, and the total separately.

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
| `map_id` | `none` | Map centerline used for route-frontier reward; `none` keeps velocity reward |
| `frontier_stagnation_seconds` | `5.0` | Time without a frontier push before truncating a mapped episode |
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
    reward_config=RewardConfig(forward_scale=1.0, collision_penalty=0.0),
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
