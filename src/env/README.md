# Layer 2 — Gymnasium environment (`src/env/`)

## Summary

Layer 2 wraps **one** Layer 1 `Racer` as a standard [Gymnasium](https://gymnasium.farama.org/) env so a learner (Layer 3 / PPO) only calls `reset()` / `step()` / `close()`.

**Rules (v1):**

- **1 env = 1 car = 1 port** (no multi-car inside one env)  
- Headless by default; full **1080** LiDAR; `frame_skip=1` (every physics tick; cannot be `0`)  
- Crashes do **not** end the episode (`terminated` stays `False`); `collision_penalty` defaults to `0`  
- Episodes end by **stagnation** truncation (and optional `max_episode_steps` if `> 0`)  

No Unity / Socket.IO code lives here — only Gym spaces, rewards, and orchestration of Layer 1.

---

## Layout

```text
src/env/
├── __init__.py         # AutoDriveEnv, RewardConfig, compute_reward, default_simulator_path
├── autodrive_env.py    # gym.Env: reset / step / close, truncation, info
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
| `RewardConfig` | Weights: `forward_scale`, `collision_penalty`, `slip_penalty`, `steer_jerk_penalty` |
| `compute_reward(...)` | Pure function: facts in → one float out |

Default formula:

```text
r  = forward_scale * v_long          # default scale 1.0
r += collision_penalty               # if collision_event; default 0
r -= slip_penalty * |slip|           # default 0
r -= steer_jerk_penalty * |Δsteer|   # default 0
```

- **`v_long`** — forward speed (m/s) from Layer 1 snapshot  
- **`collision_event`** — env-built: `snap.collision` or `collision_count` increased  
- **`slip_angle`** — from snapshot  
- **Steer jerk** — `|steering − prev_steering|` × weight (not a sensor)  
- **`forward_scale=1`** keeps reward ≈ m/s; raise only when you want speed to dominate  

`@dataclass` on `RewardConfig` only auto-builds a simple weight bag.

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
from src.env import AutoDriveEnv, RewardConfig

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
