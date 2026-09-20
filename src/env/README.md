# Layer 2 — Gymnasium environment (`src/env/`)

This folder wraps **Layer 1** (`src/racer/`) as a standard [Gymnasium](https://gymnasium.farama.org/) env so Layer 3 (PPO / Stable-Baselines3) can call `reset()` / `step()` without knowing about Unity or Socket.IO.

**Rule:** 1 `AutoDriveEnv` instance = 1 car = 1 Socket.IO port.

---

## Files in this folder

| File | Role |
|------|------|
| [`autodrive_env.py`](autodrive_env.py) | `AutoDriveEnv` — Gym API, frame skip, truncation, wiring to Layer 1 |
| [`spaces.py`](spaces.py) | Action / observation space definitions + `snapshot_to_obs` |
| [`rewards.py`](rewards.py) | `RewardConfig` + `compute_reward` (edit this most often) |
| [`__init__.py`](__init__.py) | Public exports |

---

## Quick answers (the “why” questions)

### What does `@dataclass` do?

Python decorator that turns a class into a simple **data container**.  
`RewardConfig(forward_scale=2.0)` works without writing a manual `__init__`.  
It is only for storing named numbers (weights), not for talking to the sim.

### What is `v_long`?

**Longitudinal velocity** in the car’s body frame: speed along the nose (m/s).

- Positive ≈ driving forward  
- Negative ≈ driving backward  

Computed on Layer 1 `TelemetrySnapshot.v_long` from Unity velocity + heading.  
Primary reward signal: `reward += forward_scale * v_long`.

### Where does `collision_event` come from?

**Not a Unity field.** The env builds it each `step()`:

```text
collision_event = snap.collision  OR  (collision_count increased vs last step)
```

Rewards only see `True` / `False`. Default `collision_penalty=0.0` means a hit does not change the score; stagnation truncation still ends a stuck episode.

### Where does `slip_angle` come from?

Layer 1 `TelemetrySnapshot.slip_angle` — sideslip β (radians) between heading and velocity. Large `|slip|` ≈ sliding sideways. Optional `slip_penalty` weight defaults to `0.0` (off).

### Where does steer-jerk penalty come from?

**Not a sensor.** It is invented in `rewards.py`:

```text
|steering_now − steering_previous| × steer_jerk_penalty
```

Default weight `0.0` (off). Turn it on if you want smoother steering.

### Why floats?

Throttle, steering, speeds, angles, and rewards are **continuous** real numbers. Gymnasium `Box` spaces and NumPy policies use `float32` / Python `float` for that. Integers would quantize the controls for no benefit.

### What is `forward_scale`, and why `1.0` not `100`?

It multiplies forward speed:

```text
reward += forward_scale * v_long
```

| `forward_scale` | If `v_long = 3.0` m/s |
|-----------------|------------------------|
| `1.0` (default) | `+3.0` reward |
| `100.0`         | `+300.0` reward |

`1.0` keeps units interpretable (“reward ≈ m/s forward”). Raise it only when you **want** speed to dominate other terms. Start simple; tune later.

### What is Gym `info`? (fuzzy)

Every `reset` / `step` returns an `info` **dict for humans / logging**, not for the neural net.

- **Policy sees:** `obs` only (`lidar` + `state`)  
- **You see:** `info` (`v_long`, `collision_event`, `truncate_reason`, …)

Stable-Baselines3 callbacks can read `info`; the policy does not unless you later copy fields into `obs`.

### What is “state scaling”? (fuzzy / deferred)

`obs["lidar"]` is already mapped to `[0, 1]`.  
`obs["state"]` (8 kinematics floats) is **raw** (m/s, rad/s, …) with bounds `±inf`.

“State scaling” would mean normalizing those 8 numbers (e.g. divide speed by 10) so network inputs sit in a similar range. **Not done in Layer 2 v1** — marked fuzzy in the plan. LiDAR is scaled; state is not.

---

## `frame_skip` vs `max_episode_steps` (why not default `0`?)

These are **different knobs**. Mixing them up is common.

### `frame_skip` (default **`1`**, minimum **`1`**)

How many **physics ticks** reuse the **same** action inside one `env.step()`.

| Value | Meaning |
|-------|---------|
| `1` | Act every tick (~40 Hz bridge) — Layer 2 v1 default |
| `2` | Same action held for 2 ticks (coarser control) |
| `0` | **Invalid** — would run zero physics per step |

So the default is **not** `0`. “No skipping” = `frame_skip=1`.

### `max_episode_steps` (default **`0`** = disabled)

Hard cap on env steps per episode.

| Value | Meaning |
|-------|---------|
| `0` | **No hard cap** (default) — only stagnation can truncate |
| `20000` | Truncate after 20 000 env steps even if still moving |

Stagnation (default): `|v_long| < 0.15` for **200** consecutive steps → `truncated=True`.

Crashes do **not** set `terminated` in v1.

---

## Observation and action (from `spaces.py`)

### Action — shape `(2,)`, `float32`, each in `[-1, 1]`

| Index | Name | Meaning |
|------:|------|---------|
| 0 | throttle | `+` accel, `−` brake |
| 1 | steering | left / right (sim convention) |

### Observation — `Dict`

**`lidar`:** `(1080,)` float32 in `[0, 1]` (near → far).

**`state`:** `(8,)` float32, **not** rescaled in v1:

| Index | Field | Units |
|------:|-------|-------|
| 0 | `v_long` | m/s |
| 1 | `v_lat` | m/s |
| 2 | `yaw_rate` | rad/s |
| 3 | `a_long` | m/s² |
| 4 | `a_lat` | m/s² |
| 5 | `slip_angle` | rad |
| 6 | `prev_throttle` | [-1, 1] |
| 7 | `prev_steering` | [-1, 1] |

---

## Rewards (from `rewards.py`)

```text
r  = forward_scale * v_long
r += collision_penalty          # only if collision_event (default 0)
r -= slip_penalty * |slip|      # default 0
r -= steer_jerk_penalty * |Δsteer|  # default 0
```

### `RewardConfig` defaults

| Field | Default | Notes |
|-------|--------:|-------|
| `forward_scale` | `1.0` | Main signal |
| `collision_penalty` | `0.0` | Set e.g. `-5.0` for a wall tax |
| `slip_penalty` | `0.0` | Off |
| `steer_jerk_penalty` | `0.0` | Off |

### Adding a new reward term later

1. Add a weight on `RewardConfig` (default `0.0` if unused).  
2. Pass any new fact into `compute_reward`.  
3. Have `AutoDriveEnv.step` read it from telemetry (or compute it).  
4. Add one line in `compute_reward`.

---

## `AutoDriveEnv.__init__` parameters

| Parameter | Default | Meaning |
|-----------|--------:|---------|
| `simulator_path` | auto | Unity binary; see Docker section |
| `port` | `4567` | Socket.IO port for this car |
| `auto_launch` | `True` | Spawn Unity via Layer 1 |
| `headless` | `True` | No visible window (Docker / servers) |
| `frame_skip` | `1` | Physics ticks per env step (`>= 1`) |
| `max_episode_steps` | `0` | `0` = unlimited; `>0` = hard cap |
| `stagnation_speed_threshold` | `0.15` | m/s idle threshold |
| `stagnation_steps` | `200` | Idle steps before truncate |
| `connect_timeout` | `60.0` | Seconds to wait for Unity |
| `reward_config` | `RewardConfig()` | Weights |
| `lidar_beams` | `1080` | Locked in v1 |
| `racer` | `None` | Injected Racer (tests); env won’t kill it |

Path resolution when `simulator_path=None`:

1. Env var `AICAR_SIMULATOR_PATH`  
2. First existing of: Windows `.exe`, `simulator/AutoDRIVE Simulator.x86_64`, `/app/simulator/...`  
3. Platform fallback  

---

## Docker

The Compose stack mounts `./simulator` → `/app/simulator` and runs Xvfb (`DISPLAY=:99`) via `docker/entrypoint.sh`.

**Linux binary expected at:**

```text
/app/simulator/AutoDRIVE Simulator.x86_64
```

(or set `AICAR_SIMULATOR_PATH` in `docker-compose.yml`).

`AutoDriveEnv` defaults to `headless=True` and auto-picks the Linux path inside the container. Ports: each parallel env needs its **own** `port` (and Compose `ports:` if you expose them).

Example compose env snippet:

```yaml
environment:
  - AICAR_SIMULATOR_PATH=/app/simulator/AutoDRIVE Simulator.x86_64
  - DISPLAY=:99
```

Host Windows training (no Docker): leave defaults; it picks  
`simulator/windows/AutoDRIVE Simulator.exe`.

---

## How to use

### Install deps (host)

```bash
pip install -r requirements.txt
```

### Smoke-check spaces (no Unity required)

```bash
python tests/check_gym_env.py
```

### One episode with the live simulator (host Windows)

```python
from src.env import AutoDriveEnv, RewardConfig

env = AutoDriveEnv(
    # simulator_path auto-detected; override if needed:
    # simulator_path=r"simulator\windows\AutoDRIVE Simulator.exe",
    headless=True,
    frame_skip=1,
    max_episode_steps=0,  # stagnation only
    reward_config=RewardConfig(forward_scale=1.0, collision_penalty=0.0),
)

obs, info = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # random; later = PPO policy
    obs, reward, terminated, truncated, info = env.step(action)
    print(info["v_long"], reward, info.get("truncate_reason"))

env.close()
```

### Inside Docker

```bash
docker compose build
docker compose run --rm aicar python -c "from src.env import AutoDriveEnv; e=AutoDriveEnv(); print(e.reset()[0]['lidar'].shape); e.close()"
```

Ensure the Linux `.x86_64` binary is under `./simulator` on the host so the volume mount provides it at `/app/simulator`.

### Turn on a crash tax

```python
env = AutoDriveEnv(
    reward_config=RewardConfig(collision_penalty=-5.0),
)
```

### Optional hard episode length

```python
env = AutoDriveEnv(max_episode_steps=20_000)
```

---

## What Layer 2 intentionally does **not** do (yet)

- Multi-car inside one env (use multiple envs / ports)  
- Waypoints / lap progress rewards  
- LiDAR downsampling  
- State-vector normalization (“state scaling”)  
- PPO training (Layer 3)  
- Ending the episode on crash (`terminated` stays `False`)

See root [`PLAN.md`](../../PLAN.md) §6 for the locked design decisions.
