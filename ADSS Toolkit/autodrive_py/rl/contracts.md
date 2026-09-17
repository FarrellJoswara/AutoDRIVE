# AutoDRIVE F1TENTH RL — Interface Contracts

```yaml
contracts_version: "2.0.0"
```

## 1. Purpose

This document is the **interface lock** for the AutoDRIVE F1TENTH / RoboRacer RL stack under `autodrive_py/rl/`.

Phases **3–6** (Gym FTG, Gym PPO + registry, AutoDRIVE Bridge, cross-eval) **must** implement observation, action, reward bookkeeping, episode termination semantics, metrics schema, and model artifact layout as specified here.

**Breaking change rule:** any incompatible change to obs/action shapes, discrete bin mappings, metrics field meanings, or artifact paths **requires** a `contracts_version` bump (SemVer: MAJOR for breaks, MINOR for additive non-breaking fields, PATCH for clarifications). Downstream code that reads `config.json` / `metrics.json` must refuse to load artifacts whose major version does not match the runtime contracts major.

**v2.0.0 vs v1.0.0:** observation width grew from `N_lidar + 2` → `N_lidar + 6` (speed + compact IMU). Artifacts / zips trained under `1.x` are **obsolete** and must not be loaded against a v2 env.

---

## 2. Observation space (LiDAR + legal proprioception)

### 2.1 Design intent

- Race-time observation must be buildable from **non-restricted** sensors only.
- **LiDAR ranges** remain the primary modality.
- **Previous action**, **speed** (encoders / velocity proxy), and a **compact IMU** are always included so later sensor wiring does **not** force an input-layer reshape for those channels.
- **Camera is out of contract** (legal in competition but deferred). Do **not** add raw pixels or a large `vision_pad` of zeros. Vision = future **MAJOR** contracts bump and/or a separate head.
- Prefer **excluding** restricted ground-truth topics (global pose, map occupancy as GT, opponent GT, etc.) from the observation. Reward shaping may use GT **during training only** (see §4.5).

### 2.2 Downsampled LiDAR

| Field | Contract |
| ----- | -------- |
| Source | Raw 2D LiDAR range array from gym or AutoDRIVE Bridge |
| Target length `N_lidar` | **180** (default). **240** allowed only if documented in that run’s `config.json` as `"n_lidar": 240` |
| Dtype | `float32` |
| Preprocess | (1) downsample / interpolate to `N_lidar` beams in scan order (left→right or as provided by backend; order must be consistent within a run); (2) clip ranges to **`[0, 10]` meters**; (3) normalize by dividing by `10.0` → **`[0, 1]`** |
| Invalid / NaN / Inf | Replace with `10.0` **before** normalize (treat as max range) |

Pseudo:

```text
r = downsample(raw_ranges, N_lidar)          # float32
r = clip(r, 0.0, 10.0)
r = where(isfinite(r), r, 10.0)
lidar_obs = r / 10.0                         # shape (N_lidar,), ∈ [0, 1]
```

### 2.3 Previous action

| Field | Contract |
| ----- | -------- |
| Contents | 2 floats: `(throttle, steering)` |
| Encoding | Continuous values in **`[-1, 1]`** obtained by decoding the **last** `MultiDiscrete` action via §3 tables |
| At `reset()` | `(0.0, 0.0)` |
| Dtype | `float32` |

Do **not** feed raw MultiDiscrete indices into the observation; always decode first.

Actuator feedback distinct from the commanded prev action is **not** duplicated in v2 (prev action is the clean temporal context).

### 2.4 Speed (always ON)

| Field | Contract |
| ----- | -------- |
| Contents | 1 float: forward speed proxy |
| Gym | Kinematic bicycle state `v` (m/s) |
| Bridge | Wheel-encoder angle delta → `\|ω\| * wheel_radius` (or equivalent encoder speed), documented in adapter |
| Normalize | `speed_norm = clip(v_mps / SPEED_MAX_MPS, 0, 1)` with **`SPEED_MAX_MPS = 6.0`** (must match env `v_max` / config) |
| Range in obs | **`[0, 1]`** |
| At `reset()` | `0.0` |

Restricted IPS / global pose must **not** be the race-time speed source.

### 2.5 Compact IMU (always ON, dim = 3)

| Index (within IMU slice) | Name | Phys units (pre-norm) | Normalize |
| ------------------------ | ---- | --------------------- | --------- |
| 0 | `yaw_rate` | rad/s | `clip(ω_z / 10.0, -1, 1)` — `YAW_RATE_MAX_RAD_S = 10.0` |
| 1 | `ax` | m/s² longitudinal | `clip(ax / 10.0, -1, 1)` — `ACCEL_MAX_MPS2 = 10.0` |
| 2 | `ay` | m/s² lateral | `clip(ay / 10.0, -1, 1)` |

| Backend | Source |
| ------- | ------ |
| Gym | Bicycle proxies: `yaw_rate = (v/L)*tan(δ)`; `ax = Δv/dt`; `ay = v * yaw_rate` |
| Bridge | `F1TENTH.angular_velocity[2]`, `F1TENTH.linear_acceleration[0]`, `[1]` |

`az`, `wx`, `wy` are **not** in the v2 vector (adding them is a future MAJOR bump). At `reset()`, IMU slice is zeros.

### 2.6 Total observation dimension

**Baseline (v2.0.0):**

```text
obs_dim = N_lidar + 2 + 1 + 3
#       = N_lidar + 6
# default: 180 + 6 = 186
# alt:     240 + 6 = 246
```

### 2.7 Gymnasium `Box` (concatenated)

Observation vector layout (row-major, 1-D):

```text
obs = concat([
  lidar_obs[0:N_lidar],   # [0, 1]
  prev_throttle,          # [-1, 1]
  prev_steering,          # [-1, 1]
  speed_norm,             # [0, 1]
  yaw_rate_n, ax_n, ay_n  # [-1, 1] each
])
# indices:
#   0 .. N_lidar-1
#   N_lidar, N_lidar+1
#   N_lidar+2
#   N_lidar+3 .. N_lidar+5
```

**Bounds:**

| Slice | Indices | low | high |
| ----- | ------- | --- | ---- |
| LiDAR | `0 : N_lidar` | `0.0` | `1.0` |
| Prev throttle | `N_lidar` | `-1.0` | `1.0` |
| Prev steering | `N_lidar + 1` | `-1.0` | `1.0` |
| Speed | `N_lidar + 2` | `0.0` | `1.0` |
| IMU (3) | `N_lidar + 3 : N_lidar + 6` | `-1.0` | `1.0` |

```python
from rl.observation import observation_bounds
from gymnasium.spaces import Box

low, high = observation_bounds(N_lidar=180)
observation_space = Box(low=low, high=high, dtype=np.float32)
# shape: (186,) for N_lidar=180
```

**Camera:** not part of `observation_space`. No zero `vision_pad` in v2.

---

## 3. Action space

### 3.1 Discrete space (in contract)

```python
from gymnasium.spaces import MultiDiscrete

action_space = MultiDiscrete([4, 11])
# action[0] → throttle bin index ∈ {0, 1, 2, 3}
# action[1] → steering bin index ∈ {0, 1, ..., 10}
```

Continuous `Box` control is **out of contract for v1** (do not train/serve Box policies under this version).

### 3.2 Throttle bins (4)

Exact mapping index → continuous throttle in `[-1, 1]`:

| Index `i` | Throttle |
| --------- | -------- |
| 0 | `0.00` |
| 1 | `0.33` |
| 2 | `0.66` |
| 3 | `1.00` |

```python
THROTTLE_BINS = np.array([0.00, 0.33, 0.66, 1.00], dtype=np.float32)

def decode_throttle(i: int) -> float:
    return float(THROTTLE_BINS[int(i)])
```

Notes:

- v1 throttle is **non-negative only** (no reverse). Values stay in `[0, 1] ⊂ [-1, 1]`.
- Changing bin count or values is a **breaking** change.

### 3.3 Steering bins (11)

Exact mapping: `numpy.linspace(-1.0, 1.0, 11)`:

| Index `j` | Steering |
| --------- | -------- |
| 0 | `-1.0` |
| 1 | `-0.8` |
| 2 | `-0.6` |
| 3 | `-0.4` |
| 4 | `-0.2` |
| 5 | `0.0` |
| 6 | `0.2` |
| 7 | `0.4` |
| 8 | `0.6` |
| 9 | `0.8` |
| 10 | `1.0` |

```python
STEERING_BINS = np.linspace(-1.0, 1.0, 11, dtype=np.float32)

def decode_steering(j: int) -> float:
    return float(STEERING_BINS[int(j)])
```

### 3.4 Decode API (shared by FTG wrappers, PPO, Bridge)

```python
def decode_action(action) -> tuple[float, float]:
    """MultiDiscrete [throttle_idx, steering_idx] → (throttle, steering) in [-1, 1]."""
    throttle = decode_throttle(action[0])
    steering = decode_steering(action[1])
    return throttle, steering
```

After each env step, set `prev_action = decode_action(action)` for the next observation.

Bridge / AutoDRIVE command fields (`throttle_command`, `steering_command`) receive these decoded floats.

---

## 4. Reward terms (collision-first)

Reward is a **sum of terms** per step. Defaults below are normative for comparable training; coefficients may be tuned in `config.json` but **collision penalty magnitude and termination-on-collision** are fixed for fair metrics (§6 `adjusted_time` assumes collision cost **10**).

### 4.1 Terms

| Term | Sign | Default | Definition |
| ---- | ---- | ------- | ---------- |
| Progress | `+` | `w_progress = 1.0` | Forward progress along track: encoder delta and/or pose arc-length delta **in gym**. Prefer centerline / Frenet `s` increase when available. |
| Speed | `+` | `w_speed = 0.01` | Small bonus proportional to speed (m/s or normalized speed proxy), clipped to a reasonable max to avoid reward hacking. |
| Wall proximity | `−` | `w_wall = 0.1` | If `min(raw_lidar_m) < d_wall` where `d_wall = 0.3` **meters before normalize**, apply penalty (e.g. `w_wall * (d_wall - min_range)` or flat `w_wall`). Threshold is on **meters**, not normalized `[0,1]`. |
| Collision | `−` | **`10.0`** | On collision event: add `-10.0` and **terminate** the episode (§5). |
| Time cost | `−` | `w_time = 0.001` | Small per-step cost to discourage stalling. |

```text
r_t = (
    + w_progress * progress_delta
    + w_speed    * speed
    - wall_penalty          # 0 if min_lidar_m >= 0.3
    - 10.0                  # only on collision step
    - w_time
)
```

### 4.2 Collision detection

- **Gym:** collision flag / contact from `f1tenth_gym` (or equivalent).
- **AutoDRIVE:** Bridge-exposed collision / contact signal when available; otherwise a conservative proxy (e.g. `min(raw_lidar_m) < d_collide` with `d_collide ≤ 0.15` m) may be used **only if documented** in that backend’s adapter — prefer native collision when present.

### 4.3 Wall proximity default

```text
d_wall = 0.3   # meters, applied to raw (pre-normalize) min range
```

### 4.4 Competition scoring vs training reward

Training reward (§4.1) shapes learning. **Reported** competition-style score uses:

```text
adjusted_time = lap_time_seconds + 10.0 * collision_count
```

Do not conflate dense reward returns with `adjusted_time`.

### 4.5 Ground-truth in reward (training only)

Gymnasium training **may** use ground-truth pose, centerline progress, and collision flags for **reward shaping and termination**.  

At **race / AutoDRIVE eval** time, reward shaping that depends on restricted GT is optional/unavailable; metrics (§6) must still be computable from allowed timing + collision counts. Policies must not require GT observations (§2).

---

## 5. Episode termination

An episode ends when any of the following is true:

| Condition | Default | Notes |
| --------- | ------- | ----- |
| **Collision** | always on | Terminate on the collision step after applying `-10.0` reward |
| **Timeout** | **60 s** wall/sim time **or** `max_steps` | Whichever the env configures; both must be documented in `config.json`. Suggested `max_steps` ≈ `ceil(60 / dt)` for the control rate used |
| **Lap complete** | optional | If lap detection is available (start/finish crossing / progress wrap), terminate successfully; record lap time |

`truncated` / info keys (informative, not frozen as Gym API names beyond common practice):

- `terminated`: collision or lap complete (task end)
- `truncated`: timeout / step limit
- `info["collision"]`: bool
- `info["lap_time"]`: float seconds if lap completed
- `info["timeout"]`: bool

---

## 6. Metrics schema (`metrics.json`)

Written per run under `rl/models/<run_id>/metrics.json`. Field meanings are frozen for `contracts_version` `2.0.0`.

### 6.1 Required fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `contracts_version` | string | Must be `"2.0.0"` for artifacts produced under this contract |
| `run_id` | string | Unique run directory name / id |
| `policy` | string enum | `"ftg"` \| `"ppo"` |
| `backend` | string enum | `"gym"` \| `"autodrive"` |
| `mean_lap_time` | number \| null | Mean completed-lap time in seconds; `null` if no lap completed |
| `total_collisions` | integer | Sum of collisions across evaluated episodes |
| `adjusted_time` | number \| null | `mean_lap_time + 10.0 * total_collisions` when `mean_lap_time` is not null; else `null` (or define per-episode aggregate in notes — default: null if no lap) |
| `n_episodes` | integer | Number of evaluation (or reported) episodes |
| `tracks_eval` | array of string | Track / map identifiers evaluated |
| `timestamp` | string | ISO-8601 UTC timestamp of metrics write |

### 6.2 Recommended optional fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `mean_return` | number | Mean episodic return (training reward) |
| `success_rate` | number | Fraction of episodes with lap complete and no collision |
| `seed` | integer | Eval seed |
| `n_lidar` | integer | `180` or `240` |
| `dt` | number | Control timestep seconds |
| `timeout_s` | number | Episode timeout used |

### 6.3 Example

```json
{
  "contracts_version": "2.0.0",
  "run_id": "20260316_ppo_gym_porto",
  "policy": "ppo",
  "backend": "gym",
  "mean_lap_time": 28.4,
  "total_collisions": 2,
  "adjusted_time": 48.4,
  "n_episodes": 5,
  "tracks_eval": ["porto", "berlin"],
  "timestamp": "2026-03-16T21:00:00Z",
  "n_lidar": 180,
  "seed": 0
}
```

`adjusted_time` identity:

```text
adjusted_time = mean_lap_time + 10.0 * total_collisions
# example: 28.4 + 10.0 * 2 = 48.4
```

---

## 7. Model artifact layout

```text
rl/models/<run_id>/
  best_model.zip      # SB3 / compatible policy zip (PPO); FTG may omit or store stub metadata only
  config.json         # hyperparameters, N_lidar, bin tables ref, tracks, contracts_version
  metrics.json        # schema in §6
  train_tracks.txt    # one track id / path per line used in training (eval-only runs: may list eval tracks)

rl/models/leaderboard.csv
```

### 7.1 `config.json` (minimum)

Must include at least:

```json
{
  "contracts_version": "2.0.0",
  "run_id": "20260316_ppo_gym_porto",
  "policy": "ppo",
  "backend": "gym",
  "n_lidar": 180,
  "obs_dim": 186,
  "action_space": "MultiDiscrete([4, 11])",
  "throttle_bins": [0.0, 0.33, 0.66, 1.0],
  "steering_bins": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "obs_include_speed": true,
  "obs_include_imu": true,
  "imu_dim": 3,
  "speed_max_mps": 6.0,
  "timeout_s": 60.0
}
```

### 7.2 `leaderboard.csv`

Header (column order fixed):

```text
run_id,policy,backend,adjusted_time,mean_lap_time,total_collisions,n_episodes,timestamp,contracts_version
```

Rows appended/updated by compare utilities (Phase 4+). Lower `adjusted_time` is better when present.

### 7.3 Path root

All paths above are relative to `ADSS Toolkit/autodrive_py/` (i.e. sibling of package usage: `autodrive_py/rl/models/...`).

---

## 8. Contract version

```yaml
contracts_version: "2.0.0"
```

| Version | Status |
| ------- | ------ |
| `1.0.0` | Obsolete — LiDAR 180 + prev action only (`obs_dim = N_lidar + 2`) |
| `2.0.0` | Current — LiDAR + prev action + speed + IMU×3 (`obs_dim = N_lidar + 6`); MultiDiscrete `[4,11]`; collision-first reward; metrics + artifact layout as above |

**Compatibility:** loaders must check `contracts_version`. Same major (`2.x.x`) may add optional metrics/config fields; changing obs dim meaning, bin tables, or `adjusted_time` formula requires **`3.0.0`**. Do not load `1.x` policy zips into a `2.x` env (input size mismatch).
