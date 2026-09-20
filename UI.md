# Mission Control UI — Architecture Plan

**Status:** Phases 1–3 implemented (hub + Settings/Train/Live); Phase 6 fleet canvas stubbed.  
**Depends on:** Layer 3 train CLI (`python -m src.layer3.train`) · Docker A/B (`docker/`)  
**Code home (planned):** **`src/layer4/`** — FastAPI hub + Vite/React frontend + shared types/settings  
**Infra:** stays in `docker/` / compose — **not** under `src/`

**Start here** for watch+control of training. **Layers 1–3 = learning path; Layer 4 = Mission Control infra** (hub + web). Dockerfiles/compose stay at repo root.

---

## 1. Goal / non-goals

### Goal

Ship a **Mission Control** surface that:

1. Starts / stops Layer 3 training the **same way the CLI does** (`python -m src.layer3.train …`).
2. Exposes **Settings** as a first-class page (maps 1:1 to train CLI flags).
3. Watches live training metrics **without interfering** with the step loop (pub from train → hub → browser).
4. Runs cleanly inside **brain container B** next to A×N sims.
5. **Fleet canvas** — layered bird's-eye: **map** → **cars** → **what they see** (LiDAR) → **collision X**; fed only by hub telemetry (never a second Racer stepping). Phased: train+telemetry first; fleet viz = Phase 6 once the bus is solid.

### Non-goals

| Out of scope | Scope | Why |
| :--- | :--- | :--- |
| Second Racer / UI-owned `step()` | Forever | Train owns the step loop; UI must not call Layer 1/2 step |
| RaceTrack-driven **train** path (L2/L3) | Forever on train | Train stays `make_vec_env` → `AutoDriveEnv` → one `Racer` per port. Hub must not `RaceTrack.step_all` / launch fleet for learning |
| Redis mandated for Mission Control | v1–fleet viz | FastAPI in-process bus is enough; Redis optional later behind an interface (§13) |
| `live.json` / shared-memory watch | v1 | Prefer hub POST + WebSocket |
| Hyperparameter surgery (LR, net arch) mid-run / beyond CLI | Phase N+ (**not day 1**) | v1 Settings = train CLI only; live LR/arch surgery deferred — see §7 / Phase N+ |
| Docker under `src/` | Forever | Infra stays at repo root (`docker/`, compose) |

**RaceTrack precision:** Keep RaceTrack **off** the default L2/L3 **train** path. RaceTrack **may** be used later as a **fleet display helper** (map geometry, checkpoint gates, aggregating already-published snapshots for the canvas) — never as a second stepper that fights the train child’s envs.

---

## 2. Architecture

> **LOCKED — read first (do not reverse):**
>
> 1. **Vite lives in Docker B (brain).** Docker-first. Host Vite was **Windows convenience only** — not the production or default path. **Prod:** `vite build` static assets into image B; **FastAPI serves them on the same Mission Control port `:8090`**. **Dev:** HMR **inside B**, or a `layer4` compose service on **B’s network**. Details §8 / §9.
> 2. **LiDAR readings are distances only.** To draw rays you need **where beam 0 points** and the **angular step** around the car. That calibration is **not documented in this repo yet** → Phase 6 ships **map + cars + collision X** first; **rays only after** measuring/documenting angles (§6 / Phase 6).
> 3. **Code home = `src/layer4/`** (hub + web). **Full Settings from day 1** (all train CLI flags).

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser  → http://localhost:8090  (single entry: brain B)              │
│  Vite/React ON B (not host-default)  · Settings · Train · Live · Fleet  │
└───────────────┬───────────────────────────────▲─────────────────────────┘
                │ REST (start/stop/status/settings)│ WS /ws (fan-out)
                ▼                                  │
┌─────────────────────────────────────────────────────────────────────────┐
│  Brain B  —  src/layer4/  (Mission Control)                             │
│  FastAPI hub (:8090)  ← same port serves API + Vite static (prod)       │
│  · TrainJob via subprocess.Popen                                        │
│  · TelemetryBus (in-process; Redis-swappable later)                     │
│  · POST /telemetry ← train child                                        │
│  · Prod: vite build → static on :8090 · Dev: HMR in B / layer4 svc      │
│  · GET /health (replaces ready_stub role on :8090)                      │
└───────────────┬─────────────────────────────────────────────────────────┘
                │ Popen: python -m src.layer3.train <argv from Settings>
                │ env: HUB_URL=http://127.0.0.1:8090
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Train child  (owns SB3 learn / step loop)                              │
│  · AutoDriveEnv ↔ Socket.IO ports 4567+  (not RaceTrack on train path)  │
│  · SB3 callback every N steps → POST /telemetry                         │
└───────────────┬─────────────────────────────────────────────────────────┘
                │ connect (Docker: --no-auto-launch)
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  A × N  sim containers  (compose scale)                                 │
│  Unity → brain:4567+                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Locked stack**

| Piece | Choice |
| :--- | :--- |
| Code home | **`src/layer4/`** (hub + web + shared settings/types) — **not** a top-level `ui/` folder |
| Hub | **FastAPI** (+ uvicorn) on brain **B** |
| Job runner | **`subprocess.Popen`** — same entry as CLI |
| Live path | Train **POST** → hub → **WebSocket** fan-out |
| Frontend | **Vite + React + TS in Docker B** — Docker-first; host Vite = Windows convenience only. **Prod:** `vite build` → static in B, FastAPI serves on **`:8090`**. **Dev:** HMR inside B or `layer4` service on B’s network. Plain CSS + vars, native elements, Canvas 2D — **no** Tailwind/Radix/shadcn/Pixi/Three in v1 (§8) |
| Settings | **Full Settings UI from day 1** — all `train.py` CLI flags mapped; first-class React page (not a Start/Stop stub that defers Settings). LR/net-arch surgery = Phase N+, **not day 1** |
| Bus | **In-process TelemetryBus** on FastAPI hub (v1–fleet viz); Redis optional later (§13) |
| Fleet viz | Product goal (Phase 6) — **map + cars + collision X first**; LiDAR **rays after** angle calibration (distances alone are not enough — §6) |

---

## 3. Ownership rules

| Owner | Owns | Must not |
| :--- | :--- | :--- |
| **Train child** | SB3 `learn`, env `step`/`reset`, checkpoints, TensorBoard | Depend on UI process staying alive for learning to proceed (telemetry POST is best-effort) |
| **Hub (TrainJob)** | Start/stop child, argv from Settings, status/PID/log path, TelemetryBus → WS fan-out | Call `env.step` or put `RaceTrack` on the L3 **train** path |
| **UI (browser)** | Settings edit/persist, start/stop, live charts, later fleet canvas | Drive Unity / own a `step()` loop |
| **Compose / Docker** | A×N sims + B image/GPU/ports (incl. Layer 4 Vite build into B) | Live under `src/` |
| **Layer 4 (`src/layer4/`)** | Mission Control hub + web + Settings model | Own learning steps; sit as a top-level `ui/` folder |
| **RaceTrack** | Layer 1 demos today; optional later **fleet display** helper (geometry / snapshot aggregate) | Own learning steps or replace `AutoDriveEnv` on L2/L3 train |

**Fleet viz rule:** canvas **watches hub telemetry only**; train owns `step`; RaceTrack is not on the train path (may help display geometry later). No parallel `Racer` fleet under the UI.

**One active TrainJob** in v1 (reject second `POST /train/start` while running).

---

## 4. API surface

Hub listens on **`:8090`** in container B (same host port already published). It **replaces** `docker/ready_stub.py` as the default brain command once shipped; until then stub remains.

| Method | Path | Who calls | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Compose / ops | Ready check (`OK` / JSON); supersedes stub body |
| `GET` | `/train/status` | Browser | `{state, pid, started_at, argv, exit_code, log_path, last_telemetry?}` |
| `POST` | `/train/start` | Browser | Body = Settings → build argv → Popen; 409 if already running |
| `POST` | `/train/stop` | Browser | Graceful terminate → escalate kill (see §5) |
| `GET` | `/settings` | Browser | Current Settings JSON |
| `PUT` | `/settings` | Browser | Update Settings (persist to disk) |
| `POST` | `/telemetry` | Train child only | Internal ingest; fan-out to WS clients |
| `WS` | `/ws` | Browser | Subscribe to status + telemetry events |

### Suggested event shapes (WS)

```text
{"type": "status",    "payload": { ...same as GET /train/status... }}
{"type": "telemetry", "payload": { ...see §6... }}
{"type": "log",       "payload": {"line": "..."}}   # optional later
```

### Auth (v1)

None on the compose bridge / localhost. Document that `:8090` must not be exposed publicly without a reverse proxy later.

---

## 5. TrainJob (`subprocess.Popen`)

### Start

1. Validate Settings; refuse if job `state == running`.
2. Resolve `out` dir (Settings run name / stamp under `logs/rl/`).
3. Build argv (same flags as CLI — §7).
4. Open log file e.g. `logs/rl/<run>/hub_train.log` (stdout+stderr redirect).
5. `Popen([sys.executable, "-m", "src.layer3.train", *argv], cwd=REPO_ROOT, env={**os.environ, "HUB_URL": hub_url}, …)`.
6. Record `pid`, `started_at`, `argv`, `log_path`; broadcast status on `/ws`.

Working directory = repo / `/app` in container (where `src` and `logs` mount).

### Stop

| Phase | Action | Timeout (suggested) |
| :--- | :--- | :--- |
| Graceful | `terminate()` (SIGTERM) | ~10–15 s |
| Hard | `kill()` (SIGKILL) | immediate after grace |

On child exit (any reason): set `state=exited`, store `exit_code`, and broadcast status. In Docker mode, `stop_sims_on_train_exit=True` stops compose sims through the Docker API to release RAM/CPU while the brain/UI remains available. `stop_stack_on_train_exit=True` instead stops the whole compose project, including Mission Control.

### Status fields

| Field | Notes |
| :--- | :--- |
| `state` | `idle` \| `starting` \| `running` \| `stopping` \| `exited` \| `error` |
| `pid` | OS PID while running |
| `argv` | Exact argv list used |
| `log_path` | Tailable from host via `./logs` volume |
| `exit_code` | Set when process ends |
| `hub_url` | What child sees as `HUB_URL` |

### Windows note

Popen + terminate quirks matter only when running hub **outside** Docker on Windows. **Accepted:** primary path is Linux container B; host-Windows is best-effort.

---

## 6. Telemetry bridge

### Flow

```text
SB3 BaseCallback (rollout thread)
  → sample + downsample  → bounded Queue (drop-oldest)
                            │
                            ▼  daemon publisher thread
                          HTTP POST {HUB_URL}/telemetry
                            → hub stores last sample + broadcasts on /ws
```

No shared memory, no mandated Redis, no `live.json`. Hub ingest lands on an in-process **TelemetryBus** then WS (§13). If hub is down or slow, samples are **dropped**; **training continues**.

### Back-pressure — the callback must never block the step loop

"Best-effort POST" is not enough on its own: a bare `requests.post` in `_on_step` is synchronous, and `requests` defaults to **no timeout**. A *down* hub fails fast (ECONNREFUSED), but a *hung* hub (accepted socket, no response) would stall the rollout loop indefinitely. Required shape:

| Concern | Rule |
| :--- | :--- |
| Thread | One **daemon** publisher thread; `_on_step` only enqueues and returns `True` |
| Queue | `queue.Queue(maxsize=2–4)`, **drop-oldest** on full (`get_nowait()` then `put_nowait()`) — never `put()` blocking |
| Timeout | `requests.Session` (keep-alive) + `timeout=(0.5, 1.0)` connect/read |
| Failure | Catch `Exception`, never re-raise into train; log **once** then rate-limit (~60 s) |
| Breaker | After K consecutive failures, back off 1 s → 30 s so a dead hub is not hammered |
| Data ownership | Downsample / convert to plain Python **on the train thread**; `json.dumps` on the publisher thread. The publisher never touches SB3 buffers |
| Shutdown | `_on_training_end`: sentinel + `join(timeout≈2 s)`; the daemon flag means a stuck POST still cannot block interpreter exit |

Hub side must be non-blocking too: WS fan-out uses a **bounded per-client send queue (drop-oldest)** and disconnects a client that cannot keep up. Otherwise one stalled browser tab back-pressures the broadcast → `POST /telemetry` → the publisher thread. With the bounded train-side queue, that whole chain degrades into dropped frames instead of a slow train.

### Two cadences (they are not the same number)

`telemetry_every_n` alone is wrong for the fleet canvas. At `frame_skip=1` the bridge runs ~40 steps/s, so `telemetry_every_n=100` is **one sample every ~2.5 s** — cars would teleport.

| Stream | Cadence | Default | Why |
| :--- | :--- | :--- | :--- |
| Train metrics (payload v1 below) | **step-based** `telemetry_every_n` | 100–500 | Scalars; cheap; aligns with SB3 logger |
| Fleet sim-state (payload "later" below) | **time-based** `fleet_hz` | ~15 Hz | Motion smoothness; independent of step rate |

`fleet_hz` is a **cap, not a guarantee**: effective rate = `min(fleet_hz, step_rate)`. Implement as `if monotonic() - last >= 1/fleet_hz`. Note PPO pauses stepping during each `n_steps=2048` update, so telemetry has **structural gaps of ~1–3 s**. Do **not** interpolate or extrapolate poses (it would draw physics that never happened) — render the last known pose and show a **stale** indicator when sample age > 500 ms. Fixed-delay interpolation is an optional Phase 6+ nicety, not v1.

### Callback placement

Small helper in Layer 3 (e.g. `src/layer3/hub_callback.py`) registered next to existing `CheckpointCallback` in `train.py` when `HUB_URL` is set. Keep train usable without hub (CLI parity).

### Payload (v1 — train metrics)

| Field | Type | Required | Notes |
| :--- | :--- | :--- | :--- |
| `step` | int | yes | Timesteps so far |
| `reward` | float | yes | Episode or rollout mean (document which) |
| `episode` | int | yes | Episode count if available, else 0 |
| `loss` | float \| null | no | If cheap to read from SB3 logger |
| `checkpoint` | str \| null | no | Latest ckpt path when saved |
| `run_id` | str | yes | Out dir name / stamp |
| `ts` | str | yes | UTC ISO timestamp |

**Publish interval:** every N steps (Settings or fixed default e.g. 100–500); not every env step.

### Payload (later — fleet / LiDAR viz)

Extend the same bus with **sim-state** samples, still published by the train side (or a read-only sidecar that does **not** `step`). Canvas never opens its own Socket.IO ports. **Redis stays no** for this path (§13).

Suggested fields (per sample or per-env array inside one sample):

| Field | Type | Notes |
| :--- | :--- | :--- |
| `env_id` | int | Vec-env index / car id |
| `pose` | `[x, z]` | **Unity ground plane is X–Z** (Y is up). `info["position"]` is `(x, y, z)` — take indices **0 and 2**. Metres |
| `yaw` | float | **Radians**, `TelemetrySnapshot.heading_yaw`. **Not in `info` today** — must be added to `_build_info` |
| `lidar` | list, **downsampled** | See sizing table below. Values are normalised `[0,1]`, not metres |
| `collision` | bool | True → canvas shows crashed glyph (**X**), not normal marker |
| `episode` / `episode_id` | int | Current episode / rollout id |
| `episode_return` | float | Cumulative reward this episode (or documented equivalent) |
| `speed` | float \| null | `info["true_speed"]`; helps “watch a run” panel |
| `step` | int | Align with v1 train metrics when co-published |

**Where the data comes from (no new IPC needed).** `n_envs >= 2` is `SubprocVecEnv`, so a callback cannot touch env internals cheaply. It does not have to:

- **LiDAR is already in the callback** as `self.locals["new_obs"]["lidar"]`, shape `(n_envs, 1080)` float32, already normalised to `[0,1]` by `spaces._normalize_lidar`. It rides the existing rollout transfer — **zero marginal IPC**. Do not add LiDAR to `info` (that would pickle 1080 floats per env per step across the Subproc pipe even when we publish nothing) and do not call `env_method`.
- **Pose / collision / speed** come from `self.locals["infos"][i]` (`position`, `collision`, `true_speed` already exist). Only **`yaw`** needs adding to `AutoDriveEnv._build_info`.
- To convert display values back to metres: `m = norm * (range_max - range_min) + range_min` with defaults `0.05 … 30.0`. Publish those two bounds **once** in a meta/`status` event, not per sample.

**Done-step hazard:** when `dones[i]` is true, SB3's VecEnv has already auto-reset, so `infos[i]` holds the **terminal** pose while `new_obs[i]` holds the **post-reset** obs. Mixing them makes the car flash to a wrong place. **Skip publishing that env's sample on a done step** (or tag it `reset: true` and let the canvas ignore it).

### LiDAR payload sizing (1080 beams is real — `src/layer2/spaces.py` `LIDAR_BEAMS = 1080`)

Rough per-car, per-sample cost at 15 Hz:

| Scheme | Beams | Bytes/car/sample | 1 car @15 Hz | 4 cars @15 Hz |
| :--- | :--- | :--- | :--- | :--- |
| Full, JSON floats | 1080 | ~8.6 KB | ~130 KB/s | ~520 KB/s |
| Full, binary Float32 | 1080 | 4.3 KB | ~65 KB/s | ~260 KB/s |
| **Downsampled, JSON (chosen)** | **120** | **~0.7 KB** | **~11 KB/s** | **~43 KB/s** |

**Decision: downsample on the train side to ~120 beams (min-pool groups of 9), 3 decimal places, plain JSON.** Reasoning:

- Bandwidth is not the real cost on a loopback/compose bridge — **main-thread `JSON.parse` + GC is**. Full res is ~86k floats/s at 4 cars; 120 beams is ~7k. Binary WS frames solve the wrong half of the problem and add framing complexity, so they are an **escape hatch**, not v1.
- Use **min**-pool, not stride sampling: min preserves the nearest hit per sector, so obstacles never vanish between beams.
- 120 beams over a full 360° sweep is 3° spacing — at the 30 m max range that is a ~1.6 m gap between ray tips, invisible when drawn as one filled polygon rather than discrete rays (see §8). Narrower FOV only improves this.
- Cost on the train thread is a single `arr.reshape(n_envs, 120, 9).min(axis=2)` — microseconds.

**Selection is a publish-side *budget* and a draw-side *choice*.** There is no control channel from the browser into the running train child, so the browser cannot ask the callback to change what it publishes. Therefore:

- Publish downsampled LiDAR for **all** envs up to `telemetry_lidar_max_envs` (default **4**); above that, publish LiDAR for env `0` only and pose-only for the rest. Start-of-run Setting, consistent with §7.
- The browser draws only the **selected** car's LiDAR — instant switching, no round-trip.

#### LiDAR angles (plain English) — **LOCKED sequencing**

Each LiDAR reading is just **distances** (how far until a hit per beam index). That list alone cannot aim anything on the map. To draw **rays** you also need:

1. **Where beam 0 points** relative to the car (start angle vs yaw), and  
2. The **angular step** around the car from beam to beam (and FOV / CW vs CCW).

**Calibration is not documented in this repo yet.** Layers 1–2 only expose beam count and range bounds (`LIDAR_BEAMS`, `range_min` / `range_max`). **Do not guess 360°/CCW.**

**Phase 6 order (non-negotiable):** ship **map + cars (+ collision X)** first. Draw LiDAR **rays only after** measuring and documenting those angles (park by a wall, tune `angle_min` / `angle_increment` / sign until hits land on geometry, commit constants — F7). Until then the overlay stays off.

### Map assets (Phase 6 background)

`RaceTrack.waypoints` in this repo still defaults to **`None`** — no centerline under `src/`. Official **2D** AutoDRIVE / RoboRacer assets **do** exist upstream; vendor URLs (do not commit multi-MB `.fbx` / `.skp`).

#### Candidate sources

| Source | URL | What it is | License |
| :--- | :--- | :--- | :--- |
| **AutoDRIVE RoboRacer Racetracks** (primary) | https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Racetracks | Official 1/10 tracks for AutoDRIVE Simulator | **BSD-2-Clause** |
| └ Legacy Porto | `Legacy Tracks/Porto Track/` — `Porto.pgm` + `Porto.yaml` | ROS occupancy grid, **0.05 m/px**, origin in yaml | same |
| └ Legacy Berlin | `Legacy Tracks/Berlin Track/` — `Berlin.png` + `Berlin.yaml` | Occupancy PNG + same yaml convention | same |
| └ Library previews | `Library/*.png` (~160–340 KB each) | Clean top-down / bird's-eye track outlines (Porto, Berlin, SRL 2024–2026) | same |
| └ 3D meshes | `*.fbx` / `*.skp` under Legacy + Sim Racing Tracks | Unity import only — **not** for the canvas | same |
| **f1tenth_racetracks** (secondary) | https://github.com/f1tenth/f1tenth_racetracks | 20+ tracks: `*_map.png` + yaml + `*_centerline.csv` | **GPL-3.0** — great format, may **not** match AutoDRIVE RoboRacer scenes |
| AutoDRIVE site / RCT | https://autodrive-ecosystem.github.io · [Race Control Tower](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Race-Control-Tower) | Ecosystem docs; RCT is a Socket.IO proxy/monitor — **no** fleet map pack | BSD-2-Clause (RCT) |

**Vendored samples (this repo):** `assets/maps/` — Porto occupancy + yaml, Berlin occupancy + yaml, Porto library preview. See `assets/maps/ATTRIBUTION.md`. Prefer documenting further tracks as URLs; vendor only small occupancy / preview images when needed.

#### Recommended v1 pick

**Static occupancy (or library outline) as the canvas map layer**, not waypoints-first:

1. **Default:** `assets/maps/porto/Porto.pgm` + `Porto.yaml` (or Berlin pair) — metre-accurate via `resolution` / `origin`; blit onto the **static** canvas with world→pixel from yaml (ROS map frame: origin = lower-left of image in world metres).
2. **Visual alt:** `Library/*.png` (e.g. `Porto_preview.png`) when a nicer outline is wanted — calibrate against live poses (or pair with the matching Legacy yaml) before trusting scale.
3. **Do not** pull FBX into Mission Control; if a future track has meshes only, orthographic screenshot **or** Phase 6.1 breadcrumb until a 2D asset exists.

**How Mission Control uses it:** static canvas underlayer (map) → cars → LiDAR polygon → collision **X**. Hub telemetry remains the only live source; the map file never drives `step()`.

**Attribution:** BSD-2-Clause AutoDRIVE Racetracks — retain copyright notice when redistributing (already in `ATTRIBUTION.md`). Avoid vendoring f1tenth maps into this BSD-leaning stack unless GPL coupling is an explicit choice.

#### Phased background (still useful)

| Step | Background | When |
| :--- | :--- | :--- |
| 6a | Grid + scale bar, **auto-fit** from observed pose bounds (grow-only) | Unknown / custom scene; fallback |
| 6b | **Breadcrumb trail** — visited poses → offscreen canvas | No calibrated map yet |
| 6c | **Vendored occupancy / library PNG** under cars (v1 recommended above) | Default once Settings picks a track id |
| 6d | Optional: centerline polyline from csv / recorded lap / RaceTrack display helper | Gates / Frenet later |

**Latent bug if RaceTrack is ever used as a display helper:** `track.py` `_init_track_geometry` treats waypoint columns as `[:, :2]` (x, y) while `step_single` calls `get_frenet_progress(position[0], position[2])` (x, z). Those disagree. Fix the axis convention there before reusing it for display.

---

## 7. Settings model

First-class Pydantic (or equivalent) model on the hub; **UI page is not buried**. Maps directly to `src/layer3/train.py` `build_arg_parser()` flags.

### Train / job flags

| Settings field | CLI flag | Default (today) | Notes |
| :--- | :--- | :--- | :--- |
| `n_envs` | `--n-envs` | `1` | Docker: match `--scale sim=N` |
| `base_port` | `--base-port` | `4567` | Ports `base .. base+n_envs-1` |
| `timesteps` | `--timesteps` | `10000` | |
| `out` | `--out` | stamp under `logs/rl/` | Or derive from `run_name` |
| `run_name` | (UI-only → `--out`) | optional | Convenience for Settings UI |
| `seed` | `--seed` | `0` | |
| `device` | `--device` | `auto` | `auto` \| `cpu` \| `cuda` |
| `resume` | `--resume` | `None` | Path to PPO `.zip` |

### Env kwargs (via `env_kwargs_from_args`)

| Settings field | CLI flag | Default |
| :--- | :--- | :--- |
| `headless` | `--headless` / `--no-headless` | `True` |
| `auto_launch` | `--auto-launch` / `--no-auto-launch` | `True` (**Docker: False**) |
| `connect_timeout` | `--connect-timeout` | `90.0` |
| `frame_skip` | `--frame-skip` | `1` |
| `max_episode_steps` | `--max-episode-steps` | `0` |
| `stagnation_speed_threshold` | `--stagnation-speed-threshold` | `0.15` |
| `stagnation_steps` | `--stagnation-steps` | `200` |
| `forward_scale` | `--forward-scale` | `1.0` |
| `collision_penalty` | `--collision-penalty` | `0.0` |
| `slip_penalty` | `--slip-penalty` | `0.0` |
| `steer_jerk_penalty` | `--steer-jerk-penalty` | `0.0` |

### Hub-only Settings (not train argv)

| Field | Purpose | Default |
| :--- | :--- | :--- |
| `telemetry_every_n` | Metrics publish interval, **step**-based | 200 |
| `fleet_hz` | Sim-state publish cap, **time**-based (§6) | 15 |
| `lidar_display_beams` | Downsample target (min-pool from 1080) | 120 |
| `telemetry_lidar_max_envs` | Publish LiDAR for at most this many envs | 4 |
| `docker_mode` | Preset: force `auto_launch=False` when true | — |
| `stop_sims_on_train_exit` | Stop compose sims after any train exit | True |
| `stop_stack_on_train_exit` | Stop the full stack after train exit (kills UI) | False |

### Hyperparameter surgery — locked recommendation: **later (not day 1)**

**Verdict:** do **not** put mid-run LR / `net_arch` surgery on day 1. Ship fleet canvas + hub/`Popen` first — they deliver more Mission Control value. Keep surgery as a **real Phase N+** so it is not forgotten (good later; not a day-1 blocker).

| Why later | Detail |
| :--- | :--- |
| v1 Settings = existing CLI | Mission Control Settings mirror `build_arg_parser()`; LR / `net_arch` are **not** CLI flags today |
| Hardcoded in `make_model` | `learning_rate=3e-4`, `net_arch=dict(pi=[128,128], vf=[128,128])` live in `src/layer3/train.py` |
| Mid-run surgery is heavy | Live optimizer edits + arch rebuild ≈ new run; more complexity than start/stop + telemetry |
| Higher day-1 ROI | Hub + Popen + Settings + Live + **fleet canvas** first |

**Cheap middle ground (optional Phase 4 / Phase N enrichment — not blocking 1–6):** once `train.py` accepts them, expose `learning_rate` / `net_arch` as **start-of-run** Settings → CLI/env. That is **not** live surgery — just more Settings at job start. Today those flags do not exist; add CLI first, then wire Settings.

**Phase N+** (true surgery / mid-run): only after start-of-run knobs exist and Mission Control core is solid. Document restart-required vs safe live update if SB3 allows.

### Persistence

Write JSON under e.g. `logs/layer4/settings.json` (volume-mounted). `PUT /settings` updates file; `POST /train/start` may accept an inline override body **or** use last saved Settings (pick one in impl; prefer body = full Settings snapshot for reproducibility).

### Argv build sketch

```text
python -m src.layer3.train
  --n-envs {n_envs}
  --base-port {base_port}
  --timesteps {timesteps}
  --out {out}
  --seed {seed}
  --device {device}
  [--resume {resume}]
  --headless / --no-headless
  --auto-launch / --no-auto-launch
  --connect-timeout …
  … env reward/truncation flags …
```

Identical to what an operator would type in `docker compose exec brain …`.

---

## 8. Frontend (Vite + React)

Keep v1 **simple** — **full Settings** + Train + Live from the first React ship; Fleet page in Phase 6. Do **not** ship a Start/Stop-only stub that defers Settings.

| Section | Content |
| :--- | :--- |
| **Settings** | **First-class** form bound to full Settings model (all train CLI flags §7); Save; Docker preset toggle |
| **Train** | Start / Stop; status badge; PID; link/path to log; last exit code |
| **Live** | Table or sparkline: step, reward, episode, optional loss; last checkpoint |
| **Fleet** (Phase 6) | Layered canvas (map → cars → LiDAR → collision **X**); side panel + layer toggles; WS-driven only — see Phase 6 |

### Where Vite runs — **Docker B (brain)** — **LOCKED, Docker-first**

**Vite in Docker B.** Not “Vite on the host, hub in B.” **Brain B is the single place to hit Mission Control.**

- **Host Vite = Windows convenience only** (path-mount / local Node). That was never the production path and is **not** the default anymore.
- **Production:** `vite build` → static assets **into image B**; **FastAPI serves `dist/` on the same Mission Control port `:8090`** (same-origin with REST/WS). Optional nginx in front is fine; port stays **`:8090`**.
- **Dev:** HMR **inside B** (bind-mount `src/layer4/web`) **or** a compose `layer4` service on **B’s network** that proxies API/WS to the hub — still not a host-only default.

| Mode | How it works | Browser hits |
| :--- | :--- | :--- |
| **Production** | `vite build` static into **brain B**; FastAPI serves on **`:8090`** | `http://localhost:8090` |
| **Dev** | HMR inside **B**, or `layer4` service on B’s network | Hub `:8090` and/or published Vite port on B’s network (e.g. `:8080` if HMR needs it) |

Compose may still publish `8080` for in-B Vite HMR during iterate; operators think of **B**, not the Windows host Node install, as home for the frontend.

v1 ships full Settings / Train / Live. Fleet canvas is a **planned product page** (Phase 6), not a forever-non-goal. Old PLAN sketch of UI-owned RaceTrack stepping remains superseded.

### Frontend lightness & performance

Target: a **single-operator localhost instrument panel**, not a public web app. That framing decides most calls below — no SSR, no legacy browsers, no polyfills, no a11y-audit-grade component library.

#### Locked choices

| Area | Decision | Note |
| :--- | :--- | :--- |
| Stack | **Vite + React + TS**, REST + **one** WS to the hub | No second backend, no BFF |
| Router | **None** — conditional render of 4 pages from one `useState` | Saves a dep; URLs are not a product need here |
| Styling | **Plain CSS + CSS custom properties**, 1–2 files | See tiebreaker below |
| Components | **Native elements** (`<select>`, `<input type=checkbox>`, `<dialog>`) | No Radix, no shadcn in v1 |
| Rendering | **Canvas 2D**, hand-written | **Not** PixiJS, not SVG-per-beam, not DOM nodes |
| State | **`useSyncExternalStore`** over a ~30-line store + refs for hot data | No Zustand/Redux/react-query needed at this size |
| Transport | **Native `WebSocket`** + `fetch` | Not `socket.io-client` (the *sim* uses Socket.IO; the browser must not), not `axios` |
| Charts | Hand-rolled sparkline (canvas or one SVG `<polyline>`) | No Recharts/Chart.js/D3 |
| 3D | **None** | No Three.js for v1 — bird's-eye 2D is the product |

**Tailwind — not in v1.** Tailwind is genuinely *not* a runtime cost (build-time, tree-shaken, ~8–12 KB gz here), so "heavy" is the wrong objection. The real tiebreaker is that this UI is ~4 pages with a dense-instrument look, and Tailwind adds a Vite plugin, a config, content-scan setup, and utility-class churn in every diff — to replace maybe 250 lines of CSS. Plain CSS with variables for tokens wins on total moving parts. This is cheap to reverse: revisit if the UI passes ~8 pages or gains a real design system. Dropping Tailwind + Radix also kills shadcn **by construction** — shadcn requires both.

**PixiJS — no, and the earlier "or thin PixiJS" phrasing was wrong.** Pixi is ~100–150 KB gz and pulls in a WebGL scene graph, texture/loader machinery, and a retained display-object tree. Our whole frame is: one background blit, N car glyphs, one LiDAR polygon — tens of draw calls, not thousands of sprites. Canvas 2D handles that with room to spare. Keep Pixi as an **escape hatch** only if profiling ever shows Canvas 2D missing frame budget, which at `n_envs ≤ 8` it will not.

#### Canvas rules

| Rule | Why |
| :--- | :--- |
| **Two** canvases: static (map/grid, redrawn only on resize or bounds change) + dynamic (cars + LiDAR, cleared per frame) | Avoids re-rasterising the background 15×/s; more layers = pointless compositing |
| Draw loop is **`requestAnimationFrame`**, reading a **ref** — WS messages never trigger React renders | rAF self-throttles to 0 Hz on hidden tabs, so a backgrounded tab costs nothing |
| Store the latest sim-state as **last-value-wins in a ref**, not a queue | A hidden tab must not accumulate a backlog to replay |
| Render LiDAR as **one filled `Path2D` polygon**, not ~120 `stroke()` calls | One fill vs 120 path submissions |
| Use `ctx.setTransform` for the world→screen map; scale for `devicePixelRatio` | No per-point math; no blurry canvas on HiDPI |
| Skip the frame if the sim-state ref has not changed | Nothing to redraw between telemetry samples |
| Document the axis mapping **once, in code**: world `x` → canvas `+x`, world `z` → canvas `−y`, screen angle = `−yaw` | Y-up-vs-y-down sign errors are the classic bug here |

Note that the "cap at 15–30 Hz / drop frames if WS is faster" instinct is right but aimed slightly off: with `fleet_hz ≈ 15` the WS is **never** faster than the display. The thing that actually needs capping is **WS → React state → re-render** coupling; rAF + refs removes it structurally, and the rAF cap falls out for free.

#### Memory discipline

Unbounded history is the likeliest way this UI dies during a long run. Fleet at 15 Hz × 12 h is ~650k samples.

| Buffer | Bound |
| :--- | :--- |
| Metrics history (sparkline) | `Float32Array` ring, **2000** points |
| Fleet sim-state | **Last sample only** — no history |
| Per-car trail (if drawn) | `Float32Array` ring, **256** poses, or draw into the offscreen breadcrumb canvas and keep nothing |
| Hub retention | Last sample **per type** + optional ring of ~600 metric points so a late-joining client has context. Never a growing list |

#### Connection robustness (was missing from the plan)

- **Reconnect with exponential backoff + jitter**, ~250 ms → 5 s cap. Not a bare retry loop.
- Show connection state in the header: `live` / `reconnecting` / `stale (Ns)`.
- **On every (re)connect, `GET /train/status`** to resync. Treat WS as live deltas only; REST is the source of truth. A hub restart must not leave the UI silently frozen on stale numbers.

#### Bundle discipline

| Guardrail | Concrete |
| :--- | :--- |
| Dependency budget | **≤ 6** runtime deps. Adding one is a reviewed decision, not a reflex |
| Size budget | **≤ 120 KB gz** total JS for v1 (React + react-dom is already ~45 KB gz of that) |
| Measure | `vite build` prints per-chunk gzip sizes — record the v1 baseline here and re-check on every dep add. `npx vite-bundle-visualizer` ad hoc when a number surprises you |
| Code-split | `React.lazy` + `Suspense` for the **Fleet** page (it carries the renderer and calibration constants) |
| Imports | Exact paths only — no barrel re-exports, no `import * as` of a large lib |
| Build target | `es2022`, no legacy plugin, no polyfills — it is localhost for one modern browser |
| Repro | Commit the lockfile; `npm ci` in Docker |

### Planned tree

```text
src/layer4/                 # Layer 4 — Mission Control (not a top-level ui/)
├── __init__.py
├── hub/                    # FastAPI process on brain B
│   ├── __init__.py
│   ├── app.py              # routes: health, train, settings, telemetry, WS, static
│   ├── train_job.py        # Popen lifecycle
│   └── telemetry.py        # TelemetryBus + last sample + WS fan-out
├── settings.py             # shared Settings model + load/save (CLI ↔ JSON)
├── types.py                # optional shared TS-facing / OpenAPI shapes
└── web/                    # Vite + React + TS (built into B image for prod)
    ├── package.json
    ├── vite.config.ts
    ├── index.html
    ├── src/
    │   ├── App.tsx
    │   ├── pages/Settings.tsx   # full CLI-flag form from day 1
    │   ├── pages/Train.tsx
    │   ├── pages/Live.tsx
    │   └── pages/Fleet.tsx      # Phase 6
    └── …
```

Layers 1–3 remain the learning path under `src/layer1|2|3`. Layer 4 is infra for watch/control.

---

## 9. Docker placement

**Vite-on-B is Docker-first** (same lock as §2 / §8): host Node/Vite was Windows convenience only; prod static and FastAPI share **`:8090`** on brain B.

| Concern | Decision |
| :--- | :--- |
| Where hub + frontend live | **Brain B only** — `src/layer4/` (hub + Vite) |
| Port | **8090** — replace ready stub as default `command` when hub lands; **single entry** for Mission Control (API **and** Vite static) |
| Sims | Unchanged A×N; train uses `--no-auto-launch` |
| GPU | Remains on B (`gpus: all` in compose) |
| Volumes | Existing `./src`, `./logs` mounts cover Layer 4 code + settings + train artifacts; mount `./assets` when fleet map needs it |
| Frontend assets (prod) | **`vite build` into brain image B**; **FastAPI serves static on `:8090`** (same Mission Control port) |
| Frontend (dev) | Vite HMR **inside B** or a `layer4` compose service on B’s network — **not** host-Node as the default |
| Compose note | Publish `8080` only if in-B Vite HMR needs a separate port; otherwise **`:8090` alone** |

```text
docker compose up --build --scale sim=2
# B: uvicorn src.layer4.hub.app:app --host 0.0.0.0 --port 8090
#    (+ static from layer4/web/dist in prod)
# A×2: dial brain:4567, brain:4568
# Browser → http://localhost:8090   ← only URL operators need
```

Layers 1–3 stay learning code under `src/layer*`. Layer 4 = Mission Control under `src/layer4/`. **Do not** move Dockerfiles into `src/`. **Do not** use a top-level `ui/` folder.

---

## 10. Phased milestones

Settings is **not** an optional late phase. Hub Settings API lands in Phase 1; the **full Settings React page** ships with the first browser app (Phase 3). No Start/Stop-only stub that defers Settings.

### Phase 1 — Hub + Popen + Settings API

1. [ ] FastAPI app under `src/layer4/hub/`: `/health`, `/train/start`, `/train/stop`, `/train/status`
2. [ ] `TrainJob` Popen + log file + terminate/kill
3. [ ] **Full Settings model** (§7) + `GET`/`PUT /settings` + disk JSON — maps **all** train CLI flags (not a minimal subset forever)
4. [ ] Start builds argv from Settings (Docker preset: `auto_launch=false`)
5. [ ] Swap brain `command` from `ready_stub.py` → hub (or feature-flag)
6. [ ] Smoke: start/stop + settings round-trip from `curl` inside/outside compose

**Exit:** one train run started/stopped via HTTP with argv matching Settings; CLI still works unchanged.

### Phase 2 — Telemetry WebSocket

1. [ ] `HUB_URL` + SB3 callback POST `/telemetry`
2. [ ] WS `/ws` fan-out with **bounded per-client queue** (drop-oldest, disconnect slow clients)
3. [ ] **Non-blocking publisher**: daemon thread + `maxsize<=4` drop-oldest queue + `timeout=(0.5, 1.0)` + never raise into train (§6)

**Exit:** `websocat` / browser shows updating `step`/`reward` during train — **and** killing / `SIGSTOP`-ing the hub mid-run does not slow or crash training.

### Phase 3 — React app (Settings + Train + Live)

1. [ ] Vite + React scaffold under `src/layer4/web/` — **built/served from brain B** (§8 / §9)
2. [ ] **Full Settings page** — every train CLI flag in §7 + hub-only fields; Save ↔ `PUT /settings`
3. [ ] Train + Live pages wired to REST/WS
4. [ ] **Prod:** `vite build` into B; FastAPI serves static on **`:8090`**. **Dev:** HMR inside B (or `layer4` service on B’s network) — host Vite = Windows convenience only (§8 / §9)

**Exit:** open `http://localhost:8090` on B; edit Settings; Start/Stop; live numbers move. Settings is complete for CLI parity — not deferred.

### Phase 4 — Polish

1. [ ] Log tail in UI (optional)
2. [ ] Checkpoint path links / run browser under `logs/rl/`
3. [ ] Harden stop + zombie detection
4. [ ] README / docker README: hub replaces stub; Layer 4 lives under `src/layer4/`; point to this doc
5. [ ] **Optional enrichment:** if/when `train.py` grows `--learning-rate` / `--net-arch` (or env), expose as start-of-run Settings — **not** live surgery (§7)

### Phase 6 — Fleet canvas (product UX)

**Ownership:** canvas watches **hub telemetry only**; **train owns `step`**; RaceTrack is **not** on the train path (may help **display** geometry / gates later). No UI-owned Socket.IO / second stepper.

**LiDAR angles — LOCKED order:** each sample is **distances only**. Rays need **where beam 0 points** + **angular step** around the car; that calibration is **not in this repo yet**. **Ship map + cars (+ collision X) first.** Draw LiDAR **rays only after** measuring/documenting angles (§6). Do not block the fleet page on F7.

**Canvas layers (bottom → top):**

| Z | Layer | Behavior |
| :--- | :--- | :--- |
| 1 | **Map** | Track background / geometry |
| 2 | **Cars (fleet)** | Pose markers for each vehicle |
| 3 | **What they see** | LiDAR (or perception) overlay — per car / **selected** car when `n_envs > 1` (**after** angle calibration) |
| 4 | **Collision state** | If collided, that car is an **X** (or clear crashed glyph), not a normal marker |

**Side / panel (minimal — watch a run, do not overbuild):**

| Field | Why |
| :--- | :--- |
| Episode # / `episode_id` | Which rollout |
| Steps | Progress in episode / run |
| Return / reward | Cumulative or documented mean |
| Collision flag | Matches **X** glyph |
| Speed (optional) | Situational awareness |

**Toggles / focus:**

- Layer on/off: **map** / **fleet** / **LiDAR overlay**
- When `n > 1`: **select which car’s LiDAR** to show (selected-car focus)
- **Legend:** normal marker vs collided **X**

**Checklist:**

0. [ ] Add `yaw` (`snap.heading_yaw`) to `AutoDriveEnv._build_info` — the only Layer 2 change Phase 6 needs
1. [ ] Publish per-env sim-state on TelemetryBus from `self.locals["new_obs"]["lidar"]` + `infos` (§6) — min-pool 1080→120, time-based `fleet_hz`, skip done-steps, **no UI `step()`**
2. [ ] **Measure / document LiDAR angles** (where beam 0 points + angular step / FOV / CW vs CCW vs yaw); commit `angle_min` / `angle_increment` / sign — **not in repo today**; until then **map + cars + collision X** still ship; **no rays** until documented
3. [ ] Fleet page (lazy-loaded): two canvases, `rAF` + refs, LiDAR as one filled path (§8)
4. [ ] Background: prefer vendored occupancy / library PNG under `assets/maps/` (§6); else grid+auto-fit (6a) → breadcrumb (6b)
5. [ ] Side panel: episode / steps / return / collision / optional speed / **stale indicator**
6. [ ] Toggles: map / fleet / LiDAR; selected-car LiDAR when multi-env; marker legend
7. [ ] Optional: RaceTrack **display** helpers only (waypoints / gates / snapshot aggregate) — never on the L2/L3 train launch path; fix its x/y vs x/z axis disagreement first
8. [ ] Ownership review: canvas is a consumer of hub WS events only; Redis still **no** (§13)

**Exit:** operator sees fleet on the map during a Docker train run; collided cars show as **X**; toggling LiDAR / selected car does not create a second Socket.IO owner.

**Phase 6 acceptance**

| # | Check |
|---|--------|
| F1 | Map + cars + collision **X** from hub WS (no UI step) — **ships before LiDAR angles exist**; rays are not required for F1 |
| F2 | LiDAR overlay toggles; with `n_envs > 1`, selected car controls which overlay shows — **only after** beam-0 / angular-step calibration |
| F3 | `collision=true` → car drawn as **X** / crashed glyph; legend documents normal vs crashed |
| F4 | Side panel shows at least episode, steps, return/reward, collision |
| F5 | RaceTrack (if used) is display-only; train path unchanged |
| F6 | Steps/s with fleet telemetry on is within **~2%** of the same run with `HUB_URL` unset |
| F7 | LiDAR fan lines up with walls (**where beam 0 points** + **angular step** committed, not guessed) |

### Phase N+ — Hyperparameter surgery (optional; **not day 1**)

Locked: **later**, not day 1 — see §7. Prefer start-of-run Settings (Phase 4 optional) before any mid-run controls.

1. [ ] Settings fields for LR / `net_arch` / related PPO knobs (beyond today’s CLI; requires `train.py` flags first)
2. [ ] Document: mid-run changes vs restart-required; keep IDE path valid
3. [ ] Only then: consider live surgery if SB3 supports safe update without implying a new run

**Exit:** advanced Settings can change LR/arch without editing `train.py`; still not required for Mission Control v1 / Phases 1–6.

---

## 11. Open risks / cons (accepted)

| Risk | Mitigation / acceptance |
| :--- | :--- |
| Two processes (hub + train) | **Solved by design** — hub is supervisor; train is child |
| Telemetry drops if hub restarts | Best-effort POST; training continues |
| **Telemetry POST slowing the step loop** | **Real risk, mitigated by §6**: daemon thread + bounded drop-oldest queue + hard timeouts. A *hung* (not dead) hub is the dangerous case; acceptance F6 measures it |
| Full 1080-beam LiDAR floods the WS | Min-pool to ~120 beams on the train side; `telemetry_lidar_max_envs` budget (§6) |
| LiDAR beam angles unknown in repo | Readings = distances only; rays need beam-0 + angular step. Phase 6: **map + cars + X first**; measure/document then rays (F7) — constants committed, not guessed |
| Map must match the live Unity scene | Official AutoDRIVE occupancy/library PNGs exist (§6); calibrate origin/yaw vs poses; fall back to 6a/6b if scene unknown |
| WS drops / hub restart leaves UI frozen | Backoff reconnect + `GET /train/status` resync + visible stale indicator (§8) |
| Unbounded telemetry history in browser | Fixed-capacity rings; fleet keeps last sample only (§8) |
| Stop leaves sims in bad state | Prefer graceful `vec_env.close()`; compose restart sims if needed |
| Windows Popen / signal quirks | Primary path = **Linux B**; host Windows unsupported for v1 polish |
| Port confusion 8080 vs 8090 | **8090 = Mission Control on B** (hub + static); 8080 = optional in-B Vite HMR only |
| Old PLAN.md UI = RaceTrack stepping | **Superseded** — train via Popen; fleet viz consumes telemetry only |
| Accidental UI `step()` | Ownership rules + code review; no env import on start / Fleet path |
| Premature Redis | **Deferred** — TelemetryBus interface; add Redis only if §13 triggers fire |
| Host-only Vite as default | **Corrected / locked** — Vite in Docker B; host = Windows convenience only; prod `vite build` + FastAPI on **`:8090`**; dev HMR in B or `layer4` on B’s network (§8 / §9) |

---

## 12. Acceptance tests (v1)

| # | Check |
|---|--------|
| 1 | `GET /health` returns OK with hub as brain command |
| 2 | `POST /train/start` runs equivalent of `python -m src.layer3.train …` |
| 3 | `GET /train/status` shows running PID; log file grows |
| 4 | `POST /train/stop` ends process; status `exited` |
| 5 | During train, WS clients receive telemetry samples |
| 6 | Settings round-trip matches CLI flags in §7 |
| 7 | CLI train without `HUB_URL` still works (no hub required) |
| 8 | Docker: `--scale sim=N` + Settings `n_envs=N`, `auto_launch=false` |
| 9 | **Hub killed *and* hub hung (`SIGSTOP`) mid-run → train keeps stepping at full rate**; samples drop, no exception |
| 10 | WS client killed / tab closed mid-run → hub keeps broadcasting; reopened tab resyncs via `GET /train/status` |

Fleet canvas / LiDAR acceptance lives under Phase 6 (**F1–F5** above) — not a v1 gate.

---

## 13. Pub/sub decision

**Verdict: do not set up Redis now.** FastAPI hub + in-process bus is the long-term spine for Mission Control through fleet viz; Redis is an optional backend swap later.

| Question | Answer |
| :--- | :--- |
| Is FastAPI enough as the spine? | **Yes** for start/stop, Settings, live WS, and fleet/LiDAR viz while train + hub share brain **B** |
| When is Redis worth it? | Multi-process / multi-host publishers, hub HA / restart without dropping fan-out, or several consumers outside the hub process |
| How to avoid a rewrite? | Hub exposes a small **`TelemetryBus`** (`publish` / `subscribe`); train keeps **HTTP POST** (or a thin client). v1 = in-memory bus → WS. Later Redis = new bus impl only |
| Setup Redis now? | **No** — would add compose/ops cost without a current multi-writer need |

```text
Train / publishers          Hub                         Browsers
─────────────────           ─────────────────────────   ────────
POST /telemetry  ──▶  TelemetryBus.publish  ──▶  WS /ws
(sim-state later)     (InMemory now;
                       RedisPubSub later)
```

Train and UI talk to the hub API, not to Redis. Swapping the bus backend must not change `layer3.train` argv or React REST/WS contracts.

---

## Related docs

- [`LAYER3.md`](LAYER3.md) — PPO train/play plan  
- [`PLAN.md`](PLAN.md) — system overview (UI section points here)  
- [`docker/README.md`](docker/README.md) — A/B compose  
- [`README.md`](README.md) — status + quick start  
- [`src/layer3/train.py`](src/layer3/train.py) — authoritative CLI flags  
