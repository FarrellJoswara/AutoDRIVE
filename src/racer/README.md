# Layer 1 — Simulator driver (`src/racer/`)

## Summary

Layer 1 talks to the AutoDRIVE Unity simulator over **Socket.IO (Engine.IO v4)**.  
Python hosts the Bridge server; Unity connects as a client. Each physics tick Unity sends telemetry; Python replies with throttle / steering / reset.

This layer has **no Gymnasium and no neural nets**. It only:

- launch / kill simulator processes  
- lockstep `step` / `reset`  
- parse Bridge packets into `TelemetrySnapshot`  
- manage a fleet of cars (`RaceTrack`)  
- optional CSV trajectory export  

Higher layers call this API: Layer 2 wraps one `Racer` as a Gym env; the UI (later) will call `RaceTrack` for launch / kill / live telemetry.

---

## Layout

```text
src/racer/
├── __init__.py      # Public exports: Racer, RaceTrack, TelemetrySnapshot, TrajectoryLogger
├── racer.py         # One car: Socket.IO server + sim process + step/reset/kill
├── track.py         # Fleet manager + Frenet / checkpoint helpers
├── telemetry.py     # Bridge → TelemetrySnapshot + CSV logger
└── README.md        # This file
```

**Data flow**

```text
Unity (client)  --Bridge telemetry-->  Racer (server)
Unity (client)  <--Bridge commands---  Racer.step / reset
                      │
                      ▼
               TelemetrySnapshot
                      │
            RaceTrack (optional fleet)
```

---

## Files and essentials

### `__init__.py`

Re-exports: `Racer`, `RaceTrack`, `TelemetrySnapshot`, `TrajectoryLogger`.

### `racer.py` — `Racer`

One vehicle on one Socket.IO port.

| Piece | Role |
|-------|------|
| `__init__(racer_id, port, simulator_path, auto_launch, headless, …)` | Start gevent WSGI Socket.IO server; optionally launch Unity |
| `launch_simulator()` | Spawn `.exe` / `.x86_64` with `-ip` / `-port` (and headless flags) |
| `step(throttle, steering)` | Lockstep: wait for Bridge frame, emit string commands, return snapshot |
| `reset()` | Soft reset (pose / controls); wait for ack frame |
| `kill()` | Stop server thread + terminate Unity process |
| `save_trajectory(path)` | Write this car’s CSV via `TrajectoryLogger` |
| `is_connected` / `is_alive` | Client connected? Process still running? |
| `step_counter`, `step_latency_ms` | Simple profiling |

**Protocol notes:** replies are **emitted** `Bridge` payloads with **string** values (`V1 Throttle`, `V1 Steering`, `V1 Reset`) — matches AutoDRIVE Devkit shape. Auto-connect shim handles Unity emitting Bridge before a full Socket.IO connect handshake.

### `track.py` — `RaceTrack`

Fleet of `Racer`s plus optional track geometry.

| Piece | Role |
|-------|------|
| `__init__(num_racers, base_port, simulator_path, waypoints, num_gates, …)` | Build N racers on `base_port + i` |
| `step_single` / `step_all` / `step` | Drive one or all cars |
| `reset_single` / `reset_all` | Soft reset |
| `kill_racer` / `kill_all` | Tear down one or all |
| `save_all_trajectories(dir)` | CSV per racer |
| `get_fleet_telemetry()` | Last snapshot per id |
| `get_frenet_progress(x, y)` | Along-track `s` and lateral `d` (if waypoints set) |
| `update_checkpoints(racer_id, s)` | Gate / lap bookkeeping |

### `telemetry.py`

| Piece | Role |
|-------|------|
| `TelemetrySnapshot` | Typed fields: pose, velocities, LiDAR, encoders, lap, collision, derived `v_long` / `v_lat` / `slip_angle` / `heading_yaw` |
| `TelemetrySnapshot.from_raw_dict(data, step_id)` | Parse Unity Bridge dict (lists or space-separated strings) |
| `TrajectoryLogger` | Ring buffer → `save_to_csv` |

Derived kinematics (body frame) live here so Layer 2 does not re-implement yaw math for speed / slip.

---

## How to use

```python
from src.racer import RaceTrack

track = RaceTrack(num_racers=1, base_port=4567, auto_launch=True, headless=True)
# wait until track.racers[0].is_connected
snap = track.step_single(0, throttle=1.0, steering=0.0)
track.kill_all()
```

Live CLI: `python scripts/demo.py layer1` (see root README).  
Mock tests (no Unity): `python -m pytest tests/test_layer1.py -v`.
