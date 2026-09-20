# AutoDRIVE Simulator

Binaries used by Layer 1 (`src/layer1`) and Layer 2 (`src/layer2`). Executables are **gitignored**; only docs live in git.

## Layout

| Path | Purpose |
|------|---------|
| `windows/AutoDRIVE Simulator.exe` | Local Windows demos / training |
| `AutoDRIVE Simulator.x86_64` | Linux / Docker (`/app/simulator/...` in compose) |
| `Data/` | Unity asset bundles (gitignored with binaries) |

## Usage

Prefer the Python driver (binds Socket.IO first, then spawns the process):

```bash
python scripts/demo.py layer1
python scripts/demo.py layer1 --headless --racers 1
python scripts/demo.py layer2
```

Flags the driver passes:

- **Headed:** `-ip 127.0.0.1 -port <PORT>` (then click **Connect**)
- **Headless:** `-batchmode -nographics -ip 127.0.0.1 -port <PORT>` (auto-connect)

Docker: set `AICAR_SIMULATOR_PATH=/app/simulator/AutoDRIVE Simulator.x86_64` (already in `docker-compose.yml`).

## Protocol

This RoboRacer Windows / Linux client speaks **Socket.IO over Engine.IO v4**. Layer 1 uses `python-socketio` 5.x + gevent accordingly.
