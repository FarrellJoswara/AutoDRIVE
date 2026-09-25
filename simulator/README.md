# AutoDRIVE Simulator

Binaries used by Layer 1 (`src/layer1`) and Layer 2 (`src/layer2`). Executables are
**gitignored**; only this doc lives in git.

## Easiest: use `python main.py`

From the repo root, `python main.py` downloads
[`autodrive-simulator.zip`](https://github.com/FarrellJoswara/AutoDRIVE/releases/tag/simulator-binaries)
into `./simulator/` if the Linux binary is missing, then starts Docker + Mission Control.

Manual download: extract the release zip so you have:

| Path | Purpose |
|------|---------|
| `windows/AutoDRIVE Simulator.exe` | Local Windows demos / headed play |
| `AutoDRIVE Simulator.x86_64` | Linux / Docker (`/app/simulator/...` in compose) |
| `Data/` | Unity asset bundles |
| `GameAssembly.so` / `UnityPlayer.so` | Linux Unity runtime |

## Usage (local, no Docker)

Prefer the Python driver (binds Socket.IO first, then spawns the process):

```bash
python scripts/demo.py layer1
python scripts/demo.py layer1 --headless --racers 1
python scripts/demo.py layer2
python scripts/demo.py play --model logs/rl/.../final_model.zip --no-headless
```

Flags the driver passes:

- **Headed:** `-ip 127.0.0.1 -port <PORT>` (then click **Connect**)
- **Headless:** `-batchmode -nographics -ip 127.0.0.1 -port <PORT>` (auto-connect)

Docker A/B: mount this folder into sim + brain containers; compose sets
`AICAR_SIMULATOR_PATH=/app/simulator/AutoDRIVE Simulator.x86_64`.
Sims are launched by `docker/entrypoint-sim.sh` (not by Python auto-launch).
See [`docker/README.md`](../docker/README.md).

## Protocol

This RoboRacer Windows / Linux client speaks **Socket.IO over Engine.IO v4**. Layer 1 uses `python-socketio` 5.x + gevent accordingly.
