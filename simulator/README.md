# AutoDRIVE Simulator

This directory hosts the AutoDRIVE Simulator executable used by Layer 1 (`src/racer`).

## Layout

| Path | Purpose |
| :--- | :--- |
| `windows/AutoDRIVE Simulator.exe` | Windows build used for local Layer 1 demos (gitignored) |
| `AutoDRIVE Simulator.x86_64` | Optional Linux binary for WSL / Docker |
| `Data/` | Unity asset bundles (gitignored with the binaries) |

Place the Windows build under `simulator/windows/` so `scripts/demo.py` can find it.

## Usage (Windows Layer 1)

Launch through the Python driver (it binds Socket.IO first, then spawns the process):

```bash
python scripts/demo.py
python scripts/demo.py --headless --racers 1
```

Flags the driver passes:

- Headed: `-ip 127.0.0.1 -port <PORT>` (then click **Connect** in the UI)
- Headless: `-batchmode -nographics -ip 127.0.0.1 -port <PORT>` (auto-connect)

## Upstream notes

Official AutoDRIVE bring-up examples (Linux) also document:

```bash
./AutoDRIVE\ Simulator.x86_64
xvfb-run ./AutoDRIVE\ Simulator.x86_64 -ip 127.0.0.1 -port 4567
./AutoDRIVE\ Simulator.x86_64 -batchmode -nographics -ip 127.0.0.1 -port 4567
```

This project’s RoboRacer Windows client speaks **Socket.IO over Engine.IO v4**. The Layer 1 server uses `python-socketio` 5.x + gevent accordingly.
