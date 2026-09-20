# Docker A/B — brain + scalable sims

Split layout:

| Role | Service | Image | GPU |
| :--- | :--- | :--- | :--- |
| **A — sim** | `sim` | `docker/Dockerfile.sim` | Off by default; optional via `docker-compose.sim-gpu.yml` |
| **B — brain** | `brain` | `docker/Dockerfile.brain` | On by default (NVIDIA deploy reservation) |

Network: compose bridge `aicar`. Sims dial out to `brain` with Unity flags
`-batchmode -nographics -ip <BRAIN_HOST> -port <PORT>`. Brain listens on
`0.0.0.0:4567+` (Layer 1 Socket.IO). Host publishes a **wide** port range so
you can scale many envs without editing one port at a time.

The old single-image `docker/Dockerfile` + `entrypoint.sh` are retired; use the
dual Dockerfiles below.

---

## Prerequisites

1. **Docker Desktop** (Windows/macOS) or Docker Engine + Compose plugin (Linux)
2. **NVIDIA Container Toolkit** on the host if you want GPU on brain (default) or sims
3. Linux sim binary at:
   `simulator/AutoDRIVE Simulator.x86_64`
   (mounted to `/app/simulator/...`; `AICAR_SIMULATOR_PATH` is set in compose)

### Windows install notes

If `docker` is missing in PowerShell:

1. Install Docker Desktop (Admin / UAC):  
   `winget install Docker.DockerDesktop`  
   or run `Downloads\DockerDesktopInstaller.exe`
2. Enable **WSL 2** backend; reboot if prompted
3. Start **Docker Desktop** from the Start menu and wait until it says running
4. Open a **new** PowerShell and confirm: `docker version`

---

## Quick start (1 brain + 2 sims)

```bash
# Build and start brain (ready stub :8090) + two sim containers
docker compose up --build --scale sim=2
```

Check ready stub:

```bash
curl http://localhost:8090/
# -> OK
```

Train from inside the brain (sims already connected; do not auto-launch Unity):

```bash
docker compose exec brain \
  python scripts/demo.py train \
    --n-envs 2 \
    --base-port 4567 \
    --no-auto-launch \
    --timesteps 10000 \
    --out logs/rl/docker_smoke
```

Play / layer2 smoke the same way (`--no-auto-launch` when sims are compose-managed).

Stop:

```bash
docker compose down
```

---

## Ports

| Host / container | Purpose |
| :--- | :--- |
| `4567-4582` | Socket.IO (brain ← sims). Default range fits **16** envs |
| `8090` | Brain ready stub (`docker/ready_stub.py`) |
| `8080` | Reserved for future Mission Control UI (commented in compose) |

### Widening the range

Edit **both** sides of the mapping in `docker-compose.yml`, e.g. for 32 envs:

```yaml
ports:
  - "4567-4598:4567-4598"
```

Keep `BASE_PORT=4567` on sims and pass matching `--base-port 4567 --n-envs N` on train.
Sim entrypoint assigns `PORT = BASE_PORT + replica_index - 1` from the container
hostname when `PORT` is unset (`…-sim-1` → 4567, `…-sim-2` → 4568, …).

---

## GPU

### Brain (default on)

Compose already reserves NVIDIA GPUs on `brain`. Confirm inside the container:

```bash
docker compose exec brain python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`Dockerfile.brain` installs a CUDA torch wheel (`cu124`). On the host (non-Docker), prefer:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### Sims (optional)

CPU/Xvfb by default. Enable GPU for Unity containers:

```bash
docker compose -f docker-compose.yml -f docker-compose.sim-gpu.yml up --build --scale sim=2
```

Or uncomment the `deploy.resources…` block under `sim` in `docker-compose.yml`.

---

## Simulator CLI flags

Headless (what `entrypoint-sim.sh` runs):

```text
AutoDRIVE Simulator.x86_64 -batchmode -nographics -ip <brain> -port <PORT>
```

| Flag | Meaning |
| :--- | :--- |
| `-batchmode` | Unity batch / no interactive editor loop |
| `-nographics` | No visible window (headless) |
| `-ip <host>` | Brain hostname or IP (compose service name `brain`) |
| `-port <N>` | Socket.IO port on the brain for this car |

Headed local (Windows) demos still use `-ip 127.0.0.1 -port <N>` and click **Connect** — see root README / `simulator/README.md`.

---

## Files

| Path | Role |
| :--- | :--- |
| `Dockerfile.sim` | Ubuntu + Xvfb / GL deps for Unity |
| `Dockerfile.brain` | CUDA runtime + pip requirements + ready stub default |
| `entrypoint-sim.sh` | Xvfb + launch sim with PORT/BRAIN_HOST |
| `entrypoint-brain.sh` | Thin wrapper `exec "$@"` |
| `ready_stub.py` | HTTP `OK` on `:8090` |
| `../docker-compose.yml` | `brain` + scalable `sim` |
| `../docker-compose.sim-gpu.yml` | Optional GPU for sims |

---

## Troubleshooting

- **Sim exits immediately:** missing Linux binary or wrong `AICAR_SIMULATOR_PATH`.
- **Train hangs / no connect:** scale count must match `--n-envs`; ports must not collide; brain must be up first (`depends_on` only waits for start, not for Socket.IO listen — start train after sims are running, or start train first with servers bound then scale sims).
- **GPU unavailable on brain:** install NVIDIA drivers + Container Toolkit; on Docker Desktop enable GPU support; `deploy` is honored by Compose v2 swarm-mode style reservation — if ignored, set `gpus: all` under `brain` as a fallback on Compose that supports it.
