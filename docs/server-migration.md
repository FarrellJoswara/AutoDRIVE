# Move AiCar to a GPU server

This deployment targets an x86_64 Linux server with an NVIDIA GPU, working
Docker Engine + Compose v2 (`up --wait` support), NVIDIA Container Toolkit,
and permission to use the Docker daemon. It does not require a physical monitor.
It does require GPU graphics support for Unity's camera; CUDA alone is insufficient.
If this is a university scheduler-managed machine without Docker access, use an
administrator-approved runtime rather than these commands. Do not run the WSL
Windows launcher on a Linux-only server.

### Host preflight handoff

On the inspected host (`cycle3.eecs.ku.edu`, Ubuntu 24.04, x86_64, NVIDIA L40),
the official Docker route is currently blocked: Docker/Compose and a daemon
socket are absent, sudo requires a password, rootless user namespaces are
denied, and the installed LXD client cannot reach an LXD server. No official
simulator images are cached. The local Unity player is not a substitute for
the official practice/competition images. Before continuing, ask the machine
administrator to provision Docker Engine with Compose v2 and NVIDIA container
GPU support, and grant this account daemon access; alternatively, ask for the
institution's approved container runtime and a supported deployment procedure.
Do not attempt privileged installation or daemon changes from this account.

The server image uses the repository's ROS Jazzy/CUDA 12.6 PyTorch runtime,
pinned AutoDRIVE Devkit commit, and bundled frontend. The application requirements
are not a full transitive lockfile. Record the built image IDs/digests for a run;
rebuilds may resolve newer unpinned dependencies.

## 1. Export from this PC

Stop active training through Mission Control and wait for its final checkpoint.
Do not copy a live run. This export reads files without changing or committing
your current working tree. Git alone misses ignored checkpoints and datasets.

The working continuation config currently refers to this PPO checkpoint:
`logs/rl/official_run_6723c5cf9cfa/checkpoints/post_optimizer_update_00000527_steps_000000301056.zip`.
The more recent run pointer is `official_run_1ce2764a4ce6`, which had a saved
350,208-step checkpoint and telemetry at 356,352 steps during preparation.
It may still be running; no training was stopped to prepare this guide.
These are recorded targets, not a claim that either is the latest or best
model. Their ancestry includes `porto_bc_starting_point`; preserve full folders and
sidecars. The known BC checkpoint is also preserved separately in `porto_bc_20261030_fit`.

From PowerShell at the repository root:

```powershell
python scripts/prepare_migration.py `
  --include-run logs/rl/official_run_1ce2764a4ce6 `
  --include-run logs/rl/official_run_6723c5cf9cfa `
  --include-run logs/rl/porto_bc_starting_point `
  --include-run logs/rl/porto_bc_20261030_fit `
  --include-run logs/rl/porto_bc_20261030_collect `
  --include-run logs/rl/porto_ppo_native_continuation_20261103
```

Repeat `--include-run` for other runs to keep. `--dry-run` lists files and total
uncompressed size. All existing `logs/rl` data totals about 69 GiB on this PC;
the exporter intentionally requires an explicit selection. It includes tracked
and nonignored working files, including uncommitted source changes, but omits
local environments, editor/build caches, simulator players/the ignored Unity editor project,
outputs, credentials directories, `.env` files, and live hub registries.
Review the dry-run list for any other sensitive files before transferring.
Custom map files under `simulator/maps` and maintained Unity source patches
under `simulator/unity` are included. This is not a backup of
every file on the PC.

A source-only snapshot is already available at
`outputs/migration/aicar-source-migration.tar.gz` with a SHA-256 companion file.
It contains the working code and these migration tools, but no trained models.
Use the export command above after stopping training for the actual model move.

Copy the archive and its checksum to the server home directory:

```powershell
scp outputs/migration/aicar-migration.tar.gz outputs/migration/aicar-migration.tar.gz.sha256 USER@SERVER:~/
```

## 2. Restore on the server

Use a fresh destination so old run registries and Windows process IDs are not
adopted. The commands read the original base commit directly from the archive.

```bash
cd ~
sha256sum -c aicar-migration.tar.gz.sha256
base_commit=$(tar -xOf aicar-migration.tar.gz migration-manifest.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["base_commit"])')
git clone https://github.com/FarrellJoswara/AutoDRIVE.git AiCar
cd AiCar
git checkout "$base_commit"
tar -xzf ~/aicar-migration.tar.gz
python3 scripts/verify_migration.py
```

The verifier also reports tracked deletions from the original PC. Apply those
deletions manually if reported; the overlay itself cannot remove clone files.
The archive does not include `.git` or change remotes. Keep the archive as a
snapshot of the uncommitted work.

## 3. Build, check, start

```bash
bash scripts/server.sh build
bash scripts/server.sh doctor
bash scripts/server.sh start
bash scripts/server.sh status
```

`build` may take substantial time and disk space. It builds one ROS training/hub
image and pulls the official practice and competition simulator images.
`doctor` checks architecture, Docker CLI and Compose v2 availability, daemon
access, host NVIDIA status, and Compose configuration before running any image
commands. It then checks ROS/Python imports and an actual CUDA tensor operation
inside the training image. It does not
prove Unity graphics or camera correctness. Host ROS, Python venvs, and Node
are not required; the image contains them. Docker GPU access must already be
configured by the machine administrator. This repository does not change host
drivers or install privileged system services.

Use this server entrypoint, not `python main.py`: the latter retains the local
desktop/legacy image workflow. The standalone Compose file explicitly selects
the Docker backend because Linux otherwise defaults to the Windows/WSL native
backend. Only the hub service has a restart policy; active workers are separate
containers managed by the hub, not Compose services.

## 4. Open Mission Control from your PC

Keep this command running in a separate local terminal:

```powershell
ssh -N -L 8090:127.0.0.1:8090 USER@SERVER
```

Open http://127.0.0.1:8090 locally. If local port 8090 is occupied, forward
`8091:127.0.0.1:8090` and browse port 8091 instead. The hub binds to server
loopback; it has Docker host control and should remain behind SSH.
Training runs in server containers and survives SSH disconnection.
API workers join a dedicated control network to publish telemetry directly to
`brain:8090`; simulator discovery remains on each run's separate network.

## 5. Validate before scaling

Start with one environment and a short, separately named run. Explicitly set
`simulator_backend=docker`, `track_profile=practice`, `simulator_executable=null`,
`device=cuda`, and a finite `total_timesteps` such as 2048. Saved local configs
currently select `windows_external`, so do not reuse those fields unchanged.
Keep observation/action/reward settings from the model you intend to resume.
The existing settings schema supports 1–8 environments per learner; more CPUs
or GPUs do not automatically enable more environments or distributed PPO.
Increase to 2, 4, then 8 only after measuring throughput and memory.

Checkpoint paths in API worker containers use `/runs/rl/...`. Select the
transferred checkpoint through the UI and verify its ancestry before resume.
The server image retains `/app/logs -> /runs` and the original WSL repository
path as a symlink to `/app`, allowing existing Windows/WSL path aliases in
authenticated lineage records to resolve without editing hashed artifacts.
If moving artifacts from a different original PC path, change the
`AICAR_LEGACY_REPO_ROOT` build argument in `docker-compose.server.yml` and rebuild.
Do not rewrite hashed summaries or bypass lineage verification to force resume.

The Linux official simulator uses Xvfb with graphics enabled; never add
`-nographics` for camera observations. Confirm changing, nonblack camera frames,
fresh LiDAR/pose, progressing policy steps, checkpoints, and clean stop before a
long run. The prior 10-lap Porto result used Windows GUI rendering; performance
and sensor equivalence on this new Linux host remain to be established. A
healthy dashboard or CUDA check does not establish simulator readiness. Xvfb
can use software rendering on some hosts: inspect simulator logs and GPU use
before concluding the simulator uses the GPU.

```bash
bash scripts/server.sh logs
docker ps
nvidia-smi
docker image inspect aicar-server-runtime:local --format '{{.Id}}'
docker image inspect autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice --format '{{json .RepoDigests}}'
```

Use Mission Control Stop for individual runs. To stop all active official runs
cooperatively and then stop the hub:

```bash
bash scripts/server.sh stop
```

Do not use `docker compose down` as a training stop: run workers are dynamically
created outside Compose. Preserve the entire selected run directory, including
checkpoint sidecars and original BC evidence, when copying results back.
