# Controlled Porto evaluation on a PC

The public release asset is `porto-pc-checkpoint.zip`. Install it into a fresh
checkout using the repository installer below; it verifies the bundle manifest
and relocates the checkpoint and its run metadata under `logs/rl`.

## Purpose and scope

Evaluate the exact saved PPO checkpoint on the user's PC, using the repository's
`porto_practice_diagnostic` path and three fresh attempts of three scored laps
each. The checkpoint is an immutable evaluation input: do not resume training,
run PPO updates, or overwrite it. This is a practice diagnostic, not an IROS
competition score or a promotion of a competition-best model.

The planned source is the public `origin` repository. The destination branch is
`codex/porto-pc-handoff`, and the planned release tag is
`porto-pc-handoff-20261114`. The checkpoint release asset must be named exactly
`porto-pc-checkpoint.zip`.

## Exact checkpoint

Use the audited checkpoint from run `cd103472d411`:

- Source checkpoint: `logs/rl/official_run_cd103472d411/checkpoints/post_optimizer_update_00000026_steps_000001195886.zip`
- PPO step: 1,195,886 (optimizer update 26)
- SHA-256: `eb0b921a7133ff0ee20cc33564c15f5d93aa2d11ca43d473cd7043a455e6b70f`
- Route: `competition/iros2026/official_centerline.csv`, SHA-256
  `87d26be59ccc88ff6fc273905038906b8ed9e883b111c33de008a38f83a15bb3`

There is another archive at the same step in run `dbe197c60e03`, but its SHA-256
is `979966979c9ec680bd596ec6d54e042d37fc444c57492a43555f6c6d81ef7d92`.
It is not the selected parent. The release helper must package the `cd103472d411`
file above and preserve its exact hash. After download and extraction, verify
that hash before evaluating; do not infer checkpoint identity from its filename
or step count alone.

## Runtime and simulator

This repository documents two different Porto workflows. For this handoff,
follow the native Windows practice simulator workflow in
[`porto-native-runtime.md`](porto-native-runtime.md): install/use the official
2026-IROS practice build on Windows, and run the ROS 2 Jazzy Devkit bridge and
relay in WSL. Docker is not required for that workflow. The separate
[Windows Mission Control + Docker guide](official-docker-windows-host.md) is an
alternative app workflow and is not the evaluator command documented here.

Set `AICAR_OFFICIAL_SIMULATOR_EXE` to the actual Windows path of the installed
official Porto practice simulator executable. The hub uses that explicit path
for its native Windows simulator backend. The practice route is verified by
the evaluator against the checked-in route hash above. Use GUI rendering; the
documented practice build produced black camera frames in batch mode.

Use WSL with ROS 2 Jazzy and the AutoDRIVE Devkit set up at `~/autodrive_ws`. Run `bash scripts/setup_wsl_native_training.sh` to prepare the Python environment. The hub starts its own bridge on ROS domain 86, runs the neutral startup check, and launches a fresh GUI simulator for each attempt. Close any manually started bridge or simulator before submitting the replay.

## Install the artifact

Download the public asset from PowerShell:

```powershell
gh release download porto-pc-handoff-20261114 `
  --repo FarrellJoswara/AutoDRIVE `
  --pattern porto-pc-checkpoint.zip `
  --dir .\artifacts
Get-FileHash .\artifacts\porto-pc-checkpoint.zip -Algorithm SHA256
```

The release ZIP SHA-256 is
`6e7bd63220124c8e0ba6d60d351e656743a72e1479f0d8a3fa8703f970484069`.

Clone this branch into a fresh folder before downloading the asset:

```bash
git clone --branch codex/porto-pc-handoff https://github.com/FarrellJoswara/AutoDRIVE.git AutoDRIVE-Porto
cd AutoDRIVE-Porto
```

From the checkout, install the archive into the checkout root.
The installer verifies every bundled file against `artifact_manifest.json`,
copies the repo-relative `logs/rl` tree, and preserves checkpoint ZIP bytes:

```bash
python scripts/install_porto_pc_checkpoint.py artifacts/porto-pc-checkpoint.zip .
sha256sum logs/rl/official_run_cd103472d411/checkpoints/post_optimizer_update_00000026_steps_000001195886.zip
```

The resulting checkpoint SHA-256 must equal
`eb0b921a7133ff0ee20cc33564c15f5d93aa2d11ca43d473cd7043a455e6b70f` before
evaluation. The installer refuses to overwrite an existing `logs/rl` tree;
use a fresh clone. Keep the release ZIP, its hash, and the evaluation records.

## Start Mission Control and submit the diagnostic

In WSL, set the actual Windows simulator executable path, then start the
repository's native Mission Control launcher. The value must be the Windows
path for this PC; do not reuse another machine's path:

```bash
export AICAR_OFFICIAL_SIMULATOR_EXE='C:\Users\<user>\AppData\Local\AiCar\official-simulator\2026-iros-practice\autodrive_simulator\AutoDRIVE Simulator.exe'
HUB_HOST=127.0.0.1 bash scripts/start_native_mission_control_wsl.sh
```

Submit the evaluation to the running hub at `http://127.0.0.1:8090`. Set
`model_id` to the installed checkpoint path relative to `logs/rl`:

```bash
curl --fail-with-body -X POST http://127.0.0.1:8090/api/replay/start \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "official_run_cd103472d411/checkpoints/post_optimizer_update_00000026_steps_000001195886.zip",
    "map_id": "none",
    "device": "cpu",
    "trace_steps": 500,
    "validation_mode": "porto_practice_diagnostic"
  }'
```

Do not pass `--attempts` to the direct evaluator CLI: that CLI rejects three
attempts for this diagnostic. The hub Porto service owns the batch and starts
three fresh simulator attempts, each with three scored laps. Keep this
noncompetitive `porto_practice_diagnostic` mode and CPU policy runtime.

Check progress with `curl -fsS http://127.0.0.1:8090/api/replay/status`.
The hub records attempt IDs and paths to the per-run `evaluation.json` reports.

## Required report

Keep the complete evaluator JSON and a concise summary alongside it. Record:

- release tag, artifact SHA-256, extracted checkpoint SHA-256 and checkpoint
  step;
- repository commit, Windows version, GPU/driver, simulator executable and
  scene hashes, practice route hash, ROS domain, relay address/port, and
  evaluator command;
- for each of the three attempts: fresh simulator identity, readiness result,
  three lap times, completed laps, race/raw collisions, stop reason, camera
  fresh/missing/stale counts, invalid route samples, motion verification,
  control-gap faults, ROS stamp anomalies, median control interval and camera
  age when available;
- all nine lap times, total valid laps, collision count, median flying-lap
  time, and whether the diagnostic readiness/clean gate passed;
- report/log paths, any interruption or infrastructure fault, and cleanup
  result.

For a clean diagnostic, all three attempts must pass readiness and complete
exactly three positive finite timed laps each, with zero collisions, fresh
camera samples, valid route projection, verified motion, and no monotonic
control-gap faults. Treat missing metrics as missing, not zero. Preserve failed
attempt reports; do not rerun over or discard them. If an attempt fails due to
simulator startup, sensor readiness, or transport, record it as infrastructure
failure separately from policy behavior. If the evaluator exits early, the
result is incomplete, not a passing 3×3 evaluation.

## Findings before the PC handoff

The same selected checkpoint has conflicting recent evidence:

| Evaluation evidence | Result |
| --- | --- |
| Earlier replay, run `2de825038a08` | 9/9 clean laps across three attempts; median flying lap 24.768 s |
| Later replay, run `6b686ed46208` and companion attempts | 0 completed laps across three attempts; all three collided before lap one |
| Fresh parent control, evaluation `ab37b9aa65294a5d` | 0/9 completed laps; three impacts across three attempts; clean gate failed |

The earlier clean and later failed replay used the same checkpoint hash
`eb0b921a…e6b70f`, simulator executable hash, and recorded effective manager
settings. Later control intervals were about 72.7–76.8 ms versus 59.8–60.4 ms
in the earlier clean runs; camera frames were also older on average. The
November 12 instrumented replay reproduced collisions, but its instrumentation
could not identify when the simulator applied each command. The timing shift is
real in the recorded traces; its cause remains unresolved. Xorg display-number
correlations do not establish causation.

Therefore this PC run is a controlled portability and runtime comparison, not
a presumption that the model is good or bad. Preserve the exact artifact and
all telemetry. A different PC can change timing and rendering conditions, so a
pass or failure there does not by itself identify the cause of the earlier
regression.

## Caveats and interpretation

- A `porto_practice_diagnostic` result is an engineering practice result;
  `valid_for_comparison` is intentionally false and it is not a competition
  score.
- The selected step-1,195,886 checkpoint has recent reproducible collision
  evidence as well as earlier clean evidence. Do not promote it or start
  production PPO from this handoff alone.
- Route geometry is the official Porto practice centerline; repository map
  assets are not a substitute for the separately installed official simulator
  scene.
- Fresh processes, sensor freshness, neutral start, and timing traces are part
  of the experiment. Reusing a stale simulator session weakens the comparison.

## Final handoff note

Use the installer in a fresh clone, verify the selected checkpoint SHA-256,
set `AICAR_OFFICIAL_SIMULATOR_EXE` to this PC's Windows executable path, start
`scripts/start_native_mission_control_wsl.sh`, and submit the specified
`/api/replay/start` request. The hub performs all three Porto diagnostic starts.

The archive was installed and its lineage verified in a different directory on Linux. The Windows replay itself remains to be run on the PC.
