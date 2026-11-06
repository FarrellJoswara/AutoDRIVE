# Native Linux learner and Mission Control

The native learner uses RoboStack Jazzy Python 3.12, its NumPy/OpenCV and
`cv_bridge`, plus a filtered overlay of the CUDA learner venv. Always invoke
the selected `$AICAR_NATIVE_PYTHON`; running the learner venv Python directly
does not satisfy the native backend's interpreter identity check.

```bash
cd /home/f337j378/F1Tenth
source scripts/native-linux-env.sh
"$AICAR_NATIVE_PYTHON" -c 'import torch, rclpy, cv_bridge; print(torch.ones(1, device="cuda"))'
```

Current user-space paths:

- `AICAR_NATIVE_ROOT=/tmp/aicar-ros-O15Zu3`
- ROS prefix: `/tmp/aicar-ros-O15Zu3/jazzy`
- Built official Devkit: `/tmp/aicar-ros-O15Zu3/autodrive_ws`
- Learner venv: `/home/f337j378/.venvs/aicar`
- Filtered package overlay: `/tmp/aicar-ros-O15Zu3/venv-overlay`
- Runtime: `/home/f337j378/F1Tenth/.native-linux`, currently a symlink to
  `/tmp/aicar-runtime-yNEyb7/native-linux`

Both `/tmp` roots are ephemeral. Preserve or provision them again after host
cleanup or reboot. The activation script accepts `AICAR_NATIVE_ROOT` and
`AICAR_LEARNER_VENV` overrides, but an installed Conda prefix should be
recreated at its intended location rather than moved without relocation.
RoboStack owns NumPy/OpenCV; the overlay excludes the venv copies to keep
`cv_bridge` on its matching compiled libraries. `PYTHONNOUSERSITE=1` is set.
The official Humble/Ubuntu 22 Python 3.10 image must not be overlaid onto this
Jazzy/Python 3.12 environment.

Start Mission Control with the wrapper:

```bash
cd /home/f337j378/F1Tenth
HUB_HOST=127.0.0.1 HUB_PORT=8090 scripts/native-linux-hub.sh
```

The wrapper sources the environment and executes Uvicorn with the actual
RoboStack Python. `/health` and `/` previously returned HTTP 200 using this
wrapper. CUDA tensor creation, real `rclpy`/`cv_bridge` imports, a bridge image
conversion, a separate ROS publisher/subscriber roundtrip, and the official
Devkit headless launch/listener have passed. The source checkout in the
scratch workspace has a port override patch for `AICAR_DEVKIT_PORT`.

Native Linux training uses the extracted official **practice** release and is
verified for practice training. Competition-track training is not offered on
this backend. Competitive evaluation selects the separately allowlisted
official competition runtime. A bounded competition Replay smoke has now
verified player startup, camera/telemetry, and route/runtime provenance; a full
valid three-attempt competition evaluation has not been completed. Selecting
Linux in Train does not make the practice player competition-capable.

Automatic competitive evaluation from Linux practice training is supported
with the allowlisted official competition runtime. One bounded live cycle
verified pause acknowledgement and a post-optimizer checkpoint, retirement of
the practice player, competition player identity and inference, a nonblack
camera stream, relaunch of the practice player, readiness on its fresh SID,
resume acknowledgement, and a subsequent PPO optimizer update. The report is
`.native-linux/runtime/autocycle-corrected-run-CTiI4o/execution-continuation.json`;
all 11 phase checks passed in 248.06 seconds. The pause checkpoint was at step
832/update 36, and post-resume training reached step 896/update 40. The
competition player matched the pinned `sha256:749fbef…4663d` image.

The monitor intentionally stopped the competitive evaluation after 204.7
seconds; it finalized as incomplete and produced no valid race score. This
cycle verifies one bounded lifecycle, not a completed three-attempt race,
competition performance, or every failure/recovery path. A prior
`.native-linux/runtime/single-auto-cycle-held-a2fb8z85/execution.json` run
ended at its 448-step training limit while its pause handshake was still
pending. The monitor now needs enough training headroom for evaluation and
resume before the configured training limit. Keep the full-race result
unverified until a completed evaluation produces a valid score.

Activation defaults `AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S=0.20`; an
explicit value such as `0.0` remains unchanged. The smoke request explicitly
sets its `safe_neutral_fallback_s` config from that environment value and
records it in the result, so the manager's settings serialization preserves
the actual value. This is not the Mission Control/API default: the application
settings model defaults to `0.0`. UI/API runs must explicitly set the field to
`0.20` if that fallback is intended; sourcing the native activation script
alone does not change a separately supplied UI/API setting.

## Bounded integration smoke

The exporter owns `runtime/IMAGE-IDENTITY.json` and the OCI completion marker.
Wait for its verified identity. The smoke script does not manufacture those
markers or use import shims. The earlier `runtime/camera-probe.json` is a
failed probe (`probe_passed: false`, zero frames, and a cleanup `AttributeError`);
it is not evidence of camera output. The later
`.native-linux/runtime/wrapper-camera-probe-47942.json` records 10 distinct,
nonblack frames from the official wrapper.

With the hub running, and after the runtime worker has released its player:

```bash
cd /home/f337j378/F1Tenth
source scripts/native-linux-env.sh
"$AICAR_NATIVE_PYTHON" scripts/native-linux-smoke.py --launch-preflight
"$AICAR_NATIVE_PYTHON" scripts/native-linux-smoke.py --run --steps 2048
```

The preflight starts the official bridge, relay and attested launcher briefly,
publishes neutral controls, and requires recent nonblack camera, valid LiDAR,
and finite position messages. It cleans up its player/bridge processes before
the manager creates a new practice run. `--run` uses one environment, CUDA,
64-step PPO rollouts, a finite timestep budget and a 15 minute wall limit.
Automatic competitive evaluation and BC are disabled for this PPO smoke.
The script asks the manager to stop on exit and lists saved ZIP checkpoints.
New results default to unique `.native-linux/runtime/native-smoke-<time>-<pid>.json`
files; an existing explicit `--state-file` is rejected. Each accepted run
also has `native_smoke_evidence.json` in its run directory. Bridge/player
logs go to the corresponding `logs/rl/official_run_preflight-*` directory.

The ROS gate requires at least two messages per sensor, advancing nonzero
camera/LiDAR ROS header stamps, header ages and current receive ages within
one second by default, a nonblack image, valid scan ranges, and finite pose.
Evidence includes per-message ROS stamps, wall/monotonic receive times and
sample ages. IPS uses `geometry_msgs/Point`, which has no header; its live
volatile ROS receipt times are recorded explicitly instead of inventing a
sensor timestamp. Older historical reports predate this stronger gate.
Checkpoint save, resume and replay are only considered verified after an
actual successful run. `--resume /absolute/logs/checkpoint.zip` submits a
manager-owned resume, with `--steps` as the additional transition budget;
checkpoint metadata must reach the parent count plus that budget. Replay is
not automated.

Do not run concurrently with the runtime worker's camera probe or another
player. The preflight checks active manager runs and player manifests and
uses `runtime/.native-runtime-smoke.lock`; the runtime launcher controls its
own display. Its lock must be honored by any external probe as well.
Without `--launch-preflight` or `--run`, the script inspects an existing ROS
stream in the caller's `ROS_DOMAIN_ID` and launches no player.

## Integration results, 2026-11-04

The attested runtime passed the backend identity validator. Actual ROS
preflight received a nonblack 192x108 `rgb8` camera image, 1035 valid LiDAR
ranges of 1081, and finite IPS position (0.8008, 3.1583, 0.0592).
Evidence: `.native-linux/runtime/ros-preflight-pass.json`.

Manager run `7a83287920c0` requested 2048 CUDA PPO transitions with one
environment and 64-step rollouts. It saved a genuine post-update checkpoint
at 64 transitions / four optimizer epochs, then failed on an episode reset:
`src/layer2/official_race_env.py:281` rejected a position 0.80 m from the
configured IPS spawn (limit 0.75 m). At that time, no application source had
been edited. The later reset change sends neutral throttle and steering before
training reset. Live runs below exercised it through repeated collision
resets without another spawn rejection. The traceback remains in
`logs/rl/official_run_7a83287920c0/hub_train.log`.

A separately configured 64-step run, `842fa7612335`, reached its requested
step limit and saved `logs/rl/official_run_842fa7612335/official_policy_latest.zip`.
The final verification report and checkpoint reload evidence are under
`.native-linux/runtime/integration-final-verification.json` and
`.native-linux/runtime/ppo2048-checkpoint-load.json`. Its report records CUDA,
64 transitions, four PPO updates, and checkpoint SHA256
`b455ec8ffc6dd8f2c1222a66f56c1f71ddfccca8d83796b758444033c6385056`. It was
manually connected through the GUI; its `simulator_stream_health.json` records
three camera frames. This is not unattended-startup evidence or proof of nine
changing frames. It is superseded for unattended-startup verification by the
later runs below.

The first 64-step evidence above required manual GUI connection. The later
256-step run used the wrapper's automatic Connect path without manual input.
Its player log records a verified TCP connection, and the manager's stream
health recorded camera frames and Devkit commands.

The exact environment, historical process IDs, ports, HTTP routes, ROS topics,
route/spawn metadata and source fingerprints are preserved in
`.native-linux/runtime/reviewer-environment-evidence.json`. These process
IDs are historical, not running jobs. The successful checkpoint is preserved
with SHA256 `b455ec8ffc6dd8f2c1222a66f56c1f71ddfccca8d83796b758444033c6385056`.

## Unattended practice training and resume verification

The integration report `.native-linux/runtime/integration-MLhAvQ/summary.json`
records two unattended CUDA PPO runs with safe-neutral fallback `0.20` and no
manual GUI input. Run `8cf52e2cfca4` completed 256 transitions, four
optimizer-audit records, and four collision episodes; its post-update
checkpoints reached 64/128/192/256 steps. The reset guard did not reject a
spawn during those resets. The checkpoint was reloaded on CUDA at 256 steps /
16 PPO updates with
SHA-256 `1453c3e30072b9b66af064a300f9bad6e4f94f65c8688bc8ec70083c8d1d5be0`.

Resume run `78b8f4332e50` loaded that checkpoint and stopped after two bounded
rollouts at step 387. The manager reported exit code 0 and no cleanup error;
this was an intentional operator stop before the full additional 256-step
budget. It recorded one collision reset and saved a 387-step checkpoint. The
summary confirms no remaining process-group members, closed simulator/relay
ports, exited player/Xorg processes, and shutdown of the test hub.

The additional two-environment CUDA infrastructure smoke, run
`138873696d0b`, completed its 128-step limit with the practice player, fallback
`0.20`, and a saved checkpoint. Its monitor captured nonblack camera frames
from both environments and separate ROS domains 0 and 1. The report is
`.native-linux/runtime/two-env-final-ngztauz5/monitor.json`; it verifies
multi-environment startup and telemetry, not long-run training performance.

The earlier unattended 128-step attempt in
`.native-linux/runtime/resumed-unattended128-20261104-1749.json` still records
a real connection failure (zero Socket.IO clients and frames). It predates the
successful wrapper change; preserve it as failed historical evidence. The
current successful 256-step report and per-run stream-health records are the
evidence for unattended connection and reset/resume behavior.

## Evaluation and model migration status

Native Linux training is verified on the official practice release. The
allowlisted competition runtime is configured and its identity is validated.
The bounded competitive Replay smoke in
`.native-linux/runtime/competition-replay-final-1q7upbpk/execution.json` used
checkpoint SHA-256
`6a6d311464240c761803e090e60d6392c6089211918b6aeff0453850431dd2c7` without
manual GUI input. Its first attempt launched the official competition player,
received 96 fresh camera samples, and ended in a collision before completing a
lap; the second was stopped at 40 observed steps. The batch stopped after two
of three attempts, with zero completed laps and no valid comparison score.
This verifies bounded Replay startup and telemetry, not a complete competition
evaluation or a competition score.

The attempt report `logs/rl/official_run_f3b292ce917a/evaluation.json` records
the competition image digest `sha256:749fbef…4663d`, the process executable
under `.native-linux/competition-runtime/rootfs`, and the attested identity
file. Its identity-file SHA-256 matches the current marker. Route provenance
matches the source archive: archive SHA-256
`bc350d1f3211083c251d292de20fef6f9024ca5b6fe4a02a4985a754df727954`, archived
route SHA-256
`62307684f1fd2813e50222f0c97e3545dc7c2808ef8544c1e161f7744d1ba4fb`; the
working-tree LF route is accepted only after reconstructing those exact CRLF
bytes.

The recent practice diagnostic request returned HTTP 400 because Replay's HTTP
validation still rejects `porto_practice_diagnostic`; these PPO checkpoints
also lack the authenticated BC provenance required by that diagnostic service.
No practice diagnostic result is claimed.

The source-only migration archive
`outputs/migration/aicar-source-migration.tar.gz` exists and its SHA-256
companion verifies. Its migration manifest has `included_runs: []`, so it
contains no trained-model run artifacts. The inspected `logs/rl` outputs
contain the new Linux practice checkpoints; no legacy Windows checkpoint was
present in the model artifacts reviewed. This does not establish that no copy
exists elsewhere. Resume or replay from a legacy Windows checkpoint is
therefore not verified or currently available from these artifacts.
