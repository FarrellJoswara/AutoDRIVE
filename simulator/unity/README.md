# AutoDRIVE Unity source + AiCar TrackLoader (Phase 3)

AiCar patches live under `Assets/Scripts/AiCar/`. Full AutoDRIVE-Simulator is
cloned locally to `simulator/unity/AutoDRIVE/` (gitignored).

## Scripts (tracked in this repo)

| File | Role |
| :--- | :--- |
| `TrackLoaderBootstrap.cs` | Injects TrackLoader + ForceConnect at boot |
| `AiCarSimulationGate.cs` | Holds simulator physics during PPO policy updates |
| `TrackLoader.cs` | Load `maps/<id>/mesh/*.obj` + MeshCollider (ROS metres, identity XZ) |
| `ForceConnect.cs` | Auto-activate Socket.IO when `-ip`/`-port` or batchmode |
| `MapConfig.cs` | `-map-id` / `AICAR_MAP_ID` / `.active_map.json` |
| `Hotfix/ForceConnectHotfix.cs` | Optional Windows Mono drop-in (see below) |
| `Editor/AiCarPlayerBuild.cs` | Batch Windows + Linux player builds |

## Why ForceConnect?

F1TENTH.unity wires `SocketConnection` (UI **Connect** button) and inactive
`Socket` / `SocketIO` objects, but **does not include `CLIManager`**. Stock
`-ip` / `-port` therefore only prefill UI fields when CLIManager is present;
`SocketIOComponent.Connect()` never runs until the button is clicked. Docker
`-batchmode` expected CLIManager to `SetActive` the sockets — without it, the
brain sees zero TCP and pose/lidar stay empty.

`ForceConnect` parses `-ip`/`-port`, writes InputFields + `IPAddress`/`PortNumber`,
activates `Socket`/`SocketIO`, bounces the GameObject so `OnEnable` rebuilds the
WebSocket URL, and invokes `Connect` / `ToggleSocketConnection` via reflection.

## Build

Needs **Unity 2022.3.52f1** + Windows/Linux standalone support.

Copy tracked scripts into the AutoDRIVE project first:

```powershell
Copy-Item -Recurse -Force simulator\unity\Assets\Scripts\AiCar `
  simulator\unity\AutoDRIVE\Assets\Scripts\AiCar
Copy-Item -Recurse -Force simulator\unity\Assets\Plugins\SocketIO `
  simulator\unity\AutoDRIVE\Assets\Plugins\
```

The tracked Socket.IO overrides use UTC for worker-thread timestamps and avoid
Unity's crashing local-time formatter when a bridge connection closes.

Then:

```powershell
git clone --single-branch --branch AutoDRIVE-Simulator --depth 1 `
  https://github.com/Tinker-Twins/AutoDRIVE.git simulator\unity\AutoDRIVE
bash simulator/unity/AutoDRIVE/Tools/unzip-and-clean.sh
# copy Assets/Scripts/AiCar and the tracked SocketIO overrides into AutoDRIVE
# (see above), then:
& "C:\Program Files\Unity\Hub\Editor\2022.3.52f1\Editor\Unity.exe" `
  -batchmode -nographics -quit `
  -projectPath "$PWD\simulator\unity\AutoDRIVE" `
  -executeMethod AiCar.Editor.AiCarPlayerBuild.BuildAll `
  -logFile logs\layer4\unity_build_all.log
```

Windows-only / Linux-only: `BuildWindowsOnly` / `BuildLinuxOnly`.
For a Linux player build that stays in `simulator/_build/linux-cpu-experiment/` without
replacing the currently running Docker simulator, use
`BuildLinuxStagingOnly`.

Outputs: `simulator/windows/` (Mono), Linux `.x86_64` + `Data/` (IL2CPP),
and `simulator/.aicar_trackloader`.

## Camera capture and training compatibility

The local AutoDRIVE `Socket.cs` has an experimental
`AICAR_DISABLE_CAMERA_STREAM=1` switch for batch mode. Keep this switch unset
for existing training runs. It omits front/rear (or left/right) image fields and
skips their readback, JPEG encoding, and Base64 encoding. It does not disable
the camera component or alter the LiDAR raycasts. Python's current RL pipeline
uses LiDAR and vehicle state, and does not consume these image fields.

The capture work still affects simulation timing. Socket.IO applies commands
in `FixedUpdate`, and the dispatcher generates telemetry in `Update`. Unity's
physics clock advances independently between Bridge exchanges; there is no
fixed number of physics ticks per `Racer.step`. Removing synchronous capture
therefore changes the simulated time covered by the same sequence of RL
actions. A frame rate cap can match one workload's average while changing its
interval distribution or another workload's timing. A camera omission mode
compatible with the legacy asynchronous timing has not been established.

### Optional fixed-duration camera experiment

The fixed-step staging player adds an explicit action clock. Opt in through
`Racer(action_interval_s=...)`, `AutoDriveEnv(action_interval_s=...)`, or the
trainer's `--action-interval-s` option. Each action and reset is acknowledged
with its step ID, simulated clock, and physics tick count. The client rejects
missing or inconsistent acknowledgments. Between actions, physics is paused
while socket events remain responsive. The physical timestep and LiDAR stay
unchanged. Rewards, route timers, and evaluation use the simulated duration.

`frame_skip` multiplies this interval: at `0.086` seconds and `frame_skip=4`,
one policy decision lasts `0.344` simulated seconds. This is a deliberate
control-timing migration; it cannot exactly recreate the legacy mode's
load-dependent timing. Existing checkpoints need driving evaluation at the
chosen interval before resuming training. Defaults remain in legacy mode.

The fixed-step staging build sets Unity's `job-worker-count=1` in its Linux
`boot.config`. Fixed-camera training caps `Application.Update` polling only
while waiting for the next action: normally 30 FPS, or 10 FPS for camera-off
pools of eight or more. During an action, Unity remains uncapped and runs the
same fixed-tick batch. Legacy simulator mode does not set either optimization.
Set `AICAR_ACTION_IDLE_TARGET_FPS` to override the idle cap when launching the
fixed-step player directly; it must be a positive integer.

Prepare the local vendor project first with
`python simulator/unity/scripts/prepare_action_step_project.py`. This applies
the Bridge hooks and copies the tracked gate, Socket.IO, and build overrides.
Build only the candidate with `BuildLinuxFixedStepStagingOnly`; it stays under
`simulator/_build/linux-fixed-step-experiment/`. The camera suppression switch
is separate: `AICAR_DISABLE_CAMERA_STREAM=1` skips readback/encoding, leaving
the camera component enabled. Compare both values with
`scripts/test_camera_fixed_interval.py` using the same staged player. The
benchmark compares full LiDAR, vehicle state, observations, rewards, and
scripted or deterministic policy actions, and reports Unity CPU seconds per
simulated second. Set `AICAR_SIMULATOR_PATH` to the staged executable only in
the shell running the experiment; publication is a separate action.

`scripts/test_camera_telemetry_compatibility.py` verifies that image fields do
not affect parsed telemetry, observations, or reward calculations given the
same other inputs. It does not test simulator timing equivalence.
`scripts/probe_camera_bridge_timing.py` measures actual Unity fixed time and
Bridge IDs between returned steps using an isolated diagnostic player. It
starts only probe simulators, with no CPU limit or RL model updates, and shuts
them down afterward. The diagnostic player must emit `AICAR Probe Fixed Time`
and `AICAR Probe Frame`; the active training player should not be replaced by
that instrumented build.

The October 4, 2026 diagnostic results are retained in
`logs/diagnostics/camera_capture_fourway_fixedtime_unpaced.json` and
`logs/diagnostics/camera_capture_fourway_fixedtime_capped16.json`. Each compares
camera modes within one binary, with four concurrent Unity processes and the
same commands. `fixed_time_per_returned_step_s` includes the normal Bridge
callback scheduling; `fixed_time_per_bridge_mean_s` also accounts for returned
steps spanning more than one Bridge ID by dividing elapsed fixed time by the
observed ID increase. This is an aggregate mean; intermediate event timestamps
are not sampled separately.

The October 4, 2026 fixed-step scaling check compared four camera-off
environments at 60 FPS with five at 30 FPS, using the same fixed 0.086 s action
interval and scripted actions. The five-at-30 configuration delivered 3.271
aggregate simulated environment-seconds per wall second versus 3.140 for
four-at-60 (+4.2%), while using about 3.842 versus 3.508 Unity CPU cores
(+9.5%). Per-environment throughput was lower at five-at-30 (0.654 versus
0.785 simulated seconds per wall second). This was a short simulator-only
measurement, excluding PPO update time and longer-run resource contention; it
suggests five-at-30 can modestly increase total collection throughput if CPU
headroom exists, but does not establish higher end-to-end training throughput.

## Linux Dedicated Server evaluation (October 4, 2026)

`BuildLinuxFixedStepServerStagingOnly` builds an isolated Linux Dedicated Server
candidate at `simulator/_build/linux-fixed-step-server-experiment/`; it never
publishes over the active simulator. The target must use
`NamedBuildTarget.Server` when setting the IL2CPP backend. Setting only the
regular `Standalone` target produced a Mono server build, which is not a fair
comparison with the training player and was discarded. The candidate also uses
`job-worker-count=1`, matching the fixed-step baseline.

The normal player and IL2CPP Dedicated Server were compared with camera
streaming disabled, the same seeded scripted controls, Porto map, and 0.086 s
action interval. Three alternating-order pairs ran for 30 simulated seconds
each (349 actions per build). Every pair had exact matches for all recorded
vehicle telemetry, all 1081 LiDAR beams, observations, rewards, actions,
protocol IDs, and physics ticks per action; simulated-time deltas also matched
exactly. The full report and raw traces are in
`logs/diagnostics/dedicated_server_equivalence_long.json` and `.npz`.

This did not reduce training CPU demand in the test. Median Unity CPU use was
0.771 CPU seconds per simulated second for the normal player and 0.781 for the
server candidate; the median paired comparison used 1.5% more CPU in the server
build. Median simulation throughput was 0.940 versus 0.931 environment-seconds
per wall second. The server package was about 2.0% smaller on disk. These runs
were serial and low priority while the user's
training run remained active; PPO update cost was not measured. The exact
behavior match is confirmed, but the CPU and throughput results do not justify
using the Dedicated Server build as a training optimization. Keep the normal
fixed-step player for training unless later isolated tests show a repeatable
end-to-end gain.

The repeatable runtime comparison is
`scripts/test_dedicated_server_equivalence.py`; it compares both staged players
without changing the active simulator. Unity 2022.3.52f1 can build the Server
subtarget after Linux Dedicated Server Build Support is installed. The Unity
CLI module inventory may still show that module as available even when the
server player variants are present; a successful staged build is the check used
here.

## Idle action-polling cap evaluation (October 4, 2026)

The fixed-step player caps `Application.Update` only while it is paused between
actions. Each action still runs the same uncapped fixed-tick batch. Lowering
`AICAR_ACTION_IDLE_TARGET_FPS` therefore saves idle-loop CPU but adds wait time
for the next Bridge action. It leaves the action interval, physics timestep,
LiDAR, and vehicle scripts unchanged.

The A/B used the same staged player, camera disabled, Porto map, 0.086-second
actions, deterministic controls, and a separate seed per worker. Probes ran at
low process priority while the user's six-environment training run stayed
active. Across three 20-simulated-second repeats, four environments at 30 FPS
delivered 2.625 aggregate simulated environment-seconds per wall second and
used 1.856 Unity CPU cores on average. Eight environments at 10 FPS delivered
3.064 environment-seconds per wall second (+16.7%) and used 1.871 Unity CPU
cores (+0.8%). Unity CPU per simulated second fell from 0.707 to 0.611 (-13.6%).

All 24 paired worker traces matched exactly: actions, protocol IDs, physics
ticks, simulated time, vehicle telemetry, full 1081-beam LiDAR, observations,
and rewards. No simulator payload hash changed. The raw report and per-step
traces are in `logs/diagnostics/idle_action_fps_4x30_vs_8x10_long.json` and
`.npz`. The experiment is reproducible with
`scripts/test_idle_action_fps.py`.

This is a CPU-for-parallelism tradeoff, not a free speedup. With the same eight
environments, 30 FPS delivered 4.929 environment-seconds per wall second;
10 FPS delivered 3.068 (-37.8%) while using about half the Unity CPU per wall
second. Keep 30 FPS for a fixed-size pool when CPU headroom exists. For the
camera-off training mode, the hub now selects 10 FPS at eight or more training
environments and retains 30 FPS below eight. That threshold is based on the
measured four-versus-eight pool comparison and does not affect legacy or
camera-on modes. It only changes how quickly an idle Unity player notices the
next action; it does not alter physics or the response to a given action trace.

The pool test measures simulator collection, not complete PPO throughput. The
training run's policy inference, rollout/update cost, and longer-term contention
can change the end-to-end result. Recheck training FPS and learning progress
when scaling above the current environment count. The settings regression is
covered by `scripts/test_simulator_idle_fps_settings.py`.

## CPU Profiler capture (October 4, 2026)

`BuildLinuxFixedStepProfilerStagingOnly` creates a Unity Development player in
`simulator/_build/linux-fixed-step-profiler-experiment/` and never publishes it
over the training simulator. `scripts/capture_unity_cpu_timeline.py` records a
bounded fixed-step run with Unity's `-profiler-log-file` option. Load the `.raw`
capture into the Editor Profiler to inspect the CPU Timeline; the accompanying
`AiCarProfilerExport.ExportLoadedProfile` method exports the CPU hierarchy to
JSON for repeatable, headless summaries. These Development-build timings are
for locating hot paths, not for estimating release-player throughput.

The startup capture covered 300 frames, 93 actions, and 7,998 physics ticks over
8 simulated seconds. During the first 12 seconds, `TrackLoader.Update` used
2.91 seconds of main-thread self time and allocated frequently while its
15-second baked-track suppression loop repeatedly searched scene objects. Its
per-frame cost fell to 0.1 ms total in the later capture, after that one-time
window ended; it is therefore a startup cost rather than a sustained per-step
cost for long runs.

The later capture covered the final 300 frames of a 25-simulated-second run.
The main thread spent 2.11 seconds inclusive in `Physics.Simulate` across those
frames. `LIDAR.FixedUpdate` took 308 ms inclusive, including 110 ms in 137,287
raycasts. This makes the physics engine the main persistent simulator hot path;
the current LiDAR raycast workload is measurable but smaller. Compression
methods do not appear as individual CPU samples in this non-deep capture, so
telemetry serialization still needs marker-based inspection before drawing a
conclusion about its share.

The Development player was compared with the normal fixed-step player in three
20-simulated-second pairs (233 actions per run). All compared actions, physics
ticks, protocol IDs, simulated-time deltas, vehicle and LiDAR telemetry,
observations, and rewards matched exactly (zero value mismatches). The
Development player's measured throughput varied between pairs and is not a
performance result. Raw captures and exports are local under
`logs/diagnostics/unity_cpu_timeline*`; they are intentionally not source
artifacts because raw captures exceed 300 MB.

## Unity job-worker count A/B (October 4, 2026)

The staged fixed-step Linux player defaults to one Unity job worker. The
`scripts/test_simulator_job_workers.py` harness changes only the staged
player's `boot.config`, restores it in a `finally` block, and compares complete
fixed-step traces for a serial pool of environments. At both four environments
(three 20-simulated-second paired repeats) and eight environments (two
12-second paired repeats), raising the Unity job-worker count from one to two
preserved actions, physics ticks, protocol step IDs, simulated time, all
vehicle/LiDAR telemetry, observations, and rewards exactly. The staged player
payload hashes remained unchanged and its original boot configuration was
restored.

The change did not improve pool throughput: at four environments the median
throughput ratio (two workers / one) was 0.989, while Unity CPU per simulated
second rose 17.1%. At eight environments the median throughput ratio was 0.997
and CPU per simulated second rose 9.6%. This knob therefore does not reduce
CPU per training interval or increase collection throughput on the measured
machine. Keep the one-worker setting. Reports and traces are in
`logs/diagnostics/job_worker_1_vs_2_4env.json` and
`logs/diagnostics/job_worker_1_vs_2_8env.json`.

## Telemetry serialization profiling (October 4, 2026)

Temporary `ProfilerMarker`s around `DataCompressor.CompressArray` and the
main-thread telemetry callback were compiled into a separate Development
player. The original ignored AutoDRIVE source files were restored immediately
after the staged build; the training players were not rebuilt or restarted.
The instrumented player matched the normal fixed-step player exactly in two
12-simulated-second pairs (140 actions and 12,040 physics ticks per pair):
vehicle telemetry, full LiDAR, observations, rewards, actions, protocol IDs,
and simulated time had zero mismatches. The instrumented build is diagnostic
only; its timing is not a release performance benchmark.

The capture recorded the final 300 frames after the startup scan window. Over
10 simulated seconds and 6,450 physics ticks, `Physics.Simulate` used 1.21 s
self time / 2.09 s inclusive. `LIDAR.FixedUpdate` used 176 ms self / 320 ms
inclusive. Telemetry callbacks used 55 ms self / 102 ms inclusive across 75
calls. The 150 range and intensity gzip calls used 32 ms self / 34.6 ms total.
Compression is therefore about 1.7% of Physics.Simulate inclusive time; even
caching the constant intensity payload could save only a small fraction of
total per-environment CPU. Telemetry serialization is not a promising next
optimization for the current workload. The raw capture and hierarchy export
are local at `logs/diagnostics/unity_cpu_timeline_telemetry_steady.raw` and
`.json`; the exact-trace report is
`logs/diagnostics/telemetry_profiler_exact_behavior_ab.json`.

## LiDAR allocation candidate A/B (October 4, 2026)

An isolated release IL2CPP candidate reused the per-scan ray-direction and
hit arrays and formatted the constant intensity once per scan. It left the
ray-angle formula, raycast order, mask, range, physics calls, and serialized
sensor values unchanged. The temporary edit was restored from backup after
the build; only the staged player and test harness remain. The candidate
matched the baseline exactly across all compared LiDAR beams, vehicle
telemetry, observations, rewards, actions, physics ticks, protocol IDs, and
simulated time: 12 paired environment traces at four environments (three
20-second repeats) and 16 traces at eight environments (two 12-second
repeats). Player payload hashes were unchanged.

The candidate did not improve performance. At four environments its median
pool throughput was 2.6% lower and CPU per simulated second was 2.1% higher;
at eight environments throughput was 4.0% lower and CPU per simulated second
was 5.5% higher. All five paired throughput comparisons favored the baseline.
This allocation/formatting change should not be adopted. The results are
simulator-pool measurements under the live training workload, not a claim
about PPO update throughput. Reports and per-step traces are in
`logs/diagnostics/lidar_allocation_4env.json` / `.npz` and
`logs/diagnostics/lidar_allocation_8env.json` / `.npz`.

The next low-risk simulator avenue is a separate experiment with Unity 2022.3's
`RaycastCommand.ScheduleBatch` for the existing 1,081 LiDAR rays. Keep every
direction, mask, range, and ray count identical, complete the batch before
formatting telemetry, and accept the candidate only if exact trace equality
holds and release-pool CPU per simulated second improves repeatably. Unity
documents that these commands run asynchronously in parallel and their result
buffers cannot be read before the returned `JobHandle` completes; timing and
result ordering therefore require direct measurement rather than assumption.

## CPU hot-path trace and next candidate (October 4, 2026)

The steady-state CPU Timeline capture above is 300 frames from a Unity 2022.3.52f1
Development player with camera output disabled. It includes 6,450 physics steps.
Use per-thread self time to rank work; parent totals include their children and
must not be added together. The Main Thread spent 1.21 s inside `Physics.Simulate`
(0.188 ms per physics step), with 2.11 s inclusive. Main-thread PhysX self-time
was led by contact-manager updates (198 ms), broad phase (90 ms), vehicle update
(85 ms), post-broad-phase work (76 ms), island generation (73 ms), and lost
contact processing (70 ms each for two stages). The Worker 0 thread also did
measurable PhysX work (about 400 ms across its larger stages); the other worker
threads were mostly idle. This points to real physics/contact work as the largest
steady cost, rather than telemetry compression or graphics.

The most specific avoidable-looking cost is transform synchronization around
vehicle updates. `VehicleController.FixedUpdate` took 398 ms inclusive / 55 ms
self across 6,450 calls. Nested below it, Unity recorded 25,800
`Physics.SyncColliderTransform` calls (four per update) totaling 337 ms
inclusive / 140 ms self, with 170 ms in `TransformChangeSystem`. The source calls
`WheelCollider.GetWorldPose` for each of four wheels and then writes each wheel
mesh transform's position and rotation. That exact four-per-tick match is strong
evidence these writes are associated with the sync work, but profiler hierarchy
alone cannot prove which individual property setter causes each native sync.
Project settings also set `m_AutoSyncTransforms: 1`. Unity's 2022.3 API
documentation says Auto Sync can incur repeated synchronization when transforms
change and recommends disabling it for newer projects; when disabled, changes
are synchronized before simulation, and queries that depend on same-frame
transform changes need an explicit sync. This is a promising setting to A/B,
not yet a safe optimization result: LiDAR uses physics raycasts, and exact sensor
and vehicle traces must confirm all queries see the same world.

LiDAR is the second clear software-side hotspot: 308 ms inclusive / 164 ms self
in 6,450 `FixedUpdate` calls, including 137,287 raycasts (82 ms self / 110 ms
inclusive). Its 1,081-beam scan is configured at 7 Hz in the parallel MARL scene.
The previous isolated array-reuse candidate preserved every captured value but
was slower in both pool tests, so allocation removal is not supported by the
data. Batching raycasts remains a separate candidate with a larger theoretical
upside, but it must preserve scan ordering, output, and timing exactly.

Other measured costs are smaller: wheel encoders used 36 ms self across 12,900
calls; IMU used 22 ms across 6,450; telemetry callbacks used 55 ms self and
gzip compression 32 ms across the 10-second telemetry capture. Main-thread
`WaitForTargetFPS` occupied 4.81 s across 225 frames; this is deliberate pacing
while idle, not active simulation work. The Development capture's Dispatcher
thread spent 1.58 s writing profiler data, which is capture overhead and should
not be interpreted as normal player cost. Startup `TrackLoader` scanning was
already established as a temporary first-15-second cost, not a long-run hot
path.

Recommended next experiment: build a staged player with only Auto Sync Transforms
disabled, retain all four wheel pose updates, and compare it against the current
staged fixed-step baseline with the existing exact-trace harness. Verify actions,
physics-step counts, protocol IDs, simulated-time deltas, vehicle state, complete
LiDAR arrays, observations, and rewards; benchmark paired CPU seconds per
simulated second and pool throughput at four and eight environments. Reject it
if any behavior differs or the CPU reduction does not repeat. Do not modify the
active training players. If it fails, test batched LiDAR raycasts next. Neither
candidate has been adopted based on this profile alone.

The Auto Sync-off candidate was tested in three four-environment, 20-simulated-
second paired runs after the training simulator containers had stopped. Actions,
physics-step counts, protocol IDs, and simulated-time deltas matched, but vehicle
positions and velocities diverged and 249,813 LiDAR beam values per environment
differed across the captured traces; observations differed as a result. Rewards
happened to match in this scripted route, which does not make the differing
observations acceptable. The candidate used a median 4.7% less Unity CPU per
simulated second, while median pool throughput changed by only +0.7%. Reject it:
the saved CPU came with changed query/sensor and vehicle data. The staged player
and full trace report are local at
`simulator/_build/linux-fixed-step-autosync-off-experiment/` and
`logs/diagnostics/autosync_off_4env.json` / `.npz`; active training players were
not changed. This confirms that Auto Sync Transforms is required for the current
raycast and vehicle-update sequence unless explicit synchronization points are
introduced and separately verified.

The first batched LiDAR candidate used the unchanged 12 m direction vectors;
Unity's job-query path interpreted their magnitude differently and produced
invalid sensor data, so that version was discarded. Normalizing the same rays
restored their effective directions and ranges, but it still changed 8,708 of
151,340 LiDAR values and 8,662 observation values per 12-second trace. The
largest finite range discrepancy was only about 0.000002 m, but exact input
data did not match. Across two paired single-environment repeats while the
training pool was active, median CPU per simulated second was about 0.9% higher
for the batch candidate and throughput was effectively unchanged. The result is
both behaviorally unacceptable under the exact-data requirement and not a
repeatable speedup. The strict JSON comparison harness now encodes infinite
error metrics as strings so LiDAR miss transitions are still reported rather
than aborting serialization. Reports and full traces are local at
`logs/diagnostics/lidar_batch_behavior_ab.json` / `.npz`; neither batch build is
published.

An explicit-sync candidate disabled Auto Sync Transforms and called
`Physics.SyncTransforms()` once after `VehicleController` updated all four wheel
poses. It still failed the exact behavior check in both 12-second paired traces:
vehicle pose, velocity, encoders, and 149,651 LiDAR values per environment
differed. Median CPU per simulated second was 1.3% higher and throughput was
0.5% lower than baseline. A single explicit synchronization therefore does not
preserve the current query/physics ordering and is rejected. The local full-trace
report is `logs/diagnostics/explicit_sync_behavior_ab.json`; the staged player
is under `simulator/_build/linux-fixed-step-explicit-sync-experiment/`.

These tests rule out the most visible synchronization and raycast batching
changes under the exact-data requirement. The remaining largest PhysX stages
are collision detection/contact generation and vehicle solving; changing
collision geometry, solver iterations, physics timestep, or query count would
change simulation results and is outside this optimization goal. Small
data-path opportunities can still be measured separately, such as avoiding
repeated compression of an unchanged LiDAR intensity array.

**Docker Linux and Windows players both need a rebuild** to pick up
`ForceConnect`, `AiCarSimulationGate`, and LapTimer batchmode silencing. The
training process uses the gate to stop cached controls from moving cars while
PPO updates its policy; the Socket.IO loop stays responsive so training can
resume without resetting the cars. IL2CPP Linux cannot load a drop-in Managed
DLL.

## Windows Mono hotfix (optional, no full rebuild)

`Hotfix/ForceConnectHotfix.cs` is Doorstop-free: compile against the player
`Managed/*.dll` refs, copy `AiCar.ForceConnect.dll` into
`simulator/windows/AutoDRIVE Simulator_Data/Managed/`, then append the assembly
name to `ScriptingAssemblies.json` and a `RuntimeInitializeOnLoads.json` entry
for `AiCar.Hotfix.ForceConnectHotfixBootstrap.Boot` (`loadTypes: 0` =
AfterSceneLoad). Prefer `BuildWindowsOnly` when Unity is available.

## Runtime

Activate in Mission Control writes `maps/.active_map.json` and restarts sims.
Entrypoint sets `AICAR_MAP_ID` via env/file only — **do not** pass `-map-id` on
Unity argv (stock parsers mishandle unknown flags). Bridge / L1–L3 unchanged.
