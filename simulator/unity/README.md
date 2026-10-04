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
