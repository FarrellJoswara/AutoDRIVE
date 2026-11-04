# Porto native practice runtime

The Porto BC run uses the official Windows practice simulator, with the ROS 2
Devkit bridge and sensor relay running natively in WSL. It does not need the
Docker API or simulator images.

## Verified simulator and route

- Original run config: [`logs/rl/porto_bc_aef213965b2b/launch_config.json`](../logs/rl/porto_bc_aef213965b2b/launch_config.json)
- Resolved Porto preset: [`logs/rl/porto_bc_aef213965b2b/resolved_config.json`](../logs/rl/porto_bc_aef213965b2b/resolved_config.json), SHA-256 `addbe0fc10bb4ff2e725b1b25e208951c69668175ebdb963d96cc66d8392b69f`
- Resolved native runtime environment: [`logs/rl/porto_bc_aef213965b2b/resolved_runtime_preset.json`](../logs/rl/porto_bc_aef213965b2b/resolved_runtime_preset.json), including ROS domain/base 86 and fallback 0.20 s
- Preset provenance: [`logs/rl/porto_bc_aef213965b2b/resolved_config.provenance.json`](../logs/rl/porto_bc_aef213965b2b/resolved_config.provenance.json)
- Release evidence: `%LOCALAPPDATA%/AiCar/official-simulator/2026-iros-practice/runtime_evidence.json`
- Practice executable: `AutoDRIVE Simulator.exe`, SHA-256 `f5efe8c38b78a72a37d44f07ea35e184cf1c41e4dd4e33de008a38f83a15bb3`
- Practice scene: `Data/level0`, SHA-256 `5edbd1bacc15647746e6e89d9bcf1bb9e202d62d47e76ae352497be1404b3f8d` (`WINDOWS_SCENES.practice`)
- Route: [`competition/iros2026/official_centerline.csv`](../competition/iros2026/official_centerline.csv), SHA-256 `87d26be59ccc88ff6fc273905038906b8ed9e883b111c33de008a38f83a15bb3`
- Runtime evidence: [`runtime_preflight.json`](../logs/rl/porto_bc_aef213965b2b/runtime_preflight.json) and the fresh retry probe [`sensor_preflight_retry1.json`](../logs/rl/porto_bc_aef213965b2b/sensor_preflight_retry1.json)

The observed practice spawn is approximately `(0.802, 3.158)` in ROS XY,
matching the official route metadata `(0.8, 3.1583)`. Windows practice runs must
set `track_profile: "practice"` and the separate `simulator_executable` path;
the native backend verifies the executable's scene hash before launch. The
competition executable path is not changed.

## Native preflight launch

From WSL, start the bridge with
[`scripts/start_porto_native_bridge.sh`](../scripts/start_porto_native_bridge.sh).
Set `AICAR_PORTO_RUN_DIR` to the run evidence folder before launching it. The
script starts the WSL Jazzy Devkit on port 4567 and the relay on port 4586 in ROS
domain 86, exports `AICAR_ROS_DOMAIN_ID_BASE=86` for native trainer processes,
then keeps the relay in the foreground. Ctrl+C stops both processes.
It sets `PYTHONNOUSERSITE=1` so the user's NumPy 2 package does not override
ROS Jazzy's compatible NumPy. The verified runtime uses
`/home/farrell/.venvs/aicar/bin/python` for the relay and teacher CLI.
In each separate WSL terminal used for a teacher, collector, fit, or evaluation
CLI, source `/opt/ros/jazzy/setup.bash` and `~/autodrive_ws/install/setup.bash`,
then set `ROS_DOMAIN_ID=86`, `AICAR_ROS_DOMAIN_ID_BASE=86`, and
`PYTHONNOUSERSITE=1` before invoking the venv Python. The base domain must match
the running bridge. The resolved preset enables the opt-in
`safe_neutral_fallback_s: 0.20`. It sends a fresh all-zero throttle and steering
command when a step has waited 200 ms for its next sensor frame; the existing
250 ms hard command-expiry guard and stale-frame timeout remain in force. For
direct teacher, collector, and student-evaluation CLI processes, export the
same value explicitly so all of them use the same Layer 1 sensor-only watchdog:

```bash
export ROS_DOMAIN_ID=86
export AICAR_ROS_DOMAIN_ID_BASE=86
export AICAR_OFFICIAL_SAFE_NEUTRAL_FALLBACK_S=0.20
export PYTHONNOUSERSITE=1
```

The historical `launch_config.json` is preserved as run provenance; it predates
this opt-in setting and did not itself carry the fallback value. The resolved
preset records the change and the source file hash.

Launch the practice GUI from PowerShell with the same relay address and port:

```powershell
$exe = "$env:LOCALAPPDATA\AiCar\official-simulator\2026-iros-practice\autodrive_simulator\AutoDRIVE Simulator.exe"
Start-Process -FilePath $exe -ArgumentList @(
  '-runInBackground', '-screen-fullscreen', '0', '-screen-width', '960',
  '-screen-height', '540', '-logFile', 'logs/rl/porto_bc_<run-id>/simulator.log',
  '-ip', '127.0.0.1', '-port', '4586'
)
```

When the GUI does not connect from command-line arguments alone, use
`scripts/connect_official_simulator.ps1` to set its connection panel. This only
connects the official simulator to the relay; it does not send driving controls.
Use GUI mode for camera observations: the practice build emitted black camera
frames in `-batchmode`, while visible GUI mode produced changing nonzero frames.

## Readiness evidence

Check `http://127.0.0.1:4586/healthz` for the current simulator session's fresh
complete frames, camera frames, Devkit commands, and response timeouts. Confirm
the ROS `front_camera`, `lidar`, and `ips` topics have advancing stamps. The
preflight found 192×108 RGB camera frames with changing nonzero pixels, 1081-ray
LiDAR frames, fresh IPS data, and a Direct3D 12 device on the RTX 3060. The
fresh retry used Windows simulator PID 61852, a single active simulator
session, and zero relay response timeouts. The 5.96-second ROS probe received
113 camera frames with changing nonzero pixels, 113 LiDAR frames with 1035 of
1081 rays finite, and 114 IPS messages at approximately `(0.811, 3.158)`. The
preflight sent no actuator commands. Runtime and trainer ROS domains were both
86. The previous GUI process shutdown log contained a SocketIO/WebSocket
`ThreadAbortException` during shutdown and Unity allocator statistics; it did
not show a fatal renderer exception. The retry of the same verified executable
connected successfully through the GUI helper.
