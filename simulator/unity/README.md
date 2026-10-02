# AutoDRIVE Unity source + AiCar TrackLoader (Phase 3)

AiCar patches live under `Assets/Scripts/AiCar/`. Full AutoDRIVE-Simulator is
cloned locally to `simulator/unity/AutoDRIVE/` (gitignored).

## Scripts (tracked in this repo)

| File | Role |
| :--- | :--- |
| `TrackLoaderBootstrap.cs` | Injects TrackLoader + ForceConnect at boot |
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

Outputs: `simulator/windows/` (Mono), Linux `.x86_64` + `Data/` (IL2CPP),
and `simulator/.aicar_trackloader`.

**Docker Linux and Windows players both need a rebuild** to pick up
`ForceConnect` (and LapTimer batchmode silencing). IL2CPP Linux cannot load a
drop-in Managed DLL.

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
