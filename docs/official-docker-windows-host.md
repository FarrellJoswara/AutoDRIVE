# Official Docker runs from Windows Mission Control

The WSL Mission Control launcher uses `/var/run/docker.sock`, which is owned
by `root:docker`. When the WSL user does not have Docker socket access, start
the hub on Windows instead. Docker Desktop's `desktop-linux` named pipe keeps
Engine access under the current Windows user's normal ACL; this setup does not
add a WSL group membership or expose the Engine over TCP.

From the repository root, run:

```powershell
.\scripts\start_mission_control_windows_docker.ps1
```

The script checks Docker Desktop and verifies the official Porto practice image
digest and the current API image before starting the hub. It selects
`aicar-iros2026:porto-api-20261030`, a separate API image built from
`docker/Dockerfile.brain`; it does not change `aicar-iros2026:latest` or any
older API tag. Its relay dependencies are verified at startup. Open Mission
Control at `http://localhost:8090`.
The selected `docker` backend plus `practice` track profile resolves to
`autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice`. Its pinned
image digest is `sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76`;
the official scene is `Porto Track` and uses the standard mode with action-step
mode disabled. Camera runs use graphics through Xvfb, never `-nographics`.

The official practice profile uses
`competition/iros2026/official_centerline.csv` (SHA-256
`87d26be59ccc88ff6fc273905038906b8ed9e883b111c33de008a38f83a15bb3`).
Docker practice runs persist `map_id=porto` in their run row and telemetry so
Watch resolves the Porto map even if the active catalog selection changes.

PPO continuation settings with `total_timesteps=0` and
`max_duration_hours=0` have no fixed step or duration ceiling. The 10-lap
setting is the episode/performance target, not a PPO transition cutoff.
