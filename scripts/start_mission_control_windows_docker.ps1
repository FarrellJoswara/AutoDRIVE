param(
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runtimeConfigPath = Join-Path $repoRoot "logs\rl\porto_ppo_continuation_20261030\runtime_config.json"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Windows project environment not found: $python"
}
if (-not (Test-Path -LiteralPath $runtimeConfigPath)) {
    throw "Pinned Porto runtime config not found: $runtimeConfigPath"
}
$runtimeConfig = Get-Content -LiteralPath $runtimeConfigPath -Raw | ConvertFrom-Json
if ($runtimeConfig.track_profile -ne "practice" -or $runtimeConfig.map_id -ne "porto" -or $runtimeConfig.action_step_mode -ne $false) {
    throw "Pinned Docker runtime config must select Porto practice with ActionStepMode disabled"
}

# The WSL hub uses a Unix socket owned by root:docker. Run this hub on Windows
# so Docker Desktop's current-user named-pipe ACL controls Engine access.
$env:AICAR_RUN_BACKEND = "docker"
$env:AICAR_DOCKER_HOST = "npipe:////./pipe/dockerDesktopLinuxEngine"
$env:AICAR_OFFICIAL_API_IMAGE = $runtimeConfig.api_image_tag
$env:AICAR_EXPECTED_API_IMAGE_ID = $runtimeConfig.api_image_id
$env:AICAR_EXPECTED_SIMULATOR_IMAGE_ID = $runtimeConfig.simulator_image_id
$env:HUB_HOST = if ($env:HUB_HOST) { $env:HUB_HOST } else { "0.0.0.0" }
$env:HUB_PORT = if ($env:HUB_PORT) { $env:HUB_PORT } else { "8090" }
$env:AICAR_HUB_PUBLIC_URL = if ($env:AICAR_HUB_PUBLIC_URL) {
    $env:AICAR_HUB_PUBLIC_URL
} else {
    "http://host.docker.internal:$($env:HUB_PORT)"
}

Set-Location -LiteralPath $repoRoot
@'
from src.layer4.hub import docker_control
from src.layer4.hub.official_run_manager import API_IMAGE, PORTO_PRACTICE_SIM_IMAGE
import os

info = docker_control._request("GET", "/info")
def encoded_image_name(value):
    return value.replace("/", "%2F").replace(":", "%3A")

api_image = docker_control._request(
    "GET", f"/images/{encoded_image_name(os.environ['AICAR_OFFICIAL_API_IMAGE'])}/json"
)
image = docker_control._request(
    "GET", f"/images/{encoded_image_name(PORTO_PRACTICE_SIM_IMAGE)}/json"
)
expected_api = os.environ["AICAR_EXPECTED_API_IMAGE_ID"]
expected = os.environ["AICAR_EXPECTED_SIMULATOR_IMAGE_ID"]
if api_image.get("Id") != expected_api or API_IMAGE != os.environ["AICAR_OFFICIAL_API_IMAGE"]:
    raise SystemExit(f"Current API image tag/digest mismatch: {api_image.get('Id')} / {API_IMAGE}")
if image.get("Id") != expected:
    raise SystemExit(f"Official Porto practice image digest mismatch: {image.get('Id')}")
if PORTO_PRACTICE_SIM_IMAGE != "autodriveecosystem/autodrive_roboracer_sim:2026-iros-practice":
    raise SystemExit("Mission Control Porto practice image mapping is unexpected")
print(f"Docker Desktop {info['ServerVersion']} ({info['Name']}); official Porto image digest verified")
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop preflight failed with exit code $LASTEXITCODE" }
& docker.exe run --rm --pull=never --entrypoint python $env:AICAR_OFFICIAL_API_IMAGE -c "from importlib.metadata import version; import aiohttp, socketio, engineio; from src.layer1.official_bridge_relay import OfficialBridgeRelay; assert tuple(map(int,version('python-socketio').split('.')[:2])) >= (5,10); assert tuple(map(int,version('python-engineio').split('.')[:1])) >= (4,); print('API deps',version('python-socketio'),version('python-engineio'),aiohttp.__version__,OfficialBridgeRelay.__name__)"
if ($LASTEXITCODE -ne 0) { throw "Current API image dependency preflight failed with exit code $LASTEXITCODE" }
if ($PreflightOnly) { return }

& $python -m src.layer4.hub.app
exit $LASTEXITCODE
