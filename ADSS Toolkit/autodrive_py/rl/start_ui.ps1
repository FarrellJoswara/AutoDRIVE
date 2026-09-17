# Minimal RL control panel launcher (Windows).
# Usage:
#   & "ADSS Toolkit\autodrive_py\rl\start_ui.ps1"
#   python -m rl.control_ui

param(
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

# Avoid UnicodeEncodeError on Windows cp1252 consoles when Python prints.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$RlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AutodrivePy = Split-Path -Parent $RlDir
$VenvPython = Join-Path $RlDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at $VenvPython" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== AutoDRIVE RL - control UI ===" -ForegroundColor Cyan
Write-Host "  open    : http://${HostAddr}:${Port}/"
Write-Host "  toggle  : Start/Stop training (one button; Start also launches TensorBoard)"
Write-Host "  start   : refuses a second trainer, and refuses [HOLDOUT]/[VALIDATION] maps"
Write-Host "            unless you tick the override; preview card shows exactly what will run"
Write-Host "  stop    : kills the train_ppo tree + UI-spawned TensorBoard, verifies it died,"
Write-Host "            and clears the run lock so Continue can resume"
Write-Host "  continue: resumes a run from its last complete checkpoint (same run_id)"
Write-Host "  models  : Load a saved zip into Watch, or Delete a run (live runs are protected)"
Write-Host "  maps    : Generate procedural tracks into the map pack (sealed maps untouched)"
Write-Host "  presets : Debug / Quick / Overnight fill timesteps + workers"
Write-Host "  coach   : banner (Idle/Learning/Saving/Stopping/Stale/Crashed) + steps/sec + crash rate"
Write-Host "  watch   : launches python -m rl.watch --follow in a new console"
Write-Host "  note    : training stays headless (no OpenCV in train)"
Write-Host "  tb      : Open TensorBoard -> http://127.0.0.1:6006/"
Write-Host ""
Write-Host "Offline checks (no training, no browser): python -m rl.ui_selftest" -ForegroundColor Yellow
Write-Host ""

# Kill stale control_ui on this port (duplicates keep serving old in-memory HTML).
$killed = @()
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match 'python' -and
        $_.CommandLine -and
        $_.CommandLine -match 'rl\.control_ui' -and
        $_.CommandLine -match ("--port\s+$Port")
    } |
    ForEach-Object {
        Write-Host "  stopping stale control_ui pid $($_.ProcessId)" -ForegroundColor Yellow
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        $killed += $_.ProcessId
    }
# Also free the listen socket if something else holds it.
Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object {
        $owner = $_.OwningProcess
        if ($owner -and ($killed -notcontains $owner)) {
            Write-Host "  freeing port $Port (pid $owner)" -ForegroundColor Yellow
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
        }
    }
if ($killed.Count -gt 0) { Start-Sleep -Seconds 1 }

# control_ui is imported as python -m rl.control_ui; cwd must be autodrive_py
Push-Location $AutodrivePy
try {
    & $VenvPython -u -m rl.control_ui --host $HostAddr --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
