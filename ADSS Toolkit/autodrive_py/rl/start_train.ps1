# Dead-simple PPO trainer launcher (Windows).
# Usage (from anywhere):
#   & "ADSS Toolkit\autodrive_py\rl\start_train.ps1"
# Optional overrides:
#   & "...\start_train.ps1" -Map map1 -Timesteps 100000 -Device cuda

param(
    [string]$Map = "map0",
    [int]$Timesteps = 300000,
    [string]$Device = "auto",
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"

$RlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AutodrivePy = Split-Path -Parent $RlDir
$VenvPython = Join-Path $RlDir ".venv\Scripts\python.exe"
$RunsDir = Join-Path $RlDir "runs"
$LogsDir = Join-Path $RlDir "logs"
$ModelsDir = Join-Path $RlDir "models"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: venv python not found at:" -ForegroundColor Red
    Write-Host "  $VenvPython"
    Write-Host "Create it from ADSS Toolkit/autodrive_py:"
    Write-Host "  py -3.13 -m venv rl/.venv"
    Write-Host "  .\rl\.venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r rl/requirements.txt"
    exit 1
}

New-Item -ItemType Directory -Force -Path $RunsDir, $LogsDir, $ModelsDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "${stamp}_ppo_gym_${Map}"
}
$LogFile = Join-Path $LogsDir "$RunId.log"

Write-Host ""
Write-Host "=== AutoDRIVE RL — PPO training ===" -ForegroundColor Cyan
Write-Host "  map         : $Map"
Write-Host "  timesteps   : $Timesteps"
Write-Host "  device      : $Device  (auto = CUDA if available)"
Write-Host "  run_id      : $RunId"
Write-Host "  venv python : $VenvPython"
Write-Host "  log file    : $LogFile"
Write-Host "  models save : $ModelsDir\$RunId\"
Write-Host "  TB logdir   : $RunsDir"
Write-Host ""
Write-Host "What's happening:" -ForegroundColor Yellow
Write-Host "  1) Gym RacingEnv steps on CPU (LiDAR / physics)."
Write-Host "  2) PPO policy updates on GPU when CUDA torch is present."
Write-Host "  3) Checkpoints + metrics land under rl/models/<run_id>/."
Write-Host "  4) TensorBoard scalars go under rl/runs/<run_id>/."
Write-Host ""
Write-Host "TensorBoard (separate terminal):" -ForegroundColor Yellow
Write-Host "  & `"$VenvPython`" -m tensorboard --logdir `"$RunsDir`""
Write-Host "  then open http://localhost:6006"
Write-Host ""
Write-Host "Stop training: Ctrl+C in this window (or kill the python PID)."
Write-Host ""

Push-Location $AutodrivePy
try {
    $trainArgs = @(
        "-u", "-m", "rl.train_ppo",
        "--map", $Map,
        "--timesteps", "$Timesteps",
        "--device", $Device,
        "--run_id", $RunId
    )
    & $VenvPython @trainArgs 2>&1 | Tee-Object -FilePath $LogFile
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($code -ne 0) {
    Write-Host "Training exited with code $code. See log: $LogFile" -ForegroundColor Red
    exit $code
}

Write-Host ""
Write-Host "Done. Model: $ModelsDir\$RunId\best_model.zip" -ForegroundColor Green
Write-Host "Log: $LogFile"
exit 0
