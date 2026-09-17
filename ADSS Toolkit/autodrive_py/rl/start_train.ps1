# Dead-simple PPO trainer launcher (Windows).
# Usage (from anywhere):
#   & "ADSS Toolkit\autodrive_py\rl\start_train.ps1"
# Optional overrides:
#   & "...\start_train.ps1" -Map map1 -Timesteps 1000000 -NEnvs 16 -NetArch "512,512"

param(
    [string]$Map = "map0",
    [int]$Timesteps = 500000,
    [string]$Device = "auto",
    [string]$RunId = "",
    [int]$NEnvs = 8,
    [string]$VecEnv = "subproc",  # ~3-4x FPS vs dummy here; use -VecEnv dummy if spawn fails
    [int]$NSteps = 2048,
    [int]$BatchSize = 1024,
    [int]$NEpochs = 10,
    [string]$NetArch = "512,512"
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
Write-Host "=== AutoDRIVE RL — PPO training (hard) ===" -ForegroundColor Cyan
Write-Host "  map         : $Map"
Write-Host "  timesteps   : $Timesteps"
Write-Host "  device      : $Device  (auto = CUDA if available)"
Write-Host "  n_envs      : $NEnvs  (vec=$VecEnv)"
Write-Host "  n_steps     : $NSteps  batch=$BatchSize  epochs=$NEpochs"
Write-Host "  net_arch    : $NetArch"
Write-Host "  run_id      : $RunId"
Write-Host "  venv python : $VenvPython"
Write-Host "  log file    : $LogFile"
Write-Host "  models save : $ModelsDir\$RunId\"
Write-Host "  TB logdir   : $RunsDir"
Write-Host ""
Write-Host "What's happening:" -ForegroundColor Yellow
Write-Host "  1) Gym RacingEnv steps on CPU (LiDAR / physics) — often the bottleneck."
Write-Host "  2) Parallel envs + larger net/batch push more work onto the GPU update."
Write-Host "  3) Checkpoints + latest_model.zip land under rl/models/<run_id>/."
Write-Host "  4) live_status.json + TensorBoard scalars go under rl/runs/<run_id>/."
Write-Host "  5) Training is headless — OpenCV never runs in this process."
Write-Host ""
Write-Host "Crank harder:" -ForegroundColor Yellow
Write-Host "  -NEnvs 16  -Timesteps 1000000  -NetArch `"512,512`"  -BatchSize 2048  -NEpochs 15"
Write-Host ""
Write-Host "Watch progress (second terminal, lags behind — does not slow train):" -ForegroundColor Yellow
Write-Host "  & `"$VenvPython`" -m rl.watch --follow --map $Map"
Write-Host ""
Write-Host "TensorBoard (separate terminal):" -ForegroundColor Yellow
Write-Host "  & `"$VenvPython`" -m tensorboard --logdir `"$RunsDir`""
Write-Host "  then open http://localhost:6006"
Write-Host ""
Write-Host "Stop training: Ctrl+C in this window."
Write-Host "  Or kill the tree:  Get-CimInstance Win32_Process -Filter `"Name='python.exe'`" |"
Write-Host "    Where-Object { `$_.CommandLine -like '*train_ppo*' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }"
Write-Host ""

Push-Location $AutodrivePy
try {
    $trainArgs = @(
        "-u", "-m", "rl.train_ppo",
        "--map", $Map,
        "--timesteps", "$Timesteps",
        "--device", $Device,
        "--run_id", $RunId,
        "--n-envs", "$NEnvs",
        "--vec-env", $VecEnv,
        "--n-steps", "$NSteps",
        "--batch-size", "$BatchSize",
        "--n-epochs", "$NEpochs",
        "--net-arch", $NetArch
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
