# Dead-simple PPO trainer launcher (Windows).
# Usage (from anywhere):
#   & "ADSS Toolkit\autodrive_py\rl\start_train.ps1"
# Optional overrides:
#   & "...\start_train.ps1" -Map map1 -Timesteps 1000000 -NEnvs 16 -NetArch "512,512"

param(
    [string]$Map = "map0",
    [int]$Timesteps = 100000,     # short honest runs until the beat-FTG gate; -Timesteps 500000 for overnight
    [string]$Device = "auto",
    [string]$RunId = "",
    [int]$NEnvs = 8,
    # auto = subproc when NEnvs > 1, else dummy (same rule as the control UI).
    [ValidateSet("auto", "subproc", "dummy")]
    [string]$VecEnv = "auto",
    [int]$NSteps = 2048,
    [int]$BatchSize = 1024,
    [int]$NEpochs = 10,
    [string]$NetArch = "512,512",
    [string]$Resume = "",          # run_id (or zip path) to continue from its last complete checkpoint
    [switch]$ForkResume,           # resume into a NEW run_id instead of continuing the same one
    [switch]$AllowHoldout,         # train on a sealed holdout map (voids official claims)
    [int]$EarlyStopPatience = 0,   # 0=off; stop after N validation race-evals with no *meaningful* improvement
    [double]$EarlyStopMinImprove = 0.5,  # finishers: adjusted_time seconds to reset patience; 0=any tiny gain
    [int]$RaceEvalEvery = 0,       # 0=end-only (or auto when EarlyStopPatience>0)
    [switch]$UnlimitedTimesteps    # ignore timesteps as a stop (needs EarlyStopPatience > 0); 50M safety ceiling
)

$ErrorActionPreference = "Stop"

# Avoid UnicodeEncodeError on Windows cp1252 consoles when Python prints.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

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

if ($UnlimitedTimesteps -and $EarlyStopPatience -le 0) {
    Write-Host "ERROR: -UnlimitedTimesteps requires -EarlyStopPatience > 0" -ForegroundColor Red
    Write-Host "  Otherwise training never stops except Ctrl+C. Try -EarlyStopPatience 3"
    exit 1
}

if ($VecEnv -eq "auto") {
    $VecEnv = if ($NEnvs -gt 1) { "subproc" } else { "dummy" }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
if ([string]::IsNullOrWhiteSpace($RunId)) {
    if (-not [string]::IsNullOrWhiteSpace($Resume) -and -not $ForkResume) {
        # Continue writes back into the run being resumed (same id, same folders).
        $RunId = $Resume
    } else {
        $RunId = "${stamp}_ppo_gym_${Map}"
    }
}
$LogFile = Join-Path $LogsDir "$RunId.log"

Write-Host ""
Write-Host "=== AutoDRIVE RL - PPO training (hard) ===" -ForegroundColor Cyan
Write-Host "  map         : $Map"
if ($UnlimitedTimesteps) {
    Write-Host "  timesteps   : unlimited (early-stop only; 50M safety ceiling)" -ForegroundColor Green
} else {
    Write-Host "  timesteps   : $Timesteps"
}
Write-Host "  device      : $Device  (auto = CUDA if available)"
Write-Host "  n_envs      : $NEnvs  (vec=$VecEnv)"
if (-not [string]::IsNullOrWhiteSpace($Resume)) {
    $resumeMode = if ($ForkResume) { "fork into new run_id" } else { "continue same run_id" }
    Write-Host "  resume      : $Resume  ($resumeMode, last complete checkpoint)" -ForegroundColor Green
}
if ($AllowHoldout) {
    Write-Host "  holdout     : ALLOWED - official claims on this map are void" -ForegroundColor Red
}
if ($EarlyStopPatience -gt 0) {
    Write-Host "  early-stop  : patience=$EarlyStopPatience  min_improve=${EarlyStopMinImprove}s adj" -ForegroundColor Green
}
if ($RaceEvalEvery -gt 0) {
    Write-Host "  race-eval   : every $RaceEvalEvery timesteps" -ForegroundColor Green
}
Write-Host "  n_steps     : $NSteps  batch=$BatchSize  epochs=$NEpochs"
Write-Host "  net_arch    : $NetArch"
Write-Host "  run_id      : $RunId"
Write-Host "  venv python : $VenvPython"
Write-Host "  log file    : $LogFile"
Write-Host "  models save : $(Join-Path $ModelsDir $RunId)"
Write-Host "  TB logdir   : $RunsDir"
Write-Host ""
Write-Host "Whats happening:" -ForegroundColor Yellow
Write-Host "  1. Gym RacingEnv steps on CPU (LiDAR + speed/IMU proprio) - often the bottleneck."
Write-Host "  2. Parallel envs + larger net/batch push more work onto the GPU update."
Write-Host "  3. Checkpoints + latest_model.zip land under rl/models/<run_id>/."
Write-Host "  4. live_status.json + TensorBoard scalars go under rl/runs/<run_id>/."
Write-Host "  5. Training is headless - OpenCV never runs in this process."
Write-Host "  6. contracts 2.0.0 obs_dim=186 (LiDAR+prev+speed+IMU) - v1 zips obsolete."
Write-Host "  7. Sealed holdout maps are refused unless -AllowHoldout (keeps official scores honest)."
Write-Host ""
Write-Host "Continue an interrupted run (same run_id, resumes last complete checkpoint):" -ForegroundColor Yellow
Write-Host '  .\start_train.ps1 -Resume 20260917_011531_ppo_gym_map0 -Timesteps 100000'
Write-Host "  (or press Continue in the control panel: .\start_ui.ps1)"
Write-Host ""
Write-Host "Crank harder:" -ForegroundColor Yellow
Write-Host '  -NEnvs 16  -Timesteps 1000000  -NetArch "512,512"  -BatchSize 2048  -NEpochs 15'
Write-Host ""
Write-Host "Watch progress (second terminal, lags behind - does not slow train):" -ForegroundColor Yellow
Write-Host "  $VenvPython -m rl.watch --follow --map $Map"
Write-Host ""
Write-Host "TensorBoard (separate terminal):" -ForegroundColor Yellow
Write-Host "  $VenvPython -m tensorboard --logdir $RunsDir"
Write-Host "  then open http://localhost:6006"
Write-Host ""
Write-Host "Stop training: Ctrl+C in this window."
Write-Host "  Or kill train_ppo python processes via Task Manager / Get-CimInstance."
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
    if (-not [string]::IsNullOrWhiteSpace($Resume)) { $trainArgs += @("--resume", $Resume) }
    if ($ForkResume) { $trainArgs += "--fork-resume" }
    if ($AllowHoldout) { $trainArgs += "--allow-holdout" }
    if ($UnlimitedTimesteps) { $trainArgs += "--unlimited-timesteps" }
    if ($EarlyStopPatience -gt 0) {
        $trainArgs += @("--early-stop-patience", "$EarlyStopPatience")
        $trainArgs += @("--early-stop-min-improve", "$EarlyStopMinImprove")
    }
    if ($RaceEvalEvery -gt 0) { $trainArgs += @("--race-eval-every", "$RaceEvalEvery") }
    # SB3 may print UserWarnings on stderr; do not treat those as terminating errors.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $VenvPython @trainArgs 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.ToString()
            } else {
                $_
            }
        } | Tee-Object -FilePath $LogFile
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEap
    }
} finally {
    Pop-Location
}

if ($code -ne 0) {
    Write-Host "Training exited with code $code. See log: $LogFile" -ForegroundColor Red
    exit $code
}

Write-Host ""
Write-Host "Done. Model: $(Join-Path (Join-Path $ModelsDir $RunId) 'best_model.zip')" -ForegroundColor Green
Write-Host "Log: $LogFile"
exit 0
