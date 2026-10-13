param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$SimulatorExePath,

    [Parameter(Mandatory = $true)]
    [string]$PortsFile,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{12}$')]
    [string]$RunId,

    [string]$HubUrl = "http://127.0.0.1:8091",

    [string]$BridgeIpAddress = "127.0.0.1",

    [Parameter(Mandatory = $true)]
    [string]$PidFile,

    [Parameter(Mandatory = $true)]
    [string]$PauseFile,

    [switch]$BatchMode
)

$ErrorActionPreference = "Stop"
$SimulatorExePath = (Resolve-Path -LiteralPath $SimulatorExePath).Path
$Ports = @((Get-Content -LiteralPath $PortsFile -Raw).Trim().Split(',') | ForEach-Object { [int]$_.Trim() })
if ($Ports.Count -lt 1 -or $Ports.Count -gt 8 -or @($Ports | Where-Object { $_ -lt 1 -or $_ -gt 65535 }).Count -gt 0) {
    throw "Ports must contain one to eight valid TCP port numbers."
}
$connectHelper = Join-Path $PSScriptRoot "connect_official_simulator.ps1"
if (-not (Test-Path -LiteralPath $connectHelper -PathType Leaf)) {
    throw "The official simulator connection helper is missing: $connectHelper"
}
$processes = @()
$runDirectory = (Resolve-Path -LiteralPath (Split-Path -Parent $PidFile)).Path
if ([string]::Compare((Split-Path -Leaf $runDirectory), "official_run_$RunId", $true) -ne 0 -or
    [string]::Compare([IO.Path]::GetFullPath($PidFile), [IO.Path]::GetFullPath((Join-Path $runDirectory 'windows_simulator_pids.txt')), $true) -ne 0) {
    throw "Run identity and simulator PID file must resolve inside logs/rl/official_run_$RunId."
}
if ([string]::Compare([IO.Path]::GetFullPath($PauseFile), [IO.Path]::GetFullPath((Join-Path $runDirectory 'evaluation_pause.flag')), $true) -ne 0 -or
    [string]::Compare([IO.Path]::GetFullPath($PortsFile), [IO.Path]::GetFullPath((Join-Path $runDirectory 'windows_simulator_ports.csv')), $true) -ne 0) {
    throw "Pause and port manifests must resolve inside logs/rl/official_run_$RunId."
}
$IdentityManifestPath = Join-Path $runDirectory "windows_simulator_processes.json"
$identityManifest = [ordered]@{
    schema_version = 1
    run_id = $RunId
    simulator_exe_path = $SimulatorExePath
    bridge_ip = $BridgeIpAddress
    ports = @($Ports)
    records = @()
}
if (Test-Path -LiteralPath $IdentityManifestPath -PathType Leaf) {
    $existingManifest = Get-Content -LiteralPath $IdentityManifestPath -Raw | ConvertFrom-Json
    if ($existingManifest.run_id -ne $RunId -or
        [string]::Compare([string]$existingManifest.simulator_exe_path, $SimulatorExePath, $true) -ne 0 -or
        $existingManifest.bridge_ip -ne $BridgeIpAddress -or
        (@($existingManifest.ports) -join ',') -ne (@($Ports) -join ',')) {
        throw "Existing simulator identity manifest does not match this run and executable."
    }
    $identityManifest.records = @($existingManifest.records)
}

function Save-IdentityManifest {
    $temporaryPath = "$IdentityManifestPath.tmp"
    $identityManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryPath -Encoding UTF8
    Move-Item -LiteralPath $temporaryPath -Destination $IdentityManifestPath -Force
}

function Stop-SimulatorFleet {
    foreach ($process in $script:processes) {
        try {
            $process.Refresh()
            if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
        } catch { }
    }
    $script:processes = @()
    if (Test-Path -LiteralPath $PidFile) {
        Clear-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue
    }
}

function Start-SimulatorFleet {
    $script:processes = @()
    for ($environment = 0; $environment -lt $Ports.Count; $environment++) {
        $port = $Ports[$environment]
        $arguments = @(
            "-ip", $BridgeIpAddress, "-port", [string]$port
        )
        if ($BatchMode) {
            # Unity batch mode suppresses its UI but keeps the GPU device active.
            # Deliberately omit -nographics so the official camera sensor renders.
            $simLog = Join-Path (Split-Path -Parent $PidFile) "official_simulator_env$environment.log"
            $arguments = @("-batchmode", "-logFile", $simLog) + $arguments
        } else {
            $arguments = @(
                "-runInBackground", "-screen-fullscreen", "0",
                "-screen-width", "960", "-screen-height", "540"
            ) + $arguments
        }
        $process = Start-Process -FilePath $SimulatorExePath -ArgumentList $arguments -PassThru
        $script:processes += $process
        $process.Refresh()
        $cimProcess = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($process.Id)"
        if (-not $cimProcess -or [string]::Compare([string]$cimProcess.ExecutablePath, $SimulatorExePath, $true) -ne 0) {
            throw "Could not verify the launched official simulator PID $($process.Id) against its executable path."
        }
        $identityManifest.records += [ordered]@{
            pid = [int]$process.Id
            environment = [int]$environment
            port = [int]$port
            bridge_ip = $BridgeIpAddress
            executable_path = [string]$cimProcess.ExecutablePath
            command_line = [string]$cimProcess.CommandLine
            parent_pid = [int]$cimProcess.ParentProcessId
            process_creation_utc = $process.StartTime.ToUniversalTime().ToString("o")
            process_creation_cim = [string]$cimProcess.CreationDate
            launched_utc = [DateTime]::UtcNow.ToString("o")
            batch_mode = [bool]$BatchMode
        }
        Save-IdentityManifest
        $script:processes.Id | Set-Content -LiteralPath $PidFile -Encoding Ascii
        Write-Output "Started official simulator PID $($process.Id) for environment $environment on port $port."
        if (-not $BatchMode) {
            $windowDeadline = (Get-Date).AddSeconds(90)
            do {
                $process.Refresh()
                if ($process.HasExited) { throw "Official simulator PID $($process.Id) exited before opening its window." }
                if ($process.MainWindowHandle -ne 0) { break }
                Start-Sleep -Milliseconds 500
            } while ((Get-Date) -lt $windowDeadline)
            if ($process.MainWindowHandle -eq 0) { throw "Official simulator PID $($process.Id) did not open a window within 90 seconds." }
            Start-Sleep -Seconds 3
            & $connectHelper -ProcessId $process.Id -IpAddress $BridgeIpAddress -Port $port -ConnectTimeoutSeconds 90
        }
        $deadline = (Get-Date).AddSeconds(90)
        do {
            $process.Refresh()
            if ($process.HasExited) { throw "Official simulator PID $($process.Id) exited before connecting to port $port." }
            $connected = Get-NetTCPConnection -OwningProcess $process.Id -State Established -ErrorAction SilentlyContinue |
                Where-Object { $_.RemoteAddress -eq $BridgeIpAddress -and $_.RemotePort -eq $port } |
                Select-Object -First 1
            if ($connected) { break }
            Start-Sleep -Milliseconds 500
        } while ((Get-Date) -lt $deadline)
        if (-not $connected) { throw "Official simulator PID $($process.Id) did not connect to ${BridgeIpAddress}:$port within 90 seconds." }
        Write-Output "Environment $environment connected on port $port."
    }
}

function Get-RunStatus {
    try {
        return Invoke-RestMethod -Uri "$($HubUrl.TrimEnd('/'))/train/runs/$RunId" -TimeoutSec 10
    } catch {
        Start-Sleep -Seconds 2
        return $null
    }
}

try {
    Start-SimulatorFleet
    Write-Output "All official simulator environments are connected. Leave this window open to stop them when the run ends."
    while ($true) {
        $run = Get-RunStatus
        if ($null -eq $run) { continue }
        if ($run.state -notin @("starting", "running", "stopping")) {
            Write-Output "Run $RunId is $($run.state); stopping its simulator processes."
            break
        }
        if (Test-Path -LiteralPath $PauseFile) {
            Write-Output "Evaluation requested; stopping this training simulator fleet temporarily."
            Stop-SimulatorFleet
            while (Test-Path -LiteralPath $PauseFile) {
                $run = Get-RunStatus
                if ($null -eq $run) { continue }
                if ($run.state -notin @("starting", "running", "stopping")) { break }
                Start-Sleep -Seconds 1
            }
            if ($run.state -in @("starting", "running")) {
                Write-Output "Evaluation finished; restoring the official training simulator fleet."
                Start-SimulatorFleet
                Write-Output "All official simulator environments are connected. Leave this window open to stop them when the run ends."
            }
            continue
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Stop-SimulatorFleet
    Remove-Item -LiteralPath $PauseFile -Force -ErrorAction SilentlyContinue
}
