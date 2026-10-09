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
    [string]$PidFile
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

try {
    for ($environment = 0; $environment -lt $Ports.Count; $environment++) {
        $port = $Ports[$environment]
        $arguments = @(
            "-runInBackground", "-screen-fullscreen", "0",
            "-screen-width", "960", "-screen-height", "540",
            "-ip", $BridgeIpAddress, "-port", [string]$port
        )
        $process = Start-Process -FilePath $SimulatorExePath -ArgumentList $arguments -PassThru
        $processes += $process
        $processes.Id | Set-Content -LiteralPath $PidFile -Encoding Ascii
        Write-Output "Started official simulator PID $($process.Id) for environment $environment on port $port."
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

    Write-Output "All official simulator environments are connected. Leave this window open to stop them when the run ends."
    while ($true) {
        $run = Invoke-RestMethod -Uri "$($HubUrl.TrimEnd('/'))/train/runs/$RunId" -TimeoutSec 10
        if ($run.state -notin @("starting", "running", "stopping")) {
            Write-Output "Run $RunId is $($run.state); stopping its simulator processes."
            break
        }
        Start-Sleep -Seconds 3
    }
}
finally {
    foreach ($process in $processes) {
        try {
            $process.Refresh()
            if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
        } catch { }
    }
    if (Test-Path -LiteralPath $PidFile) {
        Clear-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue
    }
}
