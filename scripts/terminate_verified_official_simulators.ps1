[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-f0-9]{12}$')]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [switch]$ApiStopAttempted,

    [Parameter(Mandatory = $true)]
    [switch]$GraceExpired,

    [switch]$HardDeadlineReached,

    [switch]$ValidateOnly,

    [string]$ProjectRoot = (Join-Path $PSScriptRoot '..'),

    [string]$HubUrl = 'http://127.0.0.1:8090'
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$runDirectory = [IO.Path]::GetFullPath((Join-Path $project "logs/rl/official_run_$RunId"))
$expectedParent = [IO.Path]::GetFullPath((Join-Path $project 'logs/rl')).TrimEnd('\') + '\'
if (-not $runDirectory.StartsWith($expectedParent, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Resolved run directory escaped the project logs/rl directory.'
}

$result = [ordered]@{
    schema_version = 1
    run_id = $RunId
    run_directory = $runDirectory
    result = 'failed'
    hard_deadline_reached = [bool]$HardDeadlineReached
    api_stop_attempted = [bool]$ApiStopAttempted
    grace_expired = [bool]$GraceExpired
    api_state = $null
    targets = @()
    terminated_pids = @()
    already_exited_pids = @()
    rejected_pids = @()
    errors = @()
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
}

function Complete-Result([int]$ExitCode) {
    $evidenceName = 'windows_deadline_cleanup_{0}_{1}.json' -f ([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')), $PID
    $evidencePath = Join-Path $runDirectory $evidenceName
    try {
        if (Test-Path -LiteralPath $runDirectory -PathType Container) {
            $tmp = "$evidencePath.tmp"
            $result | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $tmp -Encoding UTF8
            Move-Item -LiteralPath $tmp -Destination $evidencePath -Force
        }
    } catch {
        $result.errors += "Failed to persist cleanup evidence: $($_.Exception.Message)"
        $ExitCode = 1
    }
    $result.evidence_path = $evidencePath
    Write-Output ($result | ConvertTo-Json -Depth 10 -Compress)
    exit $ExitCode
}

if (-not $ApiStopAttempted -or -not $GraceExpired) {
    $result.errors += 'Refusing cleanup until the normal API stop was attempted and its grace period expired.'
    Complete-Result 2
}
if (-not (Test-Path -LiteralPath $runDirectory -PathType Container)) {
    $result.errors += 'Run directory does not exist.'
    Complete-Result 2
}

if (-not $HardDeadlineReached) {
    try {
        $runStatus = Invoke-RestMethod -Uri "$($HubUrl.TrimEnd('/'))/train/runs/$RunId" -TimeoutSec 8
        $result.api_state = [string]$runStatus.state
        if ($result.api_state -in @('starting', 'running', 'stopping')) {
            $result.errors += "Run is still active according to Mission Control ($($result.api_state)); hard-deadline authorization is required to interrupt it."
            Complete-Result 3
        }
    } catch {
        if ($result.errors.Count -gt 0) { Complete-Result 3 }
        $result.errors += "Could not confirm terminal API state: $($_.Exception.Message)"
        Complete-Result 3
    }
} else {
    $result.api_state = 'hard_deadline_override'
}

$pidFile = Join-Path $runDirectory 'windows_simulator_pids.txt'
$manifestPath = Join-Path $runDirectory 'windows_simulator_processes.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    $result.errors += 'Per-run simulator identity manifest is missing; refusing to infer process ownership.'
    Complete-Result 4
}

try {
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or $manifest.run_id -ne $RunId -or
        [string]::IsNullOrWhiteSpace([string]$manifest.simulator_exe_path) -or
        [string]::IsNullOrWhiteSpace([string]$manifest.bridge_ip)) {
        throw 'Identity manifest schema or run identity is invalid.'
    }
    $simulatorPath = [IO.Path]::GetFullPath([string]$manifest.simulator_exe_path)
    if (-not (Test-Path -LiteralPath $simulatorPath -PathType Leaf)) {
        throw 'Manifest simulator executable no longer exists at its recorded path.'
    }
    $records = @($manifest.records)
    $pidsFromFile = @()
    if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
        $pidsFromFile = @(Get-Content -LiteralPath $pidFile | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object {
            $parsed = 0
            if (-not [int]::TryParse($_.Trim(), [ref]$parsed) -or $parsed -le 0) { throw "Invalid PID entry '$($_)' in run PID file." }
            $parsed
        })
    }
} catch {
    $result.errors += "Could not validate per-run simulator manifest: $($_.Exception.Message)"
    Complete-Result 4
}

# The PID file describes the currently launched simulator session. The append-only
# identity manifest also lets hard-deadline cleanup recover a launcher that died
# before clearing that file. Every live candidate still needs its original exact
# executable, command line, port, PID, and creation time to match.
$candidatePids = @(@($pidsFromFile) + @($records | ForEach-Object { [int]$_.pid }) | Sort-Object -Unique)
$invalid = $false
foreach ($pidValue in $candidatePids) {
    $matches = @($records | Where-Object { [int]$_.pid -eq [int]$pidValue })
    if ($matches.Count -eq 0) {
        $invalid = $true
        $result.rejected_pids += [ordered]@{ pid = [int]$pidValue; reason = 'PID has no matching record in the run identity manifest.' }
        continue
    }
    $record = $matches[-1]
    $target = [ordered]@{
        pid = [int]$pidValue
        environment = [int]$record.environment
        port = [int]$record.port
        bridge_ip = [string]$record.bridge_ip
        executable_path = [string]$record.executable_path
        command_line = [string]$record.command_line
        process_creation_utc = [string]$record.process_creation_utc
        process_creation_cim = [string]$record.process_creation_cim
        live = $false
        remote_port_connection_present = $false
        action = 'none'
        reason = $null
    }
    $current = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
    if (-not $current) {
        $target.reason = 'PID is already absent.'
        $result.already_exited_pids += [int]$pidValue
        $result.targets += $target
        continue
    }

    $target.live = $true
    $expectedExe = [IO.Path]::GetFullPath([string]$record.executable_path)
    $cmd = [string]$current.CommandLine
    $expectedPortArg = '(?i)(?:^|\s)-port\s+' + [regex]::Escape([string]$record.port) + '(?:\s|"|$)'
    $expectedIpArg = '(?i)(?:^|\s)-ip\s+' + [regex]::Escape([string]$record.bridge_ip) + '(?:\s|"|$)'
    $exeMatches = [string]::Compare([IO.Path]::GetFullPath([string]$current.ExecutablePath), $expectedExe, $true) -eq 0 -and
        [string]::Compare($expectedExe, $simulatorPath, $true) -eq 0
    $commandMatches = [string]::Compare($cmd, [string]$record.command_line, [StringComparison]::OrdinalIgnoreCase) -eq 0 -and
        $cmd -match $expectedPortArg -and $cmd -match $expectedIpArg
    $createdMatches = -not [string]::IsNullOrWhiteSpace([string]$record.process_creation_cim) -and
        [string]::Compare([string]$current.CreationDate, [string]$record.process_creation_cim, [StringComparison]::OrdinalIgnoreCase) -eq 0
    $portConnection = @(Get-NetTCPConnection -OwningProcess ([int]$pidValue) -State Established -ErrorAction SilentlyContinue |
        Where-Object { $_.RemoteAddress -eq [string]$record.bridge_ip -and $_.RemotePort -eq [int]$record.port })
    $target.remote_port_connection_present = $portConnection.Count -gt 0
    if (-not $exeMatches -or -not $commandMatches -or -not $createdMatches) {
        $invalid = $true
        $target.reason = 'Live PID identity did not match the recorded executable, exact command line/port, and creation time.'
        $result.rejected_pids += [ordered]@{
            pid = [int]$pidValue
            reason = $target.reason
            actual_executable_path = [string]$current.ExecutablePath
            actual_command_line = $cmd
            actual_creation_date = [string]$current.CreationDate
            recorded_creation_date = [string]$record.process_creation_cim
            executable_matches = $exeMatches
            command_line_matches = $commandMatches
            creation_time_matches = $createdMatches
        }
    } else {
        $target.reason = if ($target.remote_port_connection_present) { 'Exact run identity verified; expected relay port connection is established.' } else { 'Exact run identity verified; expected port is present in the original command line.' }
        $target.action = 'verified'
    }
    $result.targets += $target
}

if ($invalid) {
    $result.errors += 'At least one live candidate failed identity validation; no simulator process was terminated.'
    Complete-Result 5
}

foreach ($target in @($result.targets | Where-Object { $_.live -and $_.action -eq 'verified' })) {
    try {
        if ($ValidateOnly) {
            $target.action = 'validated_only'
            $target.reason = 'Identity was fully verified; ValidateOnly prevented termination.'
            continue
        }
        Stop-Process -Id ([int]$target.pid) -Force -ErrorAction Stop
        $deadline = [DateTime]::UtcNow.AddSeconds(10)
        do {
            Start-Sleep -Milliseconds 200
            $stillThere = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($target.pid)" -ErrorAction SilentlyContinue
        } while ($stillThere -and [DateTime]::UtcNow -lt $deadline)
        if ($stillThere) {
            $target.action = 'termination_unconfirmed'
            $target.reason = 'Stop-Process returned but the exact verified PID remained alive after 10 seconds.'
            $result.errors += "PID $($target.pid) remained alive after the bounded termination wait."
        } else {
            $target.action = 'terminated'
            $target.reason = 'Exact verified simulator PID was terminated and confirmed absent.'
            $result.terminated_pids += [int]$target.pid
        }
    } catch {
        $target.action = 'termination_failed'
        $target.reason = $_.Exception.Message
        $result.errors += "Could not terminate verified PID $($target.pid): $($_.Exception.Message)"
    }
}

if ($result.errors.Count -eq 0) { $result.result = if ($ValidateOnly) { 'validated_only' } else { 'success' } }
Complete-Result $(if ($result.errors.Count -eq 0) { 0 } else { 6 })
