# Detached campaign keep-alive — survives Cursor agent shell teardown.
# Cadence: 10 minutes (600s). Ticks → logs/campaign_loop_ticks.log
$ErrorActionPreference = 'Continue'
$intervalSec = 600
$deadlineLocal = [DateTime]::Parse('2026-09-17T12:54:00')
$tz = [TimeZoneInfo]::FindSystemTimeZoneById('Central Standard Time')
$rlDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$logPath = Join-Path $rlDir 'logs\campaign_loop_ticks.log'
$pidPath = Join-Path $rlDir 'logs\campaign_keepalive.pid'
New-Item -ItemType Directory -Force -Path (Split-Path $logPath) | Out-Null
$PID | Set-Content -Path $pidPath -NoNewline

$prompt = 'CAMPAIGN KEEP-ALIVE TICK (10m aggressive): Execute ADSS Toolkit/autodrive_py/rl/campaign_tick.md in full. (1) PARALLEL: Resume Researcher f4ac2241-2ce6-4f1b-b7ea-719eb29035dc + Racer b0ab82af-c52e-400e-aa7c-0481311ab3cb + Minimalist 6e80ad0d-65b8-4b3b-9760-da999439843c + Reliability a9ade664-e7ec-4df9-a918-9ddb0acf9a9c for NEXT proposal batch; append CAMPAIGN_BOARD.md Go/No-Go. (2) Resume Orchestrator 4a5b7bd3-a4ff-42b9-ae21-c6461b5e8931 to implement MULTIPLE board-approved small features in parallel (spawn sibling implementers); protect overnight_soak_20260917_082739; commit+push often. (3) If training dead, Continue restart same run_id. (4) If past 12:54 America/Chicago 2026-09-17, STOP loop (kill loop shell, do not re-arm). Throughput: ship real features every tick — no padding.'
$escaped = $prompt.Replace('\', '\\').Replace('"', '\"')

$boot = "Loop armed FULL BOARD 10m pid=$PID interval=${intervalSec}s deadline=12:54Chicago log=$logPath"
Write-Host $boot
Add-Content -Path $logPath -Value "$(Get-Date -Format o) BOOT $boot"

while ($true) {
  Start-Sleep -Seconds $intervalSec
  $nowChi = [TimeZoneInfo]::ConvertTimeFromUtc((Get-Date).ToUniversalTime(), $tz)
  if ($nowChi -ge $deadlineLocal) {
    $line = 'AGENT_LOOP_TICK_autodrive-rl-campaign {"prompt":"STOP CAMPAIGN LOOP: past 12:54 America/Chicago 2026-09-17. Kill this loop shell and do not re-arm. Finalize any push, then end the campaign."}'
    Write-Output $line
    Add-Content -Path $logPath -Value "$(Get-Date -Format o) $line"
    break
  }
  $line = "AGENT_LOOP_TICK_autodrive-rl-campaign {`"prompt`":`"$escaped`"}"
  Write-Output $line
  Add-Content -Path $logPath -Value "$(Get-Date -Format o) $line"
}

Add-Content -Path $logPath -Value "$(Get-Date -Format o) EXIT pid=$PID"
Remove-Item -Path $pidPath -ErrorAction SilentlyContinue
