Describe 'terminate_verified_official_simulators.ps1' {
    BeforeEach {
        $script:runId = ([guid]::NewGuid().ToString('N')).Substring(0, 12)
        $script:fixtureRoot = Join-Path $env:TEMP ("AICAR-official-cleanup-test-" + [guid]::NewGuid().ToString('N'))
        $script:runDirectory = Join-Path $script:fixtureRoot "logs/rl/official_run_$script:runId"
        New-Item -ItemType Directory -Path $script:runDirectory -Force | Out-Null
        $script:simulatorPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $argumentString = '-NoProfile -Command "Start-Sleep -Seconds 180; # -ip 127.0.0.1 -port 45678"'
        $script:child = Start-Process -FilePath $script:simulatorPath -ArgumentList $argumentString -PassThru -WindowStyle Hidden
        Start-Sleep -Milliseconds 500
        $script:childCim = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $($script:child.Id)"
        if (-not $script:childCim) { throw 'Test fixture process did not start.' }
        $script:record = [ordered]@{
            pid = [int]$script:child.Id
            environment = 0
            port = 45678
            bridge_ip = '127.0.0.1'
            executable_path = [string]$script:childCim.ExecutablePath
            command_line = [string]$script:childCim.CommandLine
            parent_pid = [int]$script:childCim.ParentProcessId
            process_creation_utc = $script:child.StartTime.ToUniversalTime().ToString('o')
            process_creation_cim = [string]$script:childCim.CreationDate
            launched_utc = [DateTime]::UtcNow.ToString('o')
            batch_mode = $true
        }
        Set-Content -LiteralPath (Join-Path $script:runDirectory 'windows_simulator_pids.txt') -Value ([string]$script:child.Id) -Encoding Ascii
        $script:manifest = [ordered]@{
            schema_version = 1
            run_id = $script:runId
            simulator_exe_path = $script:simulatorPath
            bridge_ip = '127.0.0.1'
            ports = @(45678)
            records = @($script:record)
        }
        $script:manifestPath = Join-Path $script:runDirectory 'windows_simulator_processes.json'
        $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $script:manifestPath -Encoding UTF8
    }

    AfterEach {
        if ($script:child) { Stop-Process -Id $script:child.Id -Force -ErrorAction SilentlyContinue }
        if ($script:fixtureRoot) {
            $resolved = [IO.Path]::GetFullPath($script:fixtureRoot)
            $tempPrefix = [IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\AICAR-official-cleanup-test-'
            if ($resolved.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -and (Test-Path -LiteralPath $resolved)) {
                Remove-Item -LiteralPath $resolved -Recurse -Force
            }
        }
    }

    It 'validates the exact per-run PID, executable, command line, creation time, and port without stopping it in validate-only mode' {
        $helper = Join-Path $PSScriptRoot '..\scripts\terminate_verified_official_simulators.ps1'
        $windowsPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $output = & $windowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $helper -RunId $script:runId -ApiStopAttempted -GraceExpired -HardDeadlineReached -ValidateOnly -ProjectRoot $script:fixtureRoot
        $exitCode = $LASTEXITCODE
        $payload = ($output -join "`n") | ConvertFrom-Json
        $exitCode | Should Be 0
        $payload.result | Should Be 'validated_only'
        $payload.targets.Count | Should Be 1
        $payload.targets[0].pid | Should Be $script:child.Id
        $payload.targets[0].action | Should Be 'validated_only'
        (Get-Process -Id $script:child.Id -ErrorAction SilentlyContinue) | Should Not Be $null
    }

    It 'fails closed when a live PID command line differs from its recorded run identity' {
        $script:manifest.records[0].command_line = $script:manifest.records[0].command_line + ' stale'
        $script:manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $script:manifestPath -Encoding UTF8
        $helper = Join-Path $PSScriptRoot '..\scripts\terminate_verified_official_simulators.ps1'
        $windowsPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $output = & $windowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $helper -RunId $script:runId -ApiStopAttempted -GraceExpired -HardDeadlineReached -ValidateOnly -ProjectRoot $script:fixtureRoot
        $exitCode = $LASTEXITCODE
        $payload = ($output -join "`n") | ConvertFrom-Json
        $exitCode | Should Be 5
        $payload.result | Should Be 'failed'
        $payload.rejected_pids[0].pid | Should Be $script:child.Id
        (Get-Process -Id $script:child.Id -ErrorAction SilentlyContinue) | Should Not Be $null
    }

    It 'terminates only the exact verified per-run PID after the grace and deadline gates' {
        $helper = Join-Path $PSScriptRoot '..\scripts\terminate_verified_official_simulators.ps1'
        $windowsPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $output = & $windowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $helper -RunId $script:runId -ApiStopAttempted -GraceExpired -HardDeadlineReached -ProjectRoot $script:fixtureRoot
        $exitCode = $LASTEXITCODE
        $payload = ($output -join "`n") | ConvertFrom-Json
        $exitCode | Should Be 0
        $payload.result | Should Be 'success'
        (@($payload.terminated_pids | ForEach-Object { [int]$_ }) -contains [int]$script:child.Id) | Should Be $true
        (Get-Process -Id $script:child.Id -ErrorAction SilentlyContinue) | Should Be $null
    }

    It 'fails closed when terminal API state cannot be confirmed before the hard deadline' {
        $helper = Join-Path $PSScriptRoot '..\scripts\terminate_verified_official_simulators.ps1'
        $windowsPowerShell = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
        $output = & $windowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $helper -RunId $script:runId -ApiStopAttempted -GraceExpired -ValidateOnly -ProjectRoot $script:fixtureRoot -HubUrl 'http://127.0.0.1:1'
        $exitCode = $LASTEXITCODE
        $payload = ($output -join "`n") | ConvertFrom-Json
        $exitCode | Should Be 3
        $payload.result | Should Be 'failed'
        $payload.terminated_pids.Count | Should Be 0
        (Get-Process -Id $script:child.Id -ErrorAction SilentlyContinue) | Should Not Be $null
    }
}
