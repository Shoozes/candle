Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$runnerScript = Join-Path $repoRoot "scripts\gpt-oss\run-performance.ps1"
. $runnerScript -AsLibrary

$testRoot = Join-Path ([IO.Path]::GetTempPath()) ("candle-gpt-oss-performance-tests-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$testsPassed = 0

function Assert-Test {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "test failure: $Message" }
    $script:testsPassed++
}

try {
    $plan = Get-PerformancePlan -LengthSpec "8,auto" -Target 128 -Generated 16 -MeasurementMode "autoregressive"
    Assert-Test ($plan.Count -eq 2 -and $plan[1] -eq 112) "valid progressive plan did not resolve auto target"

    try { Get-PerformancePlan -LengthSpec "113" -Target 128 -Generated 16 -MeasurementMode "autoregressive" | Out-Null; throw "over-target plan unexpectedly succeeded" } catch { Assert-Test ($_.Exception.Message -match "exceeds target") "over-target configuration was not rejected" }
    try { Get-PerformancePlan -LengthSpec "8" -Target 16 -Generated 16 -MeasurementMode "teacher-forced" | Out-Null; throw "invalid target/decode plan unexpectedly succeeded" } catch { Assert-Test ($_.Exception.Message -match "leave at least one prompt") "invalid target/decode configuration was not rejected" }

    Assert-Test ((Get-TerminalStatus -RunnerStatus "cancelled") -eq "cancelled") "cooperative cancellation was not classified as cancelled"
    Assert-Test ((Get-TerminalStatus -RunnerStatus "timeout") -eq "timeout") "deadline cancellation was not classified as timeout"
    Assert-Test ((Get-TerminalStatus -RunnerStatus "model_error" -MonitorFailure) -eq "monitor_error") "monitor failure did not take precedence"

    $staleOutput = Join-Path $testRoot "stale-report.json"
    [IO.File]::WriteAllText($staleOutput, "owner evidence")
    $configuration = @{
        Model = "missing-model.gguf"; Tokenizer = "missing-tokenizer.json"; Output = $staleOutput
        Lengths = "8"; TargetContextTokens = 128; DecodeTokens = 16; Mode = "autoregressive"
        ProfileId = "test-profile"; BuildIdentity = "test-build"; DeviceIndex = 0
        MaxWeightBytes = [UInt64]1; MaxCacheBytes = [UInt64]1; MaxTotalDeviceBytes = [UInt64]1
        OverallDeadlineMs = [UInt64]1000; GracePeriodMs = 1000; SampleIntervalMs = 100
    }
    try { Assert-PerformanceConfiguration -Configuration $configuration | Out-Null; throw "stale output unexpectedly succeeded" } catch { Assert-Test ($_.Exception.Message -match "stale performance output") "stale output was not rejected" }
    Assert-Test ((Get-Content -LiteralPath $staleOutput -Raw) -eq "owner evidence") "stale output was modified"

    $progressPath = Join-Path $testRoot "monitor-progress.json"
    Write-MonitorProgress -Path $progressPath -RunId "test-run" -ProfileId "test-profile" -BuildIdentity "test-build" -Mode "autoregressive" -Samples @([pscustomobject]@{ phase = "case_8_ready"; gpu_used_bytes = 1 }) -Phase "case_8_ready" -CompletedCases 1 -TotalCases 3
    $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
    Assert-Test ($progress.status -eq "partial" -and $progress.completed_cases -eq 1) "one completed case was not preserved as partial evidence"

    $terminalPath = Join-Path $testRoot "terminal.json"
    Write-RunTerminal -Path $terminalPath -Status "forced_termination" -Phase "case_8_ready" -RunId "test-run" -ProfileId "test-profile" -BuildIdentity "test-build" -Mode "autoregressive" -CompletedCases 1 -TotalCases 3 -ErrorMessage "simulated interruption" -RunnerTerminalPath (Join-Path $testRoot "runner-terminal.json")
    $terminal = Get-Content -LiteralPath $terminalPath -Raw | ConvertFrom-Json
    Assert-Test ($terminal.status -eq "forced_termination" -and $terminal.success_claim -eq $false -and $terminal.completed_cases -eq 1) "interruption after one case was misreported"

    Write-Output "gpt-oss-performance tests: $testsPassed passed"
} finally {
    if (Test-Path -LiteralPath $testRoot) { Remove-Item -LiteralPath $testRoot -Recurse -Force }
}
