[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Tokenizer,
    [string]$Output = "artifacts/gpt-oss/performance/report.json",
    [string]$Lengths = "8,512,2048,8192,16384,auto",
    [int]$TargetContextTokens = 32768,
    [int]$DecodeTokens = 4,
    [string]$Prompt = "The quick brown fox jumps over the lazy dog. Performance characterization prompt.",
    [int]$DeviceIndex = 0,
    [UInt64]$MaxWeightBytes = [UInt64]::MaxValue,
    [int]$PostUnloadHoldMs = 10000,
    [int]$SampleIntervalMs = 1000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($DecodeTokens -lt 1) {
    throw "DecodeTokens must be greater than zero."
}
if ($TargetContextTokens -lt 1) {
    throw "TargetContextTokens must be greater than zero."
}
if ($SampleIntervalMs -lt 100) {
    throw "SampleIntervalMs must be at least 100 ms."
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $repoRoot
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$tokenizerPath = (Resolve-Path -LiteralPath $Tokenizer).Path
$outputPath = [IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
$outputDirectory = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$nvidiaCommand = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
if ($null -eq $nvidiaCommand) {
    throw "nvidia-smi.exe is required for GPU peak and recovery evidence."
}

Write-Host "Building gpt-oss-performance..."
& cargo build --locked --features cuda -p candle-examples --example gpt-oss-performance
if ($LASTEXITCODE -ne 0) {
    throw "gpt-oss-performance build failed with exit code $LASTEXITCODE."
}

$executable = Join-Path $repoRoot "target\debug\examples\gpt-oss-performance.exe"
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "Built performance executable was not found: $executable"
}

function Get-GpuSample {
    param([int]$Index)
    $line = & $nvidiaCommand.Source --id $Index --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw "nvidia-smi returned no memory sample for device $Index."
    }
    $parts = ([string]$line).Split(',') | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 2) {
        throw "Unexpected nvidia-smi memory output: $line"
    }
    [pscustomobject]@{
        used_bytes = [UInt64]([double]$parts[0] * 1MB)
        total_bytes = [UInt64]([double]$parts[1] * 1MB)
    }
}

function Get-SystemSample {
    param(
        [int]$ProcessId = 0,
        [string]$Phase = "external"
    )
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $total = [UInt64]$os.TotalVisibleMemorySize * 1KB
    $available = [UInt64]$os.FreePhysicalMemory * 1KB
    $process = $null
    if ($ProcessId -gt 0) {
        $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    }
    $gpu = Get-GpuSample -Index $DeviceIndex
    [pscustomobject]@{
        timestamp_utc = [DateTimeOffset]::UtcNow.ToString("o")
        phase = $Phase
        host_ram_total_bytes = $total
        host_ram_available_bytes = $available
        host_ram_used_bytes = $total - $available
        process_private_bytes = if ($null -eq $process) { $null } else { [UInt64]$process.PrivateMemorySize64 }
        process_working_set_bytes = if ($null -eq $process) { $null } else { [UInt64]$process.WorkingSet64 }
        gpu_used_bytes = $gpu.used_bytes
        gpu_total_bytes = $gpu.total_bytes
    }
}

function Quote-ProcessArgument {
    param([string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

$stdoutPath = Join-Path $outputDirectory ".gpt-oss-performance.stdout.log"
$stderrPath = Join-Path $outputDirectory ".gpt-oss-performance.stderr.log"
Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue

$baselineSample = Get-SystemSample -Phase "pre_launch"
$arguments = @(
    "--model", $modelPath,
    "--tokenizer", $tokenizerPath,
    "--output", $outputPath,
    "--lengths", $Lengths,
    "--target-context-tokens", $TargetContextTokens.ToString(),
    "--decode-tokens", $DecodeTokens.ToString(),
    "--prompt", $Prompt,
    "--device-index", $DeviceIndex.ToString(),
    "--max-weight-bytes", $MaxWeightBytes.ToString(),
    "--post-unload-hold-ms", $PostUnloadHoldMs.ToString()
)
$argumentString = ($arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join " "
$process = Start-Process -FilePath $executable -ArgumentList $argumentString -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$samples = [System.Collections.Generic.List[object]]::new()
$phase = "process_start"
$postUnloadSeen = $false

try {
    while (-not $process.HasExited) {
        $process.Refresh()
        $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue } else { "" }
        if ($null -eq $stdout) {
            $stdout = ""
        }
        $phaseMatches = [regex]::Matches($stdout, "PERF_PHASE (?<phase>[^\r\n ]+)")
        if ($phaseMatches.Count -gt 0) {
            $phase = $phaseMatches[$phaseMatches.Count - 1].Groups["phase"].Value
        }
        $sample = Get-SystemSample -ProcessId $process.Id -Phase $phase
        $samples.Add($sample)
        if ($phase -eq "post_unload") {
            $postUnloadSeen = $true
        }
        Start-Sleep -Milliseconds $SampleIntervalMs
    }
    $process.Refresh()
    $exitCode = $process.ExitCode
} finally {
    if (-not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
}

$afterProcessSample = Get-SystemSample -Phase "after_process_exit"
if ($exitCode -ne 0) {
    $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
    throw "Performance runner exited with code $exitCode. stderr: $stderr"
}
if (-not $postUnloadSeen) {
    throw "Performance runner exited without a post_unload phase marker."
}
if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
    throw "Performance report was not written: $outputPath"
}

$report = Get-Content -LiteralPath $outputPath -Raw | ConvertFrom-Json
$allSamples = @($baselineSample) + @($samples) + @($afterProcessSample)
$runnerSamples = @($samples)
$postUnloadSamples = @($runnerSamples | Where-Object { $_.phase -eq "post_unload" })
$privateSamples = @($runnerSamples | Where-Object { $null -ne $_.process_private_bytes })
$workingSetSamples = @($runnerSamples | Where-Object { $null -ne $_.process_working_set_bytes })
$peakPrivate = if ($privateSamples.Count -eq 0) { $null } else { [UInt64](($privateSamples | Measure-Object -Property process_private_bytes -Maximum).Maximum) }
$peakWorkingSet = if ($workingSetSamples.Count -eq 0) { $null } else { [UInt64](($workingSetSamples | Measure-Object -Property process_working_set_bytes -Maximum).Maximum) }
$peakHostUsed = [UInt64](($allSamples | Measure-Object -Property host_ram_used_bytes -Maximum).Maximum)
$peakGpuUsed = [UInt64](($allSamples | Measure-Object -Property gpu_used_bytes -Maximum).Maximum)
$maxCache = [UInt64](($report.cases | ForEach-Object { $_.cache_capacity_bytes_after_decode } | Measure-Object -Maximum).Maximum)
$staticBytes = [UInt64]$report.static_resident_bytes
$baselineGpu = [UInt64]$baselineSample.gpu_used_bytes
$gpuWorkspaceOverlap = if ($peakGpuUsed -gt ($baselineGpu + $staticBytes + $maxCache)) {
    $peakGpuUsed - $baselineGpu - $staticBytes - $maxCache
} else { [UInt64]0 }
$baselinePrivate = if ($null -eq $baselineSample.process_private_bytes) { [UInt64]0 } else { [UInt64]$baselineSample.process_private_bytes }
$hostTemporaryOverlap = if ($null -ne $peakPrivate -and $peakPrivate -gt ($baselinePrivate + [UInt64]$report.model_bytes)) {
    $peakPrivate - $baselinePrivate - [UInt64]$report.model_bytes
} else { [UInt64]0 }
$postUnloadSample = if ($postUnloadSamples.Count -gt 0) { $postUnloadSamples[$postUnloadSamples.Count - 1] } else { $null }

$monitor = [ordered]@{
    schema = "candle.gpt_oss_performance_monitor.v1"
    sample_interval_ms = $SampleIntervalMs
    baseline = $baselineSample
    peak = [ordered]@{
        host_ram_used_bytes = $peakHostUsed
        process_private_bytes = $peakPrivate
        process_working_set_bytes = $peakWorkingSet
        gpu_used_bytes = $peakGpuUsed
    }
    configured_vs_observed = [ordered]@{
        configured_context_length = $report.configured_context_length
        maximum_actual_prompt_tokens = [UInt64](($report.cases | Measure-Object -Property actual_prompt_tokens -Maximum).Maximum)
        configured_cache_bytes = $report.configured_cache_bytes
        maximum_allocated_cache_bytes = $maxCache
    }
    workspace_temporary_overlap = [ordered]@{
        gpu_bytes = $gpuWorkspaceOverlap
        gpu_definition = "max(0, peak GPU used - pre-launch GPU used - static resident bytes - maximum logical cache capacity)"
        host_bytes = $hostTemporaryOverlap
        host_definition = "max(0, peak runner private bytes - pre-launch runner private bytes - model file bytes); this is an upper-bound attribution, not a direct allocator census"
    }
    post_unload_recovery = [ordered]@{
        marker_seen = $postUnloadSeen
        sample = $postUnloadSample
        after_process_exit = $afterProcessSample
        gpu_delta_from_baseline_bytes = if ($null -eq $postUnloadSample) { $null } else { [Int64]$postUnloadSample.gpu_used_bytes - [Int64]$baselineGpu }
        host_ram_delta_from_baseline_bytes = if ($null -eq $postUnloadSample) { $null } else { [Int64]$postUnloadSample.host_ram_used_bytes - [Int64]$baselineSample.host_ram_used_bytes }
    }
    samples = $runnerSamples
}
$report | Add-Member -NotePropertyName monitor -NotePropertyValue $monitor -Force
$json = $report | ConvertTo-Json -Depth 12
[IO.File]::WriteAllText($outputPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Write-Host "gpt-oss-performance: passed; report=$outputPath"
