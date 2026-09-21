[CmdletBinding()]
param(
    [string]$Model,
    [string]$Tokenizer,
    [string]$Output = "artifacts/gpt-oss/performance/report.json",
    [string]$Lengths = "8,512,2048,8192,16384,auto",
    [int]$TargetContextTokens = 32768,
    [int]$DecodeTokens = 32,
    [ValidateSet("autoregressive", "teacher-forced")][string]$Mode = "autoregressive",
    [string]$Prompt = "The quick brown fox jumps over the lazy dog. Performance characterization prompt.",
    [int]$DeviceIndex = 0,
    [UInt64]$MaxWeightBytes = 0,
    [UInt64]$MaxCacheBytes = 0,
    [UInt64]$MaxTotalDeviceBytes = 0,
    [UInt64]$OverallDeadlineMs = 3600000,
    [int]$GracePeriodMs = 30000,
    [int]$PostUnloadHoldMs = 10000,
    [int]$SampleIntervalMs = 1000,
    [string]$ProfileId = "gpt-oss-20b-mxfp4-cuda-f32-ar-v2",
    [string]$BuildIdentity,
    [string]$ExecutableOverride,
    [switch]$SkipBuild,
    [switch]$AsLibrary
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-JsonEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText($Path, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
}

function Get-PerformancePlan {
    param(
        [Parameter(Mandatory = $true)][string]$LengthSpec,
        [Parameter(Mandatory = $true)][int]$Target,
        [Parameter(Mandatory = $true)][int]$Generated,
        [Parameter(Mandatory = $true)][string]$MeasurementMode
    )
    if ($Target -lt 1) { throw "TargetContextTokens must be greater than zero." }
    if ($Generated -lt 1) { throw "DecodeTokens must be greater than zero." }
    if ($Target -le $Generated) {
        throw "TargetContextTokens must leave at least one prompt token after DecodeTokens."
    }
    if ($MeasurementMode -eq "autoregressive" -and $Generated -lt 16) {
        throw "autoregressive qualification requires at least 16 generated tokens."
    }
    $availablePrompt = $Target - $Generated
    $lengths = [System.Collections.Generic.List[int]]::new()
    foreach ($item in $LengthSpec.Split(',')) {
        $trimmed = $item.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed)) {
            throw "Lengths contains an empty item."
        }
        if ($trimmed -ieq "auto") {
            $length = $availablePrompt
        } else {
            $parsed = 0
            if (-not [int]::TryParse($trimmed, [Globalization.NumberStyles]::Integer, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
                throw "Invalid benchmark length '$trimmed'."
            }
            $length = $parsed
        }
        if ($length -lt 1 -or $length -gt $availablePrompt) {
            throw "Benchmark prompt length $length exceeds target context $Target after $Generated generated/decode tokens."
        }
        $lengths.Add($length)
    }
    if ($lengths.Count -eq 0) { throw "Lengths must contain at least one prompt length." }
    return @($lengths)
}

function Assert-PerformanceConfiguration {
    param(
        [Parameter(Mandatory = $true)][hashtable]$Configuration,
        [switch]$CheckInputFiles
    )
    if ([string]::IsNullOrWhiteSpace([string]$Configuration.Model)) { throw "Model is required." }
    if ([string]::IsNullOrWhiteSpace([string]$Configuration.Tokenizer)) { throw "Tokenizer is required." }
    if ([string]::IsNullOrWhiteSpace([string]$Configuration.Output)) { throw "Output is required." }
    if ([string]::IsNullOrWhiteSpace([string]$Configuration.ProfileId)) { throw "ProfileId is required." }
    if ([string]::IsNullOrWhiteSpace([string]$Configuration.BuildIdentity)) { throw "BuildIdentity is required." }
    if ([int]$Configuration.DeviceIndex -lt 0) { throw "DeviceIndex must be non-negative." }
    if ([UInt64]$Configuration.MaxWeightBytes -eq 0 -or [UInt64]$Configuration.MaxCacheBytes -eq 0 -or [UInt64]$Configuration.MaxTotalDeviceBytes -eq 0) {
        throw "MaxWeightBytes, MaxCacheBytes, and MaxTotalDeviceBytes are required explicit positive budgets."
    }
    if ([UInt64]$Configuration.OverallDeadlineMs -eq 0) { throw "OverallDeadlineMs must be greater than zero." }
    if ([int]$Configuration.SampleIntervalMs -lt 100) { throw "SampleIntervalMs must be at least 100 ms." }
    if ([int]$Configuration.GracePeriodMs -lt 1000) { throw "GracePeriodMs must be at least 1000 ms." }
    $outputPath = [IO.Path]::GetFullPath([string]$Configuration.Output)
    if (Test-Path -LiteralPath $outputPath) {
        throw "refusing to overwrite stale performance output: $outputPath"
    }
    $outputParent = Split-Path -Parent $outputPath
    if (Test-Path -LiteralPath $outputParent -PathType Leaf) {
        throw "performance output parent is a file: $outputParent"
    }
    if ($CheckInputFiles) {
        if (-not (Test-Path -LiteralPath ([string]$Configuration.Model) -PathType Leaf)) {
            throw "Model is not a regular file: $($Configuration.Model)"
        }
        if (-not (Test-Path -LiteralPath ([string]$Configuration.Tokenizer) -PathType Leaf)) {
            throw "Tokenizer is not a regular file: $($Configuration.Tokenizer)"
        }
    }
    $plan = Get-PerformancePlan -LengthSpec ([string]$Configuration.Lengths) -Target ([int]$Configuration.TargetContextTokens) -Generated ([int]$Configuration.DecodeTokens) -MeasurementMode ([string]$Configuration.Mode)
    return $plan
}

function New-PerformanceRunDirectory {
    param([Parameter(Mandatory = $true)][string]$OutputPath)
    $parent = Split-Path -Parent $OutputPath
    $runs = Join-Path $parent "runs"
    New-Item -ItemType Directory -Force -Path $runs | Out-Null
    $runId = "{0}-{1}" -f ([DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")), ([guid]::NewGuid().ToString("N").Substring(0, 12))
    $runDirectory = Join-Path $runs $runId
    New-Item -ItemType Directory -Path $runDirectory | Out-Null
    return $runDirectory
}

function Get-TerminalStatus {
    param(
        [string]$RunnerStatus,
        [switch]$MonitorFailure,
        [switch]$ForcedTermination,
        [switch]$Interrupted
    )
    if ($ForcedTermination -or $Interrupted) { return "forced_termination" }
    if ($MonitorFailure) { return "monitor_error" }
    if ($RunnerStatus -eq "timeout") { return "timeout" }
    if ($RunnerStatus -eq "cancelled") { return "cancelled" }
    if ($RunnerStatus -eq "success") { return "success" }
    return "model_error"
}

function Write-RunTerminal {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$BuildIdentity,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)][int]$CompletedCases,
        [Parameter(Mandatory = $true)][int]$TotalCases,
        [string]$ErrorMessage,
        [string]$RunnerTerminalPath,
        [int]$ExitCode = -1
    )
    Write-JsonEvidence -Path $Path -Value ([ordered]@{
        schema = "candle.gpt_oss_performance_terminal.v2"
        status = $Status
        phase = $Phase
        run_id = $RunId
        profile_id = $ProfileId
        build_identity = $BuildIdentity
        mode = $Mode
        completed_cases = $CompletedCases
        total_cases = $TotalCases
        exit_code = $ExitCode
        error = $ErrorMessage
        runner_terminal = $RunnerTerminalPath
        success_claim = ($Status -eq "success")
    })
}

function Write-MonitorProgress {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$RunId,
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$BuildIdentity,
        [Parameter(Mandatory = $true)][string]$Mode,
        [Parameter(Mandatory = $true)]$Samples,
        [Parameter(Mandatory = $true)][string]$Phase,
        [Parameter(Mandatory = $true)][int]$CompletedCases,
        [Parameter(Mandatory = $true)][int]$TotalCases
    )
    Write-JsonEvidence -Path $Path -Value ([ordered]@{
        schema = "candle.gpt_oss_performance_monitor_progress.v2"
        status = "partial"
        run_id = $RunId
        profile_id = $ProfileId
        build_identity = $BuildIdentity
        mode = $Mode
        phase = $Phase
        completed_cases = $CompletedCases
        total_cases = $TotalCases
        sample_count = @($Samples).Count
        samples = @($Samples)
    })
}

function Get-GpuSample {
    param([Parameter(Mandatory = $true)][int]$Index)
    $line = & nvidia-smi.exe --id $Index --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace([string]$line)) { throw "nvidia-smi returned no memory sample for device $Index." }
    $parts = ([string]$line).Split(',') | ForEach-Object { $_.Trim() }
    if ($parts.Count -lt 2) { throw "Unexpected nvidia-smi memory output: $line" }
    [pscustomobject]@{
        used_bytes = [UInt64]([double]$parts[0] * 1MB)
        total_bytes = [UInt64]([double]$parts[1] * 1MB)
    }
}

function Get-SystemSample {
    param(
        [Parameter(Mandatory = $true)][int]$DeviceIndex,
        [int]$ProcessId = 0,
        [string]$Phase = "external"
    )
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $total = [UInt64]$os.TotalVisibleMemorySize * 1KB
    $available = [UInt64]$os.FreePhysicalMemory * 1KB
    $process = $null
    if ($ProcessId -gt 0) { $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue }
    $gpu = Get-GpuSample -Index $DeviceIndex
    return [pscustomobject]@{
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

function Get-GitBuildIdentity {
    $identity = (& git rev-parse HEAD 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace([string]$identity)) {
        throw "unable to resolve an explicit Git build identity"
    }
    return ([string]$identity).Trim()
}

function Quote-ProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-GptOssPerformance {
    param(
        [string]$Model,
        [string]$Tokenizer,
        [string]$Output,
        [string]$Lengths,
        [int]$TargetContextTokens,
        [int]$DecodeTokens,
        [string]$Mode,
        [string]$Prompt,
        [int]$DeviceIndex,
        [UInt64]$MaxWeightBytes,
        [UInt64]$MaxCacheBytes,
        [UInt64]$MaxTotalDeviceBytes,
        [UInt64]$OverallDeadlineMs,
        [int]$GracePeriodMs,
        [int]$PostUnloadHoldMs,
        [int]$SampleIntervalMs,
        [string]$ProfileId,
        [string]$BuildIdentity,
        [string]$ExecutableOverride,
        [switch]$SkipBuild
    )
    if ([string]::IsNullOrWhiteSpace($BuildIdentity)) { $BuildIdentity = Get-GitBuildIdentity }
    $preConfiguration = @{
        Model = $Model; Tokenizer = $Tokenizer; Output = $Output; Lengths = $Lengths
        TargetContextTokens = $TargetContextTokens; DecodeTokens = $DecodeTokens; Mode = $Mode
        ProfileId = $ProfileId; BuildIdentity = $BuildIdentity
        DeviceIndex = $DeviceIndex; MaxWeightBytes = $MaxWeightBytes; MaxCacheBytes = $MaxCacheBytes
        MaxTotalDeviceBytes = $MaxTotalDeviceBytes; OverallDeadlineMs = $OverallDeadlineMs
        GracePeriodMs = $GracePeriodMs; SampleIntervalMs = $SampleIntervalMs
    }
    $outputPath = [IO.Path]::GetFullPath($Output)
    $plan = Assert-PerformanceConfiguration -Configuration $preConfiguration -CheckInputFiles
    $modelPath = (Resolve-Path -LiteralPath $Model).Path
    $tokenizerPath = (Resolve-Path -LiteralPath $Tokenizer).Path
    $nvidia = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) { throw "nvidia-smi.exe is required for GPU peak and recovery evidence." }
    $runDirectory = New-PerformanceRunDirectory -OutputPath $outputPath
    $runId = Split-Path -Leaf $runDirectory
    $runnerOutput = Join-Path $runDirectory "runner-report.json"
    $runnerProgress = Join-Path $runDirectory "runner-progress.json"
    $runnerTerminal = Join-Path $runDirectory "runner-terminal.json"
    $monitorProgress = Join-Path $runDirectory "monitor-progress.json"
    $terminalPath = Join-Path $runDirectory "terminal.json"
    $stdoutPath = Join-Path $runDirectory "runner.stdout.log"
    $stderrPath = Join-Path $runDirectory "runner.stderr.log"
    Write-RunTerminal -Path $terminalPath -Status "running" -Phase "setup" -RunId $runId -ProfileId $ProfileId -BuildIdentity $BuildIdentity -Mode $Mode -CompletedCases 0 -TotalCases $plan.Count -RunnerTerminalPath $runnerTerminal
    $process = $null
    $samples = [System.Collections.Generic.List[object]]::new()
    $phase = "setup"
    $completedCases = 0
    $runnerStatus = $null
    $runnerExitCode = -1
    $monitorFailure = $false
    $forcedTermination = $false
    $interrupted = $false
    $terminalError = $null
    $finalStatus = "model_error"
    try {
        if (-not $SkipBuild -and [string]::IsNullOrWhiteSpace($ExecutableOverride)) {
            Write-Host "Building gpt-oss-performance..."
            & cargo build --locked --features cuda -p candle-examples --example gpt-oss-performance
            if ($LASTEXITCODE -ne 0) { throw "gpt-oss-performance build failed with exit code $LASTEXITCODE." }
        }
        $executable = if ([string]::IsNullOrWhiteSpace($ExecutableOverride)) {
            Join-Path ((Get-Location).Path) "target\debug\examples\gpt-oss-performance.exe"
        } else { (Resolve-Path -LiteralPath $ExecutableOverride).Path }
        if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { throw "Performance executable was not found: $executable" }
        $binaryHash = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
        $baselineSample = Get-SystemSample -DeviceIndex $DeviceIndex -Phase "pre_launch"
        $arguments = @(
            "--model", $modelPath, "--tokenizer", $tokenizerPath, "--output", $runnerOutput,
            "--lengths", $Lengths, "--target-context-tokens", $TargetContextTokens.ToString(),
            "--decode-tokens", $DecodeTokens.ToString(), "--mode", $Mode, "--prompt", $Prompt,
            "--device-index", $DeviceIndex.ToString(), "--max-weight-bytes", $MaxWeightBytes.ToString(),
            "--max-cache-bytes", $MaxCacheBytes.ToString(), "--max-total-device-bytes", $MaxTotalDeviceBytes.ToString(),
            "--overall-deadline-ms", $OverallDeadlineMs.ToString(), "--post-unload-hold-ms", $PostUnloadHoldMs.ToString(),
            "--profile-id", $ProfileId, "--build-identity", $BuildIdentity
        )
        $argumentString = ($arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join " "
        $process = Start-Process -FilePath $executable -ArgumentList $argumentString -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $started = [Diagnostics.Stopwatch]::StartNew()
        while (-not $process.HasExited) {
            $process.Refresh()
            if ($started.ElapsedMilliseconds -gt ([double]$OverallDeadlineMs + $GracePeriodMs)) {
                $forcedTermination = $true
                throw "overall deadline grace period expired before runner exit"
            }
            $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue } else { "" }
            if ($null -eq $stdout) { $stdout = "" }
            $phaseMatches = [regex]::Matches([string]$stdout, "PERF_PHASE (?<phase>[^\r\n ]+)")
            if ($phaseMatches.Count -gt 0) { $phase = $phaseMatches[$phaseMatches.Count - 1].Groups["phase"].Value }
            if (Test-Path -LiteralPath $runnerProgress -PathType Leaf) {
                try { $progressValue = Get-Content -LiteralPath $runnerProgress -Raw | ConvertFrom-Json; $completedCases = [int]$progressValue.completed_cases } catch { }
            }
            try {
                $sample = Get-SystemSample -DeviceIndex $DeviceIndex -ProcessId $process.Id -Phase $phase
                $samples.Add($sample)
                Write-MonitorProgress -Path $monitorProgress -RunId $runId -ProfileId $ProfileId -BuildIdentity $BuildIdentity -Mode $Mode -Samples @($samples) -Phase $phase -CompletedCases $completedCases -TotalCases $plan.Count
            } catch {
                $monitorFailure = $true
                throw "monitor failure during phase '$phase': $($_.Exception.Message)"
            }
            Start-Sleep -Milliseconds $SampleIntervalMs
        }
        $process.Refresh()
        $runnerExitCode = $process.ExitCode
        $afterProcessSample = Get-SystemSample -DeviceIndex $DeviceIndex -Phase "after_process_exit"
        $runnerTerminalValue = if (Test-Path -LiteralPath $runnerTerminal -PathType Leaf) { Get-Content -LiteralPath $runnerTerminal -Raw | ConvertFrom-Json } else { $null }
        $runnerStatus = if ($null -eq $runnerTerminalValue) { "model_error" } else { [string]$runnerTerminalValue.status }
        if ($runnerExitCode -ne 0) {
            $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { "" }
            throw "performance runner exited with code $runnerExitCode; status=$runnerStatus; stderr=$stderr"
        }
        if ($runnerStatus -ne "success") { throw "performance runner did not complete successfully: $runnerStatus" }
        if (-not (Test-Path -LiteralPath $runnerOutput -PathType Leaf)) { throw "runner report was not written: $runnerOutput" }
        $report = Get-Content -LiteralPath $runnerOutput -Raw | ConvertFrom-Json
        if ([string]$report.status -ne "passed") { throw "runner report does not claim a passed qualification" }
        $completedCases = @($report.cases).Count
        $postUnloadSamples = @($samples | Where-Object { $_.phase -eq "post_unload" })
        $privateSamples = @($samples | Where-Object { $null -ne $_.process_private_bytes })
        $workingSetSamples = @($samples | Where-Object { $null -ne $_.process_working_set_bytes })
        $peakPrivate = if ($privateSamples.Count -eq 0) { $null } else { [UInt64](($privateSamples | Measure-Object -Property process_private_bytes -Maximum).Maximum) }
        $peakWorkingSet = if ($workingSetSamples.Count -eq 0) { $null } else { [UInt64](($workingSetSamples | Measure-Object -Property process_working_set_bytes -Maximum).Maximum) }
        $allSamples = @($baselineSample) + @($samples) + @($afterProcessSample)
        $peakHostUsed = [UInt64](($allSamples | Measure-Object -Property host_ram_used_bytes -Maximum).Maximum)
        $peakGpuUsed = [UInt64](($allSamples | Measure-Object -Property gpu_used_bytes -Maximum).Maximum)
        $caseObjects = @($report.cases)
        $maxCache = if ($caseObjects.Count -eq 0) { [UInt64]0 } else { [UInt64](($caseObjects | Measure-Object -Property cache_capacity_bytes_after_generation -Maximum).Maximum) }
        $baselineGpu = [UInt64]$baselineSample.gpu_used_bytes
        $gpuWorkspaceOverlap = if ($peakGpuUsed -gt ($baselineGpu + [UInt64]$report.static_resident_bytes + $maxCache)) { $peakGpuUsed - $baselineGpu - [UInt64]$report.static_resident_bytes - $maxCache } else { [UInt64]0 }
        $baselinePrivate = if ($null -eq $baselineSample.process_private_bytes) { [UInt64]0 } else { [UInt64]$baselineSample.process_private_bytes }
        $hostTemporaryOverlap = if ($null -ne $peakPrivate -and $peakPrivate -gt ($baselinePrivate + [UInt64]$report.model_bytes)) { $peakPrivate - $baselinePrivate - [UInt64]$report.model_bytes } else { [UInt64]0 }
        $postUnloadSample = if ($postUnloadSamples.Count -gt 0) { $postUnloadSamples[$postUnloadSamples.Count - 1] } else { $null }
        $monitor = [ordered]@{
            schema = "candle.gpt_oss_performance_monitor.v2"
            sample_interval_ms = $SampleIntervalMs
            baseline = $baselineSample
            peak = [ordered]@{ host_ram_used_bytes = $peakHostUsed; process_private_bytes = $peakPrivate; process_working_set_bytes = $peakWorkingSet; gpu_used_bytes = $peakGpuUsed }
            configured_vs_observed = [ordered]@{ configured_context_length = $report.configured_context_length; target_context_tokens = $report.target_context_tokens; maximum_actual_prompt_tokens = [UInt64](($caseObjects | Measure-Object -Property actual_prompt_tokens -Maximum).Maximum); configured_cache_bytes = $report.configured_cache_bytes; target_cache_bytes = $report.target_cache_bytes; maximum_allocated_cache_bytes = $maxCache }
            attribution = [ordered]@{ logical_static_weight_bytes = [UInt64]$report.static_resident_bytes; logical_packed_weight_bytes = [UInt64]$report.packed_resident_bytes; logical_target_cache_bytes = [UInt64]$report.target_cache_bytes; sampled_peak_gpu_used_bytes = $peakGpuUsed; sampled_peak_process_private_bytes = $peakPrivate; workspace_temporary_overlap_gpu_bytes = $gpuWorkspaceOverlap; workspace_temporary_overlap_host_bytes = $hostTemporaryOverlap; workspace_definition = "upper-bound residual from sampled peaks; not a direct allocator census" }
            post_unload_recovery = [ordered]@{ marker_seen = ($postUnloadSamples.Count -gt 0); sample = $postUnloadSample; after_process_exit = $afterProcessSample; gpu_delta_from_baseline_bytes = if ($null -eq $postUnloadSample) { $null } else { [Int64]$postUnloadSample.gpu_used_bytes - [Int64]$baselineGpu }; host_ram_delta_from_baseline_bytes = if ($null -eq $postUnloadSample) { $null } else { [Int64]$postUnloadSample.host_ram_used_bytes - [Int64]$baselineSample.host_ram_used_bytes } }
            samples = @($samples)
        }
        if ($postUnloadSamples.Count -eq 0) { throw "runner completed without post_unload recovery samples" }
        $final = [ordered]@{}
        foreach ($property in $report.PSObject.Properties) { $final[$property.Name] = $property.Value }
        $final["qualification"] = [ordered]@{ profile_id = $ProfileId; build_identity = $BuildIdentity; executable_sha256 = $binaryHash; cargo_command = "cargo build --locked --features cuda -p candle-examples --example gpt-oss-performance"; requested_lengths = $plan; actual_prompt_tokens = @($caseObjects | ForEach-Object { $_.actual_prompt_tokens }); actual_generated_tokens = @($caseObjects | ForEach-Object { $_.actual_generated_tokens }); run_directory = $runDirectory }
        $final["monitor"] = $monitor
        Write-JsonEvidence -Path $outputPath -Value $final
        $finalStatus = "success"
        Write-Host "gpt-oss-performance: passed; report=$outputPath; run_directory=$runDirectory"
    } catch {
        $terminalError = $_.Exception.Message
        if ($terminalError -match "PipelineStopped|stopping") { $interrupted = $true }
    } finally {
        if ($null -ne $process) {
            try {
                $process.Refresh()
                if (-not $process.HasExited) {
                    $forcedTermination = $true
                    $process.Kill()
                    $process.WaitForExit()
                }
            } catch { }
        }
        if ($finalStatus -ne "success") {
            $finalStatus = Get-TerminalStatus -RunnerStatus $runnerStatus -MonitorFailure:$monitorFailure -ForcedTermination:$forcedTermination -Interrupted:$interrupted
        }
        Write-RunTerminal -Path $terminalPath -Status $finalStatus -Phase $phase -RunId $runId -ProfileId $ProfileId -BuildIdentity $BuildIdentity -Mode $Mode -CompletedCases $completedCases -TotalCases $plan.Count -ErrorMessage $terminalError -RunnerTerminalPath $runnerTerminal -ExitCode $runnerExitCode
        if ($samples.Count -gt 0 -and -not (Test-Path -LiteralPath $monitorProgress -PathType Leaf)) {
            try { Write-MonitorProgress -Path $monitorProgress -RunId $runId -ProfileId $ProfileId -BuildIdentity $BuildIdentity -Mode $Mode -Samples @($samples) -Phase $phase -CompletedCases $completedCases -TotalCases $plan.Count } catch { }
        }
    }
    if ($finalStatus -ne "success") { throw "performance qualification ended with status $finalStatus; evidence=$runDirectory; error=$terminalError" }
}

if (-not $AsLibrary) {
    try {
        $bound = @{
            Model = $Model; Tokenizer = $Tokenizer; Output = $Output; Lengths = $Lengths
            TargetContextTokens = $TargetContextTokens; DecodeTokens = $DecodeTokens; Mode = $Mode
            Prompt = $Prompt; DeviceIndex = $DeviceIndex; MaxWeightBytes = $MaxWeightBytes
            MaxCacheBytes = $MaxCacheBytes; MaxTotalDeviceBytes = $MaxTotalDeviceBytes
            OverallDeadlineMs = $OverallDeadlineMs; GracePeriodMs = $GracePeriodMs
            PostUnloadHoldMs = $PostUnloadHoldMs; SampleIntervalMs = $SampleIntervalMs
            ProfileId = $ProfileId; BuildIdentity = $BuildIdentity
        }
        if (-not [string]::IsNullOrWhiteSpace($ExecutableOverride)) { $bound.ExecutableOverride = $ExecutableOverride }
        if ($SkipBuild) { $bound.SkipBuild = $true }
        Invoke-GptOssPerformance @bound
    } catch {
        Write-Error $_
        exit 1
    }
}
