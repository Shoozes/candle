[CmdletBinding()]
param(
    [string]$Packet,
    [string]$OutputRoot,
    [switch]$AsLibrary
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-SessionProfilePlan {
    [pscustomobject]@{
        prompt_tokens = 8; generated_tokens = 16; forward_calls = 16
        token_steps = 23; loads = 1; requests = 1; warmups = 0; retries = 0
        max_job_memory_bytes = [UInt64]34359738368
        max_gpu_bytes = [UInt64]20000000000; deadline_seconds = 1800
    }
}

function Assert-SessionProfileIdentity {
    param([Parameter(Mandatory)]$Identity)
    if ([string]$Identity.sha256 -notmatch '^[0-9a-fA-F]{64}$' -or [UInt64]$Identity.size_bytes -eq 0) {
        throw 'Profile file requires explicit size and SHA-256'
    }
    $path = (Resolve-Path -LiteralPath ([string]$Identity.path) -ErrorAction Stop).Path
    $file = [IO.File]::OpenRead($path)
    try {
        if ([UInt64]$file.Length -ne [UInt64]$Identity.size_bytes) { throw "Profile file size mismatch: $path" }
        $hash = [Security.Cryptography.SHA256]::HashData($file)
        $actual = [Convert]::ToHexString($hash).ToLowerInvariant()
        if ($actual -ne ([string]$Identity.sha256).ToLowerInvariant()) { throw "Profile file SHA-256 mismatch: $path" }
    } finally { $file.Dispose() }
    $path
}

function Assert-SessionProfileInput {
    param([Parameter(Mandatory)]$InputValue, [Parameter(Mandatory)][string]$Source, [Parameter(Mandatory)][string]$Backend)
    $plan = Get-SessionProfilePlan
    if ($InputValue.schema -ne 'candle.gpt-oss.session-profile-input.v1' -or
        $InputValue.source_commit -ne $Source -or $InputValue.backend -ne $Backend) {
        throw 'Profile input source/backend/version mismatch'
    }
    if (@($InputValue.prompt_token_ids).Count -ne $plan.prompt_tokens -or
        [int]$InputValue.generated_tokens -ne $plan.generated_tokens) {
        throw 'Profile workload differs from frozen eight/16 allowance'
    }
    $referencePath=Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json'
    $frozen=(Get-Content -LiteralPath $referencePath -Raw | ConvertFrom-Json).reference_prefix_token_ids
    for($index=0; $index -lt $plan.prompt_tokens; $index++) {
        if ($InputValue.prompt_token_ids[$index] -ne $frozen[$index]) { throw 'Profile prompt differs from the pinned reference prefix' }
    }
    if ([string]$InputValue.model.sha256 -notmatch '^[0-9a-fA-F]{64}$' -or
        [UInt64]$InputValue.model.size_bytes -eq 0) { throw 'Profile model identity is invalid' }
    if ([int]$InputValue.device_index -lt 0 -or ($Backend -eq 'cpu' -and [int]$InputValue.device_index -ne 0)) {
        throw 'Profile device index is invalid'
    }
}

function Assert-SessionProfileCompletion {
    param([Parameter(Mandatory)]$Progress, [Parameter(Mandatory)]$Job, [Parameter(Mandatory)][string]$InputHash,
        [Parameter(Mandatory)]$BinaryIdentity, [Parameter(Mandatory)]$ModelIdentity, [Parameter(Mandatory)][string]$Backend)
    $plan = Get-SessionProfilePlan
    if ($Progress.schema -ne 'candle.gpt-oss.session-profile-progress.v1' -or
        $Progress.status -ne 'completed' -or $Progress.input_sha256 -ne $InputHash -or
        [int]$Progress.loads_attempted -ne $plan.loads -or
        [int]$Progress.requests_attempted -ne $plan.requests -or
        [int]$Progress.requests_completed -ne $plan.requests -or
        [int]$Progress.attempted_forward_calls -ne $plan.forward_calls -or
        [int]$Progress.completed_forward_calls -ne $plan.forward_calls -or
        [int]$Progress.attempted_token_steps_upper_bound -ne $plan.token_steps -or
        [int]$Progress.completed_token_steps -ne $plan.token_steps -or
        @($Progress.generated_token_ids).Count -ne $plan.generated_tokens -or
        @($Progress.calls).Count -ne $plan.forward_calls -or
        $null -eq $Progress.registry_entries_after_release -or
        [int]$Progress.registry_entries_after_release -ne 0) { throw 'Profile work/release receipt is incomplete' }
    if ($Job.contract -ne 'candle-lfm2-vl-bounded-oracle-v1' -or
        $Job.child_exit_code -ne 0 -or -not $Job.pid_absent_after_cleanup -or
        -not $Job.assigned_before_resume -or
        -not $Job.job_assigned -or -not $Job.started_suspended -or
        $Job.executable_sha256 -ne $BinaryIdentity.sha256 -or
        [UInt64]$Job.executable_bytes -ne [UInt64]$BinaryIdentity.size_bytes -or
        [int]$Job.pid -ne [int]$Progress.pid -or
        [UInt64]$Job.peak_private_memory_bytes -eq 0 -or [UInt64]$Job.peak_private_memory_bytes -gt $plan.max_job_memory_bytes -or
        [UInt64]$Job.peak_job_memory_bytes -eq 0 -or [UInt64]$Job.peak_job_memory_bytes -gt $plan.max_job_memory_bytes -or
        [UInt64]$Job.max_job_memory_bytes -ne $plan.max_job_memory_bytes -or
        [int]$Job.timeout_seconds -ne $plan.deadline_seconds) { throw 'Profile bounded Job receipt is incomplete' }
    $consumed = @($Progress.consumed_files)
    if ($consumed.Count -ne 1 -or $consumed[0].sha256 -ne $ModelIdentity.sha256 -or
        [UInt64]$consumed[0].size_bytes -ne [UInt64]$ModelIdentity.size_bytes) { throw 'Profile consumed model identity mismatch' }
    $position=0
    for ($index=0; $index -lt $plan.forward_calls; $index++) {
        $call=$Progress.calls[$index]
        $incoming=if($index -eq 0){$plan.prompt_tokens}else{1}
        $position+=$incoming
        if ($call.input_tokens -ne $incoming -or $call.committed_sequence_tokens -ne $position) { throw 'Profile per-call cache/work count mismatch' }
        foreach($name in @('forward_ms','caller_readback_ms','argmax_ms')) {
            if ($null -eq $call.$name -or -not [double]::IsFinite([double]$call.$name) -or [double]$call.$name -lt 0) { throw 'Profile returned invalid timing' }
        }
        if (($Backend -eq 'cpu' -and $null -ne $call.cuda_final_validation_ms) -or
            ($Backend -eq 'cuda' -and ($null -eq $call.cuda_final_validation_ms -or
            -not [double]::IsFinite([double]$call.cuda_final_validation_ms) -or
            [double]$call.cuda_final_validation_ms -lt 0 -or
            [double]$call.cuda_final_validation_ms -gt [double]$call.forward_ms))) { throw 'Profile validation timing/backend mismatch' }
    }
}

function Write-SessionProfileJson {
    param([string]$Path, $Value)
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 14), [Text.UTF8Encoding]::new($false))
}

function Get-SessionProfileGpuSample {
    param([int]$DeviceIndex)
    $lines = @(& nvidia-smi.exe --id $DeviceIndex --query-gpu=memory.used --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -ne 1) { throw 'GPU monitor returned no unique sample' }
    [UInt64]$mib = 0
    if (-not [UInt64]::TryParse(([string]$lines[0]).Trim(), [ref]$mib) -or
        $mib -gt ([UInt64]::MaxValue / 1048576)) { throw 'GPU monitor returned invalid memory' }
    [UInt64]$bytes = $mib * [UInt64]1048576
    $plan = Get-SessionProfilePlan
    if ($bytes -gt $plan.max_gpu_bytes) { throw "GPU usage $bytes exceeded $($plan.max_gpu_bytes) bytes" }
    [pscustomobject]@{ observed_utc=[DateTimeOffset]::UtcNow.ToString('o'); used_bytes=$bytes }
}

function Quote-SessionProfileLiteral {
    param([string]$Value)
    "'" + $Value.Replace("'", "''") + "'"
}

function Stop-SessionProfileController {
    param([Parameter(Mandatory)][Diagnostics.Process]$Controller)
    $Controller.Refresh()
    if (-not $Controller.HasExited) {
        $Controller.Kill()
        if (-not $Controller.WaitForExit(30000)) { throw 'Profile controller remained after termination' }
    }
}

function Invoke-SessionProfileBatch {
    param($Lanes, [string]$Source, [string]$Output, [string]$Helper, [string]$RepoRoot)
    if ($Lanes.Count -ne 2 -or $Lanes[0].backend -ne 'cpu' -or $Lanes[1].backend -ne 'cuda') { throw 'Profile packet requires CPU then CUDA exactly once' }
    foreach($lane in $Lanes) { if($lane.build_profile -ne 'release') { throw 'Profile packet requires optimized release binaries' } }
    foreach ($lane in $Lanes) { Invoke-SessionProfileLane $lane $Source (Join-Path $Output $lane.backend) $Helper $RepoRoot }
}

function Invoke-SessionProfileLane {
    param($Lane, [string]$Source, [string]$Directory, [string]$Helper, [string]$RepoRoot)
    $plan = Get-SessionProfilePlan
    $inputPath = Assert-SessionProfileIdentity $Lane.input
    $binaryPath = Assert-SessionProfileIdentity $Lane.binary
    $inputValue = Get-Content -LiteralPath $inputPath -Raw | ConvertFrom-Json
    Assert-SessionProfileInput $inputValue $Source ([string]$Lane.backend)
    $gpuSamples = [Collections.Generic.List[object]]::new()
    if ($Lane.backend -eq 'cuda') { $gpuSamples.Add((Get-SessionProfileGpuSample ([int]$inputValue.device_index))) }
    New-Item -ItemType Directory -Path $Directory -ErrorAction Stop | Out-Null
    $jobPath = Join-Path $Directory 'job.json'
    $progressPath = Join-Path $Directory 'progress.json'
    $command = '& ' + (Quote-SessionProfileLiteral $Helper) + ' -FilePath ' + (Quote-SessionProfileLiteral $binaryPath) +
        ' -ArgumentList @(' + ((@('--input', $inputPath, '--output', $progressPath) | ForEach-Object { Quote-SessionProfileLiteral $_ }) -join ',') + ')' +
        ' -TimeoutSeconds 1800 -MaxJobMemoryBytes 34359738368 -PollMilliseconds 250 -ConcurrencyScope Executable' +
        ' -WorkingDirectory ' + (Quote-SessionProfileLiteral $RepoRoot) +
        ' -EvidencePath ' + (Quote-SessionProfileLiteral $jobPath) +
        ' -LogPath ' + (Quote-SessionProfileLiteral (Join-Path $Directory 'runner.log'))
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $controller = $null
    $runnerIdentity = $null
    $status = 'failed'
    $failure = $null
    try {
        $controller = Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $Directory 'controller.stdout.log') -RedirectStandardError (Join-Path $Directory 'controller.stderr.log')
        $owner = [ordered]@{ owner='Candle GPT-OSS session profile'; source_commit=$Source; pid=$controller.Id; creation_utc=$controller.StartTime.ToUniversalTime().ToString('o'); executable=$controller.Path; intended_lifetime='one 1800-second bounded Job plus cleanup'; endpoints=@(); binary=$Lane.binary; input=$Lane.input; shutdown='Stop only this creation-time/executable-verified controller; closing its native Job kills the owned runner'; output_root=$Directory }
        Write-SessionProfileJson (Join-Path $Directory 'owner.json') $owner
        $timer = [Diagnostics.Stopwatch]::StartNew()
        while (-not $controller.HasExited) {
            if ($timer.Elapsed.TotalSeconds -gt ($plan.deadline_seconds + 60)) { throw 'Profile controller exceeded Job deadline plus cleanup' }
            if ($Lane.backend -eq 'cuda') { $gpuSamples.Add((Get-SessionProfileGpuSample ([int]$inputValue.device_index))) }
            if ($null -eq $runnerIdentity -and (Test-Path -LiteralPath $progressPath -PathType Leaf)) {
                # Partial progress may be in the middle of a bounded rewrite.
                $progress = $null
                try { $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json } catch { }
                if ($null -ne $progress) {
                    $native = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$progress.pid)"
                    if ($null -ne $native -and $native.ExecutablePath -eq $binaryPath) {
                        $runnerIdentity = [ordered]@{ pid=$native.ProcessId; creation_utc=$native.CreationDate.ToUniversalTime().ToString('o'); executable=$native.ExecutablePath }
                        $owner.runner = $runnerIdentity
                        Write-SessionProfileJson (Join-Path $Directory 'owner.json') $owner
                    }
                }
            }
            Start-Sleep -Milliseconds 1000
            $controller.Refresh()
        }
        if ($controller.ExitCode -ne 0) { throw "Bounded profile controller exited $($controller.ExitCode)" }
        $job = Get-Content -LiteralPath $jobPath -Raw | ConvertFrom-Json
        $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
        if ($progress.source_commit -ne $Source) { throw 'Profile runner source identity mismatch' }
        Assert-SessionProfileCompletion $progress $job ([string]$Lane.input.sha256) $Lane.binary $inputValue.model ([string]$Lane.backend)
        if ($Lane.backend -eq 'cuda') { $gpuSamples.Add((Get-SessionProfileGpuSample ([int]$inputValue.device_index))) }
        $status = 'completed'
    } catch { $failure = $_.Exception.Message } finally {
        if ($null -ne $controller) {
            try {
                # This handle identifies our controller, whose kill-on-close Job
                # contains the runner. Never terminate by name or a guessed PID.
                Stop-SessionProfileController $controller
            } catch { $failure=$_.Exception.Message; $status='failed' }
            $controller.Dispose()
        }
        if ($null -ne $runnerIdentity) {
            $live = Get-CimInstance Win32_Process -Filter "ProcessId=$($runnerIdentity.pid)"
            if ($null -ne $live -and $live.CreationDate.ToUniversalTime().ToString('o') -eq $runnerIdentity.creation_utc -and $live.ExecutablePath -eq $runnerIdentity.executable) {
                $failure = 'Owned profile runner remains after controller cleanup'; $status='failed'
            }
        }
        Write-SessionProfileJson (Join-Path $Directory 'terminal.json') ([ordered]@{status=$status;error=$failure;backend=$Lane.backend;source_commit=$Source;job=$jobPath;progress=$progressPath;gpu_samples=@($gpuSamples);runner_identity=$runnerIdentity;qualification='raw-token baseline only; no provider, BF16, 32k or speedup claim'})
    }
    if ($status -ne 'completed') { throw "Profile lane failed: $failure" }
}

if (-not $AsLibrary) {
    if ($PSVersionTable.PSVersion.Major -lt 7 -or -not $IsWindows) { throw 'Session profiling requires native Windows PowerShell 7' }
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
    Set-Location -LiteralPath $repoRoot
    if ([string]::IsNullOrWhiteSpace($Packet) -or [string]::IsNullOrWhiteSpace($OutputRoot)) { throw 'Packet and OutputRoot are required' }
    if ((Get-Item -LiteralPath $Packet).Length -gt 65536) { throw 'Profile packet exceeds 64 KiB' }
    $value = Get-Content -LiteralPath $Packet -Raw | ConvertFrom-Json
    $source = (git rev-parse HEAD).Trim()
    $tree = (git show -s --format=%T HEAD).Trim()
    if ($value.schema -ne 'candle.gpt-oss.session-profile-packet.v1' -or $value.source_commit -ne $source -or $value.source_tree -ne $tree -or @(git status --porcelain).Count -ne 0) { throw 'Profile packet requires its exact clean source checkpoint' }
    $helper = Assert-SessionProfileIdentity $value.bounded_helper
    if ($helper -ne (Join-Path $repoRoot 'scripts\lfm2-vl\run-bounded-oracle.ps1')) { throw 'Profile packet must use the Candle bounded helper' }
    $lanes = @($value.lanes)
    if ($lanes.Count -ne 2 -or $lanes[0].backend -ne 'cpu' -or $lanes[1].backend -ne 'cuda') { throw 'Profile packet requires CPU then CUDA exactly once' }
    foreach($lane in $lanes) { if($lane.build_profile -ne 'release' -or [UInt64]$lane.input.size_bytes -gt 16384) { throw 'Profile packet requires release binaries and bounded input files' } }
    $output = [IO.Path]::GetFullPath($OutputRoot)
    $allowed = (Join-Path $repoRoot 'artifacts') + [IO.Path]::DirectorySeparatorChar
    if (-not $output.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $output)) { throw 'Profile output must be a fresh Candle artifacts subdirectory' }
    New-Item -ItemType Directory -Path $output | Out-Null
    Invoke-SessionProfileBatch $lanes $source $output $helper $repoRoot
}
