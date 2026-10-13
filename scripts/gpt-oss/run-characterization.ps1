[CmdletBinding()]
param([string]$PacketPath,[string]$RunRoot,[switch]$AsLibrary)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$characterizationLibraryOnly=[bool]$AsLibrary
. (Join-Path $PSScriptRoot 'run-session-profile.ps1') -AsLibrary

function Get-CharacterizationPlan {
    [pscustomobject]@{loads=1;requests=3;forwards=22;token_steps=37;generated_tokens=16;rows=6;resets=2;
        warmups=0;retries=0;job_bytes=[UInt64]34359738368;gpu_bytes=[UInt64]20000000000;deadline_seconds=1800;
        observer_timeout_ms=5000;observer_job_bytes=268435456;timing_rounding_ms=0.001}
}
function Assert-CharacterizationPacketHeader($Value,[string]$Source,[string]$Tree) {
    if($Value.schema -ne 'candle.gpt-oss.characterization-packet.v1' -or $Value.draft -isnot [bool] -or $Value.draft -or
        $Value.source_commit -ne $Source -or $Value.source_tree -ne $Tree){throw 'Characterization needs its exact final source'}
    $counts=@{loads=2;segments=6;public_forwards=44;processed_token_steps=74;sampled_output_tokens=32;warmups=0;retries=0}
    foreach($name in $counts.Keys){Assert-SessionProfileUnsignedInteger $Value.allowance_proposal.$name "characterization allowance $name";if($Value.allowance_proposal.$name -ne $counts[$name]){throw 'Characterization allowance mismatch'}}
    $plan=Get-CharacterizationPlan
    foreach($property in $plan.PSObject.Properties){
        $declared=$Value.allowance_proposal.per_lane.($property.Name)
        if(Test-SessionProfileUnsignedInteger $property.Value){Assert-SessionProfileUnsignedInteger $declared "per-lane $($property.Name)"}
        elseif(-not (Test-SessionProfileTiming $declared)){throw 'Characterization per-lane timing is invalid'}
        if($declared -ne $property.Value){throw 'Characterization per-lane limits mismatch'}
    }
    foreach($entry in @{preparation_startup_probes=2;preparation_startup_job_bytes=2147483648;preparation_startup_deadline_seconds=15}.GetEnumerator()){
        Assert-SessionProfileUnsignedInteger $Value.allowance_proposal.($entry.Key) "preparation $($entry.Key)"
        if($Value.allowance_proposal.($entry.Key) -ne $entry.Value){throw 'Preparation qualification count/bound mismatch'}
    }
}
function Assert-CharacterizationOracleRecovery($Identity,$Oracle) {
    Assert-SessionProfileIdentity $Identity|Out-Null
    if($Identity.size_bytes -gt 32768){throw 'Oracle recovery record exceeds bound'}
    $recovery=Get-Content -LiteralPath $Identity.path -Raw|ConvertFrom-Json
    if($recovery.schema -ne 'candle.gpt-oss.reference-recovery.v1' -or $recovery.status -ne 'recovered-exact-artifact' -or
        $recovery.producer_source -ne '1537a0a8b2f8711d840878b0a0677ab2213c882c'){throw 'Oracle producer/recovery mismatch'}
    foreach($name in @('path','size_bytes','sha256')){if($recovery.reference.$name -ne $Oracle.$name){throw 'Oracle recovery identity mismatch'}}
    foreach($field in @('original_fixture','producer_packet','producer_job','raw_terminal','codec_validation','census_after')){Assert-SessionProfileIdentity $recovery.$field|Out-Null}
    if($recovery.release.native_job_natural_exit -isnot [bool] -or -not $recovery.release.native_job_natural_exit -or
        $recovery.release.worker_and_child_absent -isnot [bool] -or -not $recovery.release.worker_and_child_absent){throw 'Oracle recovery release incomplete'}
}
function Assert-CharacterizationInput($Value,[string]$Source,[string]$Backend) {
    $f=Get-Content (Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw|ConvertFrom-Json
    if($Value.schema -ne 'candle.gpt-oss.characterization-input.v1' -or $Value.source_commit -ne $Source -or $Value.backend -ne $Backend){throw 'Characterization input source/version/device mismatch'}
    Assert-SessionProfileUnsignedInteger $Value.device_index 'characterization device' 0 ([int]::MaxValue)
    Assert-SessionProfileUnsignedInteger $Value.generated_tokens 'characterization output count'
    Assert-SessionProfileUnsignedInteger $Value.model.size_bytes 'characterization model bytes' 1
    Assert-SessionProfileUnsignedInteger $Value.reference.size_bytes 'characterization reference bytes' 1
    if($Value.generated_tokens -ne 16 -or @($Value.prompt_token_ids).Count -ne 8 -or
        (@($Value.segments)-join ',') -ne 'cold,numerical,replay' -or
        $Value.reference.sha256 -ne $f.reference_file_sha256 -or $Value.reference.size_bytes -ne $f.reference_file_bytes -or
        $Value.model.sha256 -ne 'aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778' -or
        $Value.model.size_bytes -ne 13792638656 -or ($Backend -eq 'cpu' -and $Value.device_index -ne 0)){throw 'Characterization frozen input mismatch'}
    for($i=0;$i -lt 8;$i++) {
        Assert-SessionProfileUnsignedInteger $Value.prompt_token_ids[$i] 'characterization prompt ID' 0 ([UInt32]::MaxValue)
        if($Value.prompt_token_ids[$i] -ne $f.reference_prefix_token_ids[$i]){throw 'Characterization prefix mismatch'}
    }
}
function Assert-CharacterizationCompletion($Progress,$Job,$Lane,$InputValue) {
    $plan=Get-CharacterizationPlan
    foreach($name in @('pid','loads_attempted','requests_attempted','requests_completed','attempted_forward_calls',
        'completed_forward_calls','attempted_token_steps_upper_bound','completed_token_steps','registry_entries_after_release','static_weight_bytes')) {
        Assert-SessionProfileUnsignedInteger $Progress.$name "characterization $name"
    }
    if($Progress.schema -ne 'candle.gpt-oss.characterization-progress.v1' -or $Progress.status -ne 'completed' -or
        $null -ne $Progress.error -or $Progress.source_commit -ne $InputValue.source_commit -or
        $Progress.input_sha256 -ne $Lane.input.sha256 -or $Progress.loads_attempted -ne 1 -or
        $Progress.requests_attempted -ne 3 -or $Progress.requests_completed -ne 3 -or $Progress.attempted_forward_calls -ne 22 -or
        $Progress.completed_forward_calls -ne 22 -or $Progress.attempted_token_steps_upper_bound -ne 37 -or
        $Progress.completed_token_steps -ne 37 -or $Progress.registry_entries_after_release -ne 0 -or
        @($Progress.calls).Count -ne 22 -or @($Progress.generated_token_ids).Count -ne 16 -or @($Progress.rows).Count -ne 6 -or
        @($Progress.independent_comparisons).Count -ne 3 -or @($Progress.replay_max_abs_errors).Count -ne 3 -or @($Progress.reset_ms).Count -ne 2){throw 'Characterization work/release incomplete'}
    $fixture=Get-Content (Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw|ConvertFrom-Json
    foreach($id in $Progress.generated_token_ids){Assert-SessionProfileUnsignedInteger $id 'characterization generated ID' 0 ($fixture.reference_vocab_size-1)}
    for($i=0;$i -lt 3;$i++) {
        $report=$Progress.independent_comparisons[$i]
        if($report.stage -ne ('context.'+(5+$i)) -or @($report.top_reference).Count -ne 8 -or @($report.top_candle).Count -ne 8 -or
            $report.top_reference[0].token_id -ne $fixture.reference_row_top1[$i].token_id -or
            $report.top_candle[0].token_id -ne $report.top_reference[0].token_id){throw 'Independent comparison top-1/context mismatch'}
        foreach($name in @('max_abs_log_probability_error','mean_abs_log_probability_error','top_k_max_abs_log_probability_error','total_variation_probability_error')) {
            if(-not (Test-SessionProfileTiming $report.$name)){throw 'Invalid independent comparison metric'}
        }
        if($report.mean_abs_log_probability_error -gt $fixture.comparison.mean_abs_log_probability_tolerance -or
            $report.top_k_max_abs_log_probability_error -gt $fixture.comparison.top_k_log_probability_abs_tolerance -or
            $report.total_variation_probability_error -gt $fixture.comparison.total_variation_probability_tolerance){throw 'Independent comparison bounds exceeded'}
        foreach($entry in @($report.top_reference)+@($report.top_candle)) {
            Assert-SessionProfileUnsignedInteger $entry.token_id 'comparison token ID' 0 ($fixture.reference_vocab_size-1)
            if($entry.log_probability -isnot [double] -and $entry.log_probability -isnot [single] -and
                $entry.log_probability -isnot [decimal] -and $entry.log_probability -isnot [int] -and $entry.log_probability -isnot [long]){throw 'Invalid comparison log probability'}
            if(-not [double]::IsFinite($entry.log_probability)){throw 'Invalid comparison log probability'}
        }
        if(-not (Test-SessionProfileTiming $Progress.replay_max_abs_errors[$i]) -or $Progress.replay_max_abs_errors[$i] -gt 1e-5){throw 'Replay comparison bounds exceeded'}
    }
    $ceiling=if($Lane.backend -eq 'cuda'){$plan.gpu_bytes}else{$plan.job_bytes}
    Assert-SessionProfileUnsignedInteger $Progress.static_weight_bytes 'characterization static bytes' 1 $ceiling
    $consumed=@($Progress.consumed_files)
    if($consumed.Count -ne 1){throw 'Characterization consumed-file inventory mismatch'}
    Assert-SessionProfileUnsignedInteger $consumed[0].size_bytes 'characterization consumed bytes' 1
    if($consumed[0].sha256 -ne $InputValue.model.sha256 -or $consumed[0].size_bytes -ne $InputValue.model.size_bytes){throw 'Characterization consumed model mismatch'}
    foreach($name in @('pid','child_exit_code','executable_bytes','peak_private_memory_bytes','peak_job_memory_bytes','max_job_memory_bytes','timeout_seconds')) {
        Assert-SessionProfileUnsignedInteger $Job.$name "characterization Job $name"
    }
    foreach($name in @('pid_absent_after_cleanup','assigned_before_resume','job_assigned','started_suspended','resumed')) {
        if($Job.$name -isnot [bool] -or -not $Job.$name){throw 'Characterization Job ownership/release mismatch'}
    }
    if($Job.contract -ne 'candle-lfm2-vl-bounded-oracle-v1' -or $Job.child_exit_code -ne 0 -or
        $Job.termination_reason -ne 'exited' -or $null -ne $Job.error -or $Job.pid -ne $Progress.pid -or $Job.pid -le 0 -or
        $Job.executable_sha256 -ne $Lane.binary.sha256 -or $Job.executable_bytes -ne $Lane.binary.size_bytes -or
        $Job.peak_private_memory_bytes -le 0 -or $Job.peak_private_memory_bytes -gt $plan.job_bytes -or
        $Job.peak_job_memory_bytes -le 0 -or $Job.peak_job_memory_bytes -gt $plan.job_bytes -or
        $Job.max_job_memory_bytes -ne $plan.job_bytes -or $Job.timeout_seconds -ne $plan.deadline_seconds){throw 'Characterization Job receipt mismatch'}
    $aggregate=0.
    foreach($name in @('load_ms','unload_ms','cold_ms','numerical_ms','replay_ms','total_ms')) {
        if(-not (Test-SessionProfileTiming $Progress.$name)){throw 'Invalid characterization aggregate timing'}
        if($name -ne 'total_ms'){$aggregate+=[double]$Progress.$name}
    }
    foreach($time in $Progress.reset_ms){if(-not (Test-SessionProfileTiming $time)){throw 'Invalid reset timing'};$aggregate+=$time}
    if(-not [double]::IsFinite($aggregate) -or $aggregate-$Progress.total_ms -gt 8*$plan.timing_rounding_ms){throw 'Characterization aggregate sum inconsistent'}
    $sums=@{cold=0.;numerical=0.;replay=0.}
    for($i=0;$i -lt 22;$i++) {
        $call=$Progress.calls[$i];$segment=if($i -lt 16){'cold'}elseif($i -lt 19){'numerical'}else{'replay'}
        $index=if($i -lt 16){$i}else{($i-16)%3}
        $incoming=if($index -eq 0){if($segment -eq 'cold'){8}else{5}}else{1}
        $position=if($segment -eq 'cold'){8+$index}else{5+$index}
        Assert-SessionProfileUnsignedInteger $call.input_tokens 'characterization call count'
        Assert-SessionProfileUnsignedInteger $call.committed_sequence_tokens 'characterization position'
        if($call.segment -ne $segment -or $call.input_tokens -ne $incoming -or $call.committed_sequence_tokens -ne $position){throw 'Characterization segment/prefix position mismatch'}
        foreach($name in @('forward_ms','caller_readback_ms','argmax_ms','row_capture_ms','comparison_ms')) {
            if(-not (Test-SessionProfileTiming $call.$name)){throw 'Invalid characterization call timing'}
            $sums[$segment]+=$call.$name
        }
        if(($segment -eq 'cold' -and ($call.row_capture_ms -ne 0 -or $call.comparison_ms -ne 0)) -or
            ($segment -ne 'cold' -and $call.argmax_ms -ne 0)){throw 'Characterization measurements overlap segments'}
        if(($Lane.backend -eq 'cpu' -and $null -ne $call.cuda_final_validation_ms) -or
            ($Lane.backend -eq 'cuda' -and (-not (Test-SessionProfileTiming $call.cuda_final_validation_ms) -or $call.cuda_final_validation_ms-$call.forward_ms -gt 0.002))){throw 'Characterization validation timing mismatch'}
    }
    foreach($segment in @('cold','numerical','replay')) {
        $fields=if($segment -eq 'cold'){80}else{15}
        if(-not [double]::IsFinite($sums[$segment]) -or $sums[$segment]-$Progress.($segment+'_ms') -gt ($fields+1)*$plan.timing_rounding_ms){throw 'Characterization segment timing inconsistent'}
    }
}
function Assert-CharacterizationRuntimeClosure($Lane) {
    $names=@($Lane.required_modules|ForEach-Object{[string]$_.name})
    $required=@('VCRUNTIME140.dll')
    if($Lane.backend -eq 'cuda'){$required+=@('nvcuda.dll','cublas64_13.dll','cublasLt64_13.dll','curand64_10.dll')}
    foreach($name in $required){if(@($names|Where-Object{$_ -ieq $name}).Count -ne 1){throw "Runtime closure must bind $name exactly once"}}
    if($names.Count -gt 32 -or @($names|Sort-Object -Unique).Count -ne $names.Count){throw 'Runtime closure is duplicate or exceeds bound'}
    foreach($module in $Lane.required_modules){if((Split-Path -Leaf $module.path) -ine $module.name){throw 'Runtime module name/path mismatch'}}
}
function Assert-CharacterizationBuild($Record,$Lane,[string]$Source,[string]$Tree,$ExpectedSourcePaths) {
    if($Record.schema -ne 'candle.gpt-oss.characterization-build.v1' -or $Record.draft -isnot [bool] -or $Record.draft -or
        $Record.source_commit -ne $Source -or $Record.source_tree -ne $Tree -or $Record.profile -ne 'release' -or
        $Record.precision -ne 'f32' -or $Record.format -ne 'gguf-mxfp4-f32' -or ($Record.exit_code -isnot [long] -and $Record.exit_code -isnot [int]) -or
        $Record.exit_code -ne 0 -or [string]::IsNullOrWhiteSpace($Record.rustc) -or [string]::IsNullOrWhiteSpace($Record.cargo)){throw 'Build source/profile/toolchain mismatch'}
    $features=if($Lane.backend -eq 'cuda'){'cuda'}else{''}
    if((@($Record.features)-join ',') -ne $features){throw 'Build features mismatch'}
    foreach($name in @('path','size_bytes','sha256')){if($Record.binary.$name -ne $Lane.binary.$name){throw 'Build binary identity mismatch'}}
    $paths=@($Record.source_files|ForEach-Object{$_.path})
    if($paths.Count -ne @($ExpectedSourcePaths).Count -or @($paths|Sort-Object -Unique).Count -ne $paths.Count -or
        @($paths|Where-Object{$_ -notin $ExpectedSourcePaths}).Count -ne 0){throw 'Build source closure mismatch'}
    foreach($file in $Record.source_files){Assert-SessionProfileIdentity $file|Out-Null}
}
function Assert-CharacterizationBuildJob($Record) {
    Assert-SessionProfileIdentity $Record.bounded_job|Out-Null
    if($Record.bounded_job.size_bytes -gt 65536){throw 'Build Job record exceeds bound'}
    $job=Get-Content -LiteralPath $Record.bounded_job.path -Raw|ConvertFrom-Json
    if($job.contract -ne 'candle-lfm2-vl-bounded-oracle-v1' -or $job.child_exit_code -ne 0 -or $job.termination_reason -ne 'exited' -or
        $null -ne $job.error -or $job.pid_absent_after_cleanup -isnot [bool] -or -not $job.pid_absent_after_cleanup -or
        $job.assigned_before_resume -isnot [bool] -or -not $job.assigned_before_resume -or $job.timeout_seconds -ne 900 -or
        $job.max_job_memory_bytes -ne 17179869184){throw 'Build Job source/release mismatch'}
}
function Assert-CharacterizationStartupJob($Identity,$Binary) {
    Assert-SessionProfileIdentity $Identity|Out-Null
    if($Identity.size_bytes -gt 65536){throw 'Startup Job record exceeds bound'}
    $job=Get-Content -LiteralPath $Identity.path -Raw|ConvertFrom-Json
    foreach($field in @('pid','executable_bytes','peak_job_memory_bytes','timeout_seconds','max_job_memory_bytes','child_exit_code')){Assert-SessionProfileUnsignedInteger $job.$field "startup $field"}
    foreach($field in @('pid_absent_after_cleanup','assigned_before_resume','job_assigned','started_suspended','resumed')){if($job.$field -isnot [bool] -or -not $job.$field){throw 'Startup Job ownership/release incomplete'}}
    if($job.contract -ne 'candle-lfm2-vl-bounded-oracle-v1' -or $job.child_exit_code -ne 0 -or $job.termination_reason -ne 'exited' -or
        $null -ne $job.error -or $job.timeout_seconds -ne 15 -or $job.max_job_memory_bytes -ne 2147483648 -or
        $job.peak_job_memory_bytes -le 0 -or $job.peak_job_memory_bytes -gt 2147483648 -or $job.pid -le 0 -or
        $job.executable_sha256 -ne $Binary.sha256 -or $job.executable_bytes -ne $Binary.size_bytes){throw 'Startup Job artifact/limits mismatch'}
}
function Get-CharacterizationSourcePaths([string]$RepoRoot) {
    @('Cargo.lock','Cargo.toml','rust-toolchain.toml','candle-examples/Cargo.toml','candle-examples/examples/gpt-oss-characterize.rs','candle-examples/examples/gpt_oss_reference/mod.rs',
        'candle-kernels/src/gpt_oss_mxfp4.cu','tests/fixtures/gpt_oss_task3_short_parity/reference.json')|ForEach-Object{[IO.Path]::GetFullPath((Join-Path $RepoRoot $_))}
    Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'candle-transformers\src\models\gpt_oss') -File|ForEach-Object{$_.FullName}
    if(Test-Path -LiteralPath (Join-Path $RepoRoot '.cargo\config.toml')){Join-Path $RepoRoot '.cargo\config.toml'}
}
function Invoke-CharacterizationBatch($Lanes,[scriptblock]$Operation) {
    if(@($Lanes).Count -ne 2 -or $Lanes[0].backend -ne 'cpu' -or $Lanes[1].backend -ne 'cuda'){throw 'Characterization requires CPU then CUDA'}
    foreach($lane in $Lanes){& $Operation $lane}
}
function Invoke-CharacterizationRuntimeEnvironment($Lane,[scriptblock]$Operation) {
    $previous=$env:PATH
    try {
        if($Lane.backend -eq 'cuda'){
            $directories=@($Lane.required_modules|ForEach-Object{Split-Path -Parent $_.path}|Sort-Object -Unique)
            $env:PATH=($directories -join ';')+';'+$previous
        }
        & $Operation $Lane
    } finally {$env:PATH=$previous}
}
function Assert-CharacterizationRuntime($Observation,$Packet,$Lane,$Progress,[string]$PacketHash) {
    if($Observation.schema -ne 'candle.gpt-oss.runtime-observation.v1' -or $Observation.status -ne 'observed' -or
        $null -ne $Observation.error -or $Observation.source_commit -ne $Packet.source_commit -or
        $Observation.packet_sha256 -ne $PacketHash -or $Observation.backend -ne $Lane.backend -or
        $null -eq $Observation.identity -or $Observation.identity.pid -ne $Progress.pid -or
        -not [string]::Equals($Observation.identity.executable,$Lane.binary.path,[StringComparison]::OrdinalIgnoreCase)) {throw 'Loaded runtime observation is incomplete or mismatched'}
    Assert-CharacterizationRuntimeClosure $Lane
    Assert-CharacterizationObservedModules $Observation.modules $Lane.required_modules
}
function Assert-CharacterizationObservedModules($Modules,$RequiredModules) {
    foreach($required in $RequiredModules) {
        $matches=@($Modules|Where-Object{ $_.name -ieq $required.name -and [string]::Equals($_.path,$required.path,[StringComparison]::OrdinalIgnoreCase) })
        if($matches.Count -ne 1){throw "Required DLL was not observed loaded: $($required.name)"}
        Assert-SessionProfileIdentity $required | Out-Null
    }
}
function Invoke-CharacterizationLane($Lane,$Packet,[string]$PacketFile,[string]$Output,[string]$Helper,[string]$RepoRoot) {
    $directory=Join-Path $Output $Lane.backend;$observation=Join-Path $Output ($Lane.backend+'-runtime.json')
    $watcher=$null;$success=$false;$errorText=$null
    try {
        $watchCommand='& '+(Quote-SessionProfileLiteral $Packet.watcher.path)+' -PacketPath '+(Quote-SessionProfileLiteral $PacketFile)+
            ' -ObservationPath '+(Quote-SessionProfileLiteral $observation)+' -Backend '+(Quote-SessionProfileLiteral $Lane.backend)
        $watchEncoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watchCommand))
        $watcher=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$watchEncoded) -PassThru -WindowStyle Hidden
        $watchOwner=[ordered]@{owner='Candle characterization runtime watcher';pid=$watcher.Id;creation_utc=$watcher.StartTime.ToUniversalTime().ToString('o');executable=$watcher.Path;
            source_commit=$Packet.source_commit;intended_lifetime='One bounded diagnostic lane, at most1860s';endpoints=@();output_root=$observation;shutdown='Stop this exact watcher handle after terminal/runtime validation'}
        Write-SessionProfileJson (Join-Path $Output ($Lane.backend+'-watcher-owner.json')) $watchOwner
        $arm=[Diagnostics.Stopwatch]::StartNew();$armed=$false
        while(-not $armed -and $arm.Elapsed.TotalSeconds -lt 5) {
            $watcher.Refresh();if($watcher.HasExited){throw 'Runtime watcher exited before arming'}
            if(Test-Path -LiteralPath $observation){try{$state=Get-Content -LiteralPath $observation -Raw|ConvertFrom-Json;$armed=$state.status -eq 'armed'}catch{}}
            if(-not $armed){Start-Sleep -Milliseconds 50}
        }
        if(-not $armed){throw 'Runtime watcher did not arm within bound'}
        $inputCheck={param($value,$source,$backend) Assert-CharacterizationInput $value $source $backend}
        $completion={param($progress,$job,$lane,$inputValue,$dir)
            Assert-CharacterizationCompletion $progress $job $lane $inputValue
            # Verification is host F32 arithmetic; the CPU executable avoids CUDA
            # DLL import commitment inside the 256MiB observer Job.
            $verification=Invoke-SessionProfileProbe $Packet.lanes[0].binary.path @('--input',$lane.input.path,'--verify-progress',(Join-Path $dir 'progress.json')) 10000 4096
            if($verification.ExitCode -ne 0){throw "Captured numerical verification failed: $($verification.Stderr)"}
            $checked=$verification.Stdout|ConvertFrom-Json -ErrorAction Stop
            if($checked.status -ne 'verified' -or $checked.rows -ne 6 -or $checked.oracle_comparisons -ne 3 -or $checked.replay_comparisons -ne 3){throw 'Numerical verifier result mismatch'}
            Write-SessionProfileJson (Join-Path $dir 'row-verification.json') ([ordered]@{status='verified';model_free_verifier_ms=$verification.ElapsedMs;rows=6;oracle_comparisons=3;replay_comparisons=3})
        }
        $observe={ $watcher.Refresh();if($watcher.HasExited -and $watcher.ExitCode -ne 0){throw 'Runtime watcher failed during active lane'} }
        $sampler={param($index) Get-SessionProfileGpuSample $index $Packet.gpu_probe.path}
        Invoke-SessionProfileLane $Lane $Packet.source_commit $directory $Helper $RepoRoot -InputValidator $inputCheck -CompletionValidator $completion -Observe $observe -SampleGpu $sampler -Qualification 'cold timing; independent oracle; reset/replay; loaded runtime; no provider/context or causal speedup claim'
        if(-not $watcher.WaitForExit(10000) -or $watcher.ExitCode -ne 0){throw 'Runtime watcher did not complete after lane release'}
        $runtime=Get-Content -LiteralPath $observation -Raw|ConvertFrom-Json
        $progress=Get-Content -LiteralPath (Join-Path $directory 'progress.json') -Raw|ConvertFrom-Json
        Assert-CharacterizationRuntime $runtime $Packet $Lane $progress (Get-FileHash -LiteralPath $PacketFile).Hash.ToLowerInvariant()
        $success=$true
    } catch {$errorText=$_.Exception.Message}
    finally {
        if($null -ne $watcher){try{Stop-SessionProfileController $watcher}catch{$errorText=$_.Exception.Message;$success=$false};$watcher.Dispose()}
        Write-SessionProfileJson (Join-Path $Output ($Lane.backend+'-qualification.json')) ([ordered]@{schema='candle.gpt-oss.characterization-qualification.v1';status=$(if($success){'qualified-short'}else{'incomplete'});error=$errorText;source_commit=$Packet.source_commit;
            original_reference_identity='independent pinned oracle; not CPU/CUDA agreement';scope='Diagnostic short F32 only; native consumer/profile/capacity/causal speedup remain unqualified'})
    }
    if(-not $success){throw "Characterization failed: $errorText"}
}
if($characterizationLibraryOnly){return}
if(-not $IsWindows -or $PSVersionTable.PSVersion.Major -lt 7){throw 'Characterization needs native Windows PowerShell 7'}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repo
if([string]::IsNullOrWhiteSpace($PacketPath) -or [string]::IsNullOrWhiteSpace($RunRoot)){throw 'PacketPath and RunRoot are required'}
if((Get-Item -LiteralPath $PacketPath).Length -gt 65536){throw 'Characterization packet exceeds 64 KiB'}
$characterPacket=Get-Content -LiteralPath $PacketPath -Raw|ConvertFrom-Json
Assert-CharacterizationPacketHeader $characterPacket (git rev-parse HEAD).Trim() (git show -s --format=%T HEAD).Trim()
if(@(git status --porcelain).Count -ne 0){throw 'Characterization needs its exact clean source'}
$helper=Assert-SessionProfileIdentity $characterPacket.bounded_helper
if($helper -ne (Join-Path $repo 'scripts\lfm2-vl\run-bounded-oracle.ps1')){throw 'Characterization requires Candle bounded helper'}
Assert-SessionProfileIdentity $characterPacket.watcher | Out-Null
if($characterPacket.watcher.path -ne (Join-Path $repo 'scripts\gpt-oss\watch-runtime.ps1')){throw 'Characterization requires owned watcher'}
Assert-SessionProfileIdentity $characterPacket.gpu_probe | Out-Null
Assert-SessionProfileIdentity $characterPacket.oracle | Out-Null
Assert-CharacterizationOracleRecovery $characterPacket.oracle_recovery $characterPacket.oracle
Assert-SessionProfileIdentity $characterPacket.fixture | Out-Null
if($characterPacket.fixture.path -ne (Join-Path $repo 'tests\fixtures\gpt_oss_task3_short_parity\reference.json')){throw 'Characterization fixture path mismatch'}
$helperPaths=@('run-characterization.ps1','run-session-profile.ps1','observer.ps1','observer-worker.ps1','observer.cs','watch-runtime.ps1','prepare-characterization.ps1')|ForEach-Object{Join-Path $PSScriptRoot $_}
if(@($characterPacket.helper_inputs).Count -ne $helperPaths.Count -or
    @($characterPacket.helper_inputs.path|Sort-Object -Unique).Count -ne $helperPaths.Count){throw 'Helper closure is incomplete or duplicate'}
foreach($dependency in $characterPacket.helper_inputs){if($dependency.path -notin $helperPaths){throw 'Unexpected helper closure path'};Assert-SessionProfileIdentity $dependency|Out-Null}
if(@($characterPacket.lanes).Count -ne 2 -or $characterPacket.lanes[0].backend -ne 'cpu' -or $characterPacket.lanes[1].backend -ne 'cuda'){throw 'Characterization requires CPU then CUDA'}
foreach($lane in $characterPacket.lanes) {
    if($lane.build_profile -ne 'release' -or $lane.precision -ne 'f32' -or $lane.format -ne 'gguf-mxfp4-f32'){throw 'Unsupported characterization build/precision/format'}
    Assert-SessionProfileIdentity $lane.build_record | Out-Null
    Assert-SessionProfileIdentity $lane.binary | Out-Null
    if($lane.build_record.size_bytes -gt 65536){throw 'Build record exceeds bound'}
    $record=Get-Content -LiteralPath $lane.build_record.path -Raw|ConvertFrom-Json
    Assert-CharacterizationBuild $record $lane $characterPacket.source_commit $characterPacket.source_tree (Get-CharacterizationSourcePaths $repo)
    Assert-CharacterizationBuildJob $record
    Assert-CharacterizationStartupJob $lane.model_free_startup_job $lane.binary
    Assert-CharacterizationRuntimeClosure $lane
    foreach($module in $lane.required_modules){Assert-SessionProfileIdentity $module|Out-Null}
    Assert-SessionProfileIdentity $lane.input | Out-Null
    if($lane.input.size_bytes -gt 16384){throw 'Characterization input exceeds bound'}
    $value=Get-Content -LiteralPath $lane.input.path -Raw|ConvertFrom-Json
    Assert-CharacterizationInput $value $characterPacket.source_commit $lane.backend
    foreach($name in @('path','size_bytes','sha256')){if($value.reference.$name -ne $characterPacket.oracle.$name){throw 'Characterization oracle/input mismatch'}}
}
$output=[IO.Path]::GetFullPath($RunRoot);$allowed=(Join-Path $repo 'artifacts')+'\'
if(-not $output.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $output) -or $output -ne $characterPacket.proposed_output_root){throw 'Characterization requires its fresh owned output root'}
New-Item -ItemType Directory -Path $output | Out-Null
Invoke-CharacterizationBatch $characterPacket.lanes {param($lane)
    Invoke-CharacterizationRuntimeEnvironment $lane {param($selected) Invoke-CharacterizationLane $selected $characterPacket $PacketPath $output $helper $repo}
}
