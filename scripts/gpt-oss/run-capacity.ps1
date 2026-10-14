[CmdletBinding()]
param([string]$CapacityPacketPath,[string]$CapacityRunRoot,[switch]$AsLibrary)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$capacityLibraryOnly=[bool]$AsLibrary
. (Join-Path $PSScriptRoot 'run-characterization.ps1') -AsLibrary

function Get-CapacityPlan {
    [pscustomobject]@{loads=1;requests=1;public_forwards=2096;token_steps=8192;prefill_tokens=6144;
        prefill_chunk_tokens=128;prefill_calls=48;fixed_decode_tokens=2048;sampled_outputs=0;
        warmups=0;retries=0;job_bytes=[UInt64]34359738368;gpu_bytes=[UInt64]20000000000;deadline_seconds=1800}
}
function Assert-CapacityKeys($Value,[string[]]$Names) {
    $actual=@($Value.PSObject.Properties.Name)
    if($actual.Count -ne $Names.Count -or @($actual|Where-Object{$_ -cnotin $Names}).Count){throw 'Capacity object has missing/unknown fields'}
}
function Get-CapacityTokenBytes {
    $fixture=Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw|ConvertFrom-Json
    $prefix=@($fixture.reference_prefix_token_ids)
    if($prefix.Count -ne 8){throw 'Capacity prefix is not the pinned eight-token fixture'}
    $bytes=[byte[]]::new(32768)
    for($i=0;$i -lt 8192;$i++){
        $word=[BitConverter]::GetBytes([UInt32]$prefix[$i%8]);if(-not [BitConverter]::IsLittleEndian){[Array]::Reverse($word)}
        [Array]::Copy($word,0,$bytes,$i*4,4)
    }
    return ,$bytes
}
function Assert-CapacityInput($Value,[string]$Source,[string]$Backend) {
    Assert-CapacityKeys $Value @('schema','source_commit','backend','device_index','model','token_file','prefill_tokens','prefill_chunk_tokens','fixed_decode_tokens','sampled_outputs')
    foreach($identity in @($Value.model,$Value.token_file)){Assert-CapacityKeys $identity @('path','size_bytes','sha256');Assert-SessionProfileUnsignedInteger $identity.size_bytes 'capacity input size' 1}
    if($Value.schema -ne 'candle.gpt-oss.capacity-input.v1' -or $Value.source_commit -ne $Source -or
        $Source -cnotmatch '^[0-9a-f]{40}$' -or $Backend -notin @('cpu','cuda') -or $Value.backend -ne $Backend){throw 'Capacity input version/source/backend mismatch'}
    Assert-SessionProfileUnsignedInteger $Value.device_index 'capacity device' 0 0
    foreach($entry in @{prefill_tokens=6144;prefill_chunk_tokens=128;fixed_decode_tokens=2048;sampled_outputs=0}.GetEnumerator()){
        Assert-SessionProfileUnsignedInteger $Value.($entry.Key) $entry.Key
        if($Value.($entry.Key) -ne $entry.Value){throw 'Capacity frozen work mismatch'}
    }
    if(-not [string]::Equals($Value.model.path,'C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf',[StringComparison]::OrdinalIgnoreCase) -or
        $Value.model.size_bytes -ne 13792638656 -or $Value.model.sha256 -ne 'aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'){throw 'Capacity retained model identity mismatch'}
    $expected=[Convert]::ToHexString([Security.Cryptography.SHA256]::HashData((Get-CapacityTokenBytes))).ToLowerInvariant()
    if($Value.token_file.size_bytes -ne 32768 -or $Value.token_file.sha256 -ne $expected){throw 'Capacity token identity/work mismatch'}
    Assert-SessionProfileIdentity $Value.token_file|Out-Null
    # The model is rehashed only inside its separately admitted native Job.
}
function Get-CapacitySourcePaths([string]$RepoRoot) {
    @(Get-CharacterizationSourcePaths $RepoRoot)
    @('candle-kernels/src/ffi.rs','candle-examples/examples/gpt-oss-capacity-stress.rs')|ForEach-Object{Join-Path $RepoRoot $_}
    Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'candle-examples\examples\gpt_oss_capacity') -File|ForEach-Object{$_.FullName}
}
function Assert-CapacityPredecessor($Identity) {
    Assert-SessionProfileIdentity $Identity|Out-Null
    if($Identity.size_bytes -ne 11482 -or $Identity.sha256 -ne '76d36caae5765811696e5ee59de8da0064ca87a59318edbba074a5b8aa1d1a9e'){throw 'Capacity needs the exact completed CPU-first short predecessor'}
    $report=Get-Content -LiteralPath $Identity.path -Raw|ConvertFrom-Json
    if($report.schema -ne 'candle.gpt-oss.memory-short-report.v1' -or
        $report.source_commit -ne '3db05cd8eedb51d6b7777d2eed705ec481e0420b' -or -not $report.all_six_owners_absent -or
        $report.counts.public_forwards -ne 44 -or $report.counts.processed_token_steps -ne 74){throw 'Capacity short predecessor is incomplete'}
    foreach($receipt in $report.receipts){Assert-SessionProfileIdentity $receipt|Out-Null}
    foreach($lane in $report.lanes){if($lane.registry_entries_after_release -ne 0 -or @($lane.raw_replay_max_errors|Where-Object{$_ -ne 0}).Count){throw 'Capacity predecessor release/replay failed'}}
}
function Assert-CapacityPacketHeader($Value,[string]$Source,[string]$Tree) {
    if($Value.schema -ne 'candle.gpt-oss.capacity-packet.v1' -or $Value.draft -isnot [bool] -or $Value.draft -or
        $Value.source_commit -ne $Source -or $Value.source_tree -ne $Tree){throw 'Capacity requires its exact final clean source'}
    foreach($property in (Get-CapacityPlan).PSObject.Properties){
        Assert-SessionProfileUnsignedInteger $Value.allowance_proposal.($property.Name) "capacity $($property.Name)"
        if($Value.allowance_proposal.($property.Name) -ne $property.Value){throw 'Capacity allowance differs from frozen plan'}
    }
    if(@($Value.lanes).Count -ne 1 -or $Value.lanes[0].backend -ne 'cuda'){throw 'First capacity packet selects one CUDA lane after CPU-first short proof'}
}
function Assert-CapacityCompletion($Progress,$Job,$Lane,$InputValue,[string]$Directory) {
    foreach($name in @('pid','loads_attempted','requests_attempted','requests_completed','attempted_forward_calls','completed_forward_calls',
        'attempted_token_steps_upper_bound','completed_token_steps','completed_prefill_calls','completed_decode_calls','sampled_output_tokens',
        'committed_sequence_tokens','cache_bytes','cache_capacity_bytes','peak_cache_capacity_bytes','static_weight_bytes','registry_entries_after_release')){
        Assert-SessionProfileUnsignedInteger $Progress.$name "capacity $name"
    }
    Assert-SessionProfileUnsignedInteger $Progress.pid 'capacity owner PID' 1 ([int]::MaxValue)
    if(-not [string]::Equals($Progress.executable,$Lane.binary.path,[StringComparison]::OrdinalIgnoreCase)){throw 'Capacity executable path mismatch'}
    if($Progress.schema -ne 'candle.gpt-oss.capacity-progress.v1' -or $Progress.status -ne 'completed' -or $null -ne $Progress.error -or
        $Progress.source_commit -ne $InputValue.source_commit -or $Progress.input_sha256 -ne $Lane.input.sha256){throw 'Capacity progress identity/terminal mismatch'}
    foreach($entry in @{loads_attempted=1;requests_attempted=1;requests_completed=1;attempted_forward_calls=2096;completed_forward_calls=2096;
        attempted_token_steps_upper_bound=8192;completed_token_steps=8192;completed_prefill_calls=48;completed_decode_calls=2048;
        sampled_output_tokens=0;committed_sequence_tokens=8192;registry_entries_after_release=0}.GetEnumerator()){
        if($Progress.($entry.Key) -ne $entry.Value){throw 'Capacity work/cache/release count mismatch'}
    }
    foreach($field in @('pid_absent_after_cleanup','assigned_before_resume','job_assigned','started_suspended','resumed')){
        if($Job.$field -isnot [bool] -or -not $Job.$field){throw 'Capacity Job ownership/release incomplete'}
    }
    foreach($field in @('peak_private_memory_bytes','peak_job_memory_bytes')){Assert-SessionProfileUnsignedInteger $Job.$field $field 1 34359738368}
    if($Job.contract -ne 'candle-lfm2-vl-bounded-oracle-v1' -or $Job.child_exit_code -ne 0 -or $Job.termination_reason -ne 'exited' -or $null -ne $Job.error -or
        $Job.max_job_memory_bytes -ne 34359738368 -or $Job.timeout_seconds -ne 1800 -or $Job.pid -ne $Progress.pid -or
        $Job.executable_sha256 -ne $Lane.binary.sha256 -or $Job.executable_bytes -ne $Lane.binary.size_bytes){throw 'Capacity bounded Job identity/limits mismatch'}
    if($Progress.cache_bytes -le 0 -or $Progress.cache_capacity_bytes -lt $Progress.cache_bytes -or
        $Progress.peak_cache_capacity_bytes -lt $Progress.cache_capacity_bytes -or $Progress.peak_cache_capacity_bytes -gt 20000000000 -or
        $Progress.static_weight_bytes -le 0 -or $Progress.static_weight_bytes -gt 20000000000){throw 'Capacity storage receipt invalid'}
    $consumed=@($Progress.consumed_files)
    if($consumed.Count -ne 1 -or $consumed[0].sha256 -ne $InputValue.model.sha256 -or $consumed[0].size_bytes -ne $InputValue.model.size_bytes){throw 'Capacity consumed model identity mismatch'}
    $consumedPath=[string]$consumed[0].path
    if($consumedPath.StartsWith('\\?\',[StringComparison]::Ordinal)){$consumedPath=$consumedPath.Substring(4)}
    if(-not [string]::Equals($consumedPath,$InputValue.model.path,[StringComparison]::OrdinalIgnoreCase)){throw 'Capacity consumed model path mismatch'}
    foreach($name in @('path','sha256','size_bytes')){if($Progress.token_file.$name -ne $InputValue.token_file.$name){throw 'Capacity consumed token identity mismatch'}}
    foreach($name in @('load_ms','unload_ms','prefill_forward_ms','decode_forward_ms','total_ms','prefill_validation_ms','decode_validation_ms')){
        if(-not(Test-SessionProfileTiming $Progress.$name)){throw 'Capacity timing invalid'}
    }
    if($Progress.prefill_validation_ms -gt $Progress.prefill_forward_ms+4.192 -or $Progress.decode_validation_ms -gt $Progress.decode_forward_ms+4.192 -or
        $Progress.load_ms+$Progress.unload_ms+$Progress.prefill_forward_ms+$Progress.decode_forward_ms -gt $Progress.total_ms+4.192){throw 'Capacity nested/aggregate timing mismatch'}
    $verification=Invoke-SessionProfileProbe $script:capacityHostVerifier @('--input',$Lane.input.path,'--verify-progress',(Join-Path $Directory 'progress.json')) 10000 4096
    if($verification.ExitCode -ne 0){throw 'Host capacity receipt verification failed'}
    $verified=$verification.Stdout|ConvertFrom-Json
    if($verified.status -ne 'verified-capacity-progress' -or $verified.token_steps -ne 8192 -or $verified.public_forwards -ne 2096 -or $verified.sampled_outputs -ne 0){throw 'Host capacity result differs'}
    Write-SessionProfileJson (Join-Path $Directory 'host-verification.json') ([ordered]@{status='verified';observer_ms=$verification.ElapsedMs;model_forwards=0})
}
function Assert-CapacityAdmission($Value,[string]$RepoRoot) {
    Assert-CapacityPacketHeader $Value (git rev-parse HEAD).Trim() (git show -s --format=%T HEAD).Trim()
    if(@(git status --porcelain).Count){throw 'Capacity needs clean source'}
    Assert-CapacityPredecessor $Value.short_predecessor
    $helper=Assert-SessionProfileIdentity $Value.bounded_helper
    if($helper -ne (Join-Path $RepoRoot 'scripts\lfm2-vl\run-bounded-oracle.ps1')){throw 'Capacity requires Candle native helper'}
    Assert-SessionProfileIdentity $Value.watcher|Out-Null
    if($Value.watcher.path -ne (Join-Path $RepoRoot 'scripts\gpt-oss\watch-capacity.ps1')){throw 'Capacity watcher path mismatch'}
    Assert-SessionProfileIdentity $Value.gpu_probe|Out-Null
    $expectedHelpers=@('run-capacity.ps1','prepare-capacity.ps1','watch-capacity.ps1','run-characterization.ps1','run-session-profile.ps1','observer.ps1','observer.cs','observer-worker.ps1')|ForEach-Object{Join-Path $RepoRoot ('scripts\gpt-oss\'+$_)}
    if(@($Value.helper_inputs).Count -ne $expectedHelpers.Count -or @($Value.helper_inputs.path|Sort-Object -Unique).Count -ne $expectedHelpers.Count){throw 'Capacity helper closure incomplete'}
    foreach($file in $Value.helper_inputs){if($file.path -notin $expectedHelpers){throw 'Capacity helper closure unexpected path'};Assert-SessionProfileIdentity $file|Out-Null}
    $lane=$Value.lanes[0]
    foreach($selected in @($Value.host_verifier,$lane)){
        Assert-SessionProfileIdentity $selected.binary|Out-Null;Assert-SessionProfileIdentity $selected.build_record|Out-Null
        if($selected.build_record.size_bytes -gt 65536){throw 'Capacity build record too large'}
        $build=Get-Content -LiteralPath $selected.build_record.path -Raw|ConvertFrom-Json
        Assert-CharacterizationBuild $build $selected $Value.source_commit $Value.source_tree (Get-CapacitySourcePaths $RepoRoot)
        Assert-CharacterizationBuildJob $build;Assert-CharacterizationStartupJob $selected.model_free_startup_job $selected.binary
        Assert-CharacterizationRuntimeClosure $selected
        foreach($module in $selected.required_modules){Assert-SessionProfileIdentity $module|Out-Null}
        Assert-SessionProfileIdentity $selected.input|Out-Null
        if($selected.input.size_bytes -gt 16384){throw 'Capacity startup input too large'}
        Assert-CapacityInput (Get-Content -LiteralPath $selected.input.path -Raw|ConvertFrom-Json) $Value.source_commit $selected.backend
    }
    if($Value.host_verifier.backend -ne 'cpu'){throw 'Capacity verifier must avoid CUDA imports'}
    Assert-SessionProfileIdentity $lane.input|Out-Null
    if($lane.input.size_bytes -gt 16384){throw 'Capacity input too large'}
    Assert-CapacityInput (Get-Content -LiteralPath $lane.input.path -Raw|ConvertFrom-Json) $Value.source_commit 'cuda'
    Assert-SessionProfileIdentity $Value.fixture|Out-Null
    if($Value.fixture.path -ne (Join-Path $RepoRoot 'tests\fixtures\gpt_oss_task3_short_parity\reference.json')){throw 'Capacity fixture path differs'}
    $output=[IO.Path]::GetFullPath($Value.proposed_output_root)
    if(-not $output.StartsWith((Join-Path $RepoRoot 'artifacts')+'\',[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $output)){throw 'Capacity output must be fresh owned artifacts'}
}
function Invoke-CapacityLane($Lane,$Value,[string]$PacketFile,[string]$RepoRoot) {
    $watcher=$null;$success=$false;$message=$null;$root=$Value.proposed_output_root;$observation=Join-Path $root 'cuda-runtime.json'
    try {
        $command='& '+(Quote-SessionProfileLiteral $Value.watcher.path)+' -CapacityPacketPath '+(Quote-SessionProfileLiteral $PacketFile)+
            ' -CapacityObservationPath '+(Quote-SessionProfileLiteral $observation)
        $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
        $executable=(Get-Command pwsh).Source
        $watcher=Start-Process -FilePath $executable -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -WindowStyle Hidden -PassThru
        $identity=Get-SessionProfileOwnerIdentity $watcher.Id $watcher.StartTime $executable
        Write-SessionProfileJson (Join-Path $root 'cuda-watcher-owner.json') ([ordered]@{owner='Candle capacity watcher';pid=$identity.pid;creation_utc=$identity.creation_utc;executable=$identity.executable;
            source_commit=$Value.source_commit;intended_lifetime='one bounded capacity lane,at most1860s';endpoints=@();output_root=$observation;shutdown='stop exact watcher handle after lane release'})
        $timer=[Diagnostics.Stopwatch]::StartNew();$armed=$false
        while(-not $armed -and $timer.Elapsed.TotalSeconds -lt 5){
            $watcher.Refresh();if($watcher.HasExited){throw 'Capacity watcher exited before arming'}
            if(Test-Path -LiteralPath $observation){try{$armed=(Get-Content -LiteralPath $observation -Raw|ConvertFrom-Json).status -eq 'armed'}catch{}}
            if(-not $armed){Start-Sleep -Milliseconds 50}
        }
        if(-not $armed){throw 'Capacity watcher arming deadline'}
        $inputCheck={param($v,$s,$b) Assert-CapacityInput $v $s $b}
        $completion={param($p,$j,$l,$i,$d) Assert-CapacityCompletion $p $j $l $i $d}
        $observe={$watcher.Refresh();if($watcher.HasExited -and $watcher.ExitCode -ne 0){throw 'Capacity watcher failed'}}
        $sample={param($index) Get-SessionProfileGpuSample $index $Value.gpu_probe.path}
        Invoke-SessionProfileLane $Lane $Value.source_commit (Join-Path $root 'cuda') $Value.bounded_helper.path $RepoRoot -InputValidator $inputCheck -CompletionValidator $completion -Observe $observe -SampleGpu $sample -Qualification 'fixed-token8k stress only; no sampled generation,long-context numerical oracle or route qualification'
        if(-not $watcher.WaitForExit(10000) -or $watcher.ExitCode -ne 0){throw 'Capacity watcher release failed'}
        $runtime=Get-Content -LiteralPath $observation -Raw|ConvertFrom-Json
        $progress=Get-Content -LiteralPath (Join-Path $root 'cuda\progress.json') -Raw|ConvertFrom-Json
        Assert-CharacterizationRuntime $runtime $Value $Lane $progress (Get-FileHash -LiteralPath $PacketFile).Hash.ToLowerInvariant()
        $success=$true
    }catch{$message=$_.Exception.Message}
    finally{
        if($null -ne $watcher){try{Stop-SessionProfileController $watcher}catch{$message=$_.Exception.Message;$success=$false};$watcher.Dispose()}
        Write-SessionProfileJson (Join-Path $root 'qualification.json') ([ordered]@{schema='candle.gpt-oss.capacity-qualification.v1';status=$(if($success){'qualified-fixed-token-stress'}else{'incomplete'});error=$message;source_commit=$Value.source_commit;scope='8192 committed fixed tokens only; generated semantics,independent long-context numerics,16k/32k and90s/600s routes unqualified'})
    }
    if(-not $success){throw "Capacity qualification failed: $message"}
}
if($capacityLibraryOnly){return}
if(-not $IsWindows -or $PSVersionTable.PSVersion.Major -lt 7){throw 'Capacity execution requires native PowerShell7'}
$capacityRepo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path;Set-Location -LiteralPath $capacityRepo
if([string]::IsNullOrWhiteSpace($CapacityPacketPath) -or [string]::IsNullOrWhiteSpace($CapacityRunRoot)){throw 'CapacityPacketPath and CapacityRunRoot required'}
if((Get-Item -LiteralPath $CapacityPacketPath).Length -gt 65536){throw 'Capacity packet exceeds64KiB'}
$capacityValue=Get-Content -LiteralPath $CapacityPacketPath -Raw|ConvertFrom-Json
Assert-CapacityAdmission $capacityValue $capacityRepo
if([IO.Path]::GetFullPath($CapacityRunRoot) -ne $capacityValue.proposed_output_root){throw 'Capacity output differs from bound root'}
$script:capacityHostVerifier=$capacityValue.host_verifier.binary.path
New-Item -ItemType Directory -Path $capacityValue.proposed_output_root|Out-Null
Invoke-CharacterizationRuntimeEnvironment $capacityValue.lanes[0] {param($selected) Invoke-CapacityLane $selected $capacityValue $CapacityPacketPath $capacityRepo}
Write-Output 'Completed fixed-token capacity stress; its finite allowance is closed.'
