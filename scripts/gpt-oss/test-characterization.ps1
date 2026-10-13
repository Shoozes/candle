Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-characterization.ps1') -AsLibrary
$script:passed=0
function Assert-CharacterTest([bool]$Condition,[string]$Message) {
    if(-not $Condition){throw "Characterization test failed: $Message"};$script:passed++
}
function Assert-CharacterReject([scriptblock]$Operation,[string]$Pattern) {
    $caught=$null;try{& $Operation|Out-Null}catch{$caught=$_.Exception.Message}
    Assert-CharacterTest ($null -ne $caught -and $caught -match $Pattern) "Expected $Pattern; observed $caught"
}
$plan=Get-CharacterizationPlan
Assert-CharacterTest ($plan.loads -eq 1 -and $plan.requests -eq 3 -and $plan.forwards -eq 22 -and $plan.token_steps -eq 37 -and $plan.generated_tokens -eq 16) 'new explicit work allowance'
Assert-CharacterTest ($plan.warmups -eq 0 -and $plan.retries -eq 0 -and $plan.rows -eq 6 -and $plan.resets -eq 2) 'no hidden forwards or sampled fixed-segment tokens'
$f=Get-Content (Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw|ConvertFrom-Json
$source='a'*40
$inputValue=[pscustomobject]@{schema='candle.gpt-oss.characterization-input.v1';source_commit=$source;backend='cpu';device_index=0;
    model=[pscustomobject]@{path='never-opened.gguf';size_bytes=13792638656;sha256='aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'};
    reference=[pscustomobject]@{path='never-opened.logits';size_bytes=$f.reference_file_bytes;sha256=$f.reference_file_sha256};
    prompt_token_ids=$f.reference_prefix_token_ids;generated_tokens=16;segments=@('cold','numerical','replay')}
Assert-CharacterizationInput $inputValue $source 'cpu';Assert-CharacterTest $true 'new input admission'
$inputValue.schema='candle.gpt-oss.session-profile-input.v1'
Assert-CharacterReject {Assert-CharacterizationInput $inputValue $source 'cpu'} 'version'
$inputValue.schema='candle.gpt-oss.characterization-input.v1'
$inputValue.segments=@('numerical','cold','replay')
Assert-CharacterReject {Assert-CharacterizationInput $inputValue $source 'cpu'} 'frozen'
$inputValue.segments=@('cold','numerical','replay')
$old=$inputValue.model.sha256;$inputValue.model.sha256='d'*64
Assert-CharacterReject {Assert-CharacterizationInput $inputValue $source 'cpu'} 'frozen';$inputValue.model.sha256=$old
$inputValue.model.size_bytes++
Assert-CharacterReject {Assert-CharacterizationInput $inputValue $source 'cpu'} 'frozen';$inputValue.model.size_bytes--
$lane=[pscustomobject]@{backend='cpu';input=[pscustomobject]@{sha256='b'*64};binary=[pscustomobject]@{path='never-opened.exe';size_bytes=1;sha256='c'*64}}
$job=[pscustomobject]@{contract='candle-lfm2-vl-bounded-oracle-v1';pid=1;child_exit_code=0;executable_bytes=1;executable_sha256='c'*64;
    peak_private_memory_bytes=1;peak_job_memory_bytes=1;max_job_memory_bytes=$plan.job_bytes;timeout_seconds=1800;
    pid_absent_after_cleanup=$true;assigned_before_resume=$true;job_assigned=$true;started_suspended=$true;resumed=$true;termination_reason='exited';error=$null}
$calls=@(for($i=0;$i -lt 22;$i++) {
    $segment=if($i -lt 16){'cold'}elseif($i -lt 19){'numerical'}else{'replay'};$index=if($i -lt 16){$i}else{($i-16)%3}
    [pscustomobject]@{segment=$segment;input_tokens=$(if($index -eq 0){if($segment -eq 'cold'){8}else{5}}else{1});
        committed_sequence_tokens=$(if($segment -eq 'cold'){8+$index}else{5+$index});forward_ms=1.;cuda_final_validation_ms=$null;
        caller_readback_ms=0.;argmax_ms=0.;row_capture_ms=0.;comparison_ms=0.}
})
$progress=[pscustomobject]@{schema='candle.gpt-oss.characterization-progress.v1';status='completed';error=$null;source_commit=$source;
    input_sha256='b'*64;pid=1;loads_attempted=1;requests_attempted=3;requests_completed=3;attempted_forward_calls=22;
    completed_forward_calls=22;attempted_token_steps_upper_bound=37;completed_token_steps=37;registry_entries_after_release=0;
    static_weight_bytes=1;consumed_files=@([pscustomobject]@{path='never-opened.gguf';size_bytes=13792638656;sha256='aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'});
    generated_token_ids=@(1)*16;calls=$calls;rows=@([pscustomobject]@{fixture='metadata only; does not qualify rows'})*6;
    independent_comparisons=@(for($i=0;$i -lt 3;$i++){[pscustomobject]@{stage=('context.'+(5+$i));max_abs_log_probability_error=0.;mean_abs_log_probability_error=0.;top_k_max_abs_log_probability_error=0.;total_variation_probability_error=0.;
        top_reference=@([pscustomobject]@{token_id=$f.reference_row_top1[$i].token_id;log_probability=-0.1})*8;top_candle=@([pscustomobject]@{token_id=$f.reference_row_top1[$i].token_id;log_probability=-0.1})*8}});replay_max_abs_errors=@(0.,0.,0.);reset_ms=@(0.,0.);
    load_ms=1.;unload_ms=1.;cold_ms=16.;numerical_ms=3.;replay_ms=3.;total_ms=24.}
foreach($backend in @('cpu','cuda')) {
    $lane.backend=$backend;$inputValue.backend=$backend
    $progress.calls|ForEach-Object{$_.cuda_final_validation_ms=if($backend -eq 'cuda'){0.5}else{$null}}
    Assert-CharacterizationInput $inputValue $source $backend
    Assert-CharacterizationCompletion $progress $job $lane $inputValue
    Assert-CharacterTest $true "$backend typed completion"
    $roundTrip=$progress|ConvertTo-Json -Depth 12|ConvertFrom-Json
    Assert-CharacterizationCompletion $roundTrip $job $lane $inputValue
    Assert-CharacterTest $true "$backend JSON completion"
    foreach($name in @('completed_forward_calls','completed_token_steps','requests_completed','static_weight_bytes')) {
        $old=$progress.$name
        foreach($bad in @($true,'1',0.1,-1)){$progress.$name=$bad;Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'unsigned integer'}
        $progress.$name=$old
    }
    foreach($name in @('load_ms','cold_ms','numerical_ms','replay_ms','unload_ms','total_ms')) {
        $old=$progress.$name
        foreach($bad in @($null,$true,'1',-1,[double]::PositiveInfinity)){$progress.$name=$bad;Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'aggregate timing'}
        $progress.$name=$old
    }
    $progress.calls[16].committed_sequence_tokens=8
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'prefix position'
    $progress.calls[16].committed_sequence_tokens=5
    $progress.independent_comparisons[0].mean_abs_log_probability_error=0.1
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'bounds exceeded'
    $progress.independent_comparisons[0].mean_abs_log_probability_error=0.
    $progress.replay_max_abs_errors[0]=1.
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'bounds exceeded'
    $progress.replay_max_abs_errors[0]=0.
    $progress.calls[16].argmax_ms=1.
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'overlap'
    $progress.calls[16].argmax_ms=0.
    $progress.total_ms=23.
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'aggregate sum'
    $progress.total_ms=24.
    $progress.calls[0].forward_ms=20.
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'segment timing'
    $progress.calls[0].forward_ms=1.
    $progress.load_ms=[double]::MaxValue;$progress.unload_ms=[double]::MaxValue;$progress.total_ms=[double]::MaxValue
    Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'aggregate sum'
    $progress.load_ms=1.;$progress.unload_ms=1.;$progress.total_ms=24.
}
$progress.status='failed';$progress.error='injected after committed forward';$progress.requests_completed=1
Assert-CharacterReject {Assert-CharacterizationCompletion $progress $job $lane $inputValue} 'incomplete'
Assert-CharacterTest ($progress.status -eq 'failed' -and $progress.error -eq 'injected after committed forward') 'failed progress preserved'
$header=[pscustomobject]@{schema='candle.gpt-oss.characterization-packet.v1';draft=$false;source_commit=$source;source_tree=('e'*40);
    allowance_proposal=[pscustomobject]@{loads=2;segments=6;public_forwards=44;processed_token_steps=74;sampled_output_tokens=32;warmups=0;retries=0;per_lane=$plan;
        preparation_startup_probes=2;preparation_startup_job_bytes=2147483648;preparation_startup_deadline_seconds=15}}
Assert-CharacterizationPacketHeader $header $source ('e'*40);Assert-CharacterTest $true 'final packet counts and source'
foreach($bad in @($true,'false',$null)){$header.draft=$bad;Assert-CharacterReject {Assert-CharacterizationPacketHeader $header $source ('e'*40)} 'final source'};$header.draft=$false
$header.allowance_proposal.public_forwards=32
Assert-CharacterReject {Assert-CharacterizationPacketHeader $header $source ('e'*40)} 'allowance mismatch';$header.allowance_proposal.public_forwards=44
$header.allowance_proposal.per_lane.loads='1'
Assert-CharacterReject {Assert-CharacterizationPacketHeader $header $source ('e'*40)} 'unsigned integer';$header.allowance_proposal.per_lane.loads=1
$identity=[pscustomobject]@{path=(Resolve-Path (Join-Path $PSScriptRoot '../../tests/fixtures/gpt_oss_task3_short_parity/reference.json')).Path;size_bytes=0;sha256=''}
$identity.size_bytes=(Get-Item $identity.path).Length;$identity.sha256=(Get-FileHash $identity.path).Hash.ToLowerInvariant()
$build=[pscustomobject]@{schema='candle.gpt-oss.characterization-build.v1';draft=$false;source_commit=$source;source_tree=('e'*40);profile='release';precision='f32';format='gguf-mxfp4-f32';exit_code=0;rustc='fixture';cargo='fixture';features=@();binary=$lane.binary;source_files=@($identity)}
$lane.backend='cpu'
Assert-CharacterizationBuild $build $lane $source ('e'*40) @($identity.path);Assert-CharacterTest $true 'build source, binary and file bytes'
$build.draft=$true;Assert-CharacterReject {Assert-CharacterizationBuild $build $lane $source ('e'*40) @($identity.path)} 'Build source';$build.draft=$false
$build.source_commit='f'*40;Assert-CharacterReject {Assert-CharacterizationBuild $build $lane $source ('e'*40) @($identity.path)} 'Build source';$build.source_commit=$source
$build.features=@('cuda');Assert-CharacterReject {Assert-CharacterizationBuild $build $lane $source ('e'*40) @($identity.path)} 'features';$build.features=@()
$build.binary=[pscustomobject]@{path='other';size_bytes=1;sha256='c'*64}
Assert-CharacterReject {Assert-CharacterizationBuild $build $lane $source ('e'*40) @($identity.path)} 'binary identity';$build.binary=$lane.binary
Assert-CharacterReject {Assert-CharacterizationBuild $build $lane $source ('e'*40) @('other')} 'source closure'
$old=$identity.sha256;$identity.sha256='f'*64
Assert-CharacterReject {Assert-CharacterizationBuild $build $lane $source ('e'*40) @($identity.path)} 'SHA-256 mismatch';$identity.sha256=$old
$visited=[Collections.Generic.List[string]]::new()
$two=@([pscustomobject]@{backend='cpu'},[pscustomobject]@{backend='cuda'})
Assert-CharacterReject {Invoke-CharacterizationBatch $two {param($selected) $visited.Add($selected.backend);throw 'injected CPU failure after owned cleanup'}} 'injected CPU'
Assert-CharacterTest (($visited -join ',') -eq 'cpu') 'CPU failure never dispatches CUDA'
$visited.Clear();Invoke-CharacterizationBatch $two {param($selected) $visited.Add($selected.backend)}
Assert-CharacterTest (($visited -join ',') -eq 'cpu,cuda') 'successful CPU qualification precedes CUDA'
$priorPath=$env:PATH
Invoke-CharacterizationRuntimeEnvironment $two[0] {param($selected) Assert-CharacterTest ($env:PATH -eq $priorPath) 'CPU does not add CUDA directories'}
$runtimeLane=[pscustomobject]@{backend='cuda';required_modules=@([pscustomobject]@{path='C:\bound-runtime\cublas64_13.dll'})}
Invoke-CharacterizationRuntimeEnvironment $runtimeLane {param($selected) Assert-CharacterTest ($env:PATH.StartsWith('C:\bound-runtime;')) 'CUDA prefers its declared runtime directory'}
Assert-CharacterTest ($env:PATH -eq $priorPath) 'runtime PATH restored after success'
Assert-CharacterReject {Invoke-CharacterizationRuntimeEnvironment $runtimeLane {throw 'injected environment failure'}} 'injected environment'
Assert-CharacterTest ($env:PATH -eq $priorPath) 'runtime PATH restored after failure'
"characterization model-free controls: $script:passed passed"
