Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-session-profile.ps1') -AsLibrary
$taskRoot=Join-Path ([IO.Path]::GetTempPath()) ('candle-session-profile-tests-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskRoot | Out-Null
$script:taskPassed=0
$taskSuccess=$false
function Assert-ProfileTest([bool]$Condition,[string]$Message) {
    if(-not $Condition){throw "Profile test failed: $Message"}; $script:taskPassed++
}
function Assert-ProfileReject([scriptblock]$Operation,[string]$Pattern) {
    $caught=$null
    try { & $Operation | Out-Null } catch { $caught=$_.Exception.Message }
    Assert-ProfileTest ($null -ne $caught -and $caught -match $Pattern) "Expected rejection $Pattern; observed $caught"
}
try {
    $plan=Get-SessionProfilePlan
    Assert-ProfileTest ($plan.forward_calls -eq 16 -and $plan.token_steps -eq 23 -and $plan.loads -eq 1 -and $plan.requests -eq 1) 'forward/request accounting'
    Assert-ProfileTest ($plan.max_job_memory_bytes -eq 34359738368 -and $plan.max_gpu_bytes -eq 20000000000 -and $plan.deadline_seconds -eq 1800) 'frozen limits'
    Assert-ProfileTest ($plan.warmups -eq 0 -and $plan.retries -eq 0) 'no hidden work'
    Assert-ProfileTest ($plan.timing_rounding_tolerance_ms -eq 0.001) 'explicit per-field millisecond rounding tolerance'
    $source='a'*40
    $frozen=(Get-Content -LiteralPath (Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw | ConvertFrom-Json).reference_prefix_token_ids
    $inputValue=[pscustomobject]@{schema='candle.gpt-oss.session-profile-input.v1';source_commit=$source;backend='cpu';device_index=0;model=[pscustomobject]@{sha256='b'*64;size_bytes=1};prompt_token_ids=$frozen;generated_tokens=16}
    Assert-SessionProfileInput $inputValue $source 'cpu'
    Assert-ProfileTest $true 'CPU admission needs no GPU monitor'
    $inputValue.generated_tokens=17
    Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cpu' } 'workload differs'
    $inputValue.generated_tokens=16
    $inputValue.device_index=1
    Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cpu' } 'device index'
    $inputValue.device_index=0
    $inputValue.prompt_token_ids=@(1,2,3,4,5,6,7,8)
    Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cpu' } 'pinned reference prefix'
    $inputValue.prompt_token_ids=$frozen
    Assert-ProfileReject { Assert-SessionProfileInput $inputValue ('c'*40) 'cpu' } 'source/backend/version'
    $inputValue.backend='cuda'
    Assert-SessionProfileInput $inputValue $source 'cuda'
    Assert-ProfileTest $true 'explicit CUDA admission'
    foreach($value in @($true,15.6,'16',$null)) {
        $inputValue.generated_tokens=$value
        Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cuda' } 'unsigned integer'
    }
    $inputValue.generated_tokens=16
    foreach($value in @($true,0.1,'0',[UInt64]::MaxValue)) {
        $inputValue.device_index=$value
        Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cuda' } 'unsigned integer'
    }
    $inputValue.device_index=0
    foreach($value in @($true,976.1,'976',-1,[UInt64]4294967296)) {
        $inputValue.prompt_token_ids=@($value)*8
        Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cuda' } 'unsigned integer'
    }
    $inputValue.prompt_token_ids=$frozen
    $inputValue.model.size_bytes=$true
    Assert-ProfileReject { Assert-SessionProfileInput $inputValue $source 'cuda' } 'unsigned integer'
    $inputValue.model.size_bytes=1

    $file=Join-Path $taskRoot 'identity.txt'
    [IO.File]::WriteAllText($file,'owned proof')
    $identity=[pscustomobject]@{path=$file;size_bytes=11;sha256=(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()}
    Assert-ProfileTest ((Assert-SessionProfileIdentity $identity) -eq $file) 'exact file identity'
    $identity.size_bytes=12
    Assert-ProfileReject { Assert-SessionProfileIdentity $identity } 'size mismatch'
    $identity.size_bytes=11; $identity.sha256='d'*64
    Assert-ProfileReject { Assert-SessionProfileIdentity $identity } 'SHA-256 mismatch'
    foreach($value in @($true,11.4,'11',$null)) {
        $identity.size_bytes=$value
        Assert-ProfileReject { Assert-SessionProfileIdentity $identity } 'unsigned integer'
    }
    Assert-ProfileTest ((Get-Content -LiteralPath $file -Raw) -eq 'owned proof') 'identity rejection preserves evidence'
    Assert-ProfileTest ((Quote-SessionProfileLiteral "a'b") -eq "'a''b'") 'literal quoting'

    $binary=[pscustomobject]@{sha256='f'*64;size_bytes=1}
    $model=[pscustomobject]@{sha256='b'*64;size_bytes=1}
    $calls=@(for($index=0;$index -lt 16;$index++){[pscustomobject]@{input_tokens=if($index -eq 0){8}else{1};committed_sequence_tokens=8+$index;forward_ms=1.;caller_readback_ms=0.;argmax_ms=0.;cuda_final_validation_ms=$null}})
    $progress=[pscustomobject]@{schema='candle.gpt-oss.session-profile-progress.v1';status='completed';input_sha256='e'*64;loads_attempted=1;requests_attempted=1;requests_completed=1;attempted_forward_calls=16;completed_forward_calls=16;attempted_token_steps_upper_bound=23;completed_token_steps=23;generated_token_ids=@(1)*16;calls=$calls;load_ms=1.;unload_ms=1.;total_ms=20.;static_weight_bytes=1;registry_entries_after_release=0;pid=123;error=$null;consumed_files=@([pscustomobject]@{sha256=$model.sha256;size_bytes=$model.size_bytes})}
    $job=[pscustomobject]@{contract='candle-lfm2-vl-bounded-oracle-v1';child_exit_code=0;pid_absent_after_cleanup=$true;assigned_before_resume=$true;job_assigned=$true;started_suspended=$true;resumed=$true;termination_reason='exited';error=$null;max_job_memory_bytes=34359738368;timeout_seconds=1800;executable_sha256=$binary.sha256;executable_bytes=1;pid=123;peak_private_memory_bytes=1;peak_job_memory_bytes=1}
    Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu'
    Assert-ProfileTest $true 'complete exact receipt'
    $progress.status='running'
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'incomplete'
    $progress.status='completed';$progress.completed_forward_calls=15
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'incomplete'
    $progress.completed_forward_calls=16;$job.pid_absent_after_cleanup=$false
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'bounded Job receipt'
    $job.pid_absent_after_cleanup=$true;$job.max_job_memory_bytes=34359738369
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'bounded Job receipt'
    $job.max_job_memory_bytes=34359738368;$job.executable_sha256='a'*64
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'bounded Job receipt'
    $job.executable_sha256=$binary.sha256;$progress.consumed_files=@([pscustomobject]@{sha256='a'*64;size_bytes=1})
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'consumed model'
    $progress.consumed_files=@([pscustomobject]@{sha256=$model.sha256;size_bytes=$model.size_bytes});$progress.calls[1].committed_sequence_tokens=99
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'cache/work count'
    $progress.calls[1].committed_sequence_tokens=9;$progress.calls[1].forward_ms=[double]::NaN
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'invalid timing'
    $progress.calls[1].forward_ms=1.
    foreach($name in @('load_ms','unload_ms','total_ms')) {
        $original=$progress.$name
        foreach($value in @($true,'1',[double]::NaN,[double]::PositiveInfinity,-1,$null)) {
            $progress.$name=$value
            Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'invalid aggregate timing'
        }
        $progress.$name=$original
        $withoutTiming=$progress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
        $withoutTiming.PSObject.Properties.Remove($name)
        Assert-ProfileReject { Assert-SessionProfileCompletion $withoutTiming $job ('e'*64) $binary $model 'cpu' } $name
    }
    foreach($value in @($true,1.1,'1',0,-1,$null,([UInt64]$plan.max_job_memory_bytes+1))) {
        $progress.static_weight_bytes=$value
        Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
    }
    $progress.static_weight_bytes=1
    $progress.error='injected producer failure'
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'work/release receipt'
    $progress.error=$null
    $job.resumed=$false
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'bounded Job receipt'
    $job.resumed=$true
    foreach($value in @('timeout','memory_limit',$null)) {
        $job.termination_reason=$value
        Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'bounded Job receipt'
    }
    $job.termination_reason='exited';$job.error='injected Job failure'
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'bounded Job receipt'
    $job.error=$null
    foreach($name in @('loads_attempted','requests_attempted','requests_completed','attempted_forward_calls',
        'completed_forward_calls','attempted_token_steps_upper_bound','completed_token_steps','registry_entries_after_release','pid')) {
        $original=$progress.$name
        foreach($value in @($true,([double]$original+0.1),[string]$original,$null,-1)) {
            $progress.$name=$value
            Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
        }
        $progress.$name=$original
    }
    foreach($name in @('pid_absent_after_cleanup','assigned_before_resume','job_assigned','started_suspended','resumed')) {
        foreach($value in @('false','true',1,$null)) {
            $job.$name=$value
            Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'requires a boolean'
        }
        $job.$name=$true
    }
    foreach($name in @('child_exit_code','pid','executable_bytes','peak_private_memory_bytes','peak_job_memory_bytes','max_job_memory_bytes','timeout_seconds')) {
        $original=$job.$name
        $job.$name=[double]$original
        Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
        $job.$name=$original
    }
    foreach($value in @($true,'invalid',-1,[UInt64]4294967296,$null)) {
        $progress.generated_token_ids=@($value)*16
        Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
    }
    $progress.generated_token_ids=@(1)*16
    foreach($name in @('input_tokens','committed_sequence_tokens')) {
        $original=$progress.calls[0].$name
        $progress.calls[0].$name=$true
        Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
        $progress.calls[0].$name=$original
    }
    foreach($name in @('forward_ms','caller_readback_ms','argmax_ms')) {
        $original=$progress.calls[0].$name
        foreach($value in @($true,'1',[double]::PositiveInfinity,-1)) {
            $progress.calls[0].$name=$value
            Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'invalid timing'
        }
        $progress.calls[0].$name=$original
    }
    $progress.consumed_files=@([pscustomobject]@{sha256=$model.sha256;size_bytes='1'})
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
    $progress.consumed_files=@([pscustomobject]@{sha256=$model.sha256;size_bytes=$model.size_bytes})
    $binary.size_bytes=$true
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer'
    $binary.size_bytes=1
    $progress.calls | ForEach-Object { $_.cuda_final_validation_ms=0.5 }
    $progress.static_weight_bytes=[UInt64]$plan.max_gpu_bytes+1
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cuda' } 'unsigned integer'
    $progress.static_weight_bytes=1
    Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cuda'
    Assert-ProfileTest $true 'valid CUDA typed receipt'
    $progress.calls[0].cuda_final_validation_ms=$true
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cuda' } 'timing/backend'
    $progress.calls | ForEach-Object { $_.cuda_final_validation_ms=$null }
    # Real producer JSON may use Int64 counts and Double timings.
    $parsedProgress=$progress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $parsedJob=$job | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    Assert-SessionProfileCompletion $parsedProgress $parsedJob ('e'*64) $binary $model 'cpu'
    Assert-ProfileTest $true 'typed JSON round trip remains compatible'

    foreach($backend in @('cpu','cuda')) {
        $timedProgress=$progress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
        $timedJob=$job | ConvertTo-Json -Depth 8 | ConvertFrom-Json
        $timedProgress.calls | ForEach-Object { $_.cuda_final_validation_ms=if($backend -eq 'cuda'){1.}else{$null} }
        $timedProgress.total_ms=18.
        Assert-SessionProfileCompletion $timedProgress $timedJob ('e'*64) $binary $model $backend
        Assert-ProfileTest $true "$backend sequential sum; nested validation counted once"
        $roundingMs=(2+3*$plan.forward_calls+1)*$plan.timing_rounding_tolerance_ms
        $timedProgress.total_ms=18.-$roundingMs/2
        Assert-SessionProfileCompletion $timedProgress $timedJob ('e'*64) $binary $model $backend
        Assert-ProfileTest $true "$backend aggregate rounding inside tolerance"
        $timedProgress.total_ms=18.-2*$roundingMs
        Assert-ProfileReject { Assert-SessionProfileCompletion $timedProgress $timedJob ('e'*64) $binary $model $backend } 'aggregate timing is inconsistent'
        $timedProgress.total_ms=18.
        foreach($name in @('load_ms','unload_ms','total_ms','static_weight_bytes')) {
            $requiredProgress=$timedProgress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
            $requiredProgress.PSObject.Properties.Remove($name)
            Assert-ProfileReject { Assert-SessionProfileCompletion $requiredProgress $timedJob ('e'*64) $binary $model $backend } $name
            foreach($value in @($null,$true,'1',-1,[double]::PositiveInfinity)) {
                $requiredProgress=$timedProgress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
                $requiredProgress.$name=$value
                Assert-ProfileReject { Assert-SessionProfileCompletion $requiredProgress $timedJob ('e'*64) $binary $model $backend } 'aggregate timing|unsigned integer'
            }
        }
        if($backend -eq 'cuda') {
            $timedProgress.calls[0].cuda_final_validation_ms=1.+$plan.timing_rounding_tolerance_ms
            Assert-SessionProfileCompletion $timedProgress $timedJob ('e'*64) $binary $model $backend
            Assert-ProfileTest $true 'CUDA nested rounding inside tolerance'
            $timedProgress.calls[0].cuda_final_validation_ms=1.+3*$plan.timing_rounding_tolerance_ms
            Assert-ProfileReject { Assert-SessionProfileCompletion $timedProgress $timedJob ('e'*64) $binary $model $backend } 'timing/backend'
            $timedProgress.calls[0].cuda_final_validation_ms=1.
        }
        $roundTrip=$timedProgress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
        Assert-SessionProfileCompletion $roundTrip $timedJob ('e'*64) $binary $model $backend
        Assert-ProfileTest $true "$backend complete timing JSON round trip"
        $timedProgress.load_ms=[double]::MaxValue
        $timedProgress.unload_ms=[double]::MaxValue
        $timedProgress.total_ms=[double]::MaxValue
        Assert-ProfileReject { Assert-SessionProfileCompletion $timedProgress $timedJob ('e'*64) $binary $model $backend } 'aggregate timing is inconsistent'
    }
    $partialProgress=$progress | ConvertTo-Json -Depth 8 | ConvertFrom-Json
    $partialProgress.status='failed';$partialProgress.error='injected incomplete load'
    $partialProgress.load_ms=$null;$partialProgress.unload_ms=$null;$partialProgress.static_weight_bytes=$null
    Assert-ProfileReject { Assert-SessionProfileCompletion $partialProgress $job ('e'*64) $binary $model 'cpu' } 'unsigned integer|aggregate timing'
    Assert-ProfileTest ($partialProgress.status -eq 'failed' -and $partialProgress.error -eq 'injected incomplete load' -and $null -eq $partialProgress.load_ms) 'partial failure remains unchanged and incomplete'

    $script:visitedLanes=@()
    $originalLane=(Get-Command Invoke-SessionProfileLane).ScriptBlock
    function Invoke-SessionProfileLane { param($Lane,$Source,$Directory,$Helper,$RepoRoot); $script:visitedLanes += $Lane.backend; throw 'injected CPU failure' }
    $lanes=@([pscustomobject]@{backend='cpu';build_profile='release'},[pscustomobject]@{backend='cuda';build_profile='release'})
    Assert-ProfileReject { Invoke-SessionProfileBatch $lanes $source $taskRoot 'unused' 'unused' } 'injected CPU failure'
    Assert-ProfileTest ($script:visitedLanes.Count -eq 1 -and $script:visitedLanes[0] -eq 'cpu') 'CPU failure must prevent CUDA admission'
    $script:visitedLanes=@();$lanes[0].build_profile='debug'
    Assert-ProfileReject { Invoke-SessionProfileBatch $lanes $source $taskRoot 'unused' 'unused' } 'optimized release'
    Assert-ProfileTest ($script:visitedLanes.Count -eq 0) 'debug profile must reject before admission'
    Set-Item -LiteralPath Function:\Invoke-SessionProfileLane -Value $originalLane

    # Exercise the actual controller stop with a harmless native Job tree.
    # A unique copied cmd.exe avoids unrelated process-name ownership.
    $helper=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\lfm2-vl\run-bounded-oracle.ps1')).Path
    $probe=Join-Path $taskRoot 'session-profile-probe.exe'
    Copy-Item -LiteralPath $env:ComSpec -Destination $probe
    $nativeJob=Join-Path $taskRoot 'native-job.json'
    $nativeLog=Join-Path $taskRoot 'native-job.log'
    $command='& '+(Quote-SessionProfileLiteral $helper)+' -FilePath '+(Quote-SessionProfileLiteral $probe)+
        " -ArgumentList @('/d','/c','ping 127.0.0.1 -n 31 > nul') -TimeoutSeconds 15 -MaxJobMemoryBytes 268435456 -PollMilliseconds 50 -ConcurrencyScope Executable -WorkingDirectory "+(Quote-SessionProfileLiteral $taskRoot)+
        ' -EvidencePath '+(Quote-SessionProfileLiteral $nativeJob)+' -LogPath '+(Quote-SessionProfileLiteral $nativeLog)
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $controller=$null
    $ownedProcesses=@()
    try {
        $controller=Start-Process -FilePath (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'controller.stdout.log') -RedirectStandardError (Join-Path $taskRoot 'controller.stderr.log')
        Write-SessionProfileJson (Join-Path $taskRoot 'owner.json') ([ordered]@{owner='model-free session profiling test';pid=$controller.Id;creation_utc=$controller.StartTime.ToUniversalTime().ToString('o');executable=$controller.Path;intended_lifetime='15-second harmless native Job';endpoints=@();output_root=$taskRoot;shutdown='Stop exact owned controller handle; native Job kills only test probe tree'})
        $timer=[Diagnostics.Stopwatch]::StartNew()
        $native=$null
        while($null -eq $native -and $timer.Elapsed.TotalSeconds -lt 10){
            $native=Get-CimInstance Win32_Process -Filter "Name='session-profile-probe.exe'" | Where-Object {$_.ExecutablePath -eq $probe} | Select-Object -First 1
            if($null -ne $native -and @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($native.ProcessId)" | Where-Object {$_.Name -ieq 'ping.exe'}).Count -eq 0){$native=$null}
            if($null -eq $native){Start-Sleep -Milliseconds 100}
        }
        Assert-ProfileTest ($null -ne $native) 'native harmless Job probe started'
        $ownedProcesses=@($native)+@(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($native.ProcessId)")
        Stop-SessionProfileController $controller
        $timer.Restart()
        $remaining=@()
        do {
            $remaining=@(foreach($owned in $ownedProcesses){$live=Get-CimInstance Win32_Process -Filter "ProcessId=$($owned.ProcessId)";if($null -ne $live -and $live.CreationDate -eq $owned.CreationDate -and $live.ExecutablePath -eq $owned.ExecutablePath){$live}})
            if($remaining.Count -gt 0){Start-Sleep -Milliseconds 100}
        } while($remaining.Count -gt 0 -and $timer.Elapsed.TotalSeconds -lt 5)
        Assert-ProfileTest ($remaining.Count -eq 0) 'controller termination released native Job tree'
    } finally {
        if($null -ne $controller){Stop-SessionProfileController $controller;$controller.Dispose()}
    }
    $taskSuccess=$true
    Write-Output "session-profile model-free tests: $script:taskPassed passed"
} finally {
    $resolved=[IO.Path]::GetFullPath($taskRoot)
    $allowed=([IO.Path]::GetFullPath([IO.Path]::GetTempPath())).TrimEnd('\')+'\'
    if(-not $resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notlike 'candle-session-profile-tests-*'){throw 'Owned test root escaped temp scope'}
    if($taskSuccess){Remove-Item -LiteralPath $resolved -Recurse -Force}else{Write-Output "Retained unresolved test evidence: $resolved"}
}
