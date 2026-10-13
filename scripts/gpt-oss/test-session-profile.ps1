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

    $file=Join-Path $taskRoot 'identity.txt'
    [IO.File]::WriteAllText($file,'owned proof')
    $identity=[pscustomobject]@{path=$file;size_bytes=11;sha256=(Get-FileHash -LiteralPath $file).Hash.ToLowerInvariant()}
    Assert-ProfileTest ((Assert-SessionProfileIdentity $identity) -eq $file) 'exact file identity'
    $identity.size_bytes=12
    Assert-ProfileReject { Assert-SessionProfileIdentity $identity } 'size mismatch'
    $identity.size_bytes=11; $identity.sha256='d'*64
    Assert-ProfileReject { Assert-SessionProfileIdentity $identity } 'SHA-256 mismatch'
    Assert-ProfileTest ((Get-Content -LiteralPath $file -Raw) -eq 'owned proof') 'identity rejection preserves evidence'
    Assert-ProfileTest ((Quote-SessionProfileLiteral "a'b") -eq "'a''b'") 'literal quoting'

    $binary=[pscustomobject]@{sha256='f'*64;size_bytes=1}
    $model=[pscustomobject]@{sha256='b'*64;size_bytes=1}
    $calls=@(for($index=0;$index -lt 16;$index++){[pscustomobject]@{input_tokens=if($index -eq 0){8}else{1};committed_sequence_tokens=8+$index;forward_ms=1.;caller_readback_ms=0.;argmax_ms=0.;cuda_final_validation_ms=$null}})
    $progress=[pscustomobject]@{schema='candle.gpt-oss.session-profile-progress.v1';status='completed';input_sha256='e'*64;loads_attempted=1;requests_attempted=1;requests_completed=1;attempted_forward_calls=16;completed_forward_calls=16;attempted_token_steps_upper_bound=23;completed_token_steps=23;generated_token_ids=@(1)*16;calls=$calls;registry_entries_after_release=0;pid=123;consumed_files=@($model)}
    $job=[pscustomobject]@{contract='candle-lfm2-vl-bounded-oracle-v1';child_exit_code=0;pid_absent_after_cleanup=$true;assigned_before_resume=$true;job_assigned=$true;started_suspended=$true;max_job_memory_bytes=34359738368;timeout_seconds=1800;executable_sha256=$binary.sha256;executable_bytes=1;pid=123;peak_private_memory_bytes=1;peak_job_memory_bytes=1}
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
    $progress.consumed_files=@($model);$progress.calls[1].committed_sequence_tokens=99
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'cache/work count'
    $progress.calls[1].committed_sequence_tokens=9;$progress.calls[1].forward_ms=[double]::NaN
    Assert-ProfileReject { Assert-SessionProfileCompletion $progress $job ('e'*64) $binary $model 'cpu' } 'invalid timing'

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
