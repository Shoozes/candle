Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-session-profile.ps1') -AsLibrary
$testRoot=Join-Path ([IO.Path]::GetTempPath()) ('candle-observer-tests-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$script:passed=0
$success=$false
function Test-ObserverAssert([bool]$Condition,[string]$Message) {
    if(-not $Condition){throw "Observer test failed: $Message"};$script:passed++
}
function Test-ObserverReject([scriptblock]$Operation,[string]$Pattern) {
    $failure=$null;try { & $Operation | Out-Null } catch { $failure=$_.Exception.Message }
    Test-ObserverAssert ($null -ne $failure -and $failure -match $Pattern) "Expected $Pattern; observed $failure"
}
function Test-ObserverAbsent($Identity) {
    $live=Get-Process -Id $Identity.pid -ErrorAction SilentlyContinue
    if($null -eq $live){return $true}
    try { return $live.StartTime.ToUniversalTime().ToString('o') -ne $Identity.creation_utc -or $live.Path -ne $Identity.executable }
    finally { $live.Dispose() }
}
try {
    $shell=(Get-Command pwsh).Source
    # Deterministically reproduce exit during module discovery in a bounded child.
    $moduleFixture=Join-Path $testRoot 'module-exit-fixture.ps1'
    [IO.File]::WriteAllText($moduleFixture,@'
param([string]$Worker,[string]$Case)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$global:observedFixture=[pscustomobject]@{Id=12345;StartTime=[DateTime]::UnixEpoch;
    Path='C:\fixture\worker.exe';HasExited=$Case.StartsWith('exited');ChangeIdentity=$Case -eq 'live-changed';UnknownRefresh=$Case -eq 'unknown-refresh'}
if($Case -eq 'unknown-exit'){$global:observedFixture|Add-Member -Force ScriptProperty HasExited {throw 'fixture exit-state error'}}
if($Case.EndsWith('error')) {
    $global:observedFixture|Add-Member ScriptProperty Modules {throw 'fixture module access error'}
} else {
    $moduleValue=if($Case.EndsWith('empty') -or $Case.StartsWith('unknown')){$null}else{[pscustomobject]@{ModuleName='fixture.dll';FileName='C:\fixture\fixture.dll'}}
    $global:observedFixture|Add-Member NoteProperty Modules $moduleValue
}
$global:observedFixture|Add-Member ScriptMethod Refresh {if($this.UnknownRefresh){throw 'fixture refresh error'};if($this.ChangeIdentity){$this.StartTime=$this.StartTime.AddSeconds(1)}}
$global:observedFixture|Add-Member ScriptMethod Dispose {}
function Get-Process {param($Id,$ErrorAction) $global:observedFixture}
& $Worker -TargetPid 12345 -Mode modules
'@,[Text.UTF8Encoding]::new($false))
    $moduleWorker=Join-Path $PSScriptRoot 'observer-worker.ps1'
    foreach($moduleCase in @('exited-empty','exited-error','live-empty','live-error','live-changed','live-valid','unknown-refresh','unknown-exit')) {
        $moduleResult=Invoke-SessionProfileProbe $shell @('-NoProfile','-File',$moduleFixture,'-Worker',$moduleWorker,'-Case',$moduleCase) 5000 4096
        if($moduleCase.StartsWith('exited')) {
            Test-ObserverAssert ($moduleResult.ExitCode -eq 0 -and $moduleResult.Stdout.Trim() -eq 'null') "departed module snapshot returns null: $moduleCase"
        } elseif($moduleCase -eq 'live-valid') {
            $moduleSnapshot=$moduleResult.Stdout|ConvertFrom-Json
            Test-ObserverAssert ($moduleResult.ExitCode -eq 0 -and $moduleSnapshot.pid -eq 12345 -and $moduleSnapshot.modules.Count -eq 1 -and $moduleSnapshot.modules[0].name -eq 'fixture.dll') 'live module snapshot retains identity and modules'
        } else {
            $modulePattern=if($moduleCase -eq 'live-changed'){'identity changed'}else{'ModuleName|fixture module access error'}
            Test-ObserverAssert ($moduleResult.ExitCode -ne 0 -and $moduleResult.Stderr -match $modulePattern) "live observation remains fail-closed: $moduleCase ($($moduleResult.Stderr))"
        }
    }
    $valid=Get-SessionProfileGpuSample 0 $shell @('-NoProfile','-Command','[Console]::Out.WriteLine(42)') 3000
    Test-ObserverAssert ($valid.used_bytes -eq 42*1048576 -and $valid.observer_ms -ge 0) 'bounded unique GPU sample'
    foreach($code in @('exit 1','[Console]::Out.WriteLine("bad")','[Console]::Out.WriteLine("1`n2")','[Console]::Out.WriteLine("18446744073709551615")')) {
        Test-ObserverReject {Get-SessionProfileGpuSample 0 $shell @('-NoProfile','-Command',$code) 3000} 'GPU monitor'
    }
    Test-ObserverReject {Invoke-SessionProfileProbe $shell @('-NoProfile','-Command','[Console]::Out.Write("x"*65536)') 3000 4096} 'output.*bound|collection'
    Test-ObserverReject {Invoke-SessionProfileProbe $shell @('-NoProfile','-Command','[Console]::Error.Write("x"*65536)') 3000 4096} 'output.*bound|collection'
    $hangingScript=Join-Path $testRoot 'hang.ps1'
    $hangingOwner=Join-Path $testRoot 'hang-owner.json'
    [IO.File]::WriteAllText($hangingScript,@'
param([string]$Owner)
$self=Get-Process -Id $PID
$child=Start-Process (Get-Command pwsh).Source -ArgumentList @('-NoProfile','-Command','Start-Sleep -Seconds 30') -PassThru -WindowStyle Hidden
$ids=@($self,$child)|ForEach-Object{[ordered]@{pid=$_.Id;creation_utc=$_.StartTime.ToUniversalTime().ToString('o');executable=$_.Path}}
[IO.File]::WriteAllText($Owner,($ids|ConvertTo-Json))
Start-Sleep -Seconds 30
'@,[Text.UTF8Encoding]::new($false))
    Test-ObserverReject {Invoke-SessionProfileProbe $shell @('-NoProfile','-File',$hangingScript,'-Owner',$hangingOwner) 1500 4096} 'deadline|cancel'
    $hangingIds=@(Get-Content -LiteralPath $hangingOwner -Raw|ConvertFrom-Json)
    Test-ObserverAssert ($hangingIds.Count -eq 2 -and @($hangingIds|Where-Object{-not (Test-ObserverAbsent $_)}).Count -eq 0) 'hanging probe and its child released'
    $selfSnapshot=Get-SessionProfileProcessSnapshot $PID
    Test-ObserverAssert ($selfSnapshot.pid -eq $PID -and $selfSnapshot.executable -eq $shell) 'bounded process identity discovery'
    Test-ObserverReject {Get-SessionProfileProcessSnapshot $PID 'identity' 50} 'deadline|cancel'

    $ownerProcess=Get-Process -Id $PID
    try {
        # Start-Process objects can have an unavailable adapted Path property.
        $parentShape=[pscustomobject]@{Id=$ownerProcess.Id;StartTime=$ownerProcess.StartTime;Path=$null}
        $boundOwner=Get-SessionProfileOwnerIdentity $parentShape.Id $parentShape.StartTime $shell
        Test-ObserverAssert ($null -eq $parentShape.Path -and $boundOwner.pid -eq $PID -and $boundOwner.executable -eq $shell) 'owner path comes from bounded live identity despite unavailable parent Path'
        Test-ObserverReject {Get-SessionProfileOwnerIdentity $parentShape.Id $parentShape.StartTime $env:ComSpec} 'launcher identity mismatch'
        Test-ObserverReject {Get-SessionProfileOwnerIdentity $parentShape.Id ([DateTime]::UnixEpoch) $shell} 'launcher identity mismatch'
    } finally {$ownerProcess.Dispose()}

    # Exercise the real lane failure/finally path while a harmless Job is active.
    $probe=Join-Path $testRoot 'profile-job-probe.exe'
    Copy-Item -LiteralPath $env:ComSpec -Destination $probe
    $helper=(Resolve-Path (Join-Path $PSScriptRoot '..\lfm2-vl\run-bounded-oracle.ps1')).Path
    $frozen=(Get-Content (Join-Path $PSScriptRoot '..\..\tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw|ConvertFrom-Json).reference_prefix_token_ids
    $inputPath=Join-Path $testRoot 'input.json'
    $testSource='a'*40
    Write-SessionProfileJson $inputPath ([ordered]@{schema='candle.gpt-oss.session-profile-input.v1';source_commit=$testSource;backend='cuda';device_index=0;model=@{path='never-opened.gguf';size_bytes=1;sha256='b'*64};prompt_token_ids=$frozen;generated_tokens=16})
    function Test-ObserverFileIdentity([string]$Path) { [pscustomobject]@{path=$Path;size_bytes=(Get-Item -LiteralPath $Path).Length;sha256=(Get-FileHash -LiteralPath $Path).Hash.ToLowerInvariant()} }
    $lane=[pscustomobject]@{backend='cuda';build_profile='release';input=(Test-ObserverFileIdentity $inputPath);binary=(Test-ObserverFileIdentity $probe)}
    $activeRoot=Join-Path $testRoot 'active'
    $originalSample=(Get-Command Get-SessionProfileGpuSample).ScriptBlock
    $script:samples=0
    function Get-SessionProfileGpuSample {
        param($DeviceIndex)
        $script:samples++
        if($script:samples -eq 1){return [pscustomobject]@{used_bytes=0;observer_ms=0;observed_utc=[DateTimeOffset]::UtcNow.ToString('o')}}
        $wait=[Diagnostics.Stopwatch]::StartNew()
        do {
            $script:ownedJobProcesses=@(Get-Process -Name 'profile-job-probe' -ErrorAction SilentlyContinue|Where-Object{$_.Path -eq $probe}|ForEach-Object{[pscustomobject]@{pid=$_.Id;creation_utc=$_.StartTime.ToUniversalTime().ToString('o');executable=$_.Path}})
            if($script:ownedJobProcesses.Count -eq 0){Start-Sleep -Milliseconds 50}
        } while($script:ownedJobProcesses.Count -eq 0 -and $wait.Elapsed.TotalSeconds -lt 5)
        if($script:ownedJobProcesses.Count -eq 0){throw 'Harmless Job did not start'}
        Invoke-SessionProfileProbe $shell @('-NoProfile','-File',$hangingScript,'-Owner',$hangingOwner) 1500 4096 | Out-Null
    }
    function Start-Process {
        param($FilePath,$ArgumentList,[switch]$PassThru,$WindowStyle,$RedirectStandardOutput,$RedirectStandardError)
        $command='& '+(Quote-SessionProfileLiteral $helper)+' -FilePath '+(Quote-SessionProfileLiteral $probe)+
            " -ArgumentList @('/d','/c','ping 127.0.0.1 -n 31 > nul') -TimeoutSeconds 15 -MaxJobMemoryBytes 268435456 -PollMilliseconds 50 -ConcurrencyScope Executable -WorkingDirectory "+(Quote-SessionProfileLiteral $testRoot)+
            ' -EvidencePath '+(Quote-SessionProfileLiteral (Join-Path $testRoot 'active-job.json'))
        $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
        Microsoft.PowerShell.Management\Start-Process -FilePath $FilePath -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -PassThru -WindowStyle Hidden -RedirectStandardOutput $RedirectStandardOutput -RedirectStandardError $RedirectStandardError
    }
    Test-ObserverReject {Invoke-SessionProfileLane $lane $testSource $activeRoot $helper (Split-Path $PSScriptRoot -Parent)} 'deadline|cancel'
    $terminal=Get-Content (Join-Path $activeRoot 'terminal.json') -Raw|ConvertFrom-Json
    Test-ObserverAssert ($terminal.status -eq 'failed' -and $null -ne $terminal.error) 'observer failure retains incomplete terminal'
    $owner=Get-Content (Join-Path $activeRoot 'owner.json') -Raw|ConvertFrom-Json
    Test-ObserverAssert (Test-ObserverAbsent $owner) 'active controller released after probe failure'
    Test-ObserverAssert (@($script:ownedJobProcesses|Where-Object{-not (Test-ObserverAbsent $_)}).Count -eq 0) 'active harmless Job released after probe failure'
    $hangingIds=@(Get-Content -LiteralPath $hangingOwner -Raw|ConvertFrom-Json)
    Test-ObserverAssert (@($hangingIds|Where-Object{-not (Test-ObserverAbsent $_)}).Count -eq 0) 'active failing observer tree released'
    Set-Item Function:\Get-SessionProfileGpuSample -Value $originalSample
    Remove-Item Function:\Start-Process
    $success=$true
    "session-profile observer controls: $script:passed passed"
} finally {
    $resolved=[IO.Path]::GetFullPath($testRoot)
    $allowed=([IO.Path]::GetFullPath([IO.Path]::GetTempPath())).TrimEnd('\')+'\'
    if(-not $resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notlike 'candle-observer-tests-*'){throw 'Observer test root escaped owned scope'}
    if($success){Remove-Item -LiteralPath $resolved -Recurse -Force}else{"Retained unresolved observer evidence: $resolved"}
}
