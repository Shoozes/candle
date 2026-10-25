Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-characterization.ps1') -AsLibrary
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$root=Join-Path $repo ('artifacts\gpt-oss\watcher-fixture-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root|Out-Null
$watcher=$null;$worker=$null;$passed=0;$success=$false
function Assert-WatcherTest([bool]$Condition,[string]$Message){if(-not $Condition){throw "Watcher test failed: $Message"};$script:passed++}
function Wait-WatcherTestArmed {
    param([Diagnostics.Process]$Process,[string]$Path,[ValidateRange(1,5000)][int]$TimeoutMilliseconds=5000)
    $wait=[Diagnostics.Stopwatch]::StartNew()
    while($wait.Elapsed.TotalMilliseconds -lt $TimeoutMilliseconds) {
        $Process.Refresh();if($Process.HasExited){throw 'Watcher exited before arming'}
        if(Test-Path -LiteralPath $Path -PathType Leaf) {
            # The create-new reservation may still be empty or partially written.
            try {
                $state=Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json -ErrorAction Stop
                if($null -ne $state -and $state.schema -eq 'candle.gpt-oss.runtime-observation.v1' -and $state.status -eq 'armed'){return $state}
            } catch {}
        }
        Start-Sleep -Milliseconds 50
    }
    throw 'Watcher did not arm within bound'
}
try {
    $readinessFile=Join-Path $root 'readiness.json';$readinessProcess=Get-Process -Id $PID
    try {
        foreach($pending in @('', '{"schema":')) {
            [IO.File]::WriteAllText($readinessFile,$pending,[Text.UTF8Encoding]::new($false))
            $caught=$false;try{Wait-WatcherTestArmed $readinessProcess $readinessFile 100|Out-Null}catch{$caught=$_.Exception.Message -eq 'Watcher did not arm within bound'}
            Assert-WatcherTest $caught 'empty or partial reservation cannot prove arming'
        }
        Write-SessionProfileJson $readinessFile ([ordered]@{schema='candle.gpt-oss.runtime-observation.v1';status='armed'})
        $ready=Wait-WatcherTestArmed $readinessProcess $readinessFile 100
        Assert-WatcherTest ($ready.status -eq 'armed') 'completed reservation proves arming'
    } finally {$readinessProcess.Dispose()}
    $shell=(Get-Command pwsh).Source;$source='a'*40
    $output=Join-Path $root 'fake-output';$packetFile=Join-Path $root 'packet input.json';$observation=Join-Path $root 'observation.json'
    $lane=[pscustomobject]@{backend='cpu';binary=[pscustomobject]@{path=$shell};required_modules=@()}
    $testPacket=[pscustomobject]@{schema='candle.gpt-oss.characterization-packet.v1';source_commit=$source;proposed_output_root=$output;lanes=@($lane);fixture_only=$true}
    Write-SessionProfileJson $packetFile $testPacket
    $watchScript=Join-Path $PSScriptRoot 'watch-runtime.ps1'
    $command='& '+(Quote-SessionProfileLiteral $watchScript)+' -PacketPath '+(Quote-SessionProfileLiteral $packetFile)+
        ' -ObservationPath '+(Quote-SessionProfileLiteral $observation)+" -Backend cpu -DeadlineSeconds 15"
    $encoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
    $watcher=Start-Process $shell -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$encoded) -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root 'watcher.stdout.log') -RedirectStandardError (Join-Path $root 'watcher.stderr.log')
    $owner=[ordered]@{owner='model-free watcher test';pid=$watcher.Id;creation_utc=$watcher.StartTime.ToUniversalTime().ToString('o');executable=$watcher.Path;intended_lifetime='15-second passive fixture';endpoints=@();output_root=$root;shutdown='Stop exact owned watcher and fake worker handles'}
    Write-SessionProfileJson (Join-Path $root 'owner.json') $owner
    $initial=Wait-WatcherTestArmed $watcher $observation
    Assert-WatcherTest ($initial.schema -eq 'candle.gpt-oss.runtime-observation.v1') 'watcher arms before fake lane exists'
    Assert-WatcherTest ($initial.status -eq 'armed') 'armed record'
    $fakeScript=Join-Path $root 'fake-worker.ps1'
    [IO.File]::WriteAllText($fakeScript,@'
param([string]$Root,[string]$Source)
$lane=Join-Path $Root 'cpu';New-Item -ItemType Directory -Path $lane|Out-Null
$me=Get-Process -Id $PID
$progress=[ordered]@{schema='fixture-only';source_commit=$Source;pid=$PID;executable=$me.Path}
[IO.File]::WriteAllText((Join-Path $lane 'progress.json'),($progress|ConvertTo-Json))
Start-Sleep -Seconds 4
[IO.File]::WriteAllText((Join-Path $lane 'terminal.json'),' {"status":"completed"}')
'@,[Text.UTF8Encoding]::new($false))
    $fakeCommand='& '+(Quote-SessionProfileLiteral $fakeScript)+' -Root '+(Quote-SessionProfileLiteral $output)+' -Source '+(Quote-SessionProfileLiteral $source)
    $fakeEncoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($fakeCommand))
    $worker=Start-Process $shell -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$fakeEncoded) -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root 'worker.stdout.log') -RedirectStandardError (Join-Path $root 'worker.stderr.log')
    Assert-WatcherTest ($watcher.WaitForExit(15000) -and $watcher.ExitCode -eq 0) 'bounded watcher completes'
    Assert-WatcherTest ($worker.WaitForExit(5000) -and $worker.ExitCode -eq 0) 'fake worker exits'
    $runtime=Get-Content $observation -Raw|ConvertFrom-Json
    Assert-WatcherTest ($runtime.status -eq 'observed' -and $null -eq $runtime.error) 'observation result'
    Assert-WatcherTest ($runtime.identity.pid -eq $worker.Id -and $runtime.identity.executable -eq $shell -and $runtime.modules.Count -gt 0) 'actual fixture process identity and modules'
    Assert-WatcherTest ($runtime.module_probe_calls -ge 1 -and $runtime.process_probe_ms -gt 0) 'observer overhead recorded separately'
    $module=@($runtime.modules|Where-Object name -ieq 'kernel32.dll')[0]
    $required=[pscustomobject]@{name=$module.name;path=$module.path;size_bytes=(Get-Item -LiteralPath $module.path).Length;sha256=(Get-FileHash -LiteralPath $module.path).Hash.ToLowerInvariant()}
    $lane.required_modules=@($required);$progress=[pscustomobject]@{pid=$worker.Id}
    Assert-CharacterizationObservedModules $runtime.modules $lane.required_modules
    Assert-WatcherTest $true 'actual loaded-path and retained bytes verification'
    $caught=$false;try{Assert-CharacterizationRuntimeClosure $lane}catch{$caught=$_.Exception.Message -match 'closure must bind'}
    Assert-WatcherTest $caught 'kernel32-only mechanical observation cannot qualify a maintained runtime'
    $lane.required_modules=@([pscustomobject]@{name='missing.dll';path='never-loaded.dll';size_bytes=1;sha256='b'*64})
    $caught=$false;try{Assert-CharacterizationObservedModules $runtime.modules $lane.required_modules}catch{$caught=$_.Exception.Message -match 'not observed loaded'}
    Assert-WatcherTest $caught 'installed-or-inventoried-only module cannot qualify'
    $lane.required_modules=@($required);$runtime.identity.pid++
    $caught=$false;try{Assert-CharacterizationRuntime $runtime $testPacket $lane $progress (Get-FileHash -LiteralPath $packetFile).Hash.ToLowerInvariant()}catch{$caught=$_.Exception.Message -match 'mismatched'}
    Assert-WatcherTest $caught 'cross-process observation refused'
    $timeoutOutput=Join-Path $root 'timeout-output';$testPacket.proposed_output_root=$timeoutOutput
    $timeoutPacket=Join-Path $root 'timeout-packet.json';Write-SessionProfileJson $timeoutPacket $testPacket
    $timeoutObservation=Join-Path $root 'timeout-observation.json'
    $timed=Invoke-SessionProfileProbe $shell @('-NoProfile','-File',$watchScript,'-PacketPath',$timeoutPacket,'-ObservationPath',$timeoutObservation,'-Backend','cpu','-DeadlineSeconds','1') 5000 4096
    $failure=Get-Content $timeoutObservation -Raw|ConvertFrom-Json
    Assert-WatcherTest ($timed.ExitCode -ne 0 -and $failure.status -eq 'incomplete' -and $failure.error -match 'deadline') 'deadline preserves incomplete evidence'
    $failedOutput=Join-Path $root 'failed-output';$testPacket.proposed_output_root=$failedOutput
    $failedPacket=Join-Path $root 'failed-packet.json';Write-SessionProfileJson $failedPacket $testPacket
    $failedObservation=Join-Path $root 'failed-observation.json'
    $failedCommand='& '+(Quote-SessionProfileLiteral $watchScript)+' -PacketPath '+(Quote-SessionProfileLiteral $failedPacket)+
        ' -ObservationPath '+(Quote-SessionProfileLiteral $failedObservation)+' -Backend cpu -DeadlineSeconds 10'
    $failedEncoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($failedCommand))
    $watcher.Dispose()
    $watcher=Start-Process $shell -ArgumentList @('-NoProfile','-NonInteractive','-EncodedCommand',$failedEncoded) -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $root 'failed.stdout.log') -RedirectStandardError (Join-Path $root 'failed.stderr.log')
    $failedIdentity=Get-SessionProfileOwnerIdentity $watcher.Id $watcher.StartTime $shell
    $failedArmed=Wait-WatcherTestArmed $watcher $failedObservation
    Assert-WatcherTest ($failedArmed.status -eq 'armed') 'failure watcher arms before lane exists'
    New-Item -ItemType Directory -Path (Join-Path $failedOutput 'cpu')|Out-Null
    Write-SessionProfileJson (Join-Path $failedOutput 'cpu/terminal.json') ([ordered]@{status='failed';error='authored GPU rejection'})
    $failedStop=Stop-SessionProfileController $watcher -Identity $failedIdentity -GraceMilliseconds 5000 -PassThru
    $failedRuntime=Get-Content $failedObservation -Raw|ConvertFrom-Json
    Assert-WatcherTest ($failedStop.exited -and $failedStop.exit_code -eq 1 -and -not $failedStop.fallback_kill -and
        $null -eq $failedStop.error -and $failedRuntime.status -eq 'incomplete' -and
        $null -ne $failedRuntime.error -and $null -ne $failedRuntime.ended_utc) 'failed lane lets watcher retain incomplete terminal and exit'
    $success=$true
    "runtime-watcher model-free controls: $script:passed passed"
} finally {
    if($null -ne $worker){Stop-SessionProfileController $worker;$worker.Dispose()}
    if($null -ne $watcher){Stop-SessionProfileController $watcher;$watcher.Dispose()}
    $resolved=[IO.Path]::GetFullPath($root);$allowed=(Join-Path $repo 'artifacts\gpt-oss')+'\'
    if(-not $resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notlike 'watcher-fixture-*'){throw 'Watcher scratch escaped owned scope'}
    if($success){Remove-Item -LiteralPath $resolved -Recurse -Force}else{"Retained watcher failure evidence: $resolved"}
}
