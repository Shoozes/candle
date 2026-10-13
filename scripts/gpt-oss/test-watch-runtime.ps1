Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-characterization.ps1') -AsLibrary
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$root=Join-Path $repo ('artifacts\gpt-oss\watcher-fixture-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root|Out-Null
$watcher=$null;$worker=$null;$passed=0;$success=$false
function Assert-WatcherTest([bool]$Condition,[string]$Message){if(-not $Condition){throw "Watcher test failed: $Message"};$script:passed++}
try {
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
    $wait=[Diagnostics.Stopwatch]::StartNew()
    while(-not (Test-Path -LiteralPath $observation) -and $wait.Elapsed.TotalSeconds -lt 5){Start-Sleep -Milliseconds 50}
    Assert-WatcherTest (Test-Path -LiteralPath $observation) 'watcher arms before fake lane exists'
    $initial=Get-Content $observation -Raw|ConvertFrom-Json
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
    $success=$true
    "runtime-watcher model-free controls: $script:passed passed"
} finally {
    if($null -ne $worker){Stop-SessionProfileController $worker;$worker.Dispose()}
    if($null -ne $watcher){Stop-SessionProfileController $watcher;$watcher.Dispose()}
    $resolved=[IO.Path]::GetFullPath($root);$allowed=(Join-Path $repo 'artifacts\gpt-oss')+'\'
    if(-not $resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $resolved) -notlike 'watcher-fixture-*'){throw 'Watcher scratch escaped owned scope'}
    if($success){Remove-Item -LiteralPath $resolved -Recurse -Force}else{"Retained watcher failure evidence: $resolved"}
}
