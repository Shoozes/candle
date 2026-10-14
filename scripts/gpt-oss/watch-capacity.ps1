[CmdletBinding()]
param([string]$CapacityPacketPath,[string]$CapacityObservationPath,[switch]$AsLibrary)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$capacityWatchLibraryOnly=[bool]$AsLibrary
. (Join-Path $PSScriptRoot 'observer.ps1')
if($capacityWatchLibraryOnly){return}
if(-not $IsWindows -or $PSVersionTable.PSVersion.Major -lt 7){throw 'Capacity watcher requires native PowerShell7'}
if([string]::IsNullOrWhiteSpace($CapacityPacketPath) -or [string]::IsNullOrWhiteSpace($CapacityObservationPath)){throw 'Capacity watcher needs packet/observation'}
if((Get-Item -LiteralPath $CapacityPacketPath).Length -gt 65536){throw 'Capacity watcher packet too large'}
$capacityWatchData=Get-Content -LiteralPath $CapacityPacketPath -Raw|ConvertFrom-Json
if($capacityWatchData.schema -ne 'candle.gpt-oss.capacity-packet.v1' -or @($capacityWatchData.lanes).Count -ne 1 -or $capacityWatchData.lanes[0].backend -ne 'cuda'){throw 'Capacity watcher packet version/lane mismatch'}
$capacityWatchLane=$capacityWatchData.lanes[0];$capacityWatchRoot=$capacityWatchData.proposed_output_root
$capacityWatchDir=Join-Path $capacityWatchRoot 'cuda';$capacityWatchOut=[IO.Path]::GetFullPath($CapacityObservationPath)
$capacityAllowed=(Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path 'artifacts')+'\'
if(-not $capacityWatchOut.StartsWith($capacityAllowed,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $capacityWatchOut) -or (Test-Path -LiteralPath $capacityWatchDir)){throw 'Capacity watcher requires fresh owned output before lane'}
$capacityWatchSelf=Get-Process -Id $PID
$capacityWatchResult=[ordered]@{schema='candle.gpt-oss.runtime-observation.v1';status='armed';error=$null;source_commit=$capacityWatchData.source_commit;
    packet_sha256=(Get-FileHash -LiteralPath $CapacityPacketPath).Hash.ToLowerInvariant();owner=[ordered]@{pid=$PID;creation_utc=$capacityWatchSelf.StartTime.ToUniversalTime().ToString('o');executable=$capacityWatchSelf.Path};
    backend='cuda';intended_lifetime='bounded passive capacity observation,at most1860s';endpoints=@();shutdown='parent stops exact handle; watcher never launches/stops models';
    poll_milliseconds=250;deadline_seconds=1860;started_utc=[DateTimeOffset]::UtcNow.ToString('o');ended_utc=$null;identity=$null;modules=@();process_probe_ms=0;module_probe_calls=0}
function Write-CapacityWatch {
    [IO.File]::WriteAllText($capacityWatchOut,($capacityWatchResult|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
}
$capacityReservation=[IO.File]::Open($capacityWatchOut,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);$capacityReservation.Dispose()
Write-CapacityWatch
$capacityWatchClock=[Diagnostics.Stopwatch]::StartNew();$capacityWatchExit=1
try{
    while($capacityWatchClock.Elapsed.TotalSeconds -le 1860){
        $progressPath=Join-Path $capacityWatchDir 'progress.json'
        if(Test-Path -LiteralPath $progressPath){
            $observedProgress=$null
            if((Get-Item -LiteralPath $progressPath).Length -gt 16384){throw 'Capacity progress too large'}
            try{$observedProgress=Get-Content -LiteralPath $progressPath -Raw|ConvertFrom-Json}catch{$observedProgress=$null}
            if($null -ne $observedProgress){
                if($observedProgress.source_commit -ne $capacityWatchData.source_commit -or $observedProgress.executable -ne $capacityWatchLane.binary.path){throw 'Capacity progress source/executable mismatch'}
                $probe=[Diagnostics.Stopwatch]::StartNew();$snapshot=Get-SessionProfileProcessSnapshot ([int]$observedProgress.pid) 'modules'
                $capacityWatchResult.process_probe_ms+=$probe.Elapsed.TotalMilliseconds;$capacityWatchResult.module_probe_calls++
                if($null -ne $snapshot){
                    if(-not [string]::Equals($snapshot.executable,$capacityWatchLane.binary.path,[StringComparison]::OrdinalIgnoreCase)){throw 'Capacity observed executable mismatch'}
                    $identity=[ordered]@{pid=$snapshot.pid;creation_utc=$snapshot.creation_utc;executable=$snapshot.executable}
                    if($null -eq $capacityWatchResult.identity){$capacityWatchResult.identity=$identity}
                    elseif($identity.pid -ne $capacityWatchResult.identity.pid -or $identity.creation_utc -ne $capacityWatchResult.identity.creation_utc -or $identity.executable -ine $capacityWatchResult.identity.executable){throw 'Capacity observed owner identity changed'}
                    foreach($module in $snapshot.modules){
                        if([string]::IsNullOrWhiteSpace($module.name) -or [string]::IsNullOrWhiteSpace($module.path)){throw 'Capacity observed module identity incomplete'}
                        if($module.path -notin @($capacityWatchResult.modules|ForEach-Object path)){$capacityWatchResult.modules+=[ordered]@{name=$module.name;path=$module.path;observed_utc=[DateTimeOffset]::UtcNow.ToString('o')};if($capacityWatchResult.modules.Count -gt 256){throw 'Capacity modules exceed bound'}}
                    }
                    Write-CapacityWatch
                }
            }
        }
        $terminalPath=Join-Path $capacityWatchDir 'terminal.json'
        if(Test-Path -LiteralPath $terminalPath){
            if((Get-Item -LiteralPath $terminalPath).Length -gt 524288){throw 'Capacity terminal too large'}
            $observedTerminal=$null;try{$observedTerminal=Get-Content -LiteralPath $terminalPath -Raw|ConvertFrom-Json}catch{}
            if($null -ne $observedTerminal){if($observedTerminal.status -ne 'completed'){throw 'Capacity model lane incomplete'};$capacityWatchResult.status='observed';$capacityWatchExit=0;break}
        }
        Start-Sleep -Milliseconds 250
    }
    if($capacityWatchExit -ne 0){throw 'Capacity watcher deadline expired'}
}catch{$capacityWatchResult.status='incomplete';$capacityWatchResult.error=$_.Exception.Message}
finally{$capacityWatchResult.ended_utc=[DateTimeOffset]::UtcNow.ToString('o');Write-CapacityWatch;$capacityWatchSelf.Dispose()}
exit $capacityWatchExit
