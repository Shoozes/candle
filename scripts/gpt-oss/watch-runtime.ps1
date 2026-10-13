[CmdletBinding()]
param([string]$PacketPath,[string]$ObservationPath,[ValidateSet('cpu','cuda')][string]$Backend,
    [ValidateRange(50,1000)][int]$PollMilliseconds=250,
    [ValidateRange(1,1860)][int]$DeadlineSeconds=1860,[switch]$AsLibrary)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'observer.ps1')

function Assert-CharacterizationProcessIdentity($Actual,$Expected) {
    if($Actual.pid -ne $Expected.pid -or $Actual.creation_utc -ne $Expected.creation_utc -or
        -not [string]::Equals($Actual.executable,$Expected.executable,[StringComparison]::OrdinalIgnoreCase)) {
        throw 'Runtime observation process identity changed'
    }
}
function Write-CharacterizationObservation([string]$Path,$Value) {
    [IO.File]::WriteAllText($Path,($Value|ConvertTo-Json -Depth 12),[Text.UTF8Encoding]::new($false))
}
function Read-CharacterizationJson([string]$Path,[int]$Limit=131072) {
    if((Get-Item -LiteralPath $Path).Length -gt $Limit){throw 'Observation input exceeds byte bound'}
    Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json -ErrorAction Stop
}
if($AsLibrary){return}
if(-not $IsWindows -or $PSVersionTable.PSVersion.Major -lt 7){throw 'Runtime watcher requires native PowerShell 7'}
if([string]::IsNullOrWhiteSpace($PacketPath) -or [string]::IsNullOrWhiteSpace($ObservationPath) -or [string]::IsNullOrWhiteSpace($Backend)){throw 'Watcher needs packet, observation and backend'}
$watchPacket=Read-CharacterizationJson $PacketPath 65536
if($watchPacket.schema -ne 'candle.gpt-oss.characterization-packet.v1'){throw 'Watcher requires characterization packet version'}
$watchLane=@($watchPacket.lanes|Where-Object backend -eq $Backend)
if($watchLane.Count -ne 1){throw 'Watcher requires one matching lane'}
$watchLane=$watchLane[0]
$watchDirectory=Join-Path $watchPacket.proposed_output_root $Backend
$watchOutput=[IO.Path]::GetFullPath($ObservationPath)
$watchAllowed=(Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path 'artifacts')+'\'
if(-not $watchOutput.StartsWith($watchAllowed,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $watchOutput) -or (Test-Path -LiteralPath $watchDirectory)){throw 'Watcher must arm before a fresh owned lane'}
$watchSelf=Get-Process -Id $PID
$watchValue=[ordered]@{schema='candle.gpt-oss.runtime-observation.v1';status='armed';error=$null;
    source_commit=$watchPacket.source_commit;packet_sha256=(Get-FileHash -LiteralPath $PacketPath).Hash.ToLowerInvariant();
    owner=[ordered]@{pid=$PID;creation_utc=$watchSelf.StartTime.ToUniversalTime().ToString('o');executable=$watchSelf.Path};
    backend=$Backend;intended_lifetime="passive bounded observation, at most ${DeadlineSeconds}s";endpoints=@();
    shutdown='Parent stops this exact watcher handle; watcher never launches/stops a model or controller';
    poll_milliseconds=$PollMilliseconds;deadline_seconds=$DeadlineSeconds;started_utc=[DateTimeOffset]::UtcNow.ToString('o');
    ended_utc=$null;identity=$null;modules=@();process_probe_ms=0;module_probe_calls=0}
$reservation=[IO.File]::Open($watchOutput,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);$reservation.Dispose()
Write-CharacterizationObservation $watchOutput $watchValue
$watchTimer=[Diagnostics.Stopwatch]::StartNew();$watchExit=1
try {
    while($watchTimer.Elapsed.TotalSeconds -le $DeadlineSeconds) {
        $watchProgress=Join-Path $watchDirectory 'progress.json';$watchTerminal=Join-Path $watchDirectory 'terminal.json'
        if(Test-Path -LiteralPath $watchProgress -PathType Leaf) {
            $progress=$null;try {$progress=Read-CharacterizationJson $watchProgress}catch{}
            if($null -ne $progress) {
                if($progress.source_commit -ne $watchPacket.source_commit -or $progress.executable -ne $watchLane.binary.path){throw 'Runtime progress source/executable mismatch'}
                $probeStart=[Diagnostics.Stopwatch]::StartNew()
                $snapshot=Get-SessionProfileProcessSnapshot ([int]$progress.pid) 'modules'
                $watchValue.process_probe_ms+=$probeStart.Elapsed.TotalMilliseconds;$watchValue.module_probe_calls++
                if($null -ne $snapshot) {
                    if(-not [string]::Equals($snapshot.executable,$watchLane.binary.path,[StringComparison]::OrdinalIgnoreCase)){throw 'Runtime observed executable mismatch'}
                    $identity=[ordered]@{pid=$snapshot.pid;creation_utc=$snapshot.creation_utc;executable=$snapshot.executable}
                    if($null -eq $watchValue.identity){$watchValue.identity=$identity}else{Assert-CharacterizationProcessIdentity $identity $watchValue.identity}
                    foreach($module in $snapshot.modules) {
                        if([string]::IsNullOrWhiteSpace($module.name) -or [string]::IsNullOrWhiteSpace($module.path)){throw 'Loaded module lacks name/path'}
                        if($module.path -notin @($watchValue.modules|ForEach-Object path)) {
                            $watchValue.modules+= [ordered]@{name=$module.name;path=$module.path;observed_utc=[DateTimeOffset]::UtcNow.ToString('o')}
                            if($watchValue.modules.Count -gt 256){throw 'Observed module inventory exceeds bound'}
                        }
                    }
                    Write-CharacterizationObservation $watchOutput $watchValue
                }
            }
        }
        if(Test-Path -LiteralPath $watchTerminal -PathType Leaf) {
            $terminal=$null;try{$terminal=Read-CharacterizationJson $watchTerminal}catch{}
            if($null -ne $terminal){if($terminal.status -ne 'completed'){throw 'Model lane failed'};$watchValue.status='observed';$watchExit=0;break}
        }
        Start-Sleep -Milliseconds $PollMilliseconds
    }
    if($watchExit -ne 0){throw 'Runtime observation deadline expired'}
} catch {$watchValue.status='incomplete';$watchValue.error=$_.Exception.Message}
finally {$watchValue.ended_utc=[DateTimeOffset]::UtcNow.ToString('o');Write-CharacterizationObservation $watchOutput $watchValue;$watchSelf.Dispose()}
exit $watchExit
