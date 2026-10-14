Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

function Get-SessionProfileLocalSnapshot {
    param([Parameter(Mandatory)]$Target,[ValidateSet('identity','modules')][string]$Mode)
    try {
        $identity=[ordered]@{pid=$Target.Id;creation_utc=$Target.StartTime.ToUniversalTime().ToString('o');executable=$Target.Path}
        if($Mode -eq 'modules') {
            $modules=@($Target.Modules | ForEach-Object { [ordered]@{name=$_.ModuleName;path=$_.FileName} })
            if($modules.Count -gt 256){throw 'Loaded module count exceeds observer bound'}
            $Target.Refresh()
            if($Target.HasExited){return $null}
            if($Target.StartTime.ToUniversalTime().ToString('o') -ne $identity.creation_utc -or $Target.Path -ne $identity.executable){throw 'Observed process identity changed'}
            $identity.modules=$modules
        }
        return $identity
    } catch {
        $observationError=$_
        # Process.Modules can become null or throw while the target exits.
        # Only confirmed departure is absence; live/unknown failures stay errors.
        try {$Target.Refresh();$departed=$Target.HasExited} catch {throw $observationError}
        if($departed){return $null}
        throw $observationError
    }
}

function Invoke-SessionProfileProbe {
    param([Parameter(Mandatory)][string]$FilePath, [AllowEmptyCollection()][string[]]$ArgumentList=@(),
        [ValidateRange(50,10000)][int]$TimeoutMilliseconds=5000,
        [ValidateRange(1,131072)][int]$MaxOutputBytes=4096)
    if (-not ('Candle.GptOss.ObserverProbe' -as [type])) {
        Add-Type -Path (Join-Path $PSScriptRoot 'observer.cs')
    }
    $resolved=(Resolve-Path -LiteralPath $FilePath).Path
    [Candle.GptOss.ObserverProbe]::Run($resolved,$ArgumentList,$TimeoutMilliseconds,$MaxOutputBytes)
}

function Get-SessionProfileProcessSnapshot {
    param([Parameter(Mandatory)][int]$TargetPid, [ValidateSet('identity','modules')][string]$Mode='identity',
        [ValidateRange(50,10000)][int]$TimeoutMilliseconds=5000)
    $probe=Invoke-SessionProfileProbe (Get-Command pwsh).Source @('-NoProfile','-NonInteractive','-File',
        (Join-Path $PSScriptRoot 'observer-worker.ps1'),'-TargetPid',[string]$TargetPid,'-Mode',$Mode) $TimeoutMilliseconds 131072
    if($probe.ExitCode -ne 0) { throw "Process observation failed: $($probe.Stderr)" }
    $probe.Stdout | ConvertFrom-Json -ErrorAction Stop
}
