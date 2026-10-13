Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'

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
