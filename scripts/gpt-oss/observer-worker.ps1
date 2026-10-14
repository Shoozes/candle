[CmdletBinding()]
param([Parameter(Mandatory)][ValidateRange(1,2147483647)][int]$TargetPid,
    [ValidateSet('identity','modules')][string]$Mode='identity')
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'observer.ps1')
$target=Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
if($null -eq $target) { 'null'; exit 0 }
try {
    ConvertTo-Json -InputObject (Get-SessionProfileLocalSnapshot $target $Mode) -Depth 5 -Compress
} finally { $target.Dispose() }
