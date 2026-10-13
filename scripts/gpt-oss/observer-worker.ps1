[CmdletBinding()]
param([Parameter(Mandatory)][ValidateRange(1,2147483647)][int]$TargetPid,
    [ValidateSet('identity','modules')][string]$Mode='identity')
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$target=Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
if($null -eq $target) { 'null'; exit 0 }
try {
    $identity=[ordered]@{pid=$target.Id;creation_utc=$target.StartTime.ToUniversalTime().ToString('o');executable=$target.Path}
    if($Mode -eq 'modules') {
        $modules=@($target.Modules | ForEach-Object { [ordered]@{name=$_.ModuleName;path=$_.FileName} })
        if($modules.Count -gt 256) { throw 'Loaded module count exceeds observer bound' }
        $target.Refresh()
        if($target.HasExited) { 'null'; exit 0 }
        if($target.StartTime.ToUniversalTime().ToString('o') -ne $identity.creation_utc -or $target.Path -ne $identity.executable) { throw 'Observed process identity changed' }
        $identity.modules=$modules
    }
    $identity | ConvertTo-Json -Depth 5 -Compress
} finally { $target.Dispose() }
