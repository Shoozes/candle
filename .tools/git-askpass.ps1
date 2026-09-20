param([string]$Prompt)

$ErrorActionPreference = "Stop"

if ($Prompt -match "(?i)username") {
    Write-Output "x-access-token"
    exit 0
}

$tokenPath = $env:CANDLE_GIT_TOKEN_FILE
if ([string]::IsNullOrWhiteSpace($tokenPath) -or -not (Test-Path -LiteralPath $tokenPath -PathType Leaf)) {
    throw "Git token path is unavailable to askpass."
}
$token = (Get-Content -LiteralPath $tokenPath -TotalCount 1).Trim()
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "Git token file is empty."
}
Write-Output $token
