[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Condition {
    param(
        [Parameter(Mandatory)][bool]$Condition,
        [Parameter(Mandatory)][string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

$scriptPath = Join-Path $PSScriptRoot "preflight.ps1"
$json = (& $scriptPath -AsJson) -join [Environment]::NewLine
$report = $json | ConvertFrom-Json
Assert-Condition ($report.schema -eq "candle-lfm2-vl-resource-preflight-v1") "preflight schema mismatch"
Assert-Condition ($null -ne $report.admission.llama_processes_absent) "preflight omitted llama admission"
Assert-Condition ($null -ne $report.admission.physical_memory_probe_complete) "preflight omitted physical-memory admission"
Assert-Condition ($null -ne $report.admission.commit_probe_complete) "preflight omitted commit-memory admission"
Assert-Condition ([string]$report.admission.status -in @("review", "blocked")) "preflight returned an unknown admission status"
if ([string]$report.admission.status -eq "review") {
    Assert-Condition ([bool]$report.admission.physical_memory_probe_complete) "review requires a physical-memory probe"
    Assert-Condition ([bool]$report.admission.commit_probe_complete) "review requires a commit-memory probe"
}
Assert-Condition ($null -ne $report.redaction.secrets) "preflight omitted redaction contract"
Assert-Condition ([string]$report.redaction.command_lines -eq "omitted") "preflight exposed command lines"
Assert-Condition ($report.tracked_processes -is [array]) "tracked_processes must remain a JSON array"
Assert-Condition ($report.llama_processes -is [array]) "llama_processes must remain a JSON array"
foreach ($llama in @($report.llama_processes)) {
    Assert-Condition ([string]$llama.name -match '(?i)llama') "llama_processes contained a non-llama record"
}
Assert-Condition ($report.probe_errors -is [array]) "probe_errors must remain a JSON array"
Assert-Condition ($report.gpu.gpus -is [array]) "gpu.gpus must remain a JSON array"
Assert-Condition ($report.gpu.compute_processes -is [array]) "gpu.compute_processes must remain a JSON array"
Assert-Condition ([string]$report.disk.source -in @("Get-PSDrive", "System.IO.DriveInfo")) "disk evidence omitted its source"
Assert-Condition ([UInt64]$report.disk.free_bytes -gt 0) "disk evidence reported no free space"

$outputPath = Join-Path ([IO.Path]::GetTempPath()) ("candle-lfm2-vl-preflight-" + [Guid]::NewGuid().ToString("N") + ".json")
try {
    [void](& $scriptPath -OutputPath $outputPath)
    Assert-Condition (Test-Path -LiteralPath $outputPath -PathType Leaf) "preflight did not write its requested output"
    $saved = (Get-Content -LiteralPath $outputPath -Raw) | ConvertFrom-Json
    Assert-Condition ($saved.schema -eq $report.schema) "atomic output schema mismatch"

    $refused = $false
    try {
        [void](& $scriptPath -OutputPath $outputPath)
    }
    catch {
        $refused = $true
    }
    Assert-Condition $refused "preflight overwrote an existing output without -ForceOutput"

    [void](& $scriptPath -OutputPath $outputPath -ForceOutput)
    Write-Output "preflight-smoke: passed"
}
finally {
    if (Test-Path -LiteralPath $outputPath) {
        Remove-Item -LiteralPath $outputPath -Force
    }
    $temporary = Get-ChildItem -LiteralPath ([IO.Path]::GetDirectoryName($outputPath)) -Filter (([IO.Path]::GetFileName($outputPath)) + ".tmp-*") -ErrorAction SilentlyContinue
    foreach ($item in @($temporary)) {
        if ($null -ne $item -and $item.FullName -like ($outputPath + ".tmp-*")) {
            Remove-Item -LiteralPath $item.FullName -Force -ErrorAction SilentlyContinue
        }
    }
}
