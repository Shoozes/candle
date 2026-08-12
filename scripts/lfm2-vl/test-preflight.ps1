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
Assert-Condition ($null -ne $report.admission.model_processes_absent) "preflight omitted model-process admission"
Assert-Condition ($null -ne $report.admission.build_processes_absent) "preflight omitted build-process admission"
Assert-Condition ($null -ne $report.admission.python_processes_absent) "preflight omitted Python-process admission"
Assert-Condition ($null -ne $report.admission.quiet_host) "preflight omitted quiet-host admission"
Assert-Condition ($null -ne $report.admission.physical_memory_probe_complete) "preflight omitted physical-memory admission"
Assert-Condition ($null -ne $report.admission.commit_probe_complete) "preflight omitted commit-memory admission"
Assert-Condition ([string]$report.admission.status -in @("review", "blocked")) "preflight returned an unknown admission status"
if ([string]$report.admission.status -eq "review") {
    Assert-Condition ([bool]$report.admission.quiet_host) "review requires a quiet host"
    Assert-Condition ([bool]$report.admission.model_processes_absent) "review requires model processes to be absent"
    Assert-Condition ([bool]$report.admission.build_processes_absent) "review requires build processes to be absent"
    Assert-Condition ([bool]$report.admission.python_processes_absent) "review requires Python processes to be absent"
    Assert-Condition ([bool]$report.admission.physical_memory_probe_complete) "review requires a physical-memory probe"
    Assert-Condition ([bool]$report.admission.commit_probe_complete) "review requires a commit-memory probe"
}
Assert-Condition ($null -ne $report.redaction.secrets) "preflight omitted redaction contract"
Assert-Condition ([string]$report.redaction.command_lines -eq "omitted") "preflight exposed command lines"
Assert-Condition ($report.tracked_processes -is [array]) "tracked_processes must remain a JSON array"
Assert-Condition ($report.llama_processes -is [array]) "llama_processes must remain a JSON array"
Assert-Condition ($report.model_processes -is [array]) "model_processes must remain a JSON array"
Assert-Condition ($report.build_processes -is [array]) "build_processes must remain a JSON array"
Assert-Condition ($report.python_processes -is [array]) "python_processes must remain a JSON array"
foreach ($llama in @($report.llama_processes)) {
    Assert-Condition ([string]$llama.name -match '(?i)llama|mtmd') "llama_processes contained a non-llama/MTMD record"
}
foreach ($model in @($report.model_processes)) {
    Assert-Condition ([string]$model.name -match '(?i)llama|mtmd|lfm2-vl') "model_processes contained a non-model record"
}
foreach ($build in @($report.build_processes)) {
    Assert-Condition ([string]$build.name -match '(?i)cargo|rustc|ninja|cmake') "build_processes contained a non-build record"
}
foreach ($python in @($report.python_processes)) {
    Assert-Condition ([string]$python.name -match '(?i)^python(?:w)?$') "python_processes contained a non-Python record"
}
Assert-Condition ($report.probe_errors -is [array]) "probe_errors must remain a JSON array"
Assert-Condition ($report.gpu.gpus -is [array]) "gpu.gpus must remain a JSON array"
Assert-Condition ($report.gpu.compute_processes -is [array]) "gpu.compute_processes must remain a JSON array"
Assert-Condition ([string]$report.disk.source -in @("Get-PSDrive", "System.IO.DriveInfo")) "disk evidence omitted its source"
Assert-Condition ([UInt64]$report.disk.free_bytes -gt 0) "disk evidence reported no free space"

$outputPath = Join-Path ([IO.Path]::GetTempPath()) ("candle-lfm2-vl-preflight-" + [Guid]::NewGuid().ToString("N") + ".json")
try {
    $summary = (& $scriptPath -OutputPath $outputPath) -join [Environment]::NewLine
    Assert-Condition (Test-Path -LiteralPath $outputPath -PathType Leaf) "preflight did not write its requested output"
    Assert-Condition ($summary -match 'quiet=(True|False)') "preflight summary omitted quiet-host state"
    Assert-Condition ($summary -match 'model=\d+; build=\d+; python=\d+') "preflight summary omitted workload counts"
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
