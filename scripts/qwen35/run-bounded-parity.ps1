param(
    [Parameter(Mandatory = $true)][string]$TestExe,
    [Parameter(Mandatory = $true)][string]$ModelFile,
    [string]$TestFilter = 'models::qwen35::cpu::tests::exact_reference_trace_and_reset',
    [string]$OutputLabel,
    [string]$ComponentDir,
    [ValidateRange(0, 48)][int]$CaptureTokenIndex = 48,
    [ValidateRange(0, 31)][int]$CaptureLayerIndex = 2,
    [switch]$Timeline,
    [switch]$CompactQ8
)

$ErrorActionPreference = 'Stop'
$maxSeconds = 480
$maxPrivateBytes = 12884901888
$minAvailableBytes = 17179869184
$minCFreeBytes = 107374182400
$minDFreeBytes = 141733920768 # 100 GiB reserve plus 32 GiB WSL swap allowance.
$outputDir = Join-Path $PSScriptRoot '..\..\artifacts\qwen35'
$outputDir = [IO.Path]::GetFullPath($outputDir)
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$label = if ($CompactQ8) { 'compact' } else { 'default' }
$cpuMode = if ($CompactQ8) { 'forced_avx2' } else { 'process_default' }
if ($OutputLabel) {
    if ($OutputLabel -notmatch '^[a-z0-9-]+$') { throw 'Invalid output label' }
    $label = $OutputLabel
}
$stdout = Join-Path $outputDir "parity-$label.stdout.log"
$stderr = Join-Path $outputDir "parity-$label.stderr.log"
$report = Join-Path $outputDir "parity-$label-job.json"

if (-not (Test-Path -LiteralPath $TestExe -PathType Leaf)) { throw 'Test executable absent' }
if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) { throw 'GGUF absent' }
if ((Get-Item -LiteralPath $ModelFile).Length -ne 4482403200) { throw 'GGUF size changed' }
# Artifact::open hashes the whole file before parsing or loading any tensor.
if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { throw 'Available memory below 16 GiB' }
if ((Get-PSDrive C).Free -lt $minCFreeBytes) { throw 'C: free space below 100 GiB' }
if ((Get-PSDrive D).Free -lt $minDFreeBytes) { throw 'D: free space below 132 GiB' }

$env:QWEN35_GGUF_FILE = $ModelFile
if ($CompactQ8) { $env:MISTRALRS_FORCE_AVX2 = '1' }
if ($ComponentDir) {
    $env:QWEN35_COMPONENT_DIR = $ComponentDir
    $env:QWEN35_CAPTURE_TOKEN_INDEX = "$CaptureTokenIndex"
    $env:QWEN35_CAPTURE_LAYER_INDEX = "$CaptureLayerIndex"
    if ($Timeline) { $env:QWEN35_CAPTURE_TIMELINE = '1' }
}
$started = Get-Date
$reason = 'completed'
[long]$peakPrivate = 0
$child = Start-Process -FilePath $TestExe -ArgumentList @($TestFilter, '--ignored', '--nocapture') -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try {
    while (-not $child.HasExited) {
        $child.Refresh()
        $peakPrivate = [Math]::Max([long]$peakPrivate, [long]$child.PrivateMemorySize64)
        if (((Get-Date) - $started).TotalSeconds -gt $maxSeconds) { $reason = 'wall-time ceiling'; break }
        if ($child.PrivateMemorySize64 -gt $maxPrivateBytes) { $reason = 'private-memory ceiling'; break }
        if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { $reason = 'available-memory floor'; break }
        if ((Get-PSDrive C).Free -lt $minCFreeBytes) { $reason = 'C: free-space floor'; break }
        if ((Get-PSDrive D).Free -lt $minDFreeBytes) { $reason = 'D: free-space floor'; break }
        Start-Sleep -Milliseconds 500
    }
} finally {
    if (-not $child.HasExited) {
        Stop-Process -Id $child.Id -Force
        $child.WaitForExit()
    }
}
$child.Refresh()
$result = [ordered]@{
    reason = $reason
    cpu_mode = $cpuMode
    model_q8_matmul = 'scoped_ggml_q8_0'
    test_filter = $TestFilter
    exit_code = $child.ExitCode
    elapsed_seconds = ((Get-Date) - $started).TotalSeconds
    peak_private_bytes_sampled = $peakPrivate
    model_file = $ModelFile
    model_sha256 = 'fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572'
    stdout = $stdout
    stderr = $stderr
    c_free_bytes = (Get-PSDrive C).Free
}
$result | ConvertTo-Json | Set-Content -LiteralPath $report
$result | Format-List
Get-Content -LiteralPath $stdout -Tail 18
Get-Content -LiteralPath $stderr -Tail 18
if ($reason -ne 'completed' -or $child.ExitCode -ne 0) { exit 1 }
