param(
    [Parameter(Mandatory = $true)][string]$ModelFile,
    [ValidateSet(1, 32)][int]$Ubatch = 32,
    [ValidateRange(1, 49)][int]$TokenCount = 49,
    [string]$OutputLabel
)

$ErrorActionPreference = 'Stop'
$expectedHash = 'fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572'
$maxSeconds = 480
$maxPrivateBytes = 12884901888
$minAvailableBytes = 17179869184
$minCFreeBytes = 10737418240
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputDir = Join-Path $root 'artifacts\qwen35'
$exe = Join-Path $outputDir 'llama-layer-probe.exe'
$fixture = Join-Path $root 'tests\fixtures\qwen35_codename_b\reference.json'
$label = "ubatch-$Ubatch-tokens-$TokenCount"
if ($OutputLabel) {
    if ($OutputLabel -notmatch '^[a-z0-9-]+$') { throw 'Invalid output label' }
    $label = "$label-$OutputLabel"
}
$capture = Join-Path $outputDir "llama-layer-vectors-$label"
$stdout = Join-Path $outputDir "llama-layer-$label.stdout.log"
$stderr = Join-Path $outputDir "llama-layer-$label.stderr.log"
$report = Join-Path $outputDir "llama-layer-$label-job.json"

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'probe executable absent' }
if (-not (Test-Path -LiteralPath $fixture -PathType Leaf)) { throw 'reference fixture absent' }
if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) { throw 'GGUF absent' }
if ((Get-Item -LiteralPath $ModelFile).Length -ne 4482403200) { throw 'GGUF size changed' }
if ((Get-FileHash -LiteralPath $ModelFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) { throw 'GGUF hash changed' }
if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { throw 'Available memory below 16 GiB' }
if ((Get-PSDrive C).Free -lt $minCFreeBytes) { throw 'C: free space below 10 GiB' }
New-Item -ItemType Directory -Path $capture -Force | Out-Null

$started = Get-Date
$reason = 'completed'
[long]$peakPrivate = 0
$child = Start-Process -FilePath $exe -ArgumentList @($ModelFile, $fixture, $capture, $Ubatch, $TokenCount) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try {
    while (-not $child.HasExited) {
        $child.Refresh()
        $peakPrivate = [Math]::Max([long]$peakPrivate, [long]$child.PrivateMemorySize64)
        if (((Get-Date) - $started).TotalSeconds -gt $maxSeconds) { $reason = 'wall-time ceiling'; break }
        if ($child.PrivateMemorySize64 -gt $maxPrivateBytes) { $reason = 'private-memory ceiling'; break }
        if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { $reason = 'available-memory floor'; break }
        if ((Get-PSDrive C).Free -lt $minCFreeBytes) { $reason = 'C: free-space floor'; break }
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
    ubatch = $Ubatch
    token_count = $TokenCount
    exit_code = $child.ExitCode
    elapsed_seconds = ((Get-Date) - $started).TotalSeconds
    peak_private_bytes_sampled = $peakPrivate
    model_file = $ModelFile
    model_sha256 = $expectedHash
    captured_vectors = @(Get-ChildItem -LiteralPath $capture -Filter '*.bin' -File).Count
    c_free_bytes = (Get-PSDrive C).Free
}
$result | ConvertTo-Json | Set-Content -LiteralPath $report
$result | Format-List
Get-Content -LiteralPath $stdout -Tail 40
Get-Content -LiteralPath $stderr -Tail 15
if ($reason -ne 'completed' -or $child.ExitCode -ne 0) { exit 1 }
