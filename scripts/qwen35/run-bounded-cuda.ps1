param(
    [Parameter(Mandatory = $true)][ValidateSet('build', 'proof', 'check')][string]$Mode,
    [Parameter(Mandatory = $true)][string]$TargetDir,
    [Parameter(Mandatory = $true)][string]$TempDir,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9-]+$')][string]$OutputLabel,
    [ValidateSet('cuda_q8_projection_proof', 'hybrid_cuda_reference_trace_and_reset', 'hybrid_cuda_sealed_context_profile', 'token_major_attention_values_keep_ggml_dot_order', 'row_major_recurrent_matvec_keeps_ggml_dot_order', 'cuda_first_token_components', 'exact_reference_trace_and_reset')]
    [string]$TestFilter = 'cuda_q8_projection_proof',
    [switch]$CpuOnly,
    [string]$TestExe,
    [string]$ModelFile,
    [string]$ContextTokensFile,
    [string]$ContextGateFile,
    [switch]$ContextRepeat
)

$ErrorActionPreference = 'Stop'
if ($CpuOnly -and $TestFilter -ne 'exact_reference_trace_and_reset') {
    throw 'CpuOnly is limited to the exact CPU reference gate'
}
$target = [IO.Path]::GetFullPath($TargetDir)
$temp = [IO.Path]::GetFullPath($TempDir)
if (-not $target.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase) -or
    -not $temp.StartsWith(($target + '\'), [StringComparison]::OrdinalIgnoreCase) -or
    -not (Test-Path -LiteralPath $target -PathType Container) -or
    -not (Test-Path -LiteralPath $temp -PathType Container)) {
    throw 'Existing D: target and nested temp directory required'
}
if ($Mode -eq 'proof') {
    if (-not (Test-Path -LiteralPath $TestExe -PathType Leaf)) { throw 'Test executable absent' }
    if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) { throw 'GGUF absent' }
    if ((Get-Item -LiteralPath $ModelFile).Length -ne 4482403200) { throw 'GGUF size changed' }
    $hash = (Get-FileHash -LiteralPath $ModelFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne 'fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572') {
        throw 'GGUF hash changed'
    }
    if ($TestFilter -eq 'hybrid_cuda_sealed_context_profile') {
        if (-not (Test-Path -LiteralPath $ContextTokensFile -PathType Leaf) -or
            -not (Test-Path -LiteralPath $ContextGateFile -PathType Leaf)) { throw 'Sealed context input or gate absent' }
        if ((Get-FileHash -LiteralPath $ContextTokensFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            'fceba3897357a86a2e9063b38e528549ac5df93661487f2f7fda0789e014fbdb') { throw 'Sealed context input hash changed' }
        if ((Get-FileHash -LiteralPath $ContextGateFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne
            '8b1996d6347c936b4410f393b87732ad21e19ef56d2a1054d1df284d622ada4e') { throw 'Sealed context gate hash changed' }
        $env:QWEN35_CONTEXT_TOKENS_FILE = [IO.Path]::GetFullPath($ContextTokensFile)
        $env:QWEN35_CONTEXT_GATE_FILE = [IO.Path]::GetFullPath($ContextGateFile)
        $env:QWEN35_CONTEXT_REPEAT = if ($ContextRepeat) { '1' } else { '0' }
    }
}
$outDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\artifacts\qwen35'))
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$stdout = Join-Path $outDir "cuda-$OutputLabel.stdout.log"
$stderr = Join-Path $outDir "cuda-$OutputLabel.stderr.log"
$report = Join-Path $outDir "cuda-$OutputLabel.json"
foreach ($path in @($stdout, $stderr, $report)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite $path" }
}
$cBefore = ([IO.DriveInfo]::new('C')).AvailableFreeSpace
$dBefore = ([IO.DriveInfo]::new('D')).AvailableFreeSpace
if ($cBefore -lt 26843545600L -or $dBefore -lt 107374182400L) {
    throw 'Disk floor not met before CUDA job'
}
if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt 17179869184L) {
    throw 'Available memory below 16 GiB before job'
}
$env:CARGO_NET_OFFLINE = 'true'
$env:CARGO_BUILD_JOBS = '2'
$env:CARGO_INCREMENTAL = '0'
$env:CARGO_TARGET_DIR = $target
$env:TEMP = $temp
$env:TMP = $temp
$env:RAYON_NUM_THREADS = '4'
$env:CANDLE_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'
if ($Mode -eq 'build') {
    $exe = (Get-Command cargo -ErrorAction Stop).Source
    $arguments = @('test', '--release', '--locked', '--offline')
    if (-not $CpuOnly) { $arguments += @('--features', 'cuda') }
    $arguments += @('-p', 'candle-transformers', $TestFilter, '--lib', '--no-run')
    $deadline = 1200
} elseif ($Mode -eq 'check') {
    $exe = (Get-Command cargo -ErrorAction Stop).Source
    $arguments = @('check', '--locked', '--offline')
    if ($CpuOnly) {
        $arguments += @('-p', 'candle-core', '-p', 'candle-nn', '-p', 'candle-transformers', '-p', 'candle-vlm', '-p', 'candle-examples', '--example', 'quantized-lfm2')
    } else {
        $arguments += @('--features', 'cuda', '-p', 'candle-core', '-p', 'candle-transformers')
    }
    $deadline = 1200
} else {
    $env:QWEN35_GGUF_FILE = [IO.Path]::GetFullPath($ModelFile)
    $exe = [IO.Path]::GetFullPath($TestExe)
    $arguments = @($TestFilter)
    if ($TestFilter -notin @('token_major_attention_values_keep_ggml_dot_order', 'row_major_recurrent_matvec_keeps_ggml_dot_order')) { $arguments += '--ignored' }
    $arguments += @('--nocapture', '--test-threads=1')
    $deadline = if ($TestFilter -eq 'cuda_q8_projection_proof') { 240 } else { 600 }
}
$started = Get-Date
$reason = 'completed'
[long]$peakPrivate = 0
$child = Start-Process -FilePath $exe -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try {
    while (-not $child.HasExited) {
        $child.Refresh()
        $peakPrivate = [Math]::Max($peakPrivate, [long]$child.PrivateMemorySize64)
        if (((Get-Date) - $started).TotalSeconds -gt $deadline) { $reason = 'wall-time ceiling'; break }
        if (([IO.DriveInfo]::new('C')).AvailableFreeSpace -lt 26843545600L) { $reason = 'C: 25 GiB floor'; break }
        if (([IO.DriveInfo]::new('D')).AvailableFreeSpace -lt 107374182400L) { $reason = 'D: 100 GiB floor'; break }
        if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt 17179869184L) { $reason = '16 GiB available-memory floor'; break }
        if ($dBefore - ([IO.DriveInfo]::new('D')).AvailableFreeSpace -gt 17179869184L) { $reason = '16 GiB D: output growth'; break }
        if ((Get-Item -LiteralPath $stdout).Length + (Get-Item -LiteralPath $stderr).Length -gt 10485760L) { $reason = '10 MiB log ceiling'; break }
        if ($Mode -eq 'proof' -and $child.PrivateMemorySize64 -gt 12884901888L) { $reason = '12 GiB private-memory ceiling'; break }
        Start-Sleep -Milliseconds 500
    }
} finally {
    if (-not $child.HasExited) {
        & taskkill.exe /PID $child.Id /T /F | Out-Null
        $child.WaitForExit()
    }
}
$child.Refresh()
$result = [ordered]@{
    mode = $Mode
    cpu_only = [bool]$CpuOnly
    reason = $reason
    exit_code = $child.ExitCode
    elapsed_seconds = ((Get-Date) - $started).TotalSeconds
    peak_private_bytes_sampled = $peakPrivate
    c_free_before = $cBefore
    c_free_after = ([IO.DriveInfo]::new('C')).AvailableFreeSpace
    d_free_before = $dBefore
    d_free_after = ([IO.DriveInfo]::new('D')).AvailableFreeSpace
    target_dir = $target
    model_file = if ($Mode -eq 'proof') { $env:QWEN35_GGUF_FILE } else { $null }
    model_sha256 = if ($Mode -eq 'proof') { $hash } else { $null }
    command = "$exe $($arguments -join ' ')"
}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $report
$result | ConvertTo-Json -Depth 4
if ($reason -ne 'completed' -or $child.ExitCode -ne 0) { exit 1 }
