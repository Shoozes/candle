param(
    [Parameter(Mandatory = $true)][ValidateSet('build', 'benchmark')][string]$Mode,
    [Parameter(Mandatory = $true)][string]$TargetDir,
    [Parameter(Mandatory = $true)][string]$TempDir,
    [string]$TestExe,
    [string]$ModelFile,
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9-]+$')][string]$OutputLabel,
    [ValidateRange(1, 32)][int]$Threads = 4,
    [switch]$Profile
)

$ErrorActionPreference = 'Stop'
$cFloor = 107374182400L
$dFloor = 141733920768L # 100 GiB reserve plus 32 GiB WSL swap allowance.
$maxNewDBytes = 17179869184L
$minAvailableBytes = 17179869184L
$maxPrivateBytes = 12884901888L
$maxLogBytes = 10485760L
$maxSeconds = if ($Mode -eq 'build') { 1200 } else { 900 }
$outputDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\artifacts\qwen35'))
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$stdout = Join-Path $outputDir "performance-$OutputLabel.stdout.log"
$stderr = Join-Path $outputDir "performance-$OutputLabel.stderr.log"
$report = Join-Path $outputDir "performance-$OutputLabel.json"
foreach ($path in @($stdout, $stderr, $report)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite existing evidence: $path" }
}
if (-not (Test-Path -LiteralPath $TargetDir -PathType Container)) { throw 'Target directory absent' }
if (-not (Test-Path -LiteralPath $TempDir -PathType Container)) { throw 'Temp directory absent' }
$target = [IO.Path]::GetFullPath($TargetDir)
$temp = [IO.Path]::GetFullPath($TempDir)
if (-not $target.StartsWith('D:\', [StringComparison]::OrdinalIgnoreCase) -or
    -not $temp.StartsWith(($target + '\'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Target and its temp directory must be on D:'
}
if ($Mode -eq 'benchmark') {
    if (-not (Test-Path -LiteralPath $TestExe -PathType Leaf)) { throw 'Test executable absent' }
    if (-not (Test-Path -LiteralPath $ModelFile -PathType Leaf)) { throw 'GGUF absent' }
    if ((Get-Item -LiteralPath $ModelFile).Length -ne 4482403200) { throw 'GGUF size changed' }
    $hash = (Get-FileHash -LiteralPath $ModelFile -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne 'fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572') {
        throw 'GGUF hash changed'
    }
}
if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) {
    throw 'Available memory below 16 GiB'
}
$cBefore = ([IO.DriveInfo]::new('C')).AvailableFreeSpace
$dBefore = ([IO.DriveInfo]::new('D')).AvailableFreeSpace
if ($cBefore -lt $cFloor -or $dBefore -lt $dFloor) { throw 'Disk reserve not met' }

$env:CARGO_NET_OFFLINE = 'true'
$env:CARGO_BUILD_JOBS = '2'
$env:CARGO_INCREMENTAL = '0'
$env:CARGO_TARGET_DIR = $target
$env:TEMP = $temp
$env:TMP = $temp
$env:RAYON_NUM_THREADS = "$Threads"
$env:CANDLE_NUM_THREADS = "$Threads"
$env:OMP_NUM_THREADS = "$Threads"
if ($Mode -eq 'build') {
    $executable = (Get-Command cargo -ErrorAction Stop).Source
    $arguments = @('test', '--release', '--locked', '--offline', '-p', 'candle-transformers', 'release_cpu_benchmark', '--lib', '--no-run')
} else {
    $env:QWEN35_GGUF_FILE = [IO.Path]::GetFullPath($ModelFile)
    $env:QWEN35_BENCH_PROFILE = if ($Profile) { '1' } else { '0' }
    $executable = [IO.Path]::GetFullPath($TestExe)
    $arguments = @('models::qwen35::cpu::tests::release_cpu_benchmark', '--ignored', '--nocapture', '--test-threads=1')
}

$started = Get-Date
$reason = 'completed'
[long]$peakPrivate = 0
[long]$peakWorkingSet = 0
$child = Start-Process -FilePath $executable -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
try {
    while (-not $child.HasExited) {
        $child.Refresh()
        $peakPrivate = [Math]::Max($peakPrivate, [long]$child.PrivateMemorySize64)
        $peakWorkingSet = [Math]::Max($peakWorkingSet, [long]$child.WorkingSet64)
        $cFree = ([IO.DriveInfo]::new('C')).AvailableFreeSpace
        $dFree = ([IO.DriveInfo]::new('D')).AvailableFreeSpace
        if (((Get-Date) - $started).TotalSeconds -gt $maxSeconds) { $reason = 'wall-time ceiling'; break }
        if ($Mode -eq 'benchmark' -and $child.PrivateMemorySize64 -gt $maxPrivateBytes) { $reason = 'private-memory ceiling'; break }
        if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { $reason = 'available-memory floor'; break }
        if ($cFree -lt $cFloor) { $reason = 'C: reserve'; break }
        if ($dFree -lt $dFloor) { $reason = 'D: reserve'; break }
        if ($dBefore - $dFree -gt $maxNewDBytes) { $reason = '16 GiB new D: output ceiling'; break }
        if ((Get-Item -LiteralPath $stdout).Length + (Get-Item -LiteralPath $stderr).Length -gt $maxLogBytes) { $reason = '10 MiB log ceiling'; break }
        Start-Sleep -Milliseconds 500
    }
} finally {
    if (-not $child.HasExited) {
        & taskkill.exe /PID $child.Id /T /F | Out-Null
        $child.WaitForExit()
    }
}
$child.Refresh()
$benchLine = @(Get-Content -LiteralPath $stderr | Where-Object { $_.StartsWith('QWEN35_BENCH_JSON=') })
$cpu = try {
    Get-CimInstance Win32_Processor -ErrorAction Stop |
        Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors
} catch {
    $processor = Get-ItemProperty -LiteralPath 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0' -ErrorAction SilentlyContinue
    [ordered]@{
        Name = $processor.ProcessorNameString
        NumberOfCores = $null
        NumberOfLogicalProcessors = $env:NUMBER_OF_PROCESSORS
        source = 'registry_and_environment'
    }
}
$result = [ordered]@{
    mode = $Mode
    reason = $reason
    exit_code = $child.ExitCode
    elapsed_seconds = ((Get-Date) - $started).TotalSeconds
    peak_private_bytes_sampled = $peakPrivate
    peak_working_set_bytes_sampled = $peakWorkingSet
    c_free_before = $cBefore
    c_free_after = ([IO.DriveInfo]::new('C')).AvailableFreeSpace
    d_free_before = $dBefore
    d_free_after = ([IO.DriveInfo]::new('D')).AvailableFreeSpace
    target_dir = $target
    temp_dir = $temp
    model_file = if ($Mode -eq 'benchmark') { $env:QWEN35_GGUF_FILE } else { $null }
    model_sha256 = if ($Mode -eq 'benchmark') { $hash } else { $null }
    cpu = $cpu
    rayon_threads = $Threads
    candle_threads = $Threads
    omp_threads = $Threads
    profile = [bool]$Profile
    rustc = (& rustc -V)
    benchmark = if ($benchLine.Count -eq 1) { $benchLine[0].Substring('QWEN35_BENCH_JSON='.Length) | ConvertFrom-Json } else { $null }
    stdout = $stdout
    stderr = $stderr
}
$result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $report
[pscustomobject]$result | Select-Object mode, reason, exit_code, elapsed_seconds, peak_private_bytes_sampled, c_free_after, d_free_after, stdout, stderr | Format-List
Get-Content -LiteralPath $stdout -Tail 8
Get-Content -LiteralPath $stderr -Tail 8
if ($reason -ne 'completed' -or $child.ExitCode -ne 0 -or ($Mode -eq 'benchmark' -and $benchLine.Count -ne 1)) { exit 1 }
