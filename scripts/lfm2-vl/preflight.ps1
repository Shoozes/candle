[CmdletBinding()]
param(
    [Parameter()]
    [AllowEmptyString()]
    [string]$OutputPath = '',

    [Parameter()]
    [switch]$ForceOutput,

    [Parameter()]
    [switch]$AsJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$contract = "candle-lfm2-vl-resource-preflight-v1"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$probeErrors = New-Object 'System.Collections.Generic.List[string]'

if (-not ("CandleLfm2VlPreflightMemoryNative" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class CandleLfm2VlPreflightMemoryNative
{
    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatusEx
    {
        public uint Length;
        public uint MemoryLoad;
        public ulong TotalPhysical;
        public ulong AvailablePhysical;
        public ulong TotalPageFile;
        public ulong AvailablePageFile;
        public ulong TotalVirtual;
        public ulong AvailableVirtual;
        public ulong AvailableExtendedVirtual;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatusEx status);

    public static ulong[] GetSnapshot()
    {
        var status = new MemoryStatusEx();
        status.Length = (uint)Marshal.SizeOf(typeof(MemoryStatusEx));
        if (!GlobalMemoryStatusEx(ref status))
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(),
                "GlobalMemoryStatusEx failed");
        }
        return new[] {
            status.TotalPhysical,
            status.AvailablePhysical,
            status.TotalPageFile,
            status.AvailablePageFile
        };
    }
}
'@ -ErrorAction Stop
}

function Add-ProbeError {
    param([Parameter(Mandatory)][string]$Message)

    if ($probeErrors.Count -lt 32) {
        [void]$probeErrors.Add($Message.Substring(0, [Math]::Min(512, $Message.Length)))
    }
}

function Get-ParentProcessMap {
    $map = @{}
    try {
        $rows = @(Get-CimInstance Win32_Process -ErrorAction Stop)
        foreach ($row in $rows) {
            $processId = [int]$row.ProcessId
            $map[$processId] = [ordered]@{
                pid = $processId
                parent_pid = if ($null -ne $row.ParentProcessId) { [int]$row.ParentProcessId } else { $null }
                path = if ($null -ne $row.ExecutablePath) { [string]$row.ExecutablePath } else { $null }
            }
        }
    }
    catch {
        Add-ProbeError "Win32_Process parent map: $($_.Exception.Message)"
    }
    return $map
}

function Get-SafeProcessRecord {
    param(
        [Parameter(Mandatory)][Diagnostics.Process]$Process,
        [Parameter()][hashtable]$ParentMap = @{}
    )

    $name = $null
    $path = $null
    $start = $null
    $workingSet = $null
    $privateBytes = $null
    $parentPid = $null
    $parentPath = $null
    try { $name = [string]$Process.ProcessName } catch { }
    try { $path = $Process.Path } catch { }
    try { $start = $Process.StartTime.ToUniversalTime().ToString("O") } catch { }
    try { $workingSet = [UInt64]$Process.WorkingSet64 } catch { }
    try { $privateBytes = [UInt64]$Process.PrivateMemorySize64 } catch { }
    if ($ParentMap.ContainsKey($Process.Id)) {
        $parent = $ParentMap[$Process.Id]
        $parentPid = $parent.parent_pid
        $parentPath = $parent.path
    }
    [ordered]@{
        pid = [int]$Process.Id
        name = $name
        path = $path
        started_at_utc = $start
        parent_pid = $parentPid
        parent_path = $parentPath
        working_set_bytes = $workingSet
        private_bytes = $privateBytes
    }
}

function Invoke-NativeOutput {
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter()][string[]]$Arguments = @()
    )

    $previous = $ErrorActionPreference
    $lines = @()
    $exitCode = $null
    try {
        # Windows PowerShell promotes native stderr to a terminating error when
        # ErrorActionPreference is Stop; capture it as diagnostic output instead.
        $ErrorActionPreference = "Continue"
        $lines = @(& $Executable @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    }
    catch {
        $lines = @($_.Exception.Message)
        $exitCode = 1
    }
    finally {
        $ErrorActionPreference = $previous
    }
    [ordered]@{ lines = @($lines); exit_code = $exitCode }
}

function Get-GitIdentity {
    $identity = [ordered]@{
        executable = $null
        available = $false
        head = $null
        branch = $null
        error = $null
    }
    $git = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($null -eq $git) {
        $identity.error = "git.exe not found on PATH"
        return $identity
    }
    $identity.executable = $git.Source
    $headResult = Invoke-NativeOutput -Executable $git.Source -Arguments @('-C', $repoRoot, 'rev-parse', 'HEAD')
    $headOutput = @($headResult.lines)
    if ($headResult.exit_code -eq 0 -and $headOutput.Count -gt 0 -and [string]$headOutput[0] -match '^[0-9a-fA-F]{40}$') {
        $identity.available = $true
        $identity.head = ([string]$headOutput[0]).ToLowerInvariant()
        $branchResult = Invoke-NativeOutput -Executable $git.Source -Arguments @('-C', $repoRoot, 'symbolic-ref', '--short', '-q', 'HEAD')
        if ($branchResult.exit_code -eq 0 -and @($branchResult.lines).Count -gt 0) {
            $identity.branch = [string]@($branchResult.lines)[0]
        }
    }
    else {
        $identity.error = (($headOutput | ForEach-Object { [string]$_ }) -join ' ').Trim()
        if ([string]::IsNullOrWhiteSpace($identity.error)) {
            $identity.error = "git rev-parse HEAD failed with exit code $($headResult.exit_code)"
        }
    }
    return $identity
}

function Get-PhysicalMemorySnapshot {
    $total = $null
    $available = $null
    $pageTotal = $null
    $pageAvailable = $null
    $sources = New-Object 'System.Collections.Generic.List[string]'

    try {
        $computer = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
        if ([UInt64]$computer.TotalPhysicalMemory -gt 0) {
            $total = [UInt64]$computer.TotalPhysicalMemory
            [void]$sources.Add("Win32_ComputerSystem")
        }
    }
    catch {
        Add-ProbeError "Win32_ComputerSystem: $($_.Exception.Message)"
    }

    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        if ([UInt64]$os.FreePhysicalMemory -gt 0) {
            $available = [UInt64]$os.FreePhysicalMemory * 1024
            [void]$sources.Add("Win32_OperatingSystem")
        }
    }
    catch {
        Add-ProbeError "Win32_OperatingSystem: $($_.Exception.Message)"
    }

    try {
        $native = [CandleLfm2VlPreflightMemoryNative]::GetSnapshot()
        if ($null -eq $total -or $total -eq 0) { $total = [UInt64]$native[0] }
        if ($null -eq $available -or $available -eq 0) { $available = [UInt64]$native[1] }
        $pageTotal = [UInt64]$native[2]
        $pageAvailable = [UInt64]$native[3]
        [void]$sources.Add("GlobalMemoryStatusEx")
    }
    catch {
        Add-ProbeError "GlobalMemoryStatusEx: $($_.Exception.Message)"
    }

    [ordered]@{
        total_physical_bytes = $total
        available_physical_bytes = $available
        total_page_file_bytes = $pageTotal
        available_page_file_bytes = $pageAvailable
        sources = @($sources | Select-Object -Unique)
    }
}

function Get-CounterValue {
    param([Parameter(Mandatory)][string]$CounterPath)

    try {
        $sample = Get-Counter -Counter $CounterPath -ErrorAction Stop |
            Select-Object -ExpandProperty CounterSamples |
            Select-Object -First 1
        if ($null -ne $sample) {
            $value = [Math]::Round([double]$sample.CookedValue)
            if ($value -lt 0) { return [UInt64]0 }
            return [UInt64]$value
        }
    }
    catch {
        Add-ProbeError "counter ${CounterPath}: $($_.Exception.Message)"
    }
    return $null
}

function Get-GpuSnapshot {
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($null -eq $nvidia) {
        return [ordered]@{ status = "unavailable"; gpus = @(); compute_processes = @() }
    }
    $gpuResult = Invoke-NativeOutput -Executable $nvidia.Source -Arguments @('--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits')
    $gpuLines = @($gpuResult.lines)
    if ($gpuResult.exit_code -ne 0) {
        Add-ProbeError "nvidia-smi GPU query failed: $($gpuLines -join ' ')"
        return [ordered]@{ status = "error"; gpus = @(); compute_processes = @() }
    }
    $gpus = foreach ($line in $gpuLines) {
        $parts = ([string]$line -split ',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 4) {
            [ordered]@{
                name = $parts[0]
                memory_total_mib = $parts[1]
                memory_used_mib = $parts[2]
                memory_free_mib = $parts[3]
            }
        }
    }
    $computeResult = Invoke-NativeOutput -Executable $nvidia.Source -Arguments @('--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits')
    $computeLines = @($computeResult.lines)
    $compute = foreach ($line in $computeLines) {
        $parts = ([string]$line -split ',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 3) {
            [ordered]@{ pid = $parts[0]; process = $parts[1]; used_memory_mib = $parts[2] }
        }
    }
    return [ordered]@{ status = "ok"; gpus = @($gpus); compute_processes = @($compute) }
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Json,
        [Parameter(Mandatory)][bool]$Overwrite
    )
    $full = [IO.Path]::GetFullPath($Path)
    $parent = [IO.Path]::GetDirectoryName($full)
    if ([string]::IsNullOrWhiteSpace($parent) -or -not (Test-Path -LiteralPath $parent -PathType Container)) {
        throw "preflight output parent does not exist: $parent"
    }
    if ((Test-Path -LiteralPath $full) -and -not $Overwrite) {
        throw "preflight output exists; pass -ForceOutput to replace it: $full"
    }
    $temporary = "$full.tmp-$PID-$([Guid]::NewGuid().ToString('N'))"
    try {
        [IO.File]::WriteAllText($temporary, $Json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
        if ($Overwrite) {
            Move-Item -LiteralPath $temporary -Destination $full -Force
        }
        else {
            [IO.File]::Move($temporary, $full)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}

$physical = Get-PhysicalMemorySnapshot
$committedBytes = Get-CounterValue -CounterPath '\Memory\Committed Bytes'
$commitLimit = Get-CounterValue -CounterPath '\Memory\Commit Limit'
$driveName = ([IO.Path]::GetPathRoot($repoRoot)).TrimEnd('\').TrimEnd(':')
$drive = Get-PSDrive -Name $driveName -PSProvider FileSystem -ErrorAction SilentlyContinue
$diskFree = if ($null -ne $drive -and $null -ne $drive.Free) { [UInt64]$drive.Free } else { $null }
$diskUsed = if ($null -ne $drive -and $null -ne $drive.Used) { [UInt64]$drive.Used } else { $null }
$diskSource = if ($null -ne $diskFree -and $null -ne $diskUsed -and ($diskFree -gt 0 -or $diskUsed -gt 0)) { "Get-PSDrive" } else { $null }
if ($null -eq $diskSource) {
    try {
        $driveInfo = New-Object System.IO.DriveInfo(("{0}:\" -f $driveName))
        if ($driveInfo.IsReady -and [UInt64]$driveInfo.TotalSize -gt 0) {
            $diskFree = [UInt64]$driveInfo.AvailableFreeSpace
            $diskUsed = if ($driveInfo.TotalSize -ge $diskFree) {
                [UInt64]$driveInfo.TotalSize - $diskFree
            }
            else {
                $null
            }
            if ($null -ne $diskUsed) { $diskSource = "System.IO.DriveInfo" }
        }
    }
    catch {
        Add-ProbeError "disk: $($_.Exception.Message)"
    }
}
$disk = [ordered]@{
    name = $driveName
    free_bytes = $diskFree
    used_bytes = $diskUsed
    source = $diskSource
}

$parentProcessMap = Get-ParentProcessMap
$matchingProcessObjects = @(Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.ProcessName -match '(?i)llama|python|cargo|rustc|ninja|cmake' } |
    ForEach-Object { $_ })
$llamaProcessObjects = @($matchingProcessObjects |
    Where-Object { $_.ProcessName -match '(?i)llama' })
$trackedProcessObjects = @($matchingProcessObjects |
    Sort-Object -Property PrivateMemorySize64 -Descending |
    Select-Object -First 64)
$trackedProcesses = @($trackedProcessObjects |
    ForEach-Object { Get-SafeProcessRecord -Process $_ -ParentMap $parentProcessMap })
$llamaProcesses = @($llamaProcessObjects |
    ForEach-Object { Get-SafeProcessRecord -Process $_ -ParentMap $parentProcessMap })
$physicalComplete = $null -ne $physical.total_physical_bytes -and $null -ne $physical.available_physical_bytes
$commitComplete = $null -ne $committedBytes -and $null -ne $commitLimit
$report = [ordered]@{
    schema = $contract
    generated_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
    repository_root = $repoRoot
    git = Get-GitIdentity
    physical_memory = $physical
    committed_memory = [ordered]@{
        committed_bytes = $committedBytes
        limit_bytes = $commitLimit
        headroom_bytes = if ($null -ne $committedBytes -and $null -ne $commitLimit -and $commitLimit -ge $committedBytes) { $commitLimit - $committedBytes } else { $null }
    }
    disk = $disk
    gpu = Get-GpuSnapshot
    tracked_processes = @($trackedProcesses)
    llama_processes = @($llamaProcesses)
    admission = [ordered]@{
        llama_processes_absent = ($llamaProcesses.Count -eq 0)
        physical_memory_probe_complete = $physicalComplete
        commit_probe_complete = $commitComplete
        owner_review_required = $true
        status = if ($llamaProcesses.Count -gt 0) { "blocked" } elseif (-not $physicalComplete) { "blocked" } elseif (-not $commitComplete) { "blocked" } else { "review" }
    }
    probe_errors = @($probeErrors)
    redaction = [ordered]@{
        command_lines = "omitted"
        secrets = "not inspected"
    }
}
$json = $report | ConvertTo-Json -Depth 12
if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
    Write-AtomicJson -Path $OutputPath -Json $json -Overwrite ([bool]$ForceOutput)
}
if ($AsJson) {
    Write-Output $json
}
else {
    Write-Output "resource-preflight: $($report.admission.status); llama=$($llamaProcesses.Count); physical_probe=$physicalComplete; commit_probe=$commitComplete"
    if ($probeErrors.Count -gt 0) { Write-Output "probe-errors=$($probeErrors.Count)" }
}
