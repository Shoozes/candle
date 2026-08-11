[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$FilePath,

    [Parameter()]
    [AllowEmptyCollection()]
    [string[]]$ArgumentList = @(),

    [Parameter()]
    [ValidateRange(1, 7200)]
    [int]$TimeoutSeconds = 900,

    [Parameter()]
    [ValidateRange(134217728, 274877906944)]
    [UInt64]$MaxJobMemoryBytes = 25769803776,

    [Parameter()]
    [ValidateRange(50, 5000)]
    [int]$PollMilliseconds = 250,

    [Parameter()]
    [ValidateSet("Name", "Executable")]
    [string]$ConcurrencyScope = "Name",

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$WorkingDirectory = (Get-Location).Path,

    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$EvidencePath,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$LogPath,

    [Parameter()]
    [switch]$ForceEvidence,

    [Parameter()]
    [switch]$ForceLog,

    [Parameter()]
    [switch]$RedactArguments,

    [Parameter()]
    [switch]$AllowCudaGraphs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-ProcessCounterBytes {
    param([Parameter(Mandatory)][Int64]$Value)

    if ($Value -le 0) {
        return [UInt64]0
    }
    return [UInt64]$Value
}

$contract = "candle-lfm2-vl-bounded-oracle-v1"
$killOnJobClose = [UInt32]0x00002000
$processMemoryLimit = [UInt32]0x00000100
$jobMemoryLimit = [UInt32]0x00000200
$createSuspended = [UInt32]0x00000004
$createUnicodeEnvironment = [UInt32]0x00000400
$createNoWindow = [UInt32]0x08000000
$startfUseStdHandles = [UInt32]0x00000100
$genericRead = [UInt32]2147483648
$genericWrite = [UInt32]0x40000000
$fileShareRead = [UInt32]0x00000001
$fileShareWrite = [UInt32]0x00000002
$createNew = [UInt32]0x00000001
$createAlways = [UInt32]0x00000002
$openExisting = [UInt32]0x00000003
$fileAttributeNormal = [UInt32]0x00000080
$invalidSuspendCount = [UInt32]::MaxValue

if (-not ("CandleLfm2VlOracleJobNative" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Text;

[StructLayout(LayoutKind.Sequential)]
public struct CandleLfm2VlBasicLimitInformation
{
    public long PerProcessUserTimeLimit;
    public long PerJobUserTimeLimit;
    public uint LimitFlags;
    public UIntPtr MinimumWorkingSetSize;
    public UIntPtr MaximumWorkingSetSize;
    public uint ActiveProcessLimit;
    public UIntPtr Affinity;
    public uint PriorityClass;
    public uint SchedulingClass;
}

[StructLayout(LayoutKind.Sequential)]
public struct CandleLfm2VlIoCounters
{
    public ulong ReadOperationCount;
    public ulong WriteOperationCount;
    public ulong OtherOperationCount;
    public ulong ReadTransferCount;
    public ulong WriteTransferCount;
    public ulong OtherTransferCount;
}

[StructLayout(LayoutKind.Sequential)]
public struct CandleLfm2VlExtendedLimitInformation
{
    public CandleLfm2VlBasicLimitInformation BasicLimitInformation;
    public CandleLfm2VlIoCounters IoInfo;
    public UIntPtr ProcessMemoryLimit;
    public UIntPtr JobMemoryLimit;
    public UIntPtr PeakProcessMemoryUsed;
    public UIntPtr PeakJobMemoryUsed;
}

[StructLayout(LayoutKind.Sequential)]
public struct CandleLfm2VlStartupInformation
{
    public uint Size;
    public IntPtr Reserved;
    public IntPtr Desktop;
    public IntPtr Title;
    public uint X;
    public uint Y;
    public uint XSize;
    public uint YSize;
    public uint XCountChars;
    public uint YCountChars;
    public uint FillAttribute;
    public uint Flags;
    public ushort ShowWindow;
    public ushort ReservedSize;
    public IntPtr ReservedBytes;
    public IntPtr StandardInput;
    public IntPtr StandardOutput;
    public IntPtr StandardError;
}

[StructLayout(LayoutKind.Sequential)]
public struct CandleLfm2VlProcessInformation
{
    public IntPtr Process;
    public IntPtr Thread;
    public uint ProcessId;
    public uint ThreadId;
}

[StructLayout(LayoutKind.Sequential)]
public struct CandleLfm2VlSecurityAttributes
{
    public int Length;
    public IntPtr SecurityDescriptor;
    [MarshalAs(UnmanagedType.Bool)]
    public bool InheritHandle;
}

public static class CandleLfm2VlOracleJobNative
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CreateProcessW(
        string applicationName,
        StringBuilder commandLine,
        IntPtr processAttributes,
        IntPtr threadAttributes,
        bool inheritHandles,
        uint creationFlags,
        IntPtr environment,
        string currentDirectory,
        ref CandleLfm2VlStartupInformation startupInformation,
        out CandleLfm2VlProcessInformation processInformation);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool SetInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        uint informationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool QueryInformationJobObject(
        IntPtr job,
        int informationClass,
        IntPtr information,
        uint informationLength,
        IntPtr returnLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool TerminateProcess(IntPtr process, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GetExitCodeProcess(IntPtr process, out uint exitCode);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern IntPtr CreateFileW(
        string fileName,
        uint desiredAccess,
        uint shareMode,
        ref CandleLfm2VlSecurityAttributes securityAttributes,
        uint creationDisposition,
        uint flagsAndAttributes,
        IntPtr templateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CloseHandle(IntPtr handle);

    public static string QuoteCommandLineArgument(string argument)
    {
        if (argument == null)
        {
            throw new ArgumentNullException("argument");
        }
        if (argument.Length > 0 && argument.IndexOfAny(new[] { ' ', '\t', '\n', '\v', '"' }) < 0)
        {
            return argument;
        }

        StringBuilder quoted = new StringBuilder();
        quoted.Append('"');
        int backslashes = 0;
        foreach (char current in argument)
        {
            if (current == '\\')
            {
                backslashes++;
                continue;
            }
            if (current == '"')
            {
                quoted.Append('\\', backslashes * 2 + 1);
                quoted.Append('"');
                backslashes = 0;
                continue;
            }
            quoted.Append('\\', backslashes);
            quoted.Append(current);
            backslashes = 0;
        }
        quoted.Append('\\', backslashes * 2);
        quoted.Append('"');
        return quoted.ToString();
    }
}
'@ -ErrorAction Stop
}

if (-not ("CandleLfm2VlMemoryNative" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class CandleLfm2VlMemoryNative
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

    public static ulong GetTotalPhysicalMemoryBytes()
    {
        var status = new MemoryStatusEx
        {
            Length = (uint)Marshal.SizeOf<MemoryStatusEx>()
        };
        if (!GlobalMemoryStatusEx(ref status) || status.TotalPhysical == 0)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error(),
                "GlobalMemoryStatusEx did not return physical memory");
        }
        return status.TotalPhysical;
    }
}
'@ -ErrorAction Stop
}

function Get-TotalPhysicalMemory {
    try {
        $computer = Get-CimInstance Win32_ComputerSystem -ErrorAction Stop
        $bytes = [UInt64]$computer.TotalPhysicalMemory
        if ($bytes -gt 0) {
            return [pscustomobject]@{
                Bytes = $bytes
                Source = "Win32_ComputerSystem"
            }
        }
        $cimError = "Win32_ComputerSystem returned zero physical memory"
    }
    catch {
        $cimError = $_.Exception.Message
    }

    try {
        $bytes = [CandleLfm2VlMemoryNative]::GetTotalPhysicalMemoryBytes()
        return [pscustomobject]@{
            Bytes = [UInt64]$bytes
            Source = "GlobalMemoryStatusEx"
        }
    }
    catch {
        throw "unable to determine host physical memory; CIM=$cimError; GlobalMemoryStatusEx=$($_.Exception.Message)"
    }
}

function Get-ResolvedLeafPath {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Label
    )

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).ProviderPath
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
        throw "$Label is not a regular file: $resolved"
    }
    return [System.IO.Path]::GetFullPath($resolved)
}

function Get-ResolvedDirectoryPath {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Label
    )

    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).ProviderPath
    if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
        throw "$Label is not a directory: $resolved"
    }
    return [System.IO.Path]::GetFullPath($resolved)
}

function Get-EvidenceFullPath {
    param([Parameter(Mandatory)][string]$Path)

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    }
    else {
        Join-Path -Path (Get-Location).Path -ChildPath $Path
    }
    $fullPath = [System.IO.Path]::GetFullPath($candidate)
    $parent = [System.IO.Path]::GetDirectoryName($fullPath)
    if ([string]::IsNullOrWhiteSpace($parent) -or
        -not (Test-Path -LiteralPath $parent -PathType Container)) {
        throw "evidence parent directory does not exist: $parent"
    }
    return $fullPath
}

function New-BoundedJob {
    param(
        [Parameter(Mandatory)]
        [UInt64]$MemoryBytes,

        [Parameter(Mandatory)]
        [string]$Name
    )

    $job = [CandleLfm2VlOracleJobNative]::CreateJobObjectW([IntPtr]::Zero, $Name)
    if ($job -eq [IntPtr]::Zero) {
        throw [ComponentModel.Win32Exception]::new(
            [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        )
    }

    $information = [CandleLfm2VlExtendedLimitInformation]::new()
    $basic = $information.BasicLimitInformation
    $basic.LimitFlags = $script:killOnJobClose -bor
        $script:processMemoryLimit -bor
        $script:jobMemoryLimit
    $information.BasicLimitInformation = $basic
    $information.ProcessMemoryLimit = [UIntPtr]::new($MemoryBytes)
    $information.JobMemoryLimit = [UIntPtr]::new($MemoryBytes)
    $length = [Runtime.InteropServices.Marshal]::SizeOf($information)
    $buffer = [Runtime.InteropServices.Marshal]::AllocHGlobal($length)
    try {
        [Runtime.InteropServices.Marshal]::StructureToPtr($information, $buffer, $false)
        if (-not [CandleLfm2VlOracleJobNative]::SetInformationJobObject(
                $job,
                9,
                $buffer,
                [UInt32]$length
            )) {
            $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            [void][CandleLfm2VlOracleJobNative]::CloseHandle($job)
            throw [ComponentModel.Win32Exception]::new($errorCode)
        }
    }
    finally {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($buffer)
    }
    return $job
}

function Get-JobPeakMemoryBytes {
    param([Parameter(Mandatory)][IntPtr]$Job)

    $information = [CandleLfm2VlExtendedLimitInformation]::new()
    $length = [Runtime.InteropServices.Marshal]::SizeOf($information)
    $buffer = [Runtime.InteropServices.Marshal]::AllocHGlobal($length)
    try {
        if (-not [CandleLfm2VlOracleJobNative]::QueryInformationJobObject(
                $Job,
                9,
                $buffer,
                [UInt32]$length,
                [IntPtr]::Zero
            )) {
            throw [ComponentModel.Win32Exception]::new(
                [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            )
        }
        $information = [Runtime.InteropServices.Marshal]::PtrToStructure(
            $buffer,
            [type][CandleLfm2VlExtendedLimitInformation]
        )
        return $information.PeakJobMemoryUsed.ToUInt64()
    }
    finally {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($buffer)
    }
}

function New-UnicodeEnvironmentBlock {
    param([Parameter(Mandatory)][bool]$DisableCudaGraphs)

    $environment = [Collections.Generic.SortedDictionary[string, string]]::new(
        [StringComparer]::OrdinalIgnoreCase
    )
    foreach ($entry in [Environment]::GetEnvironmentVariables().GetEnumerator()) {
        $environment[[string]$entry.Key] = [string]$entry.Value
    }
    if ($DisableCudaGraphs) {
        $environment["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    }
    else {
        [void]$environment.Remove("GGML_CUDA_DISABLE_GRAPHS")
    }

    $entries = foreach ($entry in $environment.GetEnumerator()) {
        "$($entry.Key)=$($entry.Value)"
    }
    $block = ($entries -join [char]0) + [char]0 + [char]0
    $bytes = [Text.Encoding]::Unicode.GetBytes($block)
    $pointer = [Runtime.InteropServices.Marshal]::AllocHGlobal($bytes.Length)
    try {
        [Runtime.InteropServices.Marshal]::Copy($bytes, 0, $pointer, $bytes.Length)
    }
    catch {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($pointer)
        throw
    }
    return $pointer
}

function New-SuspendedProcess {
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$Directory,
        [Parameter(Mandatory)][bool]$DisableCudaGraphs,
        [Parameter()][string]$CombinedLogPath,
        [Parameter(Mandatory)][bool]$OverwriteLog
    )

    $tokens = @(
        [CandleLfm2VlOracleJobNative]::QuoteCommandLineArgument($Executable)
        foreach ($argument in $Arguments) {
            [CandleLfm2VlOracleJobNative]::QuoteCommandLineArgument($argument)
        }
    )
    $commandLine = [Text.StringBuilder]::new($tokens -join " ")
    $startup = [CandleLfm2VlStartupInformation]::new()
    $startup.Size = [UInt32][Runtime.InteropServices.Marshal]::SizeOf($startup)
    $information = [CandleLfm2VlProcessInformation]::new()
    $environment = New-UnicodeEnvironmentBlock -DisableCudaGraphs $DisableCudaGraphs
    $logHandle = [IntPtr]::Zero
    $inputHandle = [IntPtr]::Zero
    try {
        $inheritHandles = $false
        if (-not [string]::IsNullOrWhiteSpace($CombinedLogPath)) {
            $disposition = if ($OverwriteLog) {
                $script:createAlways
            }
            else {
                $script:createNew
            }
            $security = [CandleLfm2VlSecurityAttributes]::new()
            $security.Length = [Runtime.InteropServices.Marshal]::SizeOf($security)
            $security.InheritHandle = $true
            $logHandle = [CandleLfm2VlOracleJobNative]::CreateFileW(
                $CombinedLogPath,
                $script:genericWrite,
                ($script:fileShareRead -bor $script:fileShareWrite),
                [ref]$security,
                $disposition,
                $script:fileAttributeNormal,
                [IntPtr]::Zero
            )
            if ($logHandle -eq [IntPtr]::new(-1)) {
                throw [ComponentModel.Win32Exception]::new(
                    [Runtime.InteropServices.Marshal]::GetLastWin32Error(),
                    "creating bounded-oracle combined log $CombinedLogPath"
                )
            }
            $inputHandle = [CandleLfm2VlOracleJobNative]::CreateFileW(
                "NUL",
                $script:genericRead,
                ($script:fileShareRead -bor $script:fileShareWrite),
                [ref]$security,
                $script:openExisting,
                $script:fileAttributeNormal,
                [IntPtr]::Zero
            )
            if ($inputHandle -eq [IntPtr]::new(-1)) {
                throw [ComponentModel.Win32Exception]::new(
                    [Runtime.InteropServices.Marshal]::GetLastWin32Error(),
                    "opening NUL for bounded-oracle standard input"
                )
            }
            $startup.Flags = $script:startfUseStdHandles
            $startup.StandardInput = $inputHandle
            $startup.StandardOutput = $logHandle
            $startup.StandardError = $logHandle
            $inheritHandles = $true
        }
        $creationFlags = $script:createSuspended -bor $script:createUnicodeEnvironment
        if ([string]::IsNullOrWhiteSpace($CombinedLogPath)) {
            $creationFlags = $creationFlags -bor $script:createNoWindow
        }
        if (-not [CandleLfm2VlOracleJobNative]::CreateProcessW(
                $Executable,
                $commandLine,
                [IntPtr]::Zero,
                [IntPtr]::Zero,
                $inheritHandles,
                $creationFlags,
                $environment,
                $Directory,
                [ref]$startup,
                [ref]$information
            )) {
            throw [ComponentModel.Win32Exception]::new(
                [Runtime.InteropServices.Marshal]::GetLastWin32Error(),
                "creating suspended bounded process $Executable"
            )
        }
    }
    finally {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($environment)
        if ($inputHandle -ne [IntPtr]::Zero -and $inputHandle -ne [IntPtr]::new(-1)) {
            [void][CandleLfm2VlOracleJobNative]::CloseHandle($inputHandle)
        }
        if ($logHandle -ne [IntPtr]::Zero -and $logHandle -ne [IntPtr]::new(-1)) {
            [void][CandleLfm2VlOracleJobNative]::CloseHandle($logHandle)
        }
    }
    return [pscustomobject]@{
        ProcessId = [int]$information.ProcessId
        ProcessHandle = $information.Process
        ThreadHandle = $information.Thread
    }
}

function Get-NativeProcessExitCode {
    param([Parameter(Mandatory)][IntPtr]$ProcessHandle)

    $exitCode = [UInt32]0
    if (-not [CandleLfm2VlOracleJobNative]::GetExitCodeProcess(
            $ProcessHandle,
            [ref]$exitCode
        )) {
        throw [ComponentModel.Win32Exception]::new(
            [Runtime.InteropServices.Marshal]::GetLastWin32Error(),
            "reading bounded process exit code"
        )
    }
    if ($exitCode -eq [UInt32]259) {
        throw "bounded process is still active while reading its exit code"
    }
    return $exitCode
}

function Test-ExactPidAbsent {
    param([Parameter(Mandatory)][int]$ProcessId)

    try {
        $probe = [Diagnostics.Process]::GetProcessById($ProcessId)
        try {
            return $probe.HasExited
        }
        finally {
            $probe.Dispose()
        }
    }
    catch [ArgumentException] {
        return $true
    }
}

function Write-OracleEvidence {
    param(
        [Parameter(Mandatory)]
        [System.Collections.IDictionary]$Evidence,

        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [bool]$Overwrite
    )

    if ((Test-Path -LiteralPath $Path) -and -not $Overwrite) {
        throw "evidence path already exists; use -ForceEvidence to replace it: $Path"
    }
    $temporary = "$Path.tmp-$PID-$([Guid]::NewGuid().ToString('N'))"
    try {
        $json = $Evidence | ConvertTo-Json -Depth 6
        [IO.File]::WriteAllText(
            $temporary,
            $json + [Environment]::NewLine,
            [Text.UTF8Encoding]::new($false)
        )
        if ($Overwrite) {
            Move-Item -LiteralPath $temporary -Destination $Path -Force
        }
        else {
            [IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

$resolvedExecutable = Get-ResolvedLeafPath -Path $FilePath -Label "oracle executable"
$resolvedWorkingDirectory = Get-ResolvedDirectoryPath -Path $WorkingDirectory -Label "working directory"
$resolvedEvidencePath = Get-EvidenceFullPath -Path $EvidencePath
$resolvedLogPath = if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $null
}
else {
    Get-EvidenceFullPath -Path $LogPath
}
if ((Test-Path -LiteralPath $resolvedEvidencePath) -and -not $ForceEvidence) {
    throw "evidence path already exists; use -ForceEvidence to replace it: $resolvedEvidencePath"
}
if ($null -ne $resolvedLogPath) {
    if ($resolvedLogPath.Equals($resolvedEvidencePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "log path and evidence path must be different"
    }
    if ((Test-Path -LiteralPath $resolvedLogPath) -and -not $ForceLog) {
        throw "log path already exists; use -ForceLog to replace it: $resolvedLogPath"
    }
}
foreach ($argument in $ArgumentList) {
    if ($null -eq $argument -or $argument.Contains([char]0)) {
        throw "oracle arguments must be non-null and cannot contain NUL"
    }
}

$physicalMemory = Get-TotalPhysicalMemory
$totalPhysicalBytes = [UInt64]$physicalMemory.Bytes
$physicalMemorySource = [string]$physicalMemory.Source
$maximumSafeLimit = $totalPhysicalBytes - [UInt64]($totalPhysicalBytes / 4)
if ($MaxJobMemoryBytes -gt $maximumSafeLimit) {
    throw "requested job memory ceiling $MaxJobMemoryBytes exceeds 75% of host physical memory $totalPhysicalBytes"
}

$pathBytes = [Text.Encoding]::UTF8.GetBytes($resolvedExecutable.ToUpperInvariant())
$pathHasher = [Security.Cryptography.SHA256]::Create()
try {
    $pathHash = ([BitConverter]::ToString($pathHasher.ComputeHash($pathBytes))).Replace("-", "")
}
finally {
    $pathHasher.Dispose()
}
$mutexName = "Local\CandleLfm2VlOracle-$($pathHash.Substring(0, 24))"
$mutex = [Threading.Mutex]::new($false, $mutexName)
$mutexAcquired = $false
try {
    try {
        $mutexAcquired = $mutex.WaitOne(0)
    }
    catch [Threading.AbandonedMutexException] {
        $mutexAcquired = $true
    }
    if (-not $mutexAcquired) {
        throw "refusing concurrent launch: another bounded wrapper owns $resolvedExecutable"
    }

    $processName = [IO.Path]::GetFileNameWithoutExtension($resolvedExecutable)
    $sameNameProcesses = @(Get-Process -Name $processName -ErrorAction SilentlyContinue)
    if ($ConcurrencyScope -eq "Name" -and $sameNameProcesses.Count -ne 0) {
        $existingIds = ($sameNameProcesses.Id | Sort-Object) -join ", "
        throw "refusing concurrent launch: process name $processName is already running as PID(s) $existingIds"
    }
    if ($ConcurrencyScope -eq "Executable" -and $sameNameProcesses.Count -ne 0) {
        $matchingIds = [Collections.Generic.List[int]]::new()
        $unresolvedIds = [Collections.Generic.List[int]]::new()
        foreach ($candidate in $sameNameProcesses) {
            try {
                $candidatePath = $candidate.Path
                if ([string]::IsNullOrWhiteSpace($candidatePath)) {
                    $unresolvedIds.Add($candidate.Id)
                    continue
                }
                $candidateFullPath = [IO.Path]::GetFullPath($candidatePath)
                if ($candidateFullPath.Equals($resolvedExecutable, [StringComparison]::OrdinalIgnoreCase)) {
                    $matchingIds.Add($candidate.Id)
                }
            }
            catch {
                $unresolvedIds.Add($candidate.Id)
            }
        }
        if ($unresolvedIds.Count -ne 0) {
            $ids = ($unresolvedIds | Sort-Object) -join ", "
            throw "refusing concurrent launch: executable identity is unavailable for same-name PID(s) $ids"
        }
        if ($matchingIds.Count -ne 0) {
            $ids = ($matchingIds | Sort-Object) -join ", "
            throw "refusing concurrent launch: executable $resolvedExecutable is already running as PID(s) $ids"
        }
    }

    $executableInfo = Get-Item -LiteralPath $resolvedExecutable
    $executableHash = (Get-FileHash -LiteralPath $resolvedExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    $jobName = "Local\CandleLfm2VlOracleJob-$([Guid]::NewGuid().ToString('N'))"
    $job = [IntPtr]::Zero
    $process = $null
    $processId = $null
    $nativeProcessHandle = [IntPtr]::Zero
    $nativeThreadHandle = [IntPtr]::Zero
    $jobAssigned = $false
    $startedSuspended = $false
    $assignedBeforeResume = $false
    $resumed = $false
    $childExitCode = $null
    $peakPrivateBytes = [UInt64]0
    $peakWorkingSetBytes = [UInt64]0
    $peakJobMemoryBytes = [UInt64]0
    $terminationReason = "setup_failure"
    $failure = $null
    $cleanupFailure = $null
    $startedAtUtc = [DateTimeOffset]::UtcNow
    $timer = [Diagnostics.Stopwatch]::StartNew()

    try {
        $job = New-BoundedJob -MemoryBytes $MaxJobMemoryBytes -Name $jobName
        $nativeProcess = New-SuspendedProcess `
            -Executable $resolvedExecutable `
            -Arguments $ArgumentList `
            -Directory $resolvedWorkingDirectory `
            -DisableCudaGraphs (-not [bool]$AllowCudaGraphs) `
            -CombinedLogPath $resolvedLogPath `
            -OverwriteLog ([bool]$ForceLog)
        $nativeProcessHandle = $nativeProcess.ProcessHandle
        $nativeThreadHandle = $nativeProcess.ThreadHandle
        $processId = $nativeProcess.ProcessId
        $startedSuspended = $true
        if (-not [CandleLfm2VlOracleJobNative]::AssignProcessToJobObject($job, $nativeProcessHandle)) {
            $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            [void][CandleLfm2VlOracleJobNative]::TerminateProcess($nativeProcessHandle, 125)
            throw [ComponentModel.Win32Exception]::new(
                $errorCode,
                "assigning PID $processId to the bounded oracle job"
            )
        }
        $jobAssigned = $true
        $assignedBeforeResume = $true
        $process = [Diagnostics.Process]::GetProcessById($processId)
        $previousSuspendCount = [CandleLfm2VlOracleJobNative]::ResumeThread($nativeThreadHandle)
        if ($previousSuspendCount -eq $invalidSuspendCount) {
            throw [ComponentModel.Win32Exception]::new(
                [Runtime.InteropServices.Marshal]::GetLastWin32Error(),
                "resuming bounded oracle PID $processId"
            )
        }
        $resumed = $true
        if (-not [CandleLfm2VlOracleJobNative]::CloseHandle($nativeThreadHandle)) {
            throw [ComponentModel.Win32Exception]::new(
                [Runtime.InteropServices.Marshal]::GetLastWin32Error(),
                "closing bounded oracle thread handle"
            )
        }
        $nativeThreadHandle = [IntPtr]::Zero
        $terminationReason = "running"

        while (-not $process.WaitForExit($PollMilliseconds)) {
            $process.Refresh()
            $privateBytes = ConvertTo-ProcessCounterBytes -Value $process.PrivateMemorySize64
            $workingSetBytes = ConvertTo-ProcessCounterBytes -Value $process.WorkingSet64
            if ($privateBytes -gt $peakPrivateBytes) {
                $peakPrivateBytes = $privateBytes
            }
            if ($workingSetBytes -gt $peakWorkingSetBytes) {
                $peakWorkingSetBytes = $workingSetBytes
            }
            if ($privateBytes -ge $MaxJobMemoryBytes) {
                $terminationReason = "memory_limit"
                if (-not [CandleLfm2VlOracleJobNative]::TerminateJobObject($job, 137)) {
                    throw [ComponentModel.Win32Exception]::new(
                        [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                    )
                }
                break
            }
            if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
                $terminationReason = "timeout"
                if (-not [CandleLfm2VlOracleJobNative]::TerminateJobObject($job, 124)) {
                    throw [ComponentModel.Win32Exception]::new(
                        [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                    )
                }
                break
            }
        }

        if (-not $process.WaitForExit(30000)) {
            throw "bounded oracle PID $processId did not exit within 30 seconds of termination"
        }
        $childExitCode = Get-NativeProcessExitCode -ProcessHandle $nativeProcessHandle
        if ($terminationReason -eq "running") {
            $terminationReason = "exited"
        }
    }
    catch {
        $failure = $_
    }
    finally {
        $timer.Stop()
        try {
            if ($null -ne $process -and -not $process.HasExited) {
                if ($jobAssigned) {
                    [void][CandleLfm2VlOracleJobNative]::TerminateJobObject($job, 125)
                }
                elseif ($nativeProcessHandle -ne [IntPtr]::Zero) {
                    [void][CandleLfm2VlOracleJobNative]::TerminateProcess($nativeProcessHandle, 125)
                }
                [void]$process.WaitForExit(30000)
            }
            elseif ($null -eq $process -and $nativeProcessHandle -ne [IntPtr]::Zero) {
                [void][CandleLfm2VlOracleJobNative]::TerminateProcess($nativeProcessHandle, 125)
            }
            if ($null -ne $process -and $process.HasExited -and $null -eq $childExitCode) {
                if ($nativeProcessHandle -ne [IntPtr]::Zero) {
                    $childExitCode = Get-NativeProcessExitCode -ProcessHandle $nativeProcessHandle
                }
                else {
                    $childExitCode = $process.ExitCode
                }
            }
            if ($nativeThreadHandle -ne [IntPtr]::Zero) {
                [void][CandleLfm2VlOracleJobNative]::CloseHandle($nativeThreadHandle)
                $nativeThreadHandle = [IntPtr]::Zero
            }
            if ($nativeProcessHandle -ne [IntPtr]::Zero) {
                [void][CandleLfm2VlOracleJobNative]::CloseHandle($nativeProcessHandle)
                $nativeProcessHandle = [IntPtr]::Zero
            }
            if ($job -ne [IntPtr]::Zero) {
                $peakJobMemoryBytes = Get-JobPeakMemoryBytes -Job $job
                [void][CandleLfm2VlOracleJobNative]::CloseHandle($job)
                $job = [IntPtr]::Zero
            }
            if ($null -ne $process -and -not $process.HasExited) {
                [void]$process.WaitForExit(30000)
            }
        }
        catch {
            $cleanupFailure = $_
        }

        $endedAtUtc = [DateTimeOffset]::UtcNow
        $pidAbsentAfterCleanup = if ($null -eq $processId) {
            $true
        }
        else {
            Test-ExactPidAbsent -ProcessId $processId
        }
        if (-not $pidAbsentAfterCleanup -and $null -eq $cleanupFailure) {
            $cleanupFailure = [Management.Automation.ErrorRecord]::new(
                [InvalidOperationException]::new("bounded oracle PID $processId remains after cleanup"),
                "BoundedOraclePidRemains",
                [Management.Automation.ErrorCategory]::ResourceBusy,
                $processId
            )
        }

        $reportedArguments = if ($RedactArguments) {
            @("<redacted:$($ArgumentList.Count)>")
        }
        else {
            @($ArgumentList)
        }
        $logInfo = if ($null -ne $resolvedLogPath -and (Test-Path -LiteralPath $resolvedLogPath -PathType Leaf)) {
            $item = Get-Item -LiteralPath $resolvedLogPath
            [pscustomobject]@{
                Bytes = [UInt64]$item.Length
                Sha256 = (Get-FileHash -LiteralPath $resolvedLogPath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
        else {
            $null
        }
        $evidence = [ordered]@{
            contract = $contract
            executable_path = $resolvedExecutable
            executable_bytes = [UInt64]$executableInfo.Length
            executable_sha256 = $executableHash
            arguments = $reportedArguments
            arguments_redacted = [bool]$RedactArguments
            combined_log_path = $resolvedLogPath
            combined_log_bytes = if ($null -ne $logInfo) { $logInfo.Bytes } else { $null }
            combined_log_sha256 = if ($null -ne $logInfo) { $logInfo.Sha256 } else { $null }
            cuda_graphs_disabled = -not [bool]$AllowCudaGraphs
            working_directory = $resolvedWorkingDirectory
            pid = $processId
            started_at_utc = $startedAtUtc.ToString("O")
            ended_at_utc = $endedAtUtc.ToString("O")
            elapsed_milliseconds = [Int64]$timer.ElapsedMilliseconds
            timeout_seconds = $TimeoutSeconds
            max_job_memory_bytes = $MaxJobMemoryBytes
            total_physical_memory_bytes = $totalPhysicalBytes
            physical_memory_source = $physicalMemorySource
            poll_milliseconds = $PollMilliseconds
            concurrency_scope = $ConcurrencyScope
            job_assigned = $jobAssigned
            started_suspended = $startedSuspended
            assigned_before_resume = $assignedBeforeResume
            resumed = $resumed
            peak_private_memory_bytes = $peakPrivateBytes
            peak_working_set_bytes = $peakWorkingSetBytes
            peak_job_memory_bytes = $peakJobMemoryBytes
            child_exit_code = $childExitCode
            termination_reason = $terminationReason
            pid_absent_after_cleanup = $pidAbsentAfterCleanup
            error = if ($null -ne $failure) {
                $failure.Exception.Message
            }
            elseif ($null -ne $cleanupFailure) {
                $cleanupFailure.Exception.Message
            }
            else {
                $null
            }
        }
        try {
            Write-OracleEvidence -Evidence $evidence -Path $resolvedEvidencePath -Overwrite ([bool]$ForceEvidence)
        }
        catch {
            if ($null -eq $cleanupFailure) {
                $cleanupFailure = $_
            }
        }
        if ($null -ne $process) {
            $process.Dispose()
        }
    }

    if ($null -ne $failure) {
        throw $failure
    }
    if ($null -ne $cleanupFailure) {
        throw $cleanupFailure
    }
    if ($terminationReason -eq "timeout") {
        throw "bounded oracle timed out after $TimeoutSeconds seconds; PID $processId is absent"
    }
    if ($terminationReason -eq "memory_limit") {
        throw "bounded oracle reached the $MaxJobMemoryBytes-byte memory ceiling; PID $processId is absent"
    }
    if ($childExitCode -ne 0) {
        throw "bounded oracle exited with code $childExitCode"
    }
    [pscustomobject]$evidence
}
finally {
    if ($mutexAcquired) {
        $mutex.ReleaseMutex()
    }
    $mutex.Dispose()
}
