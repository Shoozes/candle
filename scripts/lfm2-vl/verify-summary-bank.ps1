[CmdletBinding()]
param(
    [string]$Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($scriptRoot)) {
    $scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if ([string]::IsNullOrWhiteSpace($Path)) {
    $Path = Join-Path -Path $scriptRoot -ChildPath "..\..\summary_bank.json"
}

function Fail-SummaryBank {
    param([Parameter(Mandatory)][string]$Message)
    throw "summary-bank: $Message"
}

function ConvertTo-PlainHashtable {
    param([AllowNull()][object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    if ($Value -is [Collections.IDictionary]) {
        $result = @{}
        foreach ($key in $Value.Keys) {
            $result[[string]$key] = ConvertTo-PlainHashtable $Value[$key]
        }
        return $result
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        $result = @()
        foreach ($item in $Value) {
            $result += ,(ConvertTo-PlainHashtable $item)
        }
        return ,$result
    }
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $properties = @($Value.PSObject.Properties)
        $result = @{}
        foreach ($property in $properties) {
            $result[[string]$property.Name] = ConvertTo-PlainHashtable $property.Value
        }
        return $result
    }
    return $Value
}

$repoRoot = [IO.Path]::GetFullPath((Join-Path -Path $scriptRoot -ChildPath "..\.."))
$repoPrefix = $repoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$bankPath = [IO.Path]::GetFullPath($Path)
if (-not (Test-Path -LiteralPath $bankPath -PathType Leaf)) {
    Fail-SummaryBank "file does not exist: $bankPath"
}

try {
    $rawBank = Get-Content -LiteralPath $bankPath -Raw -Encoding utf8 | ConvertFrom-Json
    $bank = ConvertTo-PlainHashtable $rawBank
}
catch {
    Fail-SummaryBank "invalid JSON in ${bankPath}: $($_.Exception.Message)"
}

if ($bank.schema -ne "gknome_summary_bank_v2") {
    Fail-SummaryBank "unexpected schema '$($bank.schema)'"
}
if ($bank.defaults -isnot [Collections.IDictionary]) {
    Fail-SummaryBank "defaults must be an object"
}
if ($bank.groups -isnot [Collections.IDictionary] -or $bank.groups.Count -eq 0) {
    Fail-SummaryBank "groups must be a non-empty object"
}

$maxKilobytes = [int]$bank.defaults.max_kb
if ($maxKilobytes -lt 64 -or $maxKilobytes -gt 384) {
    Fail-SummaryBank "defaults.max_kb must be between 64 and 384"
}
$maxBytes = [int64]$maxKilobytes * 1KB

$requiredSkips = @(".git", ".tools/.secrets", ".venv", "artifacts", "downloads", "models", "target", "__pycache__")
$skipSet = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($skip in @($bank.defaults.skip_dirs)) {
    [void]$skipSet.Add(([string]$skip).Replace('\', '/').Trim('/'))
}
foreach ($requiredSkip in $requiredSkips) {
    if (-not $skipSet.Contains($requiredSkip)) {
        Fail-SummaryBank "defaults.skip_dirs is missing '$requiredSkip'"
    }
}

$forbiddenPrefixes = @(".git", ".tools/.secrets", ".venv", "artifacts", "downloads", "models", "target", "__pycache__")
$memberships = @{}
$groupBytes = @{}
$groupFiles = @{}

foreach ($groupName in @($bank.groups.Keys | Sort-Object)) {
    $group = $bank.groups[$groupName]
    if ($group -isnot [Collections.IDictionary]) {
        Fail-SummaryBank "group '$groupName' must be an object"
    }
    if ([string]::IsNullOrWhiteSpace([string]$group.description)) {
        Fail-SummaryBank "group '$groupName' needs a description"
    }
    if ($group.status -notin @("active", "archived")) {
        Fail-SummaryBank "group '$groupName' has invalid status '$($group.status)'"
    }
    $paths = @($group.paths)
    if ($paths.Count -eq 0) {
        Fail-SummaryBank "group '$groupName' has no paths"
    }

    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    [int64]$bytes = 0
    foreach ($entry in $paths) {
        $relative = ([string]$entry).Replace('\', '/').Trim()
        if ([string]::IsNullOrWhiteSpace($relative)) {
            Fail-SummaryBank "group '$groupName' contains an empty path"
        }
        if ([IO.Path]::IsPathRooted($relative) -or $relative.Split('/') -contains "..") {
            Fail-SummaryBank "group '$groupName' path must stay repo-relative: '$relative'"
        }
        if ($relative.IndexOfAny([char[]]"*?[") -ge 0) {
            Fail-SummaryBank "group '$groupName' uses an unbounded wildcard path: '$relative'"
        }
        $normalized = if ($relative.StartsWith("./", [StringComparison]::Ordinal)) {
            $relative.Substring(2)
        }
        else {
            $relative
        }
        foreach ($forbidden in $forbiddenPrefixes) {
            if ($normalized -eq $forbidden -or $normalized.StartsWith("$forbidden/", [StringComparison]::OrdinalIgnoreCase)) {
                Fail-SummaryBank "group '$groupName' routes excluded path '$relative'"
            }
        }
        if ($normalized -ieq "Cargo.lock") {
            Fail-SummaryBank "group '$groupName' routes the ignored verifier-only Cargo.lock"
        }
        if (-not $seen.Add($normalized)) {
            Fail-SummaryBank "group '$groupName' repeats '$normalized'"
        }

        $nativeRelative = $normalized.Replace('/', [IO.Path]::DirectorySeparatorChar)
        $fullPath = [IO.Path]::GetFullPath((Join-Path -Path $repoRoot -ChildPath $nativeRelative))
        if (-not $fullPath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            Fail-SummaryBank "group '$groupName' escapes the repository: '$relative'"
        }
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            Fail-SummaryBank "group '$groupName' path is missing or not a file: '$relative'"
        }
        $bytes += (Get-Item -LiteralPath $fullPath).Length
        if (-not $memberships.ContainsKey($normalized)) {
            $memberships[$normalized] = [Collections.Generic.List[string]]::new()
        }
        $memberships[$normalized].Add($groupName)
    }
    if ($bytes -gt $maxBytes) {
        Fail-SummaryBank "group '$groupName' is $([Math]::Ceiling($bytes / 1KB)) KiB, above the $maxKilobytes KiB ceiling"
    }
    $groupBytes[$groupName] = $bytes
    $groupFiles[$groupName] = $seen.Count
}

$defaultNames = @($bank.defaults.groups)
if ($defaultNames.Count -eq 0) {
    Fail-SummaryBank "defaults.groups must not be empty"
}
$defaultPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($groupName in $defaultNames) {
    if (-not $bank.groups.Contains([string]$groupName)) {
        Fail-SummaryBank "default group '$groupName' does not exist"
    }
    if ($bank.groups[[string]$groupName].status -ne "active") {
        Fail-SummaryBank "default group '$groupName' is not active"
    }
    foreach ($entry in @($bank.groups[[string]$groupName].paths)) {
        $defaultPath = ([string]$entry).Replace('\', '/')
        if ($defaultPath.StartsWith("./", [StringComparison]::Ordinal)) {
            $defaultPath = $defaultPath.Substring(2)
        }
        [void]$defaultPaths.Add($defaultPath)
    }
}

[int64]$defaultBytes = 0
foreach ($relative in $defaultPaths) {
    $nativeRelative = $relative.Replace('/', [IO.Path]::DirectorySeparatorChar)
    $defaultBytes += (Get-Item -LiteralPath (Join-Path -Path $repoRoot -ChildPath $nativeRelative)).Length
}
if ($defaultBytes -gt $maxBytes) {
    Fail-SummaryBank "default union is $([Math]::Ceiling($defaultBytes / 1KB)) KiB, above the $maxKilobytes KiB ceiling"
}

foreach ($entry in $memberships.GetEnumerator()) {
    if ($entry.Value.Count -gt 4) {
        Fail-SummaryBank "path '$($entry.Key)' fans out to $($entry.Value.Count) groups: $($entry.Value -join ', ')"
    }
}

foreach ($groupName in @($groupBytes.Keys | Sort-Object)) {
    "summary-bank group={0} files={1} kib={2:N1}" -f $groupName, $groupFiles[$groupName], ($groupBytes[$groupName] / 1KB)
}
"summary-bank defaults files={0} kib={1:N1} max_kib={2}" -f $defaultPaths.Count, ($defaultBytes / 1KB), $maxKilobytes
"summary-bank: passed"
