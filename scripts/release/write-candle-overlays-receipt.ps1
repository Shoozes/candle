#Requires -Version 5.1

<##
.SYNOPSIS
    Writes the external identity receipt for Candle overlays MVP 0.2.0.

.DESCRIPTION
    This command is the final publication identity gate. It does not stage,
    commit, push, tag, create a release, or change repository rules. It
    requires a clean named main worktree at the operator-supplied commit,
    verifies the pinned Windows Rust toolchain and lock hash, runs all overlay
    inventory gates, and requires local and remote main/tag identity to match.
    The historical LFM2-VL MVP tag must still peel to its immutable commit.

    Run it only after the owner-authorized main and annotated tag publication.
    The receipt is written atomically outside the repository and contains no
    local filesystem paths.
##>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ExpectedHead,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ExpectedTree,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ReceiptPath,

    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$Distro = 'NVIDIA-Workbench',

    [ValidatePattern('^/[A-Za-z0-9._/-]+$')]
    [string]$WslRepositoryRoot = '/mnt/c/DevStuff/candle-mods'
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$releaseVersion = '0.2.0'
$releaseTag = 'candle-overlays-mvp-0.2.0'
$historicalTag = 'lfm2-vl-mvp-0.1.0'
$historicalCommit = 'ff885586f6d44a3d9b9ac1724032cdf5f0155384'
$upstreamBase = '6f74e7c390c717f8fd34f23ce02aceb058173370'
$expectedRustc = 'rustc 1.97.1 (8bab26f4f 2026-07-14)'
$expectedCargo = 'cargo 1.97.1 (c980f4866 2026-06-30)'
$expectedLockHash = '9b7aa15899ae8acf7b1a09b951ddba2f16462137eee2fed0db863a9d84707175'
$expectedRemote = 'https://github.com/Shoozes/candle.git'
$branch = 'main'

function Test-IsWithinRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root,
        [switch]$AllowRoot
    )

    $fullPath = [IO.Path]::GetFullPath($Path)
    $fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    if ($AllowRoot -and $fullPath.Equals($fullRoot, [StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    return $fullPath.StartsWith(
        $fullRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Invoke-WslGit {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = @(& wsl.exe -d $Distro git -C $WslRepositoryRoot @Arguments 2>&1)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($null -eq $exitCode) { $exitCode = 0 }
    if ($exitCode -ne 0) {
        $detail = ($output | ForEach-Object { "$_" }) -join [Environment]::NewLine
        throw "WSL Git failed (exit $exitCode): git $($Arguments -join ' ')`n$detail"
    }
    return @($output | ForEach-Object { "$_" })
}

function Get-SingleGitLine {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $lines = @(Invoke-WslGit -Arguments $Arguments | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($lines.Count -ne 1) {
        throw "Expected one Git result for '$($Arguments -join ' ')'; found $($lines.Count)."
    }
    return $lines[0].Trim()
}

function Invoke-WslVerifier {
    param([Parameter(Mandatory = $true)][string]$ScriptPath)

    $command = "cd -- '$WslRepositoryRoot' && bash '$ScriptPath'"
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = @(& wsl.exe -d $Distro bash -lc $command 2>&1)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($null -eq $exitCode) { $exitCode = 0 }
    $text = ($output | ForEach-Object { "$_" }) -join [Environment]::NewLine
    if ($exitCode -ne 0) {
        throw "Overlay verifier failed (exit $exitCode): $ScriptPath`n$text"
    }
    return $text
}

function Invoke-VersionCommand {
    param([Parameter(Mandatory = $true)][string]$FilePath)

    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = @(& $FilePath '--version' 2>&1)
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($null -eq $exitCode) { $exitCode = 0 }
    if ($exitCode -ne 0) {
        throw "$FilePath --version failed with exit code $exitCode."
    }
    return (($output | Select-Object -First 1) -as [string]).Trim()
}

function Get-RemoteRefHash {
    param([Parameter(Mandatory = $true)][string]$Ref)

    $line = Get-SingleGitLine -Arguments @('ls-remote', 'origin', $Ref)
    $parts = @($line -split '\s+' | Where-Object { $_ })
    if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[0-9a-f]{40}$' -or $parts[1] -ne $Ref) {
        throw "Could not parse remote ref '$Ref'."
    }
    return $parts[0].ToLowerInvariant()
}

function Get-RemoteTagState {
    param([Parameter(Mandatory = $true)][string]$Tag)

    $tagRef = "refs/tags/$Tag"
    $lines = @(Invoke-WslGit -Arguments @('ls-remote', '--tags', 'origin', $tagRef, "$tagRef^{}"))
    $object = ''
    $peeled = ''
    foreach ($line in $lines) {
        $parts = @($line -split '\s+' | Where-Object { $_ })
        if ($parts.Count -ne 2) { continue }
        if ($parts[1] -eq $tagRef) { $object = $parts[0].ToLowerInvariant() }
        if ($parts[1] -eq "$tagRef^{}") { $peeled = $parts[0].ToLowerInvariant() }
    }
    if ($object -notmatch '^[0-9a-f]{40}$' -or $peeled -notmatch '^[0-9a-f]{40}$') {
        throw "Remote tag '$Tag' is missing or is not annotated."
    }
    return [pscustomobject][ordered]@{ object = $object; commit = $peeled }
}

function Assert-Exact {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Actual,
        [Parameter(Mandatory = $true)][string]$Expected
    )
    if ($Actual -cne $Expected) {
        throw "$Name mismatch: expected '$Expected', found '$Actual'."
    }
}

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$receiptFullPath = [IO.Path]::GetFullPath($ReceiptPath)
$receiptParent = Split-Path -Parent $receiptFullPath
if (Test-IsWithinRoot -Path $receiptFullPath -Root $repositoryRoot -AllowRoot) {
    throw 'The release identity receipt must be written outside the repository.'
}
if ([IO.Path]::GetExtension($receiptFullPath) -ine '.json') {
    throw 'The release identity receipt must use a .json filename.'
}
if (-not (Test-Path -LiteralPath $receiptParent -PathType Container)) {
    throw "The receipt parent directory does not exist: $receiptParent"
}
if (Test-Path -LiteralPath $receiptFullPath) {
    throw "The release identity receipt already exists; refusing overwrite: $receiptFullPath"
}

$currentBranch = Get-SingleGitLine -Arguments @('branch', '--show-current')
Assert-Exact -Name 'named publication branch' -Actual $currentBranch -Expected $branch
$status = @(Invoke-WslGit -Arguments @('status', '--porcelain=v1', '--untracked-files=all'))
if (@($status | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }).Count -ne 0) {
    throw 'The worktree is not clean; commit the reviewed candidate and replay local proof before writing a release receipt.'
}

$localHead = (Get-SingleGitLine -Arguments @('rev-parse', 'HEAD')).ToLowerInvariant()
Assert-Exact -Name 'candidate commit' -Actual $localHead -Expected $ExpectedHead.ToLowerInvariant()
$localTree = (Get-SingleGitLine -Arguments @('rev-parse', 'HEAD^{tree}')).ToLowerInvariant()
Assert-Exact -Name 'candidate tree' -Actual $localTree -Expected $ExpectedTree.ToLowerInvariant()

$origin = Get-SingleGitLine -Arguments @('remote', 'get-url', 'origin')
$expectedUri = [Uri]$expectedRemote
$actualUri = $null
$expectedPath = $expectedUri.AbsolutePath.TrimEnd('/') -replace '\.git$', ''
if (-not [Uri]::TryCreate($origin, [UriKind]::Absolute, [ref]$actualUri) -or
    $actualUri.Scheme -ne 'https' -or
    $actualUri.Host -ine $expectedUri.Host -or
    ($actualUri.AbsolutePath.TrimEnd('/') -replace '\.git$', '') -ine $expectedPath) {
    throw 'Configured origin is not the reviewed Shoozes/candle HTTPS remote.'
}

$savedLocation = Get-Location
try {
    Set-Location -LiteralPath $repositoryRoot
    $rustcVersion = Invoke-VersionCommand -FilePath 'rustc'
    $cargoVersion = Invoke-VersionCommand -FilePath 'cargo'
}
finally {
    Set-Location -LiteralPath $savedLocation
}
Assert-Exact -Name 'rustc version' -Actual $rustcVersion -Expected $expectedRustc
Assert-Exact -Name 'Cargo version' -Actual $cargoVersion -Expected $expectedCargo

$inputPaths = [ordered]@{
    cargo_lock = 'Cargo.lock'
    rust_toolchain = 'rust-toolchain.toml'
    portability_workflow = '.github/workflows/rust-ci.yml'
    release_contract = 'docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md'
    overlay_registry = 'docs/FORK_OVERLAYS.md'
    lfm2_vl_manifest = 'docs/lfm2-vl/MOD_MANIFEST.md'
    snapflash_manifest = 'docs/snapflash/MOD_MANIFEST.md'
}
$inputHashes = [ordered]@{}
foreach ($entry in $inputPaths.GetEnumerator()) {
    $path = Join-Path $repositoryRoot $entry.Value
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Release input is missing: $($entry.Value)"
    }
    $inputHashes[$entry.Key] = [ordered]@{ path = $entry.Value; sha256 = Get-Sha256 -Path $path }
}
Assert-Exact -Name 'Cargo.lock SHA-256' -Actual $inputHashes.cargo_lock.sha256 -Expected $expectedLockHash

$contract = Get-Content -LiteralPath (Join-Path $repositoryRoot $inputPaths.release_contract) -Raw
foreach ($required in @($releaseTag, $historicalTag, $upstreamBase, $expectedRustc, $expectedCargo, $expectedLockHash)) {
    if (-not $contract.Contains($required)) {
        throw "Release contract omits required identity: $required"
    }
}

$localTagRef = "refs/tags/$releaseTag"
$localTagType = Get-SingleGitLine -Arguments @('cat-file', '-t', $localTagRef)
Assert-Exact -Name 'release tag object type' -Actual $localTagType -Expected 'tag'
$localTagObject = (Get-SingleGitLine -Arguments @('rev-parse', $localTagRef)).ToLowerInvariant()
$localTagCommit = (Get-SingleGitLine -Arguments @('rev-parse', "$localTagRef^{}" )).ToLowerInvariant()
Assert-Exact -Name 'release tag target' -Actual $localTagCommit -Expected $localHead

$historicalTagRef = "refs/tags/$historicalTag"
$historicalTagType = Get-SingleGitLine -Arguments @('cat-file', '-t', $historicalTagRef)
Assert-Exact -Name 'historical tag object type' -Actual $historicalTagType -Expected 'tag'
$historicalTagObject = (Get-SingleGitLine -Arguments @('rev-parse', $historicalTagRef)).ToLowerInvariant()
$historicalTagCommit = (Get-SingleGitLine -Arguments @('rev-parse', "$historicalTagRef^{}" )).ToLowerInvariant()
Assert-Exact -Name 'historical tag target' -Actual $historicalTagCommit -Expected $historicalCommit

$remoteHead = Get-RemoteRefHash -Ref 'refs/heads/main'
Assert-Exact -Name 'remote main' -Actual $remoteHead -Expected $localHead
$remoteTag = Get-RemoteTagState -Tag $releaseTag
Assert-Exact -Name 'remote release tag object' -Actual $remoteTag.object -Expected $localTagObject
Assert-Exact -Name 'remote release tag target' -Actual $remoteTag.commit -Expected $localHead
$remoteHistoricalTag = Get-RemoteTagState -Tag $historicalTag
Assert-Exact -Name 'remote historical tag object' -Actual $remoteHistoricalTag.object -Expected $historicalTagObject
Assert-Exact -Name 'remote historical tag target' -Actual $remoteHistoricalTag.commit -Expected $historicalCommit

$lfmVerifier = Invoke-WslVerifier -ScriptPath 'scripts/lfm2-vl/verify-mod-manifest.sh'
$snapVerifier = Invoke-WslVerifier -ScriptPath 'scripts/snapflash/verify-mod-manifest.sh'
$unionVerifier = Invoke-WslVerifier -ScriptPath 'scripts/verify-fork-overlays.sh'
if ($lfmVerifier -notmatch 'lfm2-vl-mod-manifest baseline=6f74e7c390c717f8fd34f23ce02aceb058173370 total=156 fork_modified=16 mod_added=140' -or
    $lfmVerifier -notmatch 'mod-manifest: passed') {
    throw 'LFM2-VL overlay verifier did not report the frozen 156/16/140 inventory.'
}
if ($snapVerifier -notmatch 'snapflash-mod-manifest baseline=6f74e7c390c717f8fd34f23ce02aceb058173370 total=20 fork_modified=8 mod_added=12' -or
    $snapVerifier -notmatch 'snapflash-mod-manifest: passed') {
    throw 'SnapFlash-derived overlay verifier did not report the frozen 20/8/12 inventory.'
}
if ($unionVerifier -notmatch 'fork-overlays baseline=6f74e7c390c717f8fd34f23ce02aceb058173370 paths=167 overlays=2 shared=13' -or
    $unionVerifier -notmatch 'fork-overlays: passed') {
    throw 'Repository overlay verifier did not report the frozen 167/2/13 inventory.'
}

$receipt = [ordered]@{
    schema = 'candle_overlays_release_identity_receipt_v1'
    product = 'Candle combined overlays'
    version = $releaseVersion
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    source = [ordered]@{
        repository = $expectedRemote
        branch = $branch
        commit = $localHead
        tree = $localTree
        upstream_base = $upstreamBase
        remote_main = $remoteHead
    }
    release_tag = [ordered]@{
        name = $releaseTag
        annotated_object = $localTagObject
        commit = $localTagCommit
        remote_object = $remoteTag.object
        remote_commit = $remoteTag.commit
    }
    immutable_historical_tag = [ordered]@{
        name = $historicalTag
        annotated_object = $historicalTagObject
        commit = $historicalTagCommit
        remote_object = $remoteHistoricalTag.object
        remote_commit = $remoteHistoricalTag.commit
    }
    toolchain = [ordered]@{
        platform = 'windows-msvc'
        rustc = $rustcVersion
        cargo = $cargoVersion
    }
    inputs = $inputHashes
    overlays = [ordered]@{
        lfm2_vl = [ordered]@{ paths = 156; fork_modified = 16; added = 140 }
        snapflash_derived = [ordered]@{ paths = 20; fork_modified = 8; added = 12 }
        union = [ordered]@{ paths = 167; overlays = 2; shared_paths = 13 }
    }
    assertions = [ordered]@{
        clean_named_main = $true
        expected_head = $true
        expected_tree = $true
        remote_main_equal = $true
        annotated_tag_equal = $true
        historical_tag_unchanged = $true
        pinned_toolchain_equal = $true
        lock_hash_equal = $true
        overlay_inventories_equal = $true
        hosted_ci_used = $false
        model_artifacts_used = $false
    }
    result = 'pass'
}

$temporaryPath = "$receiptFullPath.tmp.$([guid]::NewGuid().ToString('N'))"
try {
    $json = $receipt | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText($temporaryPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    if (Test-Path -LiteralPath $receiptFullPath) {
        throw 'The receipt target appeared during generation; refusing overwrite.'
    }
    [IO.File]::Move($temporaryPath, $receiptFullPath)
}
finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
}

Write-Host 'Candle overlays release identity receipt: pass' -ForegroundColor Green
Write-Host "Receipt: $receiptFullPath" -ForegroundColor Green
