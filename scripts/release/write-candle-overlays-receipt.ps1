#Requires -Version 5.1
<##
.SYNOPSIS
    Packages and verifies one reviewed Candle overlay source identity.
.DESCRIPTION
    Candidate mode is local-only. Published mode also checks public remote
    branch/tag identities. Neither mode commits, tags, pushes or creates a
    hosted release. Both outputs must be fresh and outside this checkout.
##>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$ExpectedHead,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{40}$')][string]$ExpectedTree,
    [Parameter(Mandatory = $true)][string]$ExpectationsPath,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$ExpectedExpectationsSha256,
    [Parameter(Mandatory = $true)][string]$ReceiptPath,
    [Parameter(Mandatory = $true)][string]$SourcePackagePath,
    [Parameter(Mandatory = $true)][ValidateSet('Candidate', 'Published')][string]$Stage,
    [string]$ExpectedRepositoryRoot = 'C:\DevStuff\candle'
)
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..')).TrimEnd('\', '/')
$releaseTag = 'candle-overlays-mvp-0.2.0'
$historicalTag = 'lfm2-vl-mvp-0.1.0'
$expectedRemote = 'https://github.com/Shoozes/candle.git'
$manifests = [ordered]@{
    lfm2_vl = 'docs/lfm2-vl/MOD_MANIFEST.md'
    snapflash_derived = 'docs/snapflash/MOD_MANIFEST.md'
    gpt_oss = 'docs/gpt-oss/MOD_MANIFEST.md'
    fork_compat = 'docs/fork-compat/MOD_MANIFEST.md'
    qwen35 = 'docs/qwen35/MOD_MANIFEST.md'
}
$inputPaths = @('Cargo.lock', 'rust-toolchain.toml', '.github/workflows/rust-ci.yml',
    'docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md', 'docs/FORK_OVERLAYS.md') + @($manifests.Values)

function Assert-Exact([string]$Name, [string]$Actual, [string]$Expected) {
    if ($Actual -cne $Expected) { throw "$Name mismatch: expected '$Expected', found '$Actual'." }
}
function Invoke-Native([string]$Command, [string[]]$Arguments) {
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(& $Command @Arguments 2>&1)
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $savedPreference }
    if ($code -ne 0) { throw "Native command failed ($code): $Command $($Arguments -join ' ')`n$($output -join "`n")" }
    return @($output | ForEach-Object { "$_" })
}
function Invoke-Git([string[]]$Arguments) {
    Invoke-Native 'git.exe' (@('-C', $repositoryRoot) + $Arguments)
}
function Get-GitLine([string[]]$Arguments) {
    $lines = @(Invoke-Git $Arguments | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($lines.Count -ne 1) { throw "Expected one Git result: $($Arguments -join ' ')" }
    return $lines[0].Trim()
}
function Get-Hash([string]$Path, [switch]$CanonicalText) {
    if (-not $CanonicalText) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
    # Match source-archive LF bytes on Windows CRLF checkouts.
    $bytes = [Text.UTF8Encoding]::new($false, $true).GetBytes([IO.File]::ReadAllText($Path).Replace("`r`n", "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
function Assert-ExternalFresh([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if ($full.Equals($repositoryRoot, [StringComparison]::OrdinalIgnoreCase) -or
        $full.StartsWith($repositoryRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Release outputs must be outside the repository.'
    }
    if (Test-Path -LiteralPath $full) { throw "Output already exists; refusing overwrite: $full" }
    if (-not (Test-Path -LiteralPath (Split-Path -Parent $full) -PathType Container)) { throw 'Output parent is missing.' }
    return $full
}
function Get-Tag([string]$Name) {
    $ref = "refs/tags/$Name"
    Assert-Exact "$Name object type" (Get-GitLine @('cat-file', '-t', $ref)) 'tag'
    return [ordered]@{ name=$Name; object=(Get-GitLine @('rev-parse', $ref)); commit=(Get-GitLine @('rev-parse', "$ref^{}")) }
}
function Get-RemoteTag([string]$Name) {
    $ref = "refs/tags/$Name"
    $rows = @(Invoke-Git @('ls-remote', '--tags', 'origin', $ref, "$ref^{}"))
    $state = @{}
    foreach ($row in $rows) {
        $parts = @($row -split '\s+')
        if ($parts.Count -ne 2 -or $parts[0] -notmatch '^[0-9a-f]{40}$' -or $state.ContainsKey($parts[1])) { throw 'Malformed remote tag result.' }
        $state[$parts[1]] = $parts[0]
    }
    if ($state.Count -ne 2 -or -not $state.ContainsKey($ref) -or -not $state.ContainsKey("$ref^{}")) { throw "Remote annotated tag missing: $Name" }
    return @{ object=$state[$ref]; commit=$state["$ref^{}"] }
}
function Resolve-GitBash {
    foreach ($root in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($root) {
            $candidate = Join-Path $root 'Git\bin\bash.exe'
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
        }
    }
    throw 'Git for Windows bash.exe is required; WSL is not the native release lane.'
}
function Invoke-Verifier([string]$Script, [string]$ExpectedSummary, [string[]]$Extra = @()) {
    $savedLocation = Get-Location
    try {
        Set-Location -LiteralPath $repositoryRoot
        $output = @(Invoke-Native $bash (@('--noprofile', '--norc', $Script) + $Extra))
    } finally { Set-Location -LiteralPath $savedLocation }
    if ($output -cnotcontains $ExpectedSummary -or -not ($output -match '^(?:[a-z0-9-]+-)?mod-manifest: passed$|^fork-overlays: passed$')) {
        throw "Frozen inventory mismatch: $Script"
    }
}
function Move-ReleaseOutputs([string]$PackageTemporary, [string]$PackageFinal,
    [string]$ReceiptTemporary, [string]$ReceiptFinal, [string]$PackageHash,
    [scriptblock]$MoveFile = { param($Source, $Destination) [IO.File]::Move($Source, $Destination) }) {
    $packageMoved = $false
    try {
        & $MoveFile $PackageTemporary $PackageFinal
        $packageMoved = $true
        # Completion receipt is last; a failed final move never claims success.
        & $MoveFile $ReceiptTemporary $ReceiptFinal
    } catch {
        $failure = $_
        if ($packageMoved -and (Test-Path -LiteralPath $PackageFinal)) {
            if ((Get-Hash $PackageFinal) -cne $PackageHash) { throw "Completion failed; package changed after creation and was retained: $PackageFinal. Primary error: $failure" }
            Remove-Item -LiteralPath $PackageFinal -Force
        }
        throw $failure
    }
}

# Refuse existing outputs before any Git or native command.
$receiptFullPath = Assert-ExternalFresh $ReceiptPath
$packageFullPath = Assert-ExternalFresh $SourcePackagePath
if ($receiptFullPath.Equals($packageFullPath, [StringComparison]::OrdinalIgnoreCase)) { throw 'Outputs must be distinct.' }
$canonicalRoot = [IO.Path]::GetFullPath($ExpectedRepositoryRoot).TrimEnd('\', '/')
if (-not $repositoryRoot.Equals($canonicalRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Canonical checkout mismatch.' }
Assert-Exact 'reviewed expectations SHA-256' (Get-Hash $ExpectationsPath -CanonicalText) $ExpectedExpectationsSha256.ToLowerInvariant()
$gitRoot = [IO.Path]::GetFullPath((Get-GitLine @('rev-parse', '--show-toplevel'))).TrimEnd('\', '/')
if (-not $gitRoot.Equals($repositoryRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Git checkout mismatch.' }
if ($env:FORK_OVERLAYS_REPO_ROOT) { throw 'Clear FORK_OVERLAYS_REPO_ROOT before release verification; inventory must use this checkout.' }
$freeze = Get-Content -LiteralPath $ExpectationsPath -Raw | ConvertFrom-Json
Assert-Exact 'expectations schema' $freeze.schema 'candle_overlays_mvp_expectations_v1'
Assert-Exact 'release tag' $freeze.release_tag $releaseTag
Assert-Exact 'named branch' (Get-GitLine @('branch', '--show-current')) 'main'
if (@(Invoke-Git @('status', '--porcelain=v1', '--untracked-files=all')).Count -ne 0) { throw 'The worktree is not clean.' }
$head = Get-GitLine @('rev-parse', 'HEAD')
$tree = Get-GitLine @('rev-parse', 'HEAD^{tree}')
Assert-Exact 'candidate commit' $head $ExpectedHead.ToLowerInvariant()
Assert-Exact 'candidate tree' $tree $ExpectedTree.ToLowerInvariant()
$origin = Get-GitLine @('remote', 'get-url', 'origin')
Assert-Exact 'origin' ($origin -replace '\.git$', '') ($expectedRemote -replace '\.git$', '')
if ($freeze.upstream_base -notmatch '^[0-9a-f]{40}$') { throw 'Invalid frozen upstream base.' }
$null = Invoke-Git @('merge-base', '--is-ancestor', $freeze.upstream_base, $head)

$savedLocation = Get-Location
try {
    Set-Location -LiteralPath $repositoryRoot
    $rustc = (Invoke-Native 'rustc' @('--version')) -join "`n"
    $cargo = (Invoke-Native 'cargo' @('--version')) -join "`n"
} finally { Set-Location -LiteralPath $savedLocation }
Assert-Exact 'rustc' $rustc $freeze.rustc
Assert-Exact 'Cargo' $cargo $freeze.cargo
if (@(Compare-Object ($inputPaths | Sort-Object) @($freeze.inputs.PSObject.Properties.Name | Sort-Object)).Count -ne 0) { throw 'Frozen input set mismatch.' }
$inputs = [ordered]@{}
foreach ($path in $inputPaths) {
    $actual = Get-Hash (Join-Path $repositoryRoot $path) -CanonicalText
    Assert-Exact "input $path SHA-256" $actual $freeze.inputs.$path
    $inputs[$path] = $actual
}

$basePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
foreach ($path in (Invoke-Git @('ls-tree', '-r', '--name-only', $freeze.upstream_base))) { [void]$basePaths.Add($path) }
$allPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$duplicates = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$inventories = [ordered]@{}
if (@(Compare-Object @($manifests.Keys | Sort-Object) @($freeze.overlays.PSObject.Properties.Name | Sort-Object)).Count -ne 0) { throw 'Frozen overlay set mismatch.' }
foreach ($entry in $manifests.GetEnumerator()) {
    $paths = @(Get-Content -LiteralPath (Join-Path $repositoryRoot $entry.Value) | ForEach-Object {
        if ($_ -match '^\| `([^`]+)` \|' -or $_ -match '^- `([^`]+)`$') { $Matches[1] }
    } | Sort-Object -Unique)
    $modified = @($paths | Where-Object { $basePaths.Contains($_) }).Count
    $actual = [ordered]@{ paths=$paths.Count; fork_modified=$modified; added=($paths.Count - $modified) }
    foreach ($key in @('paths', 'fork_modified', 'added')) { Assert-Exact "$($entry.Key) $key inventory" $actual[$key] $freeze.overlays.($entry.Key).$key }
    $inventories[$entry.Key] = $actual
    foreach ($path in $paths) { if (-not $allPaths.Add($path)) { [void]$duplicates.Add($path) } }
}
Assert-Exact 'union inventory' $allPaths.Count $freeze.union.paths
$registry = [IO.File]::ReadAllText((Join-Path $repositoryRoot 'docs/FORK_OVERLAYS.md'))
$sharedBlock = [regex]::Match($registry, '(?s)<!-- shared-paths:start -->(.*?)<!-- shared-paths:end -->')
if (-not $sharedBlock.Success) { throw 'Shared-path registry markers missing.' }
$sharedPaths = @([regex]::Matches($sharedBlock.Groups[1].Value, '(?m)^- `([^`]+)`\r?$') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
Assert-Exact 'shared inventory' $sharedPaths.Count $freeze.union.shared_paths
Assert-Exact 'overlap inventory' $duplicates.Count $freeze.union.overlapping_paths
foreach ($path in $duplicates) { if ($sharedPaths -cnotcontains $path) { throw "Undeclared overlap: $path" } }
Assert-Exact 'overlay count' $manifests.Count $freeze.union.overlays
$bash = Resolve-GitBash
$base = $freeze.upstream_base
$lfm = $inventories.lfm2_vl
$snap = $inventories.snapflash_derived
Invoke-Verifier 'scripts/lfm2-vl/verify-mod-manifest.sh' "lfm2-vl-mod-manifest baseline=$base total=$($lfm.paths) fork_modified=$($lfm.fork_modified) mod_added=$($lfm.added)" @($base)
Invoke-Verifier 'scripts/snapflash/verify-mod-manifest.sh' "snapflash-mod-manifest baseline=$base total=$($snap.paths) fork_modified=$($snap.fork_modified) mod_added=$($snap.added)" @($base)
Invoke-Verifier 'scripts/gpt-oss/verify-mod-manifest.sh' "gpt-oss-mod-manifest baseline=$base paths=$($inventories.gpt_oss.paths)" @($base)
Invoke-Verifier 'scripts/verify-fork-overlays.sh' "fork-overlays baseline=$base paths=$($allPaths.Count) overlays=5 shared=$($sharedPaths.Count)" @('--upstream-baseline', $base)
$historical = Get-Tag $historicalTag
Assert-Exact 'historical tag object' $historical.object $freeze.historical_tag.object
Assert-Exact 'historical tag target' $historical.commit $freeze.historical_tag.commit
$release = $null
if ($Stage -eq 'Published') {
    $release = Get-Tag $releaseTag
    Assert-Exact 'release tag target' $release.commit $head
    Assert-Exact 'remote main' (Get-GitLine @('ls-remote', 'origin', 'refs/heads/main')) "$head`trefs/heads/main"
    foreach ($tag in @($historical, $release)) {
        $remote = Get-RemoteTag $tag.name
        Assert-Exact "remote $($tag.name) object" $remote.object $tag.object
        Assert-Exact "remote $($tag.name) target" $remote.commit $tag.commit
    }
}

$temporaryPackage = "$packageFullPath.tmp.$([guid]::NewGuid().ToString('N'))"
$temporaryReceipt = "$receiptFullPath.tmp.$([guid]::NewGuid().ToString('N'))"
try {
    Assert-Exact 'final candidate commit' (Get-GitLine @('rev-parse', 'HEAD')) $head
    if (@(Invoke-Git @('status', '--porcelain=v1', '--untracked-files=all')).Count -ne 0) { throw 'The worktree changed during verification.' }
    $null = Invoke-Git @('archive', '--format=zip', "--prefix=$releaseTag/", "--output=$temporaryPackage", $head)
    $package = [ordered]@{ name=[IO.Path]::GetFileName($packageFullPath); bytes=(Get-Item -LiteralPath $temporaryPackage).Length; sha256=(Get-Hash $temporaryPackage); format='git-archive-zip'; source_commit=$head }
    $receipt = [ordered]@{
        schema='candle_overlays_release_identity_receipt_v2'
        stage=$Stage.ToLowerInvariant()
        generated_at_utc=[DateTime]::UtcNow.ToString('o')
        source=[ordered]@{ repository=$expectedRemote; branch='main'; commit=$head; tree=$tree; upstream_base=$base }
        release_tag=$release
        immutable_historical_tag=$historical
        toolchain=[ordered]@{ platform='windows-msvc'; rustc=$rustc; cargo=$cargo }
        expectations_sha256=(Get-Hash $ExpectationsPath -CanonicalText)
        input_hash_convention='UTF-8 without BOM, CRLF normalized to LF'
        inputs=$inputs
        overlays=$inventories
        union=$freeze.union
        source_package=$package
        publication_verified=($Stage -eq 'Published')
        model_artifacts_used=$false
        hosted_ci_used=$false
        result='pass'
    }
    [IO.File]::WriteAllText($temporaryReceipt, ($receipt | ConvertTo-Json -Depth 12) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    if ((Test-Path -LiteralPath $packageFullPath) -or (Test-Path -LiteralPath $receiptFullPath)) { throw 'An output appeared during generation; refusing overwrite.' }
    Move-ReleaseOutputs $temporaryPackage $packageFullPath $temporaryReceipt $receiptFullPath $package.sha256
} finally {
    foreach ($temporary in @($temporaryPackage, $temporaryReceipt)) {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    }
}
Write-Host "Candle overlays $Stage identity/source package: pass"
