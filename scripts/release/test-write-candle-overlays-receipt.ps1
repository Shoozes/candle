#Requires -Version 5.1
[CmdletBinding()]
param()
Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
$script:assertions = 0
function Assert-Test([bool]$Condition, [string]$Message) {
    $script:assertions++
    if (-not $Condition) { throw "receipt-test: $Message" }
}
function Write-TestText([string]$Path, [string]$Text) {
    [IO.Directory]::CreateDirectory((Split-Path -Parent $Path)) | Out-Null
    [IO.File]::WriteAllText($Path, $Text.Replace("`r`n", "`n"), [Text.UTF8Encoding]::new($false))
}
function Get-TestHash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('candle-receipt-' + [guid]::NewGuid().ToString('N'))
$fixtureRoot = Join-Path $testRoot 'checkout with spaces'
$writer = Join-Path $fixtureRoot 'scripts/release/write-candle-overlays-receipt.ps1'
$head = '1111111111111111111111111111111111111111'
$tree = '2222222222222222222222222222222222222222'
$base = 'c68b24997319b8d76e5a4e775dab197c7f26322a'
$oldCommit = 'ff885586f6d44a3d9b9ac1724032cdf5f0155384'
$oldObject = '7de0b4dc5f914e9b1d317da7f76f677d9f32b844'
$tagObject = '3333333333333333333333333333333333333333'
$wrong = '5555555555555555555555555555555555555555'
$global:ReleaseTestRoot = $fixtureRoot
$global:ReleaseTestCalls = [Collections.Generic.List[string]]::new()
$global:ReleaseTestOverrides = @{}
$global:ReleaseTestGitHook = $null
$global:ReleaseTestDefaults = @{
    'branch --show-current' = 'main'
    'status --porcelain=v1 --untracked-files=all' = @()
    'rev-parse HEAD' = $head
    'rev-parse HEAD^{tree}' = $tree
    'remote get-url origin' = 'https://github.com/Shoozes/candle.git'
    "merge-base --is-ancestor $base $head" = @()
    "ls-tree -r --name-only $base" = 'Cargo.lock'
    'cat-file -t refs/tags/candle-overlays-mvp-0.2.0' = 'tag'
    'rev-parse refs/tags/candle-overlays-mvp-0.2.0' = $tagObject
    'rev-parse refs/tags/candle-overlays-mvp-0.2.0^{}' = $head
    'cat-file -t refs/tags/lfm2-vl-mvp-0.1.0' = 'tag'
    'rev-parse refs/tags/lfm2-vl-mvp-0.1.0' = $oldObject
    'rev-parse refs/tags/lfm2-vl-mvp-0.1.0^{}' = $oldCommit
    'ls-remote origin refs/heads/main' = "$head`trefs/heads/main"
    'ls-remote --tags origin refs/tags/candle-overlays-mvp-0.2.0 refs/tags/candle-overlays-mvp-0.2.0^{}' = @("$tagObject`trefs/tags/candle-overlays-mvp-0.2.0", "$head`trefs/tags/candle-overlays-mvp-0.2.0^{}")
    'ls-remote --tags origin refs/tags/lfm2-vl-mvp-0.1.0 refs/tags/lfm2-vl-mvp-0.1.0^{}' = @("$oldObject`trefs/tags/lfm2-vl-mvp-0.1.0", "$oldCommit`trefs/tags/lfm2-vl-mvp-0.1.0^{}")
}
$fakeGit = {
    $commandArgs = @($args | ForEach-Object { [string]$_ })
    if ($commandArgs[0] -ne '-C' -or $commandArgs[1] -ne $global:ReleaseTestRoot) { throw 'receipt-test: Git used a second checkout' }
    $key = $commandArgs[2..($commandArgs.Count - 1)] -join ' '
    [void]$global:ReleaseTestCalls.Add($key)
    $global:LASTEXITCODE = 0
    if ($null -ne $global:ReleaseTestGitHook) { & $global:ReleaseTestGitHook $key }
    if ($global:ReleaseTestOverrides.ContainsKey($key)) { return $global:ReleaseTestOverrides[$key] }
    if ($key -eq 'rev-parse --show-toplevel') { return $global:ReleaseTestRoot }
    if ($key -like '-c core.autocrlf=false archive --format=zip --prefix=candle-overlays-mvp-0.2.0/*') {
        if ($commandArgs[-1] -ne '1111111111111111111111111111111111111111') { throw 'receipt-test: archive source drift' }
        $output = @($commandArgs | Where-Object { $_.StartsWith('--output=') })
        [IO.File]::WriteAllText($output[0].Substring(9), 'frozen fixture archive')
        return
    }
    if (-not $global:ReleaseTestDefaults.ContainsKey($key)) { throw "receipt-test: unexpected Git command: $key" }
    return $global:ReleaseTestDefaults[$key]
}
$fakeRustc = { $global:LASTEXITCODE = 0; 'rustc 1.97.1 (8bab26f4f 2026-07-14)' }
$fakeCargo = { $global:LASTEXITCODE = 0; 'cargo 1.97.1 (c980f4866 2026-06-30)' }
$manifestPaths = [ordered]@{
    lfm2_vl='docs/lfm2-vl/MOD_MANIFEST.md'
    snapflash_derived='docs/snapflash/MOD_MANIFEST.md'
    gpt_oss='docs/gpt-oss/MOD_MANIFEST.md'
    fork_compat='docs/fork-compat/MOD_MANIFEST.md'
    qwen35='docs/qwen35/MOD_MANIFEST.md'
}
$inputPaths = @('Cargo.lock','rust-toolchain.toml','.github/workflows/rust-ci.yml',
    'docs/releases/CANDLE_OVERLAYS_MVP_0.2.0.md','docs/FORK_OVERLAYS.md') + @($manifestPaths.Values)
$freezePath = Join-Path $fixtureRoot 'freeze.json'
function Reset-Fixture {
    $global:ReleaseTestOverrides.Clear()
    $global:ReleaseTestCalls.Clear()
    $global:ReleaseTestGitHook = $null
    foreach ($path in $inputPaths) { Write-TestText (Join-Path $fixtureRoot $path) "fixed $path`n" }
    Write-TestText (Join-Path $fixtureRoot 'docs/FORK_OVERLAYS.md') ('<!-- shared-paths:start -->' + "`n" + '- `Cargo.lock`' + "`n" + '<!-- shared-paths:end -->' + "`n")
    $overlays = [ordered]@{}
    foreach ($entry in $manifestPaths.GetEnumerator()) {
        Write-TestText (Join-Path $fixtureRoot $entry.Value) ('- `Cargo.lock`' + "`n" + '- `' + $entry.Key + '.rs`' + "`n")
        $overlays[$entry.Key] = @{ paths=2; fork_modified=1; added=1 }
    }
    $inputs = [ordered]@{}
    foreach ($path in $inputPaths) { $inputs[$path] = Get-TestHash (Join-Path $fixtureRoot $path) }
    $script:freeze = [ordered]@{
        schema='candle_overlays_mvp_expectations_v1'; release_tag='candle-overlays-mvp-0.2.0'
        upstream_base=$base; rustc='rustc 1.97.1 (8bab26f4f 2026-07-14)'; cargo='cargo 1.97.1 (c980f4866 2026-06-30)'
        historical_tag=@{ object=$oldObject; commit=$oldCommit }; inputs=$inputs; overlays=$overlays
        union=@{ paths=6; overlays=5; shared_paths=1; overlapping_paths=1 }
    }
    $verifiers = @{
        'scripts/lfm2-vl/verify-mod-manifest.sh'="lfm2-vl-mod-manifest baseline=$base total=2 fork_modified=1 mod_added=1"
        'scripts/snapflash/verify-mod-manifest.sh'="snapflash-mod-manifest baseline=$base total=2 fork_modified=1 mod_added=1"
        'scripts/gpt-oss/verify-mod-manifest.sh'="gpt-oss-mod-manifest baseline=$base paths=2"
        'scripts/verify-fork-overlays.sh'="fork-overlays baseline=$base paths=6 overlays=5 shared=1"
    }
    foreach ($entry in $verifiers.GetEnumerator()) {
        $pass = if ($entry.Key -eq 'scripts/verify-fork-overlays.sh') { 'fork-overlays: passed' } else { 'mod-manifest: passed' }
        Write-TestText (Join-Path $fixtureRoot $entry.Key) ("#!/bin/bash`nprintf '%s\n' '" + $entry.Value + "' '" + $pass + "'`n")
    }
}
function Save-Freeze { Write-TestText $freezePath ($script:freeze | ConvertTo-Json -Depth 8) }
function Assert-SafeCalls {
    Assert-Test (-not ($global:ReleaseTestCalls -match '^(fetch|push|tag|commit|checkout|reset|update-ref|clean)\b')) 'mutating Git was invoked'
    Assert-Test (@(Get-ChildItem -LiteralPath $testRoot -Filter '*.tmp.*' -File -Recurse).Count -eq 0) 'temporary output leaked'
}
function Test-Rejection([string]$Name, [string]$Pattern, [scriptblock]$Setup) {
    Reset-Fixture
    $script:options = @{ ExpectedHead=$head; ExpectedTree=$tree; ExpectedRepositoryRoot=$fixtureRoot; ExpectationsPath=$freezePath; ReceiptPath=(Join-Path $testRoot "$Name.json"); SourcePackagePath=(Join-Path $testRoot "$Name.zip"); Stage='Published' }
    & $Setup
    Save-Freeze
    if (-not $script:options.ContainsKey('ExpectedExpectationsSha256')) { $script:options.ExpectedExpectationsSha256=Get-TestHash $freezePath }
    $caught = $false
    try { & $writer @script:options }
    catch { if ($_.Exception.Message -notmatch $Pattern) { throw }; $caught=$true }
    Assert-Test $caught "$Name was accepted"
    Assert-Test (-not (Test-Path -LiteralPath $script:options.ReceiptPath)) "$Name wrote a receipt"
    Assert-Test (-not (Test-Path -LiteralPath $script:options.SourcePackagePath)) "$Name wrote a package"
    Assert-SafeCalls
}
[IO.Directory]::CreateDirectory((Split-Path -Parent $writer)) | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'write-candle-overlays-receipt.ps1') -Destination $writer
Set-Item Function:global:git.exe $fakeGit
Set-Item Function:global:rustc $fakeRustc
Set-Item Function:global:cargo $fakeCargo
try {
    Reset-Fixture
    Save-Freeze
    $receiptPath = Join-Path $testRoot 'success.json'
    $packagePath = Join-Path $testRoot 'success.zip'
    $success = @{ ExpectedHead=$head; ExpectedTree=$tree; ExpectedRepositoryRoot=$fixtureRoot; ExpectationsPath=$freezePath; ExpectedExpectationsSha256=(Get-TestHash $freezePath); ReceiptPath=$receiptPath; SourcePackagePath=$packagePath; Stage='Published' }
    & $writer @success
    $raw = [IO.File]::ReadAllText($receiptPath)
    $receipt = $raw | ConvertFrom-Json
    Assert-Test ($receipt.schema -eq 'candle_overlays_release_identity_receipt_v2' -and $receipt.publication_verified) 'published receipt boundary drift'
    Assert-Test ($receipt.source.commit -eq $head -and $receipt.source.tree -eq $tree) 'source identity drift'
    Assert-Test ($receipt.immutable_historical_tag.object -eq $oldObject -and $receipt.immutable_historical_tag.commit -eq $oldCommit) 'old tag drift'
    Assert-Test ($receipt.overlays.PSObject.Properties.Name.Count -eq 5 -and $receipt.union.paths -eq 6) 'five-overlay fixture drift'
    Assert-Test ($receipt.source_package.sha256 -eq (Get-TestHash $packagePath)) 'package not bound'
    Assert-Test ([bool]($global:ReleaseTestCalls -match '^-c core\.autocrlf=false archive --format=zip')) 'archive inherits checkout line-ending conversion'
    Assert-Test (-not $raw.Contains($testRoot)) 'receipt leaked a local path'
    Assert-SafeCalls
    $savedHash = Get-TestHash $receiptPath
    $calls = $global:ReleaseTestCalls.Count
    try { & $writer @success; throw 'receipt-test: overwrite accepted' }
    catch { if ($_.Exception.Message -notmatch 'already exists; refusing overwrite') { throw } }
    Assert-Test ((Get-TestHash $receiptPath) -eq $savedHash -and $global:ReleaseTestCalls.Count -eq $calls) 'overwrite changed receipt or reached Git'
    $success.ReceiptPath = Join-Path $testRoot 'existing-package.json'
    try { & $writer @success; throw 'receipt-test: existing package accepted' }
    catch { if ($_.Exception.Message -notmatch 'already exists; refusing overwrite') { throw } }
    Assert-Test (-not (Test-Path -LiteralPath $success.ReceiptPath)) 'existing package wrote receipt'
    Reset-Fixture
    Save-Freeze
    $success.ExpectedExpectationsSha256=Get-TestHash $freezePath
    $success.ReceiptPath=Join-Path $testRoot 'candidate.json'; $success.SourcePackagePath=Join-Path $testRoot 'candidate.zip'
    $success.Stage='Candidate'
    & $writer @success
    $candidate=Get-Content -LiteralPath $success.ReceiptPath -Raw | ConvertFrom-Json
    Assert-Test (-not $candidate.publication_verified -and $null -eq $candidate.release_tag) 'candidate masqueraded as published'
    Assert-Test (-not ($global:ReleaseTestCalls -match '^ls-remote|refs/tags/candle-overlays')) 'candidate touched public refs or required new tag'
    # A file change after admission cannot replace the reviewed in-memory freeze.
    foreach ($phase in @('before-parse', 'after-parse')) {
        Reset-Fixture
        Save-Freeze
        $approvedHash = Get-TestHash $freezePath
        if ($phase -eq 'after-parse') {
            [IO.File]::WriteAllText($freezePath, [IO.File]::ReadAllText($freezePath).Replace("`n", "`r`n"), [Text.UTF8Encoding]::new($false))
        }
        $global:ReleaseTestMutationPhase = $phase
        $global:ReleaseTestGitHook = {
            param($Key)
            if (($global:ReleaseTestMutationPhase -eq 'before-parse' -and $Key -eq 'rev-parse --show-toplevel') -or
                ($global:ReleaseTestMutationPhase -eq 'after-parse' -and $Key.StartsWith('-c core.autocrlf=false archive '))) {
                [IO.File]::WriteAllText((Join-Path $global:ReleaseTestRoot 'freeze.json'), '{"schema":"unreviewed"}')
            }
        }
        $snapshot = @{ ExpectedHead=$head; ExpectedTree=$tree; ExpectedRepositoryRoot=$fixtureRoot; ExpectationsPath=$freezePath; ExpectedExpectationsSha256=$approvedHash; ReceiptPath=(Join-Path $testRoot "$phase.json"); SourcePackagePath=(Join-Path $testRoot "$phase.zip"); Stage='Candidate' }
        & $writer @snapshot
        $snapshotReceipt = Get-Content -LiteralPath $snapshot.ReceiptPath -Raw | ConvertFrom-Json
        Assert-Test ([IO.File]::ReadAllText($freezePath) -ceq '{"schema":"unreviewed"}') "$phase mutation did not execute"
        Assert-Test ($snapshotReceipt.expectations_sha256 -ceq $approvedHash -and $snapshotReceipt.union.paths -eq 6) "$phase receipt did not bind the admitted snapshot"
        Assert-SafeCalls
    }
    Test-Rejection 'canonical-root' 'Canonical checkout mismatch' { $script:options.ExpectedRepositoryRoot=Join-Path $testRoot 'other' }
    Test-Rejection 'second-checkout' 'Git checkout mismatch' { $global:ReleaseTestOverrides['rev-parse --show-toplevel']=Join-Path $testRoot 'other' }
    Test-Rejection 'dirty' 'not clean' { $global:ReleaseTestOverrides['status --porcelain=v1 --untracked-files=all']=' M owner.rs' }
    Test-Rejection 'head' 'candidate commit mismatch' { $script:options.ExpectedHead=$wrong }
    Test-Rejection 'tree' 'candidate tree mismatch' { $script:options.ExpectedTree=$wrong }
    Test-Rejection 'remote-main' 'remote main mismatch' { $global:ReleaseTestOverrides['ls-remote origin refs/heads/main']="$wrong`trefs/heads/main" }
    Test-Rejection 'unreviewed-freeze' 'reviewed expectations SHA-256 mismatch' {
        Save-Freeze
        $script:options.ExpectedExpectationsSha256=Get-TestHash $freezePath
        $script:freeze.overlays.gpt_oss.paths=3
    }
    foreach ($path in $inputPaths) {
        Test-Rejection ('input-' + ($path -replace '\W','-')) 'SHA-256 mismatch' { Add-Content -LiteralPath (Join-Path $fixtureRoot $path) 'changed' }
    }
    foreach ($overlay in $manifestPaths.Keys) {
        Test-Rejection "inventory-$overlay" 'inventory mismatch' { $script:freeze.overlays[$overlay].paths=3 }
    }
    Test-Rejection 'union' 'union inventory mismatch' { $script:freeze.union.paths=7 }
    Test-Rejection 'shared' 'shared inventory mismatch' { $script:freeze.union.shared_paths=2 }
    Test-Rejection 'overlap' 'overlap inventory mismatch' { $script:freeze.union.overlapping_paths=2 }
    Test-Rejection 'verifier-drift' 'Frozen inventory mismatch' { Write-TestText (Join-Path $fixtureRoot 'scripts/verify-fork-overlays.sh') "printf '%s\n' 'wrong' 'fork-overlays: passed'`n" }
    Test-Rejection 'release-lightweight' 'object type mismatch' { $global:ReleaseTestOverrides['cat-file -t refs/tags/candle-overlays-mvp-0.2.0']='commit' }
    Test-Rejection 'release-missing' 'Expected one Git result' { $global:ReleaseTestOverrides['cat-file -t refs/tags/candle-overlays-mvp-0.2.0']=@() }
    Test-Rejection 'release-target' 'release tag target mismatch' { $global:ReleaseTestOverrides['rev-parse refs/tags/candle-overlays-mvp-0.2.0^{}']=$wrong }
    Test-Rejection 'historical-object' 'historical tag object mismatch' { $global:ReleaseTestOverrides['rev-parse refs/tags/lfm2-vl-mvp-0.1.0']=$wrong }
    Test-Rejection 'historical-target' 'historical tag target mismatch' { $global:ReleaseTestOverrides['rev-parse refs/tags/lfm2-vl-mvp-0.1.0^{}']=$wrong }
    foreach ($tag in @('candle-overlays-mvp-0.2.0', 'lfm2-vl-mvp-0.1.0')) {
        foreach ($field in @('object','target')) {
            Test-Rejection "$tag-remote-$field" "remote $tag $field mismatch" {
                $obj = if ($tag -eq 'lfm2-vl-mvp-0.1.0') { $oldObject } else { $tagObject }
                $commit = if ($tag -eq 'lfm2-vl-mvp-0.1.0') { $oldCommit } else { $head }
                if ($field -eq 'object') { $obj=$wrong } else { $commit=$wrong }
                $global:ReleaseTestOverrides["ls-remote --tags origin refs/tags/$tag refs/tags/$tag^{}"]=@("$obj`trefs/tags/$tag", "$commit`trefs/tags/$tag^{}")
            }
        }
    }
    # Exercise the actual pair-publication helper with a deterministic second-move failure.
    $tokens=$null; $parseErrors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($writer, [ref]$tokens, [ref]$parseErrors)
    foreach ($name in @('Get-Hash','Move-ReleaseOutputs')) {
        $functionAst=$ast.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true)
        . ([scriptblock]::Create($functionAst.Extent.Text))
    }
    $partialTemporary=Join-Path $testRoot 'partial.tmp'
    $partialFinal=Join-Path $testRoot 'partial.zip'
    $partialReceipt=Join-Path $testRoot 'partial.json'
    Write-TestText $partialTemporary 'owned package'
    $script:moveCount=0
    $caught=$false
    try {
        Move-ReleaseOutputs $partialTemporary $partialFinal 'unused-receipt.tmp' $partialReceipt (Get-TestHash $partialTemporary) {
            param($Source,$Destination)
            $script:moveCount++
            if ($script:moveCount -eq 2) { throw 'injected completion failure' }
            [IO.File]::Move($Source,$Destination)
        }
    } catch { if ($_.Exception.Message -notmatch 'injected completion failure') { throw }; $caught=$true }
    Assert-Test ($caught -and -not (Test-Path -LiteralPath $partialFinal) -and -not (Test-Path -LiteralPath $partialReceipt)) 'completion failure leaked an owned final package or receipt'
    Write-Host "Candle release receipt tests: pass ($script:assertions assertions)"
} finally {
    foreach ($name in @('git.exe','rustc','cargo')) { Remove-Item -LiteralPath "Function:global:$name" -ErrorAction SilentlyContinue }
    foreach ($name in @('ReleaseTestRoot','ReleaseTestCalls','ReleaseTestOverrides','ReleaseTestDefaults','ReleaseTestGitHook','ReleaseTestMutationPhase')) { Remove-Variable -Name $name -Scope Global -ErrorAction SilentlyContinue }
    $fullRoot=[IO.Path]::GetFullPath($testRoot)
    $prefix=[IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $fullRoot.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe fixture cleanup target' }
    if (Test-Path -LiteralPath $fullRoot) { Remove-Item -LiteralPath $fullRoot -Recurse -Force }
}
