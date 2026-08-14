#Requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

$script:Assertions = 0
function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)][bool]$Condition,
        [Parameter(Mandatory = $true)][string]$Message
    )
    $script:Assertions++
    if (-not $Condition) { throw "receipt-test: $Message" }
}

function Assert-ThrowsLike {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Message
    )
    $caught = $false
    try { & $Action }
    catch {
        if ($_.Exception.Message -notmatch $Pattern) { throw }
        $caught = $true
    }
    Assert-Condition -Condition $caught -Message $Message
}

$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$receiptScript = Join-Path $PSScriptRoot 'write-candle-overlays-receipt.ps1'
$testRoot = Join-Path ([IO.Path]::GetTempPath()) ('candle-receipt-test-' + [guid]::NewGuid().ToString('N'))
$head = '1111111111111111111111111111111111111111'
$tree = '2222222222222222222222222222222222222222'
$releaseTagObject = '3333333333333333333333333333333333333333'
$historicalTagObject = '4444444444444444444444444444444444444444'
$historicalCommit = 'ff885586f6d44a3d9b9ac1724032cdf5f0155384'
$wrongHead = '5555555555555555555555555555555555555555'
$receiptPath = Join-Path $testRoot 'candle-overlays-mvp-0.2.0-receipt.json'
$mismatchReceiptPath = Join-Path $testRoot 'mismatch-receipt.json'

$global:ReceiptTestCalls = [Collections.Generic.List[string]]::new()
$global:ReceiptTestRemoteHead = $head

$fakeWsl = {
    $commandArgs = @($args | ForEach-Object { [string]$_ })
    [void]$global:ReceiptTestCalls.Add(($commandArgs -join ' '))
    $global:LASTEXITCODE = 0

    if ($commandArgs.Count -ge 6 -and $commandArgs[2] -eq 'git' -and $commandArgs[3] -eq '-C') {
        $gitArgs = @($commandArgs[5..($commandArgs.Count - 1)])
        $key = $gitArgs -join ' '
        switch ($key) {
            'branch --show-current' { return 'main' }
            'status --porcelain=v1 --untracked-files=all' { return }
            'rev-parse HEAD' { return '1111111111111111111111111111111111111111' }
            'rev-parse HEAD^{tree}' { return '2222222222222222222222222222222222222222' }
            'remote get-url origin' { return 'https://github.com/Shoozes/candle.git' }
            'cat-file -t refs/tags/candle-overlays-mvp-0.2.0' { return 'tag' }
            'rev-parse refs/tags/candle-overlays-mvp-0.2.0' { return '3333333333333333333333333333333333333333' }
            'rev-parse refs/tags/candle-overlays-mvp-0.2.0^{}' { return '1111111111111111111111111111111111111111' }
            'cat-file -t refs/tags/lfm2-vl-mvp-0.1.0' { return 'tag' }
            'rev-parse refs/tags/lfm2-vl-mvp-0.1.0' { return '4444444444444444444444444444444444444444' }
            'rev-parse refs/tags/lfm2-vl-mvp-0.1.0^{}' { return 'ff885586f6d44a3d9b9ac1724032cdf5f0155384' }
            'ls-remote origin refs/heads/main' {
                return "$global:ReceiptTestRemoteHead`trefs/heads/main"
            }
            'ls-remote --tags origin refs/tags/candle-overlays-mvp-0.2.0 refs/tags/candle-overlays-mvp-0.2.0^{}' {
                return @(
                    ('3333333333333333333333333333333333333333' + "`trefs/tags/candle-overlays-mvp-0.2.0")
                    ('1111111111111111111111111111111111111111' + "`trefs/tags/candle-overlays-mvp-0.2.0^{}")
                )
            }
            'ls-remote --tags origin refs/tags/lfm2-vl-mvp-0.1.0 refs/tags/lfm2-vl-mvp-0.1.0^{}' {
                return @(
                    ('4444444444444444444444444444444444444444' + "`trefs/tags/lfm2-vl-mvp-0.1.0")
                    ('ff885586f6d44a3d9b9ac1724032cdf5f0155384' + "`trefs/tags/lfm2-vl-mvp-0.1.0^{}")
                )
            }
            default { throw "receipt-test: unexpected fake Git command: $key" }
        }
    }

    if ($commandArgs.Count -ge 5 -and $commandArgs[2] -eq 'bash' -and $commandArgs[3] -eq '-lc') {
        $command = $commandArgs[4]
        if ($command -match 'scripts/lfm2-vl/verify-mod-manifest\.sh') {
            return @(
                'lfm2-vl-mod-manifest baseline=6f74e7c390c717f8fd34f23ce02aceb058173370 total=156 fork_modified=16 mod_added=140',
                'mod-manifest: passed'
            )
        }
        if ($command -match 'scripts/snapflash/verify-mod-manifest\.sh') {
            return @(
                'snapflash-mod-manifest baseline=6f74e7c390c717f8fd34f23ce02aceb058173370 total=20 fork_modified=8 mod_added=12',
                'snapflash-mod-manifest: passed'
            )
        }
        if ($command -match 'scripts/verify-fork-overlays\.sh') {
            return @(
                'fork-overlays baseline=6f74e7c390c717f8fd34f23ce02aceb058173370 paths=167 overlays=2 shared=13',
                'fork-overlays: passed'
            )
        }
        throw "receipt-test: unexpected fake Bash command: $command"
    }

    throw "receipt-test: unexpected fake WSL invocation: $($commandArgs -join ' ')"
}

$fakeRustc = {
    if (($args -join ' ') -ne '--version') { throw 'receipt-test: unexpected rustc arguments' }
    $global:LASTEXITCODE = 0
    return 'rustc 1.97.1 (8bab26f4f 2026-07-14)'
}
$fakeCargo = {
    if (($args -join ' ') -ne '--version') { throw 'receipt-test: unexpected Cargo arguments' }
    $global:LASTEXITCODE = 0
    return 'cargo 1.97.1 (c980f4866 2026-06-30)'
}

New-Item -ItemType Directory -Path $testRoot | Out-Null
Set-Item -Path Function:global:wsl.exe -Value $fakeWsl
Set-Item -Path Function:global:rustc -Value $fakeRustc
Set-Item -Path Function:global:cargo -Value $fakeCargo

try {
    Assert-Condition -Condition (Test-Path -LiteralPath $receiptScript -PathType Leaf) -Message 'receipt generator is missing'

    & $receiptScript -ExpectedHead $head -ExpectedTree $tree -ReceiptPath $receiptPath
    Assert-Condition -Condition (Test-Path -LiteralPath $receiptPath -PathType Leaf) -Message 'success path did not write a receipt'
    $rawReceipt = [IO.File]::ReadAllText($receiptPath)
    $receipt = $rawReceipt | ConvertFrom-Json
    Assert-Condition -Condition ([string]$receipt.schema -eq 'candle_overlays_release_identity_receipt_v1') -Message 'receipt schema drifted'
    Assert-Condition -Condition ([string]$receipt.result -eq 'pass') -Message 'receipt result is not pass'
    Assert-Condition -Condition ([string]$receipt.source.commit -eq $head) -Message 'receipt commit drifted'
    Assert-Condition -Condition ([string]$receipt.source.tree -eq $tree) -Message 'receipt tree drifted'
    Assert-Condition -Condition ([string]$receipt.release_tag.annotated_object -eq $releaseTagObject) -Message 'release tag object drifted'
    Assert-Condition -Condition ([string]$receipt.immutable_historical_tag.annotated_object -eq $historicalTagObject) -Message 'historical tag object drifted'
    Assert-Condition -Condition ([string]$receipt.immutable_historical_tag.commit -eq $historicalCommit) -Message 'historical tag commit drifted'
    Assert-Condition -Condition ([int]$receipt.overlays.lfm2_vl.paths -eq 156) -Message 'LFM2-VL inventory drifted'
    Assert-Condition -Condition ([int]$receipt.overlays.snapflash_derived.paths -eq 20) -Message 'SnapFlash inventory drifted'
    Assert-Condition -Condition ([int]$receipt.overlays.union.paths -eq 167 -and [int]$receipt.overlays.union.shared_paths -eq 13) -Message 'overlay union drifted'
    Assert-Condition -Condition ($receipt.assertions.hosted_ci_used -eq $false -and $receipt.assertions.model_artifacts_used -eq $false) -Message 'receipt claimed forbidden evidence'
    Assert-Condition -Condition (-not $rawReceipt.Contains($repositoryRoot) -and -not $rawReceipt.Contains($testRoot)) -Message 'receipt leaked a local filesystem path'
    Assert-Condition -Condition (@(Get-ChildItem -LiteralPath $testRoot -Filter '*.tmp.*' -File).Count -eq 0) -Message 'success left a temporary receipt file'
    Assert-Condition -Condition (-not ($global:ReceiptTestCalls -match '(?i)\b(fetch|push)\b')) -Message 'identity test attempted a mutating Git command'

    $receiptHash = (Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash
    $callsBeforeOverwrite = $global:ReceiptTestCalls.Count
    Assert-ThrowsLike -Pattern 'already exists; refusing overwrite' -Message 'existing receipt was not rejected' -Action {
        & $receiptScript -ExpectedHead $head -ExpectedTree $tree -ReceiptPath $receiptPath
    }
    Assert-Condition -Condition ((Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash -eq $receiptHash) -Message 'overwrite rejection changed the existing receipt'
    Assert-Condition -Condition ($global:ReceiptTestCalls.Count -eq $callsBeforeOverwrite) -Message 'overwrite rejection reached Git inspection'

    $global:ReceiptTestRemoteHead = $wrongHead
    Assert-ThrowsLike -Pattern 'remote main mismatch' -Message 'remote-main mismatch was not rejected' -Action {
        & $receiptScript -ExpectedHead $head -ExpectedTree $tree -ReceiptPath $mismatchReceiptPath
    }
    Assert-Condition -Condition (-not (Test-Path -LiteralPath $mismatchReceiptPath)) -Message 'remote mismatch wrote a receipt'
    Assert-Condition -Condition (@(Get-ChildItem -LiteralPath $testRoot -Filter '*.tmp.*' -File).Count -eq 0) -Message 'failure left a temporary receipt file'

    Write-Host "Candle release receipt tests: pass ($($script:Assertions) assertions)" -ForegroundColor Green
}
finally {
    Remove-Item -Path Function:global:wsl.exe -ErrorAction SilentlyContinue
    Remove-Item -Path Function:global:rustc -ErrorAction SilentlyContinue
    Remove-Item -Path Function:global:cargo -ErrorAction SilentlyContinue
    Remove-Variable -Name ReceiptTestCalls -Scope Global -ErrorAction SilentlyContinue
    Remove-Variable -Name ReceiptTestRemoteHead -Scope Global -ErrorAction SilentlyContinue

    $fullTestRoot = [IO.Path]::GetFullPath($testRoot)
    $fullTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\', '/')
    $tempPrefix = $fullTempRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $fullTestRoot.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "receipt-test: refusing cleanup outside the temporary root: $fullTestRoot"
    }
    if (Test-Path -LiteralPath $fullTestRoot) {
        Remove-Item -LiteralPath $fullTestRoot -Recurse -Force
    }
}
