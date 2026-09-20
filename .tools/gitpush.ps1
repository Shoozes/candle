[CmdletBinding()]
param(
    [string]$Remote = "https://github.com/Shoozes/candle.git",
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$LocalTestMode,
    [string]$LocalTestVerifier = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:GIT_TERMINAL_PROMPT = "0"

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "gitpush.ps1 requires PowerShell 7."
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $comparison = if ($IsWindows) {
        [StringComparison]::OrdinalIgnoreCase
    } else {
        [StringComparison]::Ordinal
    }
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $pathFull = [IO.Path]::GetFullPath($Path)
    return $pathFull.Equals($rootFull, $comparison) -or
        $pathFull.StartsWith($rootFull + [IO.Path]::DirectorySeparatorChar, $comparison)
}

function Resolve-LocalTestConfig {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$RemotePath,
        [Parameter(Mandatory = $true)][string]$Verifier
    )
    $tempRoot = (Resolve-Path -LiteralPath ([IO.Path]::GetTempPath())).Path
    if (-not (Test-PathWithin -Root $tempRoot -Path $RepoRoot)) {
        throw "LocalTestMode is restricted to a repository under the OS temporary directory."
    }
    $markerPath = Join-Path $RepoRoot ".tools\gitpush-test-mode.marker"
    $marker = Get-Item -LiteralPath $markerPath -ErrorAction SilentlyContinue
    if ($null -eq $marker -or $marker.PSIsContainer -or
        (($marker.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) -or
        ([IO.File]::ReadAllText($marker.FullName).Trim() -ne "candle.gitpush.test.v1")) {
        throw "LocalTestMode requires the exact regular marker .tools/gitpush-test-mode.marker."
    }
    if ([string]::IsNullOrWhiteSpace($Verifier)) {
        throw "LocalTestMode requires -LocalTestVerifier."
    }
    $verifierCandidate = if ([IO.Path]::IsPathRooted($Verifier)) { $Verifier } else { Join-Path $RepoRoot $Verifier }
    $verifierPath = (Resolve-Path -LiteralPath $verifierCandidate).Path
    $verifierItem = Get-Item -LiteralPath $verifierPath
    if (-not (Test-PathWithin -Root $RepoRoot -Path $verifierPath) -or
        $verifierItem.PSIsContainer -or
        (($verifierItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)) {
        throw "LocalTestMode verifier must be a regular file inside the temporary repository."
    }
    $remoteCandidate = if ([IO.Path]::IsPathRooted($RemotePath)) { $RemotePath } else { Join-Path $RepoRoot $RemotePath }
    $resolvedRemote = (Resolve-Path -LiteralPath $remoteCandidate).Path
    if (-not (Test-PathWithin -Root $tempRoot -Path $resolvedRemote) -or
        -not (Test-Path -LiteralPath $resolvedRemote -PathType Container)) {
        throw "LocalTestMode remote must be a local directory under the OS temporary directory."
    }
    $isBare = (& git --git-dir $resolvedRemote rev-parse --is-bare-repository 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or $isBare -ne "true") {
        throw "LocalTestMode remote must be a local bare Git repository."
    }
    return [pscustomobject]@{
        Remote = $resolvedRemote
        Verifier = $verifierPath
        VerificationCommand = "pwsh -NoProfile -File .tools/test-verifier.ps1"
    }
}

function Write-PublicationReceipt {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$Commit,
        [Parameter(Mandatory = $true)][string]$Tree,
        [Parameter(Mandatory = $true)][string]$RemoteTip,
        [Parameter(Mandatory = $true)][string]$VerificationCommand
    )
    $directory = Join-Path $RepoRoot "artifacts\publication"
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $path = Join-Path $directory "last-push.json"
    $temporary = Join-Path $directory (".last-push-" + [guid]::NewGuid().ToString("N") + ".tmp")
    $receipt = [ordered]@{
        schema = "candle.git_publication.v1"
        generated_at_utc = [DateTimeOffset]::UtcNow.ToString("o")
        repository = "Shoozes/candle"
        branch = "main"
        commit = $Commit
        tree = $Tree
        remote_tip = $RemoteTip
        verification = [ordered]@{
            command = $VerificationCommand
            exit_code = 0
            candidate_tree = $Tree
        }
    }
    try {
        [IO.File]::WriteAllText(
            $temporary,
            (($receipt | ConvertTo-Json -Depth 8) + [Environment]::NewLine),
            [Text.UTF8Encoding]::new($false)
        )
        Move-Item -LiteralPath $temporary -Destination $path -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
    return $path
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

$localConfig = $null
if ($LocalTestMode) {
    $localConfig = Resolve-LocalTestConfig -RepoRoot $repoRoot -RemotePath $Remote -Verifier $LocalTestVerifier
    $Remote = $localConfig.Remote
} elseif (-not [string]::IsNullOrWhiteSpace($LocalTestVerifier)) {
    throw "-LocalTestVerifier is valid only with -LocalTestMode."
}

if (-not $Yes) {
    $answer = Read-Host "Publish the verified clean main commit to Shoozes/candle? Type YES to continue"
    if ($answer -cne "YES") {
        Write-Host "Canceled."
        exit 0
    }
}

$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $branch -ne "main") {
    throw "Publication requires the named main branch; current branch is '$branch'."
}
$status = ((& git status --porcelain=v1 --untracked-files=all) -join "`n")
if (-not [string]::IsNullOrWhiteSpace($status)) {
    throw "Publication requires a clean reviewed worktree. Commit explicit paths first; no files were staged or pushed."
}

$originUrl = (& git remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($originUrl)) {
    throw "The origin remote is missing."
}
if ($LocalTestMode) {
    if ([IO.Path]::GetFullPath($originUrl) -ne [IO.Path]::GetFullPath($Remote)) {
        throw "LocalTestMode origin does not match the supplied bare remote."
    }
} else {
    $normalizedOrigin = $originUrl.TrimEnd('/') -replace '\.git$', ''
    $normalizedRequested = $Remote.TrimEnd('/') -replace '\.git$', ''
    if ($normalizedOrigin -cne $normalizedRequested -or $normalizedOrigin -cne "https://github.com/Shoozes/candle") {
        throw "Publication is restricted to https://github.com/Shoozes/candle.git; origin is '$originUrl'."
    }
}

$askPass = $null
$tokenPath = $null
if (-not $LocalTestMode) {
    $tokenPath = Join-Path $repoRoot ".tools\.secrets\gt.txt"
    $tokenItem = Get-Item -LiteralPath $tokenPath -ErrorAction SilentlyContinue
    if ($null -eq $tokenItem -or $tokenItem.PSIsContainer -or $tokenItem.Length -le 0) {
        throw "Missing or empty token file: .tools/.secrets/gt.txt"
    }
    & git check-ignore --quiet -- ".tools/.secrets/gt.txt"
    if ($LASTEXITCODE -ne 0) {
        throw ".tools/.secrets/gt.txt must be ignored before publication."
    }
    & git ls-files --error-unmatch -- ".tools/.secrets/gt.txt" *>$null
    if ($LASTEXITCODE -eq 0) {
        throw ".tools/.secrets/gt.txt is tracked; remove it from the index before publication."
    }
    $askPass = Join-Path $repoRoot ".tools\git-askpass.cmd"
    if (-not (Test-Path -LiteralPath $askPass -PathType Leaf)) {
        throw "Git askpass helper is missing: .tools/git-askpass.cmd"
    }
}

$gitEnvironmentNames = @(
    "GIT_ASKPASS",
    "GIT_ASKPASS_REQUIRE",
    "CANDLE_GIT_TOKEN_FILE",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_KEY_0",
    "GIT_CONFIG_VALUE_0"
)
$previousGitEnvironment = @{}
foreach ($name in $gitEnvironmentNames) {
    $previousGitEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    if ($LocalTestMode) {
        foreach ($name in $gitEnvironmentNames) {
            [Environment]::SetEnvironmentVariable($name, $null, "Process")
        }
    } else {
        $env:GIT_ASKPASS = $askPass
        $env:GIT_ASKPASS_REQUIRE = "force"
        $env:CANDLE_GIT_TOKEN_FILE = $tokenPath
        $env:GIT_CONFIG_COUNT = "1"
        $env:GIT_CONFIG_KEY_0 = "credential.helper"
        $env:GIT_CONFIG_VALUE_0 = ""
    }

    & git fetch origin
    if ($LASTEXITCODE -ne 0) {
        throw "Fetch failed. Check token, repository permission, and network access."
    }
    & git rev-parse --verify "origin/main^{commit}" *>$null
    if ($LASTEXITCODE -ne 0) {
        throw "origin/main is unavailable."
    }
    & git merge-base --is-ancestor origin/main HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Local main is behind or diverged from origin/main. Integrate it without force, review, and retry."
    }

    $candidateCommit = (& git rev-parse HEAD).Trim()
    $candidateTree = (& git rev-parse "HEAD^{tree}").Trim()
    $verificationCommand = "pwsh -NoProfile -File .tools/verify-before-push.ps1"
    $verifier = Join-Path $repoRoot ".tools\verify-before-push.ps1"
    if ($LocalTestMode) {
        $verifier = $localConfig.Verifier
        $verificationCommand = $localConfig.VerificationCommand
    }
    Write-Host "Verifying commit $candidateCommit (tree $candidateTree) ..."
    & pwsh -NoProfile -File $verifier
    if ($LASTEXITCODE -ne 0) {
        throw "Local verification failed. Nothing was pushed."
    }

    $postStatus = ((& git status --porcelain=v1 --untracked-files=all) -join "`n")
    if (-not [string]::IsNullOrWhiteSpace($postStatus)) {
        throw "Verification changed the clean candidate. Nothing was pushed."
    }
    $postCommit = (& git rev-parse HEAD).Trim()
    $postTree = (& git rev-parse "HEAD^{tree}").Trim()
    if ($postCommit -ne $candidateCommit -or $postTree -ne $candidateTree) {
        throw "HEAD changed during verification. Nothing was pushed."
    }

    & git fetch origin
    if ($LASTEXITCODE -ne 0) {
        throw "Post-verification fetch failed. Nothing was pushed."
    }
    & git merge-base --is-ancestor origin/main $candidateCommit
    if ($LASTEXITCODE -ne 0) {
        throw "origin/main changed incompatibly during verification. Nothing was pushed."
    }

    if ($DryRun) {
        Write-Host "Dry run complete: commit $candidateCommit is clean, verified, and fast-forwardable."
        exit 0
    }

    & git push -u origin "${candidateCommit}:refs/heads/main"
    if ($LASTEXITCODE -ne 0) {
        throw "Push failed. The verified commit remains local; no success receipt was written."
    }
    $remoteLine = (& git ls-remote --heads origin "refs/heads/main" | Select-Object -First 1)
    $remoteTip = if ($remoteLine) { ([string]$remoteLine -split "\s+")[0] } else { "" }
    if ($remoteTip -ne $candidateCommit) {
        throw "Remote verification failed: origin/main is '$remoteTip', expected '$candidateCommit'."
    }
    $receiptPath = Write-PublicationReceipt `
        -RepoRoot $repoRoot `
        -Commit $candidateCommit `
        -Tree $candidateTree `
        -RemoteTip $remoteTip `
        -VerificationCommand $verificationCommand
    Write-Host "Published verified commit $candidateCommit to origin/main. Receipt: $receiptPath"
} finally {
    foreach ($name in $gitEnvironmentNames) {
        [Environment]::SetEnvironmentVariable($name, $previousGitEnvironment[$name], "Process")
    }
}
