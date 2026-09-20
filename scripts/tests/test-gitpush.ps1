[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$helperSource = Join-Path $repoRoot ".tools\gitpush.ps1"
$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("candle-gitpush-" + [guid]::NewGuid().ToString("N"))

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string]$WorkingDirectory, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & git -C $WorkingDirectory @Arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed in $WorkingDirectory"
    }
}

function Assert-Equal {
    param($Actual, $Expected, [string]$Message)
    if ($Actual -cne $Expected) {
        throw "$Message`nExpected: $Expected`nActual: $Actual"
    }
}

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function New-Fixture {
    param([Parameter(Mandatory = $true)][string]$Name)
    $root = Join-Path $tempRoot $Name
    $publisher = Join-Path $root "publisher"
    $remote = Join-Path $root "remote.git"
    $race = Join-Path $root "race"
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    & git init --bare $remote | Out-Null
    & git init -b main $publisher | Out-Null
    Invoke-Git $publisher config user.name "Candle Test"
    Invoke-Git $publisher config user.email "candle-test@invalid.local"
    New-Item -ItemType Directory -Force -Path (Join-Path $publisher ".tools") | Out-Null
    Copy-Item -LiteralPath $helperSource -Destination (Join-Path $publisher ".tools\gitpush.ps1")
    [IO.File]::WriteAllText((Join-Path $publisher ".tools\gitpush-test-mode.marker"), "candle.gitpush.test.v1`n")
    [IO.File]::WriteAllText((Join-Path $publisher ".gitignore"), "artifacts/`n")
    [IO.File]::WriteAllText((Join-Path $publisher "candidate.txt"), "base`n")
    Invoke-Git $publisher add -- ".gitignore" ".tools" "candidate.txt"
    Invoke-Git $publisher commit -m "test: base"
    Invoke-Git $publisher remote add origin $remote
    Invoke-Git $publisher push -u origin main
    & git clone -b main $remote $race | Out-Null
    Invoke-Git $race config user.name "Candle Race Test"
    Invoke-Git $race config user.email "candle-race@invalid.local"
    [IO.File]::WriteAllText((Join-Path $publisher "candidate.txt"), "candidate`n")
    Invoke-Git $publisher add -- "candidate.txt"
    Invoke-Git $publisher commit -m "test: candidate"
    return [pscustomobject]@{ Root = $root; Publisher = $publisher; Remote = $remote; Race = $race }
}

function Write-Verifier {
    param([Parameter(Mandatory = $true)]$Fixture, [Parameter(Mandatory = $true)][string]$Behavior)
    $path = Join-Path $Fixture.Publisher ".tools\test-verifier.ps1"
    $observation = Join-Path $Fixture.Root "observed.txt"
    $source = @"
`$ErrorActionPreference = "Stop"
`$repo = (Resolve-Path (Join-Path `$PSScriptRoot "..")).Path
`$receipt = Join-Path `$repo "artifacts\publication\last-push.json"
if (Test-Path -LiteralPath `$receipt) { exit 90 }
`$head = (& git -C `$repo rev-parse HEAD).Trim()
`$tree = (& git -C `$repo rev-parse "HEAD^{tree}").Trim()
[IO.File]::WriteAllText("$($observation.Replace('\', '\\'))", "`$head``n`$tree``n")
switch ("$Behavior") {
    "fail" { exit 17 }
    "drift" { [IO.File]::AppendAllText((Join-Path `$repo "candidate.txt"), "drift``n"); exit 0 }
    "race" {
        [IO.File]::WriteAllText("$((Join-Path $Fixture.Race 'race.txt').Replace('\', '\\'))", "race``n")
        & git -C "$($Fixture.Race.Replace('\', '\\'))" add -- race.txt
        & git -C "$($Fixture.Race.Replace('\', '\\'))" commit -m "test: race"
        & git -C "$($Fixture.Race.Replace('\', '\\'))" push origin "HEAD:refs/heads/main"
        exit `$LASTEXITCODE
    }
    "success" { exit 0 }
    default { exit 98 }
}
"@
    [IO.File]::WriteAllText($path, $source, [Text.UTF8Encoding]::new($false))
    Invoke-Git $Fixture.Publisher add -- ".tools/test-verifier.ps1"
    Invoke-Git $Fixture.Publisher commit -m "test: verifier $Behavior"
    return [pscustomobject]@{ Path = $path; Observation = $observation }
}

function Invoke-Helper {
    param([Parameter(Mandatory = $true)]$Fixture)
    $output = & pwsh -NoProfile -File (Join-Path $Fixture.Publisher ".tools\gitpush.ps1") `
        -Remote $Fixture.Remote -Yes -LocalTestMode -LocalTestVerifier ".tools/test-verifier.ps1" 2>&1
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = ($output -join "`n") }
}

function Get-RemoteTip($Fixture) {
    return (& git --git-dir $Fixture.Remote rev-parse refs/heads/main).Trim()
}

function Get-LocalHead($Fixture) {
    return (& git -C $Fixture.Publisher rev-parse HEAD).Trim()
}

try {
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    $cases = @("fail", "drift", "race", "success")
    foreach ($case in $cases) {
        $fixture = New-Fixture -Name $case
        $initialRemote = Get-RemoteTip $fixture
        $verifier = Write-Verifier -Fixture $fixture -Behavior $case
        $candidate = Get-LocalHead $fixture
        $result = Invoke-Helper -Fixture $fixture
        $receipt = Join-Path $fixture.Publisher "artifacts\publication\last-push.json"

        if ($case -eq "success") {
            Assert-Equal $result.ExitCode 0 "success helper invocation failed: $($result.Output)"
            Assert-Equal (Get-RemoteTip $fixture) $candidate "success did not publish the verified commit"
            Assert-True (Test-Path -LiteralPath $receipt -PathType Leaf) "success receipt is missing"
            $data = Get-Content -Raw -LiteralPath $receipt | ConvertFrom-Json
            Assert-Equal $data.schema "candle.git_publication.v1" "receipt schema mismatch"
            Assert-Equal $data.commit $candidate "receipt commit mismatch"
            Assert-Equal $data.remote_tip $candidate "receipt remote tip mismatch"
        } else {
            Assert-True ($result.ExitCode -ne 0) "$case unexpectedly succeeded"
            Assert-True (-not (Test-Path -LiteralPath $receipt)) "$case wrote a success receipt"
            if ($case -in @("fail", "drift")) {
                Assert-Equal (Get-RemoteTip $fixture) $initialRemote "$case moved the remote"
            } else {
                Assert-True ((Get-RemoteTip $fixture) -ne $candidate) "race published the candidate"
            }
        }
        $observed = Get-Content -LiteralPath $verifier.Observation
        Assert-Equal $observed[0] $candidate "$case verifier observed the wrong commit"
        Write-Host "gitpush case=$case passed"
    }
    Write-Host "gitpush integration: 4 passed"
} finally {
    if (Test-Path -LiteralPath $tempRoot) {
        $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
        $osTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if ($resolvedTemp.StartsWith($osTemp, [StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
        }
    }
}
