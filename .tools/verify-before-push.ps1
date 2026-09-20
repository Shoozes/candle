[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "verify-before-push.ps1 requires PowerShell 7."
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Command
    )
    Write-Host "[candle-verify] $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

function Resolve-Python {
    foreach ($candidate in @("python", "python3")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return ,@($command.Source)
        }
    }
    $launcher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        return ,@($launcher.Source, "-3")
    }
    throw "Python is required for the module-layout verifier."
}

function Resolve-GitBash {
    $candidates = @(
        (Join-Path $env:ProgramFiles "Git\bin\bash.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Git\bin\bash.exe")
    )
    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $command = Get-Command "bash.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command -and $command.Source -match "(?i)[\\/]Git[\\/]") {
        return $command.Source
    }
    throw "Git for Windows bash.exe is required for the overlay verifier."
}

$previousEnvironment = @{
    CARGO_NET_OFFLINE = $env:CARGO_NET_OFFLINE
    CARGO_BUILD_JOBS = $env:CARGO_BUILD_JOBS
    HF_HUB_OFFLINE = $env:HF_HUB_OFFLINE
    HF_HUB_DISABLE_TELEMETRY = $env:HF_HUB_DISABLE_TELEMETRY
    PYO3_NO_PYTHON = $env:PYO3_NO_PYTHON
}

try {
    $env:CARGO_NET_OFFLINE = "true"
    $env:CARGO_BUILD_JOBS = "2"
    $env:HF_HUB_OFFLINE = "1"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:PYO3_NO_PYTHON = "1"

    Invoke-Step "format" { & cargo fmt --all -- --check }
    Invoke-Step "maintained library check" {
        & cargo check --locked --offline -j 2 -p candle-core -p candle-nn -p candle-transformers -p candle-vlm
    }
    foreach ($example in @("lfm2", "quantized-lfm2", "lfm2-vl")) {
        Invoke-Step "example check: $example" {
            & cargo check --locked --offline -j 2 -p candle-examples --example $example
        }
    }
    Invoke-Step "transformer clippy" {
        & cargo clippy --locked --offline -j 2 -p candle-transformers --lib -- -D warnings
    }
    Invoke-Step "maintained library tests" {
        & cargo test --locked --offline -j 2 -p candle-core -p candle-transformers -p candle-vlm
    }
    Invoke-Step "LFM2-VL example tests" {
        & cargo test --locked --offline -j 2 -p candle-examples --example lfm2-vl
    }
    Invoke-Step "summary bank" {
        & pwsh -NoProfile -File (Join-Path $repoRoot "scripts\lfm2-vl\verify-summary-bank.ps1")
    }

    $python = Resolve-Python
    Invoke-Step "module layout" {
        if ($python.Count -eq 1) {
            & $python[0] (Join-Path $repoRoot "scripts\lfm2-vl\verify-module-layout.py")
        } else {
            & $python[0] @($python[1..($python.Count - 1)]) (Join-Path $repoRoot "scripts\lfm2-vl\verify-module-layout.py")
        }
    }

    $bash = Resolve-GitBash
    $bashRoot = $repoRoot.Replace('\', '/')
    Invoke-Step "fork overlay union" {
        & $bash -lc "cd '$bashRoot' && bash scripts/verify-fork-overlays.sh"
    }
    Invoke-Step "Git whitespace" { & git diff --check HEAD }
    Write-Host "[candle-verify] complete"
} finally {
    foreach ($entry in $previousEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}
