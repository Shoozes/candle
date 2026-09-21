[CmdletBinding()]
param(
    [string]$Model = 'C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf',
    [string]$Tokenizer = 'C:\llamacpp\models\gpt-oss-20b\tokenizer.json',
    [string]$TokenizerConfig = 'C:\llamacpp\models\gpt-oss-20b\tokenizer_config.json',
    [string]$ChatTemplate = 'C:\llamacpp\models\gpt-oss-20b\chat_template.jinja',
    [Parameter(Mandatory)][string]$ReferenceLogits,
    [string]$Fixture = 'tests\fixtures\gpt_oss_task3_short_parity\reference.json',
    [string]$Output = 'artifacts\gpt-oss\short-parity\receipt.json',
    [int]$DeviceIndex = 0,
    [UInt64]$MaxWeightBytes = 34359738368
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Assert-FileHash {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Expected,
        [Parameter(Mandatory)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label is missing: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch: expected $Expected, got $actual"
    }
    return $actual
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot

$fixturePath = (Resolve-Path -LiteralPath $Fixture).Path
$fixtureData = Get-Content -LiteralPath $fixturePath -Raw -Encoding utf8 | ConvertFrom-Json
if ($fixtureData.fixture_id -ne 'gpt-oss-task3-short-parity-c8-v1') {
    throw "unexpected short-parity fixture id: $($fixtureData.fixture_id)"
}

Assert-FileHash -Path $Model -Expected 'aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778' -Label 'GGUF model' | Out-Null
Assert-FileHash -Path $Tokenizer -Expected '0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3' -Label 'tokenizer' | Out-Null
Assert-FileHash -Path $TokenizerConfig -Expected '9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070' -Label 'tokenizer_config' | Out-Null
Assert-FileHash -Path $ChatTemplate -Expected 'a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146' -Label 'chat template' | Out-Null
Assert-FileHash -Path $ReferenceLogits -Expected ([string]$fixtureData.reference_file_sha256) -Label 'reference logits' | Out-Null

$baselineCommit = (& git rev-parse HEAD).Trim()
$baselineTree = (& git show -s --format=%T HEAD).Trim()
$statusLines = @(& git status --short --untracked-files=all)
$candidateDirty = $statusLines.Count -gt 0
$temporaryIndex = Join-Path ([IO.Path]::GetTempPath()) ("candle-gptoss-parity-index-" + [guid]::NewGuid().ToString('N'))
$oldGitIndex = $env:GIT_INDEX_FILE
try {
    $env:GIT_INDEX_FILE = $temporaryIndex
    & git read-tree HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "git read-tree failed with exit code $LASTEXITCODE"
    }
    & git add -A -- .
    if ($LASTEXITCODE -ne 0) {
        throw "git add failed with exit code $LASTEXITCODE"
    }
    $candidateTree = (& git write-tree).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "git write-tree failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($null -eq $oldGitIndex) {
        Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue
    }
    else {
        $env:GIT_INDEX_FILE = $oldGitIndex
    }
    if (Test-Path -LiteralPath $temporaryIndex -PathType Leaf) {
        Remove-Item -LiteralPath $temporaryIndex -Force
    }
}
$outputPath = [IO.Path]::GetFullPath($Output)
$outputParent = Split-Path -Parent $outputPath
New-Item -ItemType Directory -Force -Path $outputParent | Out-Null

$cargoArgs = @(
    'run', '--locked', '--features', 'cuda', '-p', 'candle-examples',
    '--example', 'gpt-oss-short-parity', '--',
    '--model', $Model,
    '--tokenizer', $Tokenizer,
    '--tokenizer-config', $TokenizerConfig,
    '--chat-template', $ChatTemplate,
    '--fixture', $fixturePath,
    '--reference-logits', $ReferenceLogits,
    '--output', $outputPath,
    '--prompt', [string]$fixtureData.prompt,
    '--baseline-commit', $baselineCommit,
    '--baseline-tree', $baselineTree,
    '--candidate-tree', $candidateTree,
    '--candidate-dirty', ([string]$candidateDirty).ToLowerInvariant(),
    '--device-index', [string]$DeviceIndex,
    '--max-weight-bytes', [string]$MaxWeightBytes
)

& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) {
    throw "gpt-oss-short-parity failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
    throw "parity runner did not write its receipt: $outputPath"
}
Write-Output "gpt-oss-short-parity: passed; receipt=$outputPath"
