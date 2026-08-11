[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Condition {
    param(
        [Parameter(Mandatory)]
        [bool]$Condition,

        [Parameter(Mandatory)]
        [string]$Message
    )

    if (-not $Condition) {
        throw $Message
    }
}

function Stop-TestProcessTree {
    param([Parameter(Mandatory)][Diagnostics.Process]$Process)

    if ($Process.HasExited) {
        return
    }
    try {
        # Process.Kill(bool) is available on modern .NET but not PS5.1/.NET
        # Framework. Fall back to the exact owned PID and its descendants.
        $Process.Kill($true)
    }
    catch [System.Management.Automation.MethodException] {
        try {
            $Process.Kill()
        }
        catch {
            # Last-resort exact-PID tree cleanup for runtimes where Kill() is
            # denied; do not turn a cleanup diagnostic into a false test pass.
            & taskkill.exe /PID $Process.Id /T /F > $null 2>&1
        }
    }
    [void]$Process.WaitForExit(5000)
}

$wrapper = Join-Path -Path $PSScriptRoot -ChildPath "run-bounded-oracle.ps1"
$wrapper = (Resolve-Path -LiteralPath $wrapper).ProviderPath
$tokens = $null
$parseErrors = $null
$wrapperAst = [Management.Automation.Language.Parser]::ParseFile(
    $wrapper,
    [ref]$tokens,
    [ref]$parseErrors
)
Assert-Condition ($parseErrors.Count -eq 0) "bounded wrapper has PowerShell parse errors"
$counterFunction = $wrapperAst.Find({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and
        $node.Name -eq "ConvertTo-ProcessCounterBytes"
    }, $true)
Assert-Condition ($null -ne $counterFunction) "bounded wrapper omitted the process-counter conversion helper"
. ([ScriptBlock]::Create($counterFunction.Extent.Text))
$aboveInt32 = [Int64]2177044480
Assert-Condition `
    ((ConvertTo-ProcessCounterBytes -Value $aboveInt32) -eq [UInt64]$aboveInt32) `
    "process-counter conversion truncated or rejected a value above Int32 max"
Assert-Condition `
    ((ConvertTo-ProcessCounterBytes -Value -1) -eq [UInt64]0) `
    "process-counter conversion did not clamp an unavailable negative counter"
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testRoot = Join-Path -Path $tempRoot -ChildPath "candle-lfm2-vl-oracle-$([Guid]::NewGuid().ToString('N'))"
$testRoot = [IO.Path]::GetFullPath($testRoot)
if (-not $testRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "refusing to create bounded-oracle test outside the system temp directory"
}

$passed = $false
$existing = $null
$alternate = $null
$owner = $null
$ownerChildId = $null
try {
    [void](New-Item -ItemType Directory -Path $testRoot -ErrorAction Stop)
    $childPath = Join-Path -Path $testRoot -ChildPath "bounded-oracle-child.exe"
    Copy-Item -LiteralPath $env:ComSpec -Destination $childPath -ErrorAction Stop
    $normalEvidencePath = Join-Path -Path $testRoot -ChildPath "normal.json"
    $normalLogPath = Join-Path -Path $testRoot -ChildPath "normal.log"
    $timeoutEvidencePath = Join-Path -Path $testRoot -ChildPath "timeout.json"
    $ownerExitEvidencePath = Join-Path -Path $testRoot -ChildPath "owner-exit.json"
    $ownerReadyPath = Join-Path -Path $testRoot -ChildPath "owner-ready.txt"
    $refusalEvidencePath = Join-Path -Path $testRoot -ChildPath "refusal.json"
    $exactRefusalEvidencePath = Join-Path -Path $testRoot -ChildPath "exact-refusal.json"
    $executableScopeEvidencePath = Join-Path -Path $testRoot -ChildPath "executable-scope.json"
    $common = @{
        FilePath = $childPath
        TimeoutSeconds = 5
        MaxJobMemoryBytes = [UInt64]268435456
        PollMilliseconds = 50
        WorkingDirectory = $testRoot
    }

    & $wrapper @common `
        -ArgumentList @("/d", "/c", 'if "%GGML_CUDA_DISABLE_GRAPHS%"=="1" (echo bounded-stdout & echo bounded-stderr 1>&2 & ping 127.0.0.1 -n 2 > nul & exit /b 0) else exit /b 9') `
        -LogPath $normalLogPath `
        -EvidencePath $normalEvidencePath | Out-Null
    $normal = Get-Content -LiteralPath $normalEvidencePath -Raw | ConvertFrom-Json
    Assert-Condition ($normal.contract -eq "candle-lfm2-vl-bounded-oracle-v1") "normal run has the wrong evidence contract"
    Assert-Condition ($normal.job_assigned -eq $true) "normal run was not assigned to the job"
    Assert-Condition ($normal.started_suspended -eq $true) "normal run was not created suspended"
    Assert-Condition ($normal.assigned_before_resume -eq $true) "normal run resumed before job assignment"
    Assert-Condition ($normal.resumed -eq $true) "normal run was not resumed"
    Assert-Condition ($normal.termination_reason -eq "exited") "normal run did not exit normally"
    Assert-Condition ($normal.child_exit_code -eq 0) "normal run has a nonzero child exit code"
    Assert-Condition ($normal.cuda_graphs_disabled -eq $true) "normal run did not disable CUDA graphs"
    Assert-Condition ($normal.physical_memory_source -in @("Win32_ComputerSystem", "GlobalMemoryStatusEx")) "normal run did not record a supported physical-memory source"
    Assert-Condition ($normal.pid_absent_after_cleanup -eq $true) "normal run left its child PID present"
    Assert-Condition ($normal.peak_job_memory_bytes -gt 0) "normal run did not record peak job memory"
    Assert-Condition ($normal.combined_log_path -eq $normalLogPath) "normal run did not record its combined log path"
    Assert-Condition ($normal.combined_log_bytes -gt 0) "normal run recorded an empty combined log"
    Assert-Condition ($normal.combined_log_sha256.Length -eq 64) "normal run did not hash its combined log"
    $normalLog = Get-Content -LiteralPath $normalLogPath -Raw
    Assert-Condition ($normalLog -like "*bounded-stdout*") "normal log omitted stdout"
    Assert-Condition ($normalLog -like "*bounded-stderr*") "normal log omitted stderr"

    $treeCommand = 'start "" /b "{0}" /d /c "ping 127.0.0.1 -n 30 > nul" & ping 127.0.0.1 -n 30 > nul' -f $childPath
    $timeoutObserved = $false
    try {
        & $wrapper `
            -FilePath $childPath `
            -ArgumentList @("/d", "/c", $treeCommand) `
            -TimeoutSeconds 1 `
            -MaxJobMemoryBytes ([UInt64]268435456) `
            -PollMilliseconds 50 `
            -WorkingDirectory $testRoot `
            -EvidencePath $timeoutEvidencePath | Out-Null
    }
    catch {
        if ($_.Exception.Message -notlike "*timed out*") {
            throw
        }
        $timeoutObserved = $true
    }
    Assert-Condition $timeoutObserved "timeout run unexpectedly succeeded"
    $timeout = Get-Content -LiteralPath $timeoutEvidencePath -Raw | ConvertFrom-Json
    Assert-Condition ($timeout.termination_reason -eq "timeout") "timeout run has the wrong termination reason"
    Assert-Condition ($timeout.pid_absent_after_cleanup -eq $true) "timeout run left its parent PID present"
    Start-Sleep -Milliseconds 200
    $remaining = @(Get-Process -Name "bounded-oracle-child" -ErrorAction SilentlyContinue)
    Assert-Condition ($remaining.Count -eq 0) "timeout run left a descendant process present"

    $ownerStartInfo = [Diagnostics.ProcessStartInfo]::new()
    $ownerStartInfo.FileName = (Get-Command pwsh -ErrorAction Stop).Source
    $ownerStartInfo.UseShellExecute = $false
    $ownerStartInfo.CreateNoWindow = $true
    $quotedWrapper = $wrapper.Replace("'", "''")
    $quotedChildPath = $childPath.Replace("'", "''")
    $quotedTestRoot = $testRoot.Replace("'", "''")
    $quotedOwnerEvidencePath = $ownerExitEvidencePath.Replace("'", "''")
    $ownerInvocation = @"
& '$quotedWrapper' ``
    -FilePath '$quotedChildPath' ``
    -ArgumentList @('/d', '/c', 'echo ready>owner-ready.txt & ping 127.0.0.1 -n 30 > nul') ``
    -TimeoutSeconds 30 ``
    -MaxJobMemoryBytes ([UInt64]268435456) ``
    -PollMilliseconds 50 ``
    -WorkingDirectory '$quotedTestRoot' ``
    -EvidencePath '$quotedOwnerEvidencePath'
"@
    $encodedOwnerInvocation = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($ownerInvocation)
    )
    # ProcessStartInfo.ArgumentList is unavailable on Windows PowerShell 5.1's
    # .NET Framework. The encoded command is base64/whitespace-safe, so the
    # legacy Arguments string is equivalent on both supported runtimes.
    $ownerStartInfo.Arguments = "-NoProfile -NonInteractive -EncodedCommand $encodedOwnerInvocation"
    $owner = [Diagnostics.Process]::new()
    $owner.StartInfo = $ownerStartInfo
    Assert-Condition ($owner.Start()) "failed to start the owner-exit wrapper process"

    $ownerChild = $null
    $ownerChildReady = $false
    $ownerChildDeadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
    while (-not $ownerChildReady -and [DateTimeOffset]::UtcNow -lt $ownerChildDeadline) {
        Start-Sleep -Milliseconds 50
        $ownerChildren = @(Get-Process -Name "bounded-oracle-child" -ErrorAction SilentlyContinue)
        if ($ownerChildren.Count -gt 0) {
            $ownerChild = $ownerChildren[0]
            $ownerChildId = $ownerChild.Id
        }
        $ownerChildReady = $null -ne $ownerChild -and (Test-Path -LiteralPath $ownerReadyPath)
    }
    Assert-Condition ($null -ne $ownerChild) "owner-exit run did not start its bounded child"
    Assert-Condition $ownerChildReady "owner-exit child did not reach its post-assignment resume handshake"
    Assert-Condition (-not $owner.HasExited) "owner-exit wrapper stopped before the kill-on-close test"
    $ownerChild.Dispose()
    $owner.Kill()
    Assert-Condition ($owner.WaitForExit(5000)) "owner-exit wrapper did not stop"
    $owner.Dispose()
    $owner = $null

    $ownerChildAbsent = $false
    $ownerCleanupDeadline = [DateTimeOffset]::UtcNow.AddSeconds(5)
    while (-not $ownerChildAbsent -and [DateTimeOffset]::UtcNow -lt $ownerCleanupDeadline) {
        Start-Sleep -Milliseconds 50
        $ownerChildAbsent = $null -eq (Get-Process -Id $ownerChildId -ErrorAction SilentlyContinue)
    }
    Assert-Condition $ownerChildAbsent "owner exit left bounded child PID $ownerChildId present"
    $ownerChildId = $null
    Assert-Condition (-not (Test-Path -LiteralPath $ownerExitEvidencePath)) "forced owner exit wrote misleading completion evidence"

    $existing = Start-Process `
        -FilePath $childPath `
        -ArgumentList @("/d", "/c", "ping 127.0.0.1 -n 30 > nul") `
        -PassThru `
        -WindowStyle Hidden
    Start-Sleep -Milliseconds 200
    $refusalObserved = $false
    try {
        & $wrapper @common `
            -ArgumentList @("/d", "/c", "exit /b 0") `
            -EvidencePath $refusalEvidencePath | Out-Null
    }
    catch {
        if ($_.Exception.Message -notlike "*refusing concurrent launch*") {
            throw
        }
        $refusalObserved = $true
    }
    Assert-Condition $refusalObserved "concurrent process-name launch was not refused"
    Assert-Condition (-not (Test-Path -LiteralPath $refusalEvidencePath)) "refused launch wrote misleading execution evidence"

    $exactRefusalObserved = $false
    try {
        & $wrapper @common `
            -ConcurrencyScope Executable `
            -ArgumentList @("/d", "/c", "exit /b 0") `
            -EvidencePath $exactRefusalEvidencePath | Out-Null
    }
    catch {
        if ($_.Exception.Message -notlike "*refusing concurrent launch*") {
            throw
        }
        $exactRefusalObserved = $true
    }
    Assert-Condition $exactRefusalObserved "concurrent exact-executable launch was not refused"
    Assert-Condition (-not (Test-Path -LiteralPath $exactRefusalEvidencePath)) "exact-executable refusal wrote misleading execution evidence"

    Stop-TestProcessTree -Process $existing
    $existing.Dispose()
    $existing = $null
    $alternateRoot = Join-Path -Path $testRoot -ChildPath "alternate"
    [void](New-Item -ItemType Directory -Path $alternateRoot -ErrorAction Stop)
    $alternatePath = Join-Path -Path $alternateRoot -ChildPath ([IO.Path]::GetFileName($childPath))
    Copy-Item -LiteralPath $env:ComSpec -Destination $alternatePath -ErrorAction Stop
    $alternate = Start-Process `
        -FilePath $alternatePath `
        -ArgumentList @("/d", "/c", "ping 127.0.0.1 -n 30 > nul") `
        -PassThru `
        -WindowStyle Hidden
    Start-Sleep -Milliseconds 200
    & $wrapper @common `
        -ConcurrencyScope Executable `
        -ArgumentList @("/d", "/c", "exit /b 0") `
        -EvidencePath $executableScopeEvidencePath | Out-Null
    $executableScope = Get-Content -LiteralPath $executableScopeEvidencePath -Raw | ConvertFrom-Json
    Assert-Condition ($executableScope.concurrency_scope -eq "Executable") "executable-scope run did not record its concurrency policy"
    Assert-Condition ($executableScope.child_exit_code -eq 0) "executable-scope run did not exit normally"
    Assert-Condition ($executableScope.pid_absent_after_cleanup -eq $true) "executable-scope run left its child PID present"

    $passed = $true
    "bounded-oracle-smoke: passed"
}
finally {
    if ($null -ne $ownerChildId) {
        $remainingOwnerChild = Get-Process -Id $ownerChildId -ErrorAction SilentlyContinue
        if ($null -ne $remainingOwnerChild) {
            try {
                if ($remainingOwnerChild.ProcessName -eq "bounded-oracle-child") {
                    Stop-TestProcessTree -Process $remainingOwnerChild
                }
            }
            finally {
                $remainingOwnerChild.Dispose()
            }
        }
    }
    if ($null -ne $owner) {
        try {
            if (-not $owner.HasExited) {
                Stop-TestProcessTree -Process $owner
            }
        }
        finally {
            $owner.Dispose()
        }
    }
    if ($null -ne $existing) {
        try {
            if (-not $existing.HasExited) {
                Stop-TestProcessTree -Process $existing
            }
        }
        finally {
            $existing.Dispose()
        }
    }
    if ($null -ne $alternate) {
        try {
            if (-not $alternate.HasExited) {
                Stop-TestProcessTree -Process $alternate
            }
        }
        finally {
            $alternate.Dispose()
        }
    }
    if ($passed -and (Test-Path -LiteralPath $testRoot)) {
        $resolvedTestRoot = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $testRoot).ProviderPath)
        if (-not $resolvedTestRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
            throw "refusing to remove bounded-oracle test directory outside system temp"
        }
        Remove-Item -LiteralPath $resolvedTestRoot -Recurse -Force
    }
    elseif (-not $passed) {
        Write-Warning "bounded-oracle test artifacts retained at $testRoot"
    }
}
