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

$wrapper = Join-Path -Path $PSScriptRoot -ChildPath "run-bounded-oracle.ps1"
$wrapper = (Resolve-Path -LiteralPath $wrapper).ProviderPath
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$testRoot = Join-Path -Path $tempRoot -ChildPath "candle-lfm2-vl-oracle-$([Guid]::NewGuid().ToString('N'))"
$testRoot = [IO.Path]::GetFullPath($testRoot)
if (-not $testRoot.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "refusing to create bounded-oracle test outside the system temp directory"
}

$passed = $false
$existing = $null
$owner = $null
try {
    [void](New-Item -ItemType Directory -Path $testRoot -ErrorAction Stop)
    $childPath = Join-Path -Path $testRoot -ChildPath "bounded-oracle-child.exe"
    Copy-Item -LiteralPath $env:ComSpec -Destination $childPath -ErrorAction Stop
    $normalEvidencePath = Join-Path -Path $testRoot -ChildPath "normal.json"
    $timeoutEvidencePath = Join-Path -Path $testRoot -ChildPath "timeout.json"
    $ownerExitEvidencePath = Join-Path -Path $testRoot -ChildPath "owner-exit.json"
    $refusalEvidencePath = Join-Path -Path $testRoot -ChildPath "refusal.json"
    $common = @{
        FilePath = $childPath
        TimeoutSeconds = 5
        MaxJobMemoryBytes = [UInt64]268435456
        PollMilliseconds = 50
        WorkingDirectory = $testRoot
    }

    & $wrapper @common `
        -ArgumentList @("/d", "/c", 'if not "%GGML_CUDA_DISABLE_GRAPHS%"=="1" exit /b 9 & ping 127.0.0.1 -n 2 > nul & exit /b 0') `
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
    Assert-Condition ($normal.pid_absent_after_cleanup -eq $true) "normal run left its child PID present"
    Assert-Condition ($normal.peak_job_memory_bytes -gt 0) "normal run did not record peak job memory"

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
    -ArgumentList @('/d', '/c', 'ping 127.0.0.1 -n 30 > nul') ``
    -TimeoutSeconds 30 ``
    -MaxJobMemoryBytes ([UInt64]268435456) ``
    -PollMilliseconds 50 ``
    -WorkingDirectory '$quotedTestRoot' ``
    -EvidencePath '$quotedOwnerEvidencePath'
"@
    $encodedOwnerInvocation = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($ownerInvocation)
    )
    $ownerArguments = @(
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        $encodedOwnerInvocation
    )
    foreach ($argument in $ownerArguments) {
        [void]$ownerStartInfo.ArgumentList.Add($argument)
    }
    $owner = [Diagnostics.Process]::new()
    $owner.StartInfo = $ownerStartInfo
    Assert-Condition ($owner.Start()) "failed to start the owner-exit wrapper process"

    $ownerChild = $null
    $ownerChildDeadline = [DateTimeOffset]::UtcNow.AddSeconds(5)
    while ($null -eq $ownerChild -and [DateTimeOffset]::UtcNow -lt $ownerChildDeadline) {
        Start-Sleep -Milliseconds 50
        $ownerChildren = @(Get-Process -Name "bounded-oracle-child" -ErrorAction SilentlyContinue)
        if ($ownerChildren.Count -gt 0) {
            $ownerChild = $ownerChildren[0]
        }
    }
    Assert-Condition ($null -ne $ownerChild) "owner-exit run did not start its bounded child"
    Assert-Condition (-not $owner.HasExited) "owner-exit wrapper stopped before the kill-on-close test"
    $ownerChildId = $ownerChild.Id
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

    $passed = $true
    "bounded-oracle-smoke: passed"
}
finally {
    if ($null -ne $owner) {
        try {
            if (-not $owner.HasExited) {
                $owner.Kill($true)
                [void]$owner.WaitForExit(5000)
            }
        }
        finally {
            $owner.Dispose()
        }
    }
    if ($null -ne $existing) {
        try {
            if (-not $existing.HasExited) {
                $existing.Kill($true)
                [void]$existing.WaitForExit(5000)
            }
        }
        finally {
            $existing.Dispose()
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
