param(
    [Parameter(Mandatory = $true)][string]$ModelFile,
    [ValidateRange(1, 32)][int]$BatchSize = 32,
    [ValidateRange(1, 32)][int]$UbatchSize = 1
)

$ErrorActionPreference = 'Stop'
$server = 'C:\llamacpp\tools-b11026\llama-server.exe'
$serverSha = 'bfb73ca63cbac5835681d8a489d22617995c85d4ca93ee36e2fd36508c96646d'
$modelSha = 'fd6dcc1ea357b483ae2517b482bf328181e5b371b65661c734aa133407ae2572'
$maxSeconds = 480
$maxPrivateBytes = 12884901888
$minAvailableBytes = 17179869184
$minCFreeBytes = 10737418240
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$outputDir = Join-Path $root 'artifacts\qwen35'
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
if ($UbatchSize -gt $BatchSize) { throw 'Ubatch exceeds batch size' }
$label = "batch-$BatchSize-ubatch-$UbatchSize"
$responseFile = Join-Path $outputDir "llama-$label.json"
$serverStdout = Join-Path $outputDir "llama-$label.stdout.log"
$serverStderr = Join-Path $outputDir "llama-$label.stderr.log"
$reportFile = Join-Path $outputDir "llama-$label-job.json"
$slotDir = Join-Path $outputDir "llama-$label-slots"
New-Item -ItemType Directory -Path $slotDir -Force | Out-Null

if ((Get-Item -LiteralPath $ModelFile).Length -ne 4482403200) { throw 'GGUF size mismatch' }
if ((Get-FileHash -LiteralPath $ModelFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $modelSha) { throw 'GGUF hash mismatch' }
if ((Get-FileHash -LiteralPath $server -Algorithm SHA256).Hash.ToLowerInvariant() -ne $serverSha) { throw 'Pinned llama-server hash mismatch' }
if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { throw 'Available memory below 16 GiB' }
if ((Get-PSDrive C).Free -lt $minCFreeBytes) { throw 'C: free space below 10 GiB' }

$listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = ([Net.IPEndPoint]$listener.LocalEndpoint).Port
$listener.Stop()
$base = "http://127.0.0.1:$port"
$arguments = @(
    '--model', $ModelFile, '--host', '127.0.0.1', '--port', "$port",
    '--ctx-size', '256', '--batch-size', "$BatchSize", '--ubatch-size', "$UbatchSize",
    '--parallel', '1', '--threads', '4', '--threads-batch', '4',
    '--n-gpu-layers', '0', '--device', 'none', '--fit', 'off',
    '--flash-attn', 'off', '--no-kv-offload', '--no-op-offload',
    '--cache-type-k', 'f32', '--cache-type-v', 'f32', '--no-warmup', '--no-ui',
    '--slot-save-path', $slotDir
)
$env:GGML_CUDA_DISABLE_GRAPHS = '1'
$started = Get-Date
[long]$peakPrivate = 0
$reason = 'completed'
$process = Start-Process -FilePath $server -ArgumentList $arguments -WorkingDirectory $root -PassThru -WindowStyle Hidden -RedirectStandardOutput $serverStdout -RedirectStandardError $serverStderr
try {
    $startupDeadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $startupDeadline) {
        $process.Refresh()
        if ($process.HasExited) { throw "llama-server exited during startup: $($process.ExitCode)" }
        try {
            $health = Invoke-RestMethod -Uri "$base/health" -TimeoutSec 2
            if ($health.status -eq 'ok') { break }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    if ($health.status -ne 'ok') { throw 'llama-server health deadline expired' }
    $fixture = Get-Content -LiteralPath (Join-Path $root 'tests\fixtures\qwen35_codename_b\reference.json') -Raw | ConvertFrom-Json
    $body = @{
        prompt = @($fixture.prompt_token_ids)
        id_slot = 0
        n_predict = 3
        temperature = -1
        seed = 0
        n_probs = 10
        post_sampling_probs = $false
        return_tokens = $true
        cache_prompt = $false
        stream = $false
        repeat_penalty = 1
        top_k = 0
        top_p = 1
        min_p = 0
    } | ConvertTo-Json -Depth 5 -Compress
    $client = [Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(400)
    $content = [Net.Http.StringContent]::new($body, [Text.Encoding]::UTF8, 'application/json')
    $request = $client.PostAsync("$base/completion", $content)
    while (-not $request.IsCompleted) {
        $process.Refresh()
        $peakPrivate = [Math]::Max([long]$peakPrivate, [long]$process.PrivateMemorySize64)
        if ($process.HasExited) { $reason = 'server exited before response'; break }
        if (((Get-Date) - $started).TotalSeconds -gt $maxSeconds) { $reason = 'wall-time ceiling'; break }
        if ($process.PrivateMemorySize64 -gt $maxPrivateBytes) { $reason = 'private-memory ceiling'; break }
        if ((Get-Counter '\Memory\Available Bytes').CounterSamples.CookedValue -lt $minAvailableBytes) { $reason = 'available-memory floor'; break }
        if ((Get-PSDrive C).Free -lt $minCFreeBytes) { $reason = 'C: free-space floor'; break }
        Start-Sleep -Milliseconds 500
    }
    if ($reason -ne 'completed') { throw $reason }
    $response = $request.GetAwaiter().GetResult()
    $response.EnsureSuccessStatusCode() | Out-Null
    $json = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if ($json.Length -gt 4 * 1024 * 1024) { throw 'Response exceeds 4 MiB' }
    $json | Set-Content -LiteralPath $responseFile
    $trace = $json | ConvertFrom-Json
    [pscustomobject]@{Tokens=($trace.tokens -join ',');Content=$trace.content;Response=$responseFile}
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
        $process.WaitForExit()
    }
    $report = [ordered]@{
        reason = $reason
        server_exit_code = $process.ExitCode
        elapsed_seconds = ((Get-Date) - $started).TotalSeconds
        peak_private_bytes_sampled = $peakPrivate
        model_sha256 = $modelSha
        server_sha256 = $serverSha
        batch_size = $BatchSize
        ubatch_size = $UbatchSize
        response = $responseFile
        c_free_bytes = (Get-PSDrive C).Free
    }
    $report | ConvertTo-Json | Set-Content -LiteralPath $reportFile
    $report | Format-List
}
