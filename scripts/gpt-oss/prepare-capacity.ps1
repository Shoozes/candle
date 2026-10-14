[CmdletBinding()]
param([Parameter(Mandatory)][string]$CapacityDestination,[Parameter(Mandatory)][string]$ShortReportPath,
    [Parameter(Mandatory)][string[]]$CpuRuntimeDlls,[Parameter(Mandatory)][string[]]$CudaRuntimeDlls,[switch]$Draft)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-capacity.ps1') -AsLibrary
$capacityPrepRepo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path;Set-Location -LiteralPath $capacityPrepRepo
$capacityPrepRoot=[IO.Path]::GetFullPath($CapacityDestination)
if(-not $capacityPrepRoot.StartsWith((Join-Path $capacityPrepRepo 'artifacts')+'\',[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $capacityPrepRoot)){throw 'Capacity preparation needs a fresh owned artifact root'}
if(-not $Draft -and @(git status --porcelain).Count){throw 'Final capacity preparation requires clean source'}
function Get-CapacityIdentity([string]$Path){$file=Get-Item -LiteralPath $Path;[pscustomobject]@{path=$file.FullName;size_bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant()}}
$capacityPredecessor=Get-CapacityIdentity $ShortReportPath;Assert-CapacityPredecessor $capacityPredecessor
$capacityModel=Get-Item -LiteralPath 'C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf'
if($capacityModel.Length -ne 13792638656){throw 'Capacity retained model metadata differs'}
$capacityModelIdentity=[ordered]@{path=$capacityModel.FullName;size_bytes=$capacityModel.Length;sha256='aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'}
$capacityPrepSource=(git rev-parse HEAD).Trim();$capacityPrepTree=(git show -s --format=%T HEAD).Trim()
$capacityPrepFiles=@((Get-CapacitySourcePaths $capacityPrepRepo)|ForEach-Object{Get-CapacityIdentity $_})
$capacityPrepHelpers=@('run-capacity.ps1','prepare-capacity.ps1','watch-capacity.ps1','run-characterization.ps1','run-session-profile.ps1','observer.ps1','observer.cs','observer-worker.ps1')|ForEach-Object{Get-CapacityIdentity (Join-Path $PSScriptRoot $_)}
$capacityProbe=Get-CapacityIdentity (Get-Command nvidia-smi.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $capacityPrepRoot|Out-Null
$capacityPrepComplete=$false;$capacityPrepared=@()
$oldOffline=$env:CARGO_NET_OFFLINE;$oldJobs=$env:CARGO_BUILD_JOBS;$oldPython=$env:PYO3_NO_PYTHON;$oldIncremental=$env:CARGO_INCREMENTAL
try{
    $env:CARGO_NET_OFFLINE='true';$env:CARGO_BUILD_JOBS='2';$env:PYO3_NO_PYTHON='1'
    if([string]::IsNullOrWhiteSpace($oldIncremental)){$env:CARGO_INCREMENTAL='0'}
    $tokenPath=Join-Path $capacityPrepRoot 'tokens.u32';[IO.File]::WriteAllBytes($tokenPath,(Get-CapacityTokenBytes))
    $tokenIdentity=Get-CapacityIdentity $tokenPath
    foreach($backend in @('cpu','cuda')){
        $dir=Join-Path $capacityPrepRoot $backend;New-Item -ItemType Directory -Path $dir|Out-Null
        $features=if($backend -eq 'cuda'){@('cuda')}else{@()}
        $arguments=@('build','--locked','--offline','--release','-j','2','-p','candle-examples','--example','gpt-oss-capacity-stress')
        if($backend -eq 'cuda'){$arguments+=@('--features','cuda')}
        $start=[DateTimeOffset]::UtcNow.ToString('o');$jobPath=Join-Path $dir 'build-job.json'
        & (Join-Path $capacityPrepRepo 'scripts\lfm2-vl\run-bounded-oracle.ps1') -FilePath (Get-Command cargo).Source -ArgumentList $arguments -WorkingDirectory $capacityPrepRepo -ConcurrencyScope Executable -TimeoutSeconds 900 -MaxJobMemoryBytes 17179869184 -EvidencePath $jobPath -LogPath (Join-Path $dir 'build.log')
        $target=if([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)){Join-Path $capacityPrepRepo 'target'}else{[IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)}
        $binary=Join-Path $dir 'gpt-oss-capacity-stress.exe';Copy-Item -LiteralPath (Join-Path $target 'release\examples\gpt-oss-capacity-stress.exe') -Destination $binary
        $binaryIdentity=Get-CapacityIdentity $binary;$buildPath=Join-Path $dir 'build.json'
        Write-SessionProfileJson $buildPath ([ordered]@{schema='candle.gpt-oss.characterization-build.v1';source_commit=$capacityPrepSource;source_tree=$capacityPrepTree;draft=[bool]$Draft;
            command=@('cargo')+$arguments;features=$features;profile='release';precision='f32';format='gguf-mxfp4-f32';rustc=((& rustc -Vv)-join "`n");cargo=(& cargo -V);
            cuda_compute_cap=$env:CUDA_COMPUTE_CAP;cuda_path=$env:CUDA_PATH;rustflags=$env:RUSTFLAGS;encoded_rustflags=$env:CARGO_ENCODED_RUSTFLAGS;
            target_rustflags=$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS;nvcc_ccbin=$env:NVCC_CCBIN;nvcc_flags=$env:CANDLE_NVCC_FLAGS;target_directory=$env:CARGO_TARGET_DIR;
            started_utc=$start;ended_utc=[DateTimeOffset]::UtcNow.ToString('o');exit_code=0;source_files=$capacityPrepFiles;binary=$binaryIdentity;bounded_job=(Get-CapacityIdentity $jobPath)})
        $inputPath=Join-Path $dir 'input.json'
        Write-SessionProfileJson $inputPath ([ordered]@{schema='candle.gpt-oss.capacity-input.v1';source_commit=$capacityPrepSource;backend=$backend;device_index=0;model=$capacityModelIdentity;token_file=$tokenIdentity;
            prefill_tokens=6144;prefill_chunk_tokens=128;fixed_decode_tokens=2048;sampled_outputs=0})
        $paths=if($backend -eq 'cpu'){$CpuRuntimeDlls}else{$CudaRuntimeDlls}
        $modules=@($paths|ForEach-Object{$id=Get-CapacityIdentity $_;$id|Add-Member -NotePropertyName name -NotePropertyValue (Split-Path -Leaf $id.path);$id})
        $selected=[pscustomobject]@{backend=$backend;build_profile='release';precision='f32';format='gguf-mxfp4-f32';binary=$binaryIdentity;build_record=(Get-CapacityIdentity $buildPath);
            input=(Get-CapacityIdentity $inputPath);required_modules=$modules}
        Assert-CharacterizationRuntimeClosure $selected
        $startupJob=Join-Path $dir 'startup-job.json';$startupLog=Join-Path $dir 'startup.log'
        Invoke-CharacterizationRuntimeEnvironment $selected {param($lane)
            & (Join-Path $capacityPrepRepo 'scripts\lfm2-vl\run-bounded-oracle.ps1') -FilePath $lane.binary.path -ArgumentList @('--input',$inputPath,'--check-input') -WorkingDirectory $capacityPrepRepo -ConcurrencyScope Executable -TimeoutSeconds 15 -MaxJobMemoryBytes 2147483648 -EvidencePath $startupJob -LogPath $startupLog
        }
        $checked=Get-Content -LiteralPath $startupLog -Raw|ConvertFrom-Json
        if($checked.status -ne 'verified-capacity-input' -or $checked.token_steps -ne 8192 -or $checked.public_forwards -ne 2096 -or $checked.sampled_outputs -ne 0){throw 'Capacity startup/input control failed'}
        $selected|Add-Member -NotePropertyName model_free_startup_job -NotePropertyValue (Get-CapacityIdentity $startupJob)
        $capacityPrepared+=$selected
    }
    foreach($file in $capacityPrepFiles+$capacityPrepHelpers){Assert-SessionProfileIdentity $file|Out-Null}
    if((git rev-parse HEAD).Trim() -ne $capacityPrepSource -or (-not $Draft -and @(git status --porcelain).Count)){throw 'Capacity source changed during preparation'}
    $output=Join-Path $capacityPrepRepo ('artifacts\gpt-oss\capacity8k-'+$capacityPrepSource.Substring(0,8)+'-'+[guid]::NewGuid().ToString('N'))
    $packetPath=Join-Path $capacityPrepRoot 'packet.json'
    Write-SessionProfileJson $packetPath ([ordered]@{schema='candle.gpt-oss.capacity-packet.v1';source_commit=$capacityPrepSource;source_tree=$capacityPrepTree;draft=[bool]$Draft;
        proposed_output_root=$output;bounded_helper=(Get-CapacityIdentity (Join-Path $capacityPrepRepo 'scripts\lfm2-vl\run-bounded-oracle.ps1'));watcher=(Get-CapacityIdentity (Join-Path $PSScriptRoot 'watch-capacity.ps1'));
        helper_inputs=$capacityPrepHelpers;gpu_probe=$capacityProbe;fixture=(Get-CapacityIdentity (Join-Path $capacityPrepRepo 'tests\fixtures\gpt_oss_task3_short_parity\reference.json'));
        short_predecessor=$capacityPredecessor;host_verifier=$capacityPrepared[0];lanes=@($capacityPrepared[1]);allowance_proposal=(Get-CapacityPlan);
        preparation=[ordered]@{cpu_then_cuda_release_builds=2;build_job_bytes=17179869184;build_deadline_seconds=900;model_free_startup_probes=2;startup_job_bytes=2147483648;startup_deadline_seconds=15};
        execution='Model-free preparation only; execution needs final clean-source packet review and separate finite allowance';scope='One8k fixed-token CUDA stress after CPU-first predecessor; no sampled generation,long-context oracle,16k/32k or route signoff'})
    $capacityPrepComplete=$true
    Get-CapacityIdentity $packetPath|ConvertTo-Json
}finally{
    $env:CARGO_NET_OFFLINE=$oldOffline;$env:CARGO_BUILD_JOBS=$oldJobs;$env:PYO3_NO_PYTHON=$oldPython;$env:CARGO_INCREMENTAL=$oldIncremental
    Write-SessionProfileJson (Join-Path $capacityPrepRoot 'preparation-terminal.json') ([ordered]@{source_commit=$capacityPrepSource;complete=$capacityPrepComplete;draft=[bool]$Draft;production_loads=0;forwards=0;sampled_tokens=0;
        retention='Bound CPU verifier/CUDA executable,one32KiB token stream and source/input/build/helper packet records; existing build target protected'})
}
