[CmdletBinding()]
param([Parameter(Mandatory)][string]$Destination,[Parameter(Mandatory)][string]$OraclePath,
    [Parameter(Mandatory)][string]$OracleRecoveryPath,
    [Parameter(Mandatory)][string[]]$CpuRuntimeDlls,[Parameter(Mandatory)][string[]]$CudaRuntimeDlls,
    [string]$ModelPath='C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf',[switch]$Draft)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-characterization.ps1') -AsLibrary
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path;Set-Location -LiteralPath $repo
$prepareOutput=[IO.Path]::GetFullPath($Destination)
if(-not $prepareOutput.StartsWith((Join-Path $repo 'artifacts')+'\',[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $prepareOutput)){throw 'Preparation needs a fresh owned artifacts directory'}
if(-not $Draft -and @(git status --porcelain).Count -ne 0){throw 'Final packet preparation requires clean source'}
function Get-CharacterizationFileIdentity([string]$Path) {
    $file=Get-Item -LiteralPath $Path
    [pscustomobject]@{path=$file.FullName;size_bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName).Hash.ToLowerInvariant()}
}
$fixture=Get-Content (Join-Path $repo 'tests\fixtures\gpt_oss_task3_short_parity\reference.json') -Raw|ConvertFrom-Json
$oracle=Get-CharacterizationFileIdentity $OraclePath
if($oracle.size_bytes -ne $fixture.reference_file_bytes -or $oracle.sha256 -ne $fixture.reference_file_sha256){throw 'Oracle differs from pinned reference; review regenerated authority separately'}
$recovery=Get-CharacterizationFileIdentity $OracleRecoveryPath
Assert-CharacterizationOracleRecovery $recovery $oracle
$modelFile=Get-Item -LiteralPath $ModelPath
if($modelFile.Length -ne 13792638656){throw 'Retained model metadata differs from original model identity'}
# Do not read 13.8GB of weights during model-free preparation. The session
# rehashes and owns this retained identity inside each admitted native Job.
$model=[ordered]@{path=$modelFile.FullName;size_bytes=$modelFile.Length;sha256='aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'}
$source=(git rev-parse HEAD).Trim();$tree=(git show -s --format=%T HEAD).Trim()
$closure=Get-CharacterizationSourcePaths $repo
$sourceFiles=@($closure|ForEach-Object{Get-CharacterizationFileIdentity $_})
$helpers=@('scripts/gpt-oss/run-characterization.ps1','scripts/gpt-oss/run-session-profile.ps1','scripts/gpt-oss/observer.ps1','scripts/gpt-oss/observer-worker.ps1','scripts/gpt-oss/observer.cs','scripts/gpt-oss/watch-runtime.ps1','scripts/gpt-oss/prepare-characterization.ps1')
$helperInputs=@($helpers|ForEach-Object{Get-CharacterizationFileIdentity (Join-Path $repo $_)})
$gpuProbe=Get-CharacterizationFileIdentity (Get-Command nvidia-smi.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $prepareOutput|Out-Null
$lanes=@();$oldOffline=$env:CARGO_NET_OFFLINE;$oldJobs=$env:CARGO_BUILD_JOBS;$oldPython=$env:PYO3_NO_PYTHON
$buildSuccess=$false
try {
    $env:CARGO_NET_OFFLINE='true';$env:CARGO_BUILD_JOBS='2';$env:PYO3_NO_PYTHON='1'
    foreach($backend in @('cpu','cuda')) {
        $dir=Join-Path $prepareOutput $backend;New-Item -ItemType Directory -Path $dir|Out-Null
        $features=if($backend -eq 'cuda'){@('cuda')}else{@()}
        $arguments=@('build','--locked','--offline','--release','-j','2','-p','candle-examples','--example','gpt-oss-characterize')
        if($backend -eq 'cuda'){$arguments+=@('--features','cuda')}
        $start=[DateTimeOffset]::UtcNow.ToString('o')
        $buildJob=Join-Path $dir 'build-job.json'
        & (Join-Path $repo 'scripts\lfm2-vl\run-bounded-oracle.ps1') -FilePath (Get-Command cargo).Source -ArgumentList $arguments -WorkingDirectory $repo -ConcurrencyScope Executable -TimeoutSeconds 900 -MaxJobMemoryBytes 17179869184 -EvidencePath $buildJob -LogPath (Join-Path $dir 'build.log')
        $target=if([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)){Join-Path $repo 'target'}else{[IO.Path]::GetFullPath($env:CARGO_TARGET_DIR)}
        $binary=Join-Path $dir 'gpt-oss-characterize.exe';Copy-Item -LiteralPath (Join-Path $target 'release\examples\gpt-oss-characterize.exe') -Destination $binary
        $binaryIdentity=Get-CharacterizationFileIdentity $binary
        $recordPath=Join-Path $dir 'build.json'
        Write-SessionProfileJson $recordPath ([ordered]@{schema='candle.gpt-oss.characterization-build.v1';source_commit=$source;source_tree=$tree;
            draft=[bool]$Draft;command=@('cargo')+$arguments;features=$features;profile='release';precision='f32';format='gguf-mxfp4-f32';
            rustc=((& rustc -Vv)-join "`n");cargo=(& cargo -V);cuda_compute_cap=$env:CUDA_COMPUTE_CAP;cuda_path=$env:CUDA_PATH;
            rustflags=$env:RUSTFLAGS;encoded_rustflags=$env:CARGO_ENCODED_RUSTFLAGS;target_rustflags=$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS;
            nvcc_ccbin=$env:NVCC_CCBIN;nvcc_flags=$env:CANDLE_NVCC_FLAGS;target_directory=$env:CARGO_TARGET_DIR;
            started_utc=$start;ended_utc=[DateTimeOffset]::UtcNow.ToString('o');exit_code=0;source_files=$sourceFiles;binary=$binaryIdentity;
            bounded_job=(Get-CharacterizationFileIdentity $buildJob)})
        $inputPath=Join-Path $dir 'input.json'
        Write-SessionProfileJson $inputPath ([ordered]@{schema='candle.gpt-oss.characterization-input.v1';source_commit=$source;backend=$backend;device_index=0;
            model=$model;reference=$oracle;prompt_token_ids=$fixture.reference_prefix_token_ids;generated_tokens=16;segments=@('cold','numerical','replay')})
        $runtimePaths=if($backend -eq 'cpu'){$CpuRuntimeDlls}else{$CudaRuntimeDlls}
        $runtime=@($runtimePaths|ForEach-Object{$id=Get-CharacterizationFileIdentity $_;$id|Add-Member -NotePropertyName name -NotePropertyValue (Split-Path -Leaf $id.path);$id})
        $preparedLane=[pscustomobject]@{backend=$backend;build_profile='release';precision='f32';format='gguf-mxfp4-f32';
            input=(Get-CharacterizationFileIdentity $inputPath);binary=$binaryIdentity;build_record=(Get-CharacterizationFileIdentity $recordPath);required_modules=$runtime}
        Assert-CharacterizationRuntimeClosure $preparedLane
        $startupJob=Join-Path $dir 'startup-job.json'
        Invoke-CharacterizationRuntimeEnvironment $preparedLane {param($selected)
            & (Join-Path $repo 'scripts\lfm2-vl\run-bounded-oracle.ps1') -FilePath $selected.binary.path -ArgumentList @('--inspect-reference',$oracle.path) -WorkingDirectory $repo -ConcurrencyScope Executable -TimeoutSeconds 15 -MaxJobMemoryBytes 2147483648 -EvidencePath $startupJob -LogPath (Join-Path $dir 'startup.log')
        }
        $codec=Get-Content -LiteralPath (Join-Path $dir 'startup.log') -Raw|ConvertFrom-Json
        if($codec.status -ne 'verified-reference-structure' -or $codec.rows -ne 3 -or $codec.vocabulary_size -ne 201088){throw 'Release startup/codec qualification failed'}
        $preparedLane|Add-Member -NotePropertyName model_free_startup_job -NotePropertyValue (Get-CharacterizationFileIdentity $startupJob)
        $lanes+=$preparedLane
    }
    foreach($file in $sourceFiles+$helperInputs){Assert-SessionProfileIdentity $file|Out-Null}
    if((git rev-parse HEAD).Trim() -ne $source -or (-not $Draft -and @(git status --porcelain).Count -ne 0)){throw 'Source changed during packet preparation'}
    $proposedOutput=Join-Path $repo ('artifacts\gpt-oss\characterization-'+$source.Substring(0,8)+'-'+[guid]::NewGuid().ToString('N'))
    $packetPath=Join-Path $prepareOutput 'packet.json'
    Write-SessionProfileJson $packetPath ([ordered]@{schema='candle.gpt-oss.characterization-packet.v1';source_commit=$source;source_tree=$tree;
        draft=[bool]$Draft;proposed_output_root=$proposedOutput;bounded_helper=(Get-CharacterizationFileIdentity (Join-Path $repo 'scripts\lfm2-vl\run-bounded-oracle.ps1'));
        watcher=(Get-CharacterizationFileIdentity (Join-Path $repo 'scripts\gpt-oss\watch-runtime.ps1'));helper_inputs=$helperInputs;gpu_probe=$gpuProbe;
        fixture=(Get-CharacterizationFileIdentity (Join-Path $repo 'tests\fixtures\gpt_oss_task3_short_parity\reference.json'));oracle=$oracle;oracle_recovery=$recovery;lanes=$lanes;
        allowance_proposal=[ordered]@{loads=2;segments=6;public_forwards=44;processed_token_steps=74;sampled_output_tokens=32;warmups=0;retries=0;
            per_lane=(Get-CharacterizationPlan);preparation_startup_probes=2;preparation_startup_job_bytes=2147483648;preparation_startup_deadline_seconds=15;
            observer_qualification='CPU executable performs host row verification on both lanes; additional bounded model-free process/module probes; no hidden model forwards';
            scope='Diagnostic only. Historical timing, independent oracle and reset/replay remain distinct; no consumer deadlines/capacity or causal speedup claim'};
        execution='Requires independent packet review and separate finite owner allowance; Draft is never executable'})
    $buildSuccess=$true
    Get-CharacterizationFileIdentity $packetPath|ConvertTo-Json
} finally {
    $env:CARGO_NET_OFFLINE=$oldOffline;$env:CARGO_BUILD_JOBS=$oldJobs;$env:PYO3_NO_PYTHON=$oldPython
    Write-SessionProfileJson (Join-Path $prepareOutput 'preparation-terminal.json') ([ordered]@{source_commit=$source;complete=$buildSuccess;draft=[bool]$Draft;
        production_loads=0;forwards=0;tokens=0;retention='Bound executable copies and compact build/input/packet evidence retained; shared build target protected'})
}
