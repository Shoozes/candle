Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'run-capacity.ps1') -AsLibrary
$script:capacityControls=0
function Assert-CapacityControl([bool]$Condition,[string]$Name){if(-not $Condition){throw "Capacity control failed: $Name"};$script:capacityControls++}
function Assert-CapacityReject([scriptblock]$Operation,[string]$Pattern){$message=$null;try{& $Operation|Out-Null}catch{$message=$_.Exception.Message};Assert-CapacityControl ($null -ne $message -and $message -match $Pattern) "reject $Pattern; observed $message"}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$scratch=Join-Path $repo ('artifacts\gpt-oss\capacity-controls-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch|Out-Null
$owned=@()
try{
    $tokenPath=Join-Path $scratch 'tokens.u32';[IO.File]::WriteAllBytes($tokenPath,(Get-CapacityTokenBytes));$owned+=$tokenPath
    $tokenFile=Get-Item -LiteralPath $tokenPath
    $source='a'*40;$tree='b'*40
    $value=[pscustomobject]@{schema='candle.gpt-oss.capacity-input.v1';source_commit=$source;backend='cuda';device_index=0;
        model=[pscustomobject]@{path='C:\llamacpp\models\gpt-oss-20b-mxfp4.gguf';size_bytes=13792638656;sha256='aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'};
        token_file=[pscustomobject]@{path=$tokenPath;size_bytes=$tokenFile.Length;sha256=(Get-FileHash -LiteralPath $tokenPath).Hash.ToLowerInvariant()};
        prefill_tokens=6144;prefill_chunk_tokens=128;fixed_decode_tokens=2048;sampled_outputs=0}
    Assert-CapacityInput $value $source 'cuda';Assert-CapacityControl $true 'input token admission without model reads'
    $plan=Get-CapacityPlan
    Assert-CapacityControl ($plan.prefill_tokens/$plan.prefill_chunk_tokens -eq 48 -and $plan.prefill_calls+$plan.fixed_decode_tokens -eq 2096 -and $plan.prefill_tokens+$plan.fixed_decode_tokens -eq 8192) 'exact frozen work'
    Assert-CapacityControl ($plan.sampled_outputs -eq 0 -and $plan.warmups -eq 0 -and $plan.retries -eq 0) 'fixed tokens do not imply sampled outputs'
    foreach($name in @('prefill_tokens','prefill_chunk_tokens','fixed_decode_tokens','sampled_outputs')){
        $old=$value.$name;$value.$name=$old+1;Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'frozen';$value.$name=$old
    }
    $value.device_index=-1;Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'device';$value.device_index=0
    $value.prefill_tokens='6144';Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'unsigned';$value.prefill_tokens=6144
    $value|Add-Member -NotePropertyName unknown -NotePropertyValue 1;Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'unknown';$value.PSObject.Properties.Remove('unknown')
    $value.model.sha256='c'*64;Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'model identity';$value.model.sha256='aab205256a9b6361e410c24de3086e30f907092ca6f9ba8cd4b22c8a2b025778'
    $bytes=[IO.File]::ReadAllBytes($tokenPath);$bytes[0]=$bytes[0] -bxor 1;[IO.File]::WriteAllBytes($tokenPath,$bytes)
    Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'SHA-256 mismatch';[IO.File]::WriteAllBytes($tokenPath,(Get-CapacityTokenBytes))
    $value.token_file.size_bytes=32767;Assert-CapacityReject {Assert-CapacityInput $value $source 'cuda'} 'token identity';$value.token_file.size_bytes=32768
    $controlPacket=[pscustomobject]@{schema='candle.gpt-oss.capacity-packet.v1';source_commit=$source;source_tree=$tree;draft=$false;lanes=@([pscustomobject]@{backend='cuda'});allowance_proposal=$plan}
    Assert-CapacityPacketHeader $controlPacket $source $tree;Assert-CapacityControl $true 'final packet frozen header'
    $controlPacket.draft=$true;Assert-CapacityReject {Assert-CapacityPacketHeader $controlPacket $source $tree} 'final';$controlPacket.draft=$false
    $controlPacket.allowance_proposal.public_forwards=2097;Assert-CapacityReject {Assert-CapacityPacketHeader $controlPacket $source $tree} 'allowance';$controlPacket.allowance_proposal.public_forwards=2096
    $controlPacket.lanes[0].backend='cpu';Assert-CapacityReject {Assert-CapacityPacketHeader $controlPacket $source $tree} 'CUDA lane';$controlPacket.lanes[0].backend='cuda'
    $controlPacket.schema='candle.gpt-oss.characterization-packet.v1';Assert-CapacityReject {Assert-CapacityPacketHeader $controlPacket $source $tree} 'final';$controlPacket.schema='candle.gpt-oss.capacity-packet.v1'
    $packetFile=Join-Path $scratch 'draft.json';$controlPacket.draft=$true;Write-SessionProfileJson $packetFile $controlPacket;$owned+=$packetFile
    $refusedRoot=Join-Path $scratch 'never-created'
    $runScript=Join-Path $PSScriptRoot 'run-capacity.ps1'
    $watchScript=Join-Path $PSScriptRoot 'watch-capacity.ps1'
    $entryOutput=& (Get-Command pwsh).Source -NoProfile -File $runScript 2>&1
    Assert-CapacityControl ($LASTEXITCODE -ne 0 -and ($entryOutput-join "`n") -match 'required') 'CLI executes required-argument refusal after library import'
    $entryOutput=& (Get-Command pwsh).Source -NoProfile -File $runScript -CapacityPacketPath $packetFile -CapacityRunRoot $refusedRoot 2>&1
    Assert-CapacityControl ($LASTEXITCODE -ne 0 -and ($entryOutput-join "`n") -match 'final clean source') 'draft CLI refusal'
    Assert-CapacityControl (-not(Test-Path -LiteralPath $refusedRoot)) 'refusal creates no production output'
    & (Get-Command pwsh).Source -NoProfile -File $runScript -AsLibrary
    Assert-CapacityControl ($LASTEXITCODE -eq 0) 'explicit library entry'
    $entryOutput=& (Get-Command pwsh).Source -NoProfile -File $watchScript 2>&1
    Assert-CapacityControl ($LASTEXITCODE -ne 0 -and ($entryOutput-join "`n") -match 'packet/observation') 'watcher executable refusal'
    & (Get-Command pwsh).Source -NoProfile -File $watchScript -AsLibrary
    Assert-CapacityControl ($LASTEXITCODE -eq 0) 'watcher library entry'
    Write-Output "capacity controls: passed $script:capacityControls; zero model work"
}finally{
    # Only the three files created here; no recursive cleanup or unknown owners.
    foreach($path in $owned){if(-not [IO.Path]::GetFullPath($path).StartsWith($scratch+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Control cleanup escaped owner root'};Remove-Item -LiteralPath $path -ErrorAction SilentlyContinue}
    if(@(Get-ChildItem -LiteralPath $scratch -Force).Count -eq 0){Remove-Item -LiteralPath $scratch}
}
