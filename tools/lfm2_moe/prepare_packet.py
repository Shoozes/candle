"""Prepare reviewable, source/binary-bound inputs. This program never runs a model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

p = argparse.ArgumentParser()
p.add_argument('--candle-binary', type=Path, required=True)
p.add_argument('--reference-binary', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
inputs = root / 'tests/fixtures/lfm2_moe_tiny/qualification_inputs.json'
data = json.loads(inputs.read_text(encoding='utf-8'))
model = Path('C:/llamacpp/models/LFM2.5-24B-A2B-Q4_K_M.gguf')
if model.stat().st_size != 14526258464: raise ValueError('model size changed')
if a.output.exists(): raise ValueError('packet output must be fresh')
a.output.mkdir(parents=True)
files = [inputs]
case_paths = []
for case in data['cases']:
    prefix = a.output / case['name']
    prompt = prefix.with_suffix('.txt'); prompt.write_bytes(case['rendered'].encode('utf-8'))
    forced = prefix.with_suffix('.forced.json'); forced.write_text(json.dumps(case['forced_ids'])+'\n', encoding='utf-8', newline='\n')
    files.extend([prompt, forced])
    case_paths.append((case['name'], prompt, forced))
probes = a.output / 'probe-ids.json'; probes.write_text(json.dumps(data['logit_probe_ids'])+'\n', encoding='utf-8', newline='\n'); files.append(probes)
manifest = root / 'docs/lfm2-moe/MOD_MANIFEST.md'
import re
paths = re.findall(r'^- `([^`]+)`$', manifest.read_text(encoding='utf-8'), re.M)
source = [dict(path=s, sha256=digest(root/s)) for s in paths]
source_path = a.output / 'source-inventory.json'; source_path.write_text(json.dumps(source,indent=2)+'\n', encoding='utf-8', newline='\n')
packet = dict(schema='candle.lfm2-moe.inert-parity-packet.v1', status='inert-review-required', execution_allowed=False,
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(), source_inventory_sha256=digest(source_path),
    model=dict(path=str(model),bytes=model.stat().st_size,sha256='d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d',hash_authority='retained full-file verification; fresh revalidation required at admission'),
    binaries=[dict(role=role,path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path)) for role,path in [('candle',a.candle_binary),('llamacpp_cpu_reference',a.reference_binary)]],
    inputs=[dict(path=str(path.resolve()),sha256=digest(path)) for path in files],
    limits=dict(attempts=1,retries=0,children=6,wall_seconds=900,child_seconds=180,job_bytes=32*1024**3,context_tokens=12288,prefill_chunk=128,forced_positions_per_child=8,cpu_threads=2,minimum_host_available_bytes=40*1024**3,minimum_commit_headroom_bytes=40*1024**3,minimum_cuda_free_bytes=18*1024**3),
    environment=dict(RAYON_NUM_THREADS='2',GGML_CUDA_DISABLE_GRAPHS='1',CANDLE_DEQUANTIZE_ALL='0',CANDLE_DEQUANTIZE_ALL_F16='0'),
    order=['reference_cpu','candle_cpu','candle_cuda_0'],stop_on_any_failure=True,production_runs_performed=0)
children = []
for lane in packet['order']:
    for name, prompt, forced in case_paths:
        if lane == 'reference_cpu':
            binary = a.reference_binary.resolve(); argv = [str(model),str(prompt.resolve()),str(forced.resolve()),str(probes.resolve())]
        else:
            binary = a.candle_binary.resolve(); argv = ['--gguf',str(model),'--prompt-file',str(prompt.resolve()),'--teacher-forced-ids',str(forced.resolve()),'--logit-probe-ids',str(probes.resolve()),'--context','12288']
            if lane == 'candle_cuda_0': argv += ['--cuda','0']
        children.append(dict(id=lane+'-'+name, executable=str(binary), arguments=argv, working_directory=str(root), timeout_seconds=180, max_job_memory_bytes=32*1024**3, combined_log=str(a.output/(lane+'-'+name+'.log'))))
packet['children'] = children
packet['comparisons'] = [
    dict(id=lane+'-'+name, executable=sys.executable,
         arguments=['-B', str(root/'tools/lfm2_moe/compare_probes.py'),
                    '--reference', str(a.output/('reference_cpu-'+name+'.log')),
                    '--candidate', str(a.output/(lane+'-'+name+'.log')),
                    '--inputs', str(inputs), '--case', name],
         expected_positions=len(data['cases'][index]['forced_ids']))
    for lane in ['candle_cpu', 'candle_cuda_0']
    for index, (name, _, _) in enumerate(case_paths)
]
(a.output/'packet.json').write_text(json.dumps(packet,indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(status=packet['status'],packet=str(a.output/'packet.json'),sha256=digest(a.output/'packet.json'),model_runs=0)))
