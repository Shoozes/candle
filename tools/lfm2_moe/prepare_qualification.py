"""Freeze authored prompts and teacher-forced positions before any model run."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.reference / 'transformers-source'))
import transformers
from transformers import AutoTokenizer
assert transformers.__version__ == '5.12.1'
tokenizer = AutoTokenizer.from_pretrained(a.reference, local_files_only=True)
cases = []
for name, user, continuation in [
    ('arithmetic', 'What is 17 + 25? Explain briefly, then give the answer.', 'I should add seventeen and twenty five carefully.'),
    ('unicode', 'Translate 猫 into English. Reply briefly.', 'The user asks for a short translation of the word.'),
]:
    rendered = tokenizer.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':user}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
    assert len(continuation_ids) >= 8 and len(ids) <= 256
    cases.append(dict(name=name, rendered=rendered, prompt_ids=ids, authored_continuation=continuation, forced_ids=continuation_ids[:8], probe_positions=list(range(8))))
probes = sorted(set(tokenizer.encode('42 The cat 猫\n', add_special_tokens=False) + [tokenizer.convert_tokens_to_ids('<|im_end|>')]))
result = dict(schema='candle.lfm2-moe.qualification-inputs.v1', kind='authored-before-runtime', model_revision='3dc4cd09c9a2a99e1e819693e79d3aba5e9bb4f2', cases=cases, logit_probe_ids=probes)
a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(status='prepared-not-executed', cases=len(cases), forced_positions=8)))
