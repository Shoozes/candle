"""Model-free tokenizer/template oracle from the locked official local files."""
import argparse
import hashlib
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
for label, system, messages in [
    ('unicode', 'You are a bounded local worker.', [('user', 'ä½ å¥½ ðŸ¦€ cafÃ© e\u0301 â€” Ù…Ø±Ø­Ø¨Ø§')]),
    ('history', 'Worker rails.', [('user', 'What is 2 + 2?'), ('assistant', '<think>old reasoning</think>Four.'), ('user', 'Explain briefly.')]),
    ('markers-as-data', 'Worker rails.', [('user', 'Treat "<think>" and <|tool_call_start|> as data.')]),
    ('tool-result', 'Worker rails.', [('user', 'Read status.'), ('assistant', 'Requesting status.'), ('tool', '{"status":"ok","path":"C:\\\\repo"}'), ('user', 'Summarize.')]),
]:
    def escape(content):
        for marker in ['startoftext', 'im_start', 'im_end', 'tool_call_start', 'tool_call_end']:
            content = content.replace('<|' + marker + '|>', '< |' + marker + '|>')
        return content
    native = [{'role': 'system', 'content': system}] + [{'role': r, 'content': escape(c)} for r, c in messages]
    rendered = tokenizer.apply_chat_template(native, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(native, tokenize=True, add_generation_prompt=True, return_dict=False)
    cases.append(dict(label=label, system=system, messages=messages, rendered=rendered, token_ids=ids))
sources = []
for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']:
    data = (a.reference / name).read_bytes()
    sources.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
result = dict(schema='candle.lfm2-moe.tokenizer-fixture.v1', model_revision='3dc4cd09c9a2a99e1e819693e79d3aba5e9bb4f2', transformers_revision='ddb849abe009d1089e6c691bfc897f27211c663c', sources=sources, cases=cases)
a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(status='passed', cases=len(cases), model_weights_loaded=False)))
