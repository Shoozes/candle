"""Compare the local GGUF tokenizer to the pinned oracle without loading weights."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--gguf', type=Path, required=True)
p.add_argument('--fixture', type=Path, required=True)
a = p.parse_args()
fixture = json.loads(a.fixture.read_text(encoding='utf-8'))
with tempfile.TemporaryDirectory(prefix='candle-moe-tokenizer-') as scratch:
    for case in fixture['cases']:
        prompt = Path(scratch) / 'prompt.txt'
        prompt.write_bytes(case['rendered'].encode('utf-8'))
        result = subprocess.run([str(a.binary.resolve()), '--gguf', str(a.gguf), '--tokenize-only', '--prompt-file', str(prompt)], capture_output=True, text=True, encoding='utf-8', timeout=60, check=True)
        actual = json.loads(result.stdout)
        assert actual['weight_payload_loaded'] is False
        assert actual['rendered'] == case['rendered'], case['label']
        assert actual['token_ids'] == case['token_ids'], case['label']
print(json.dumps(dict(status='passed', cases=len(fixture['cases']), weight_payload_loaded=False)))
