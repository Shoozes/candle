"""Export small goldens from the hash-verified official prompt.py, without a model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer, __version__
    if __version__ != "0.22.2":
        raise ValueError("reference tokenizers must be 0.22.2")
    raw = args.reference.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31":
        raise ValueError("reference prompt.py differs from the pinned source")
    spec = importlib.util.spec_from_file_location("d1_official_prompt", args.reference)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Adapter:
        def __init__(self):
            self.tokenizer = Tokenizer.from_file(str(args.tokenizer))
        def encode(self, text, add_special_tokens=False):
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    tokenizer = Adapter()
    fixtures = [
        ("unicode-order", {"z": "café\n雪", "a": [True, None, 2]}, {"type": "noul", "instructions": "Ready?"}, 0, False),
        ("empty-criteria", "hello", {"type": "noul", "instructions": "Ready?", "criteria": {}}, 0, False),
        ("null-criteria-value", None, {"type": "noul", "instructions": "Ready?", "criteria": {"true": None}}, 2, False),
        ("choice-order-tie", "hello", {"type": "choice", "instructions": "Pick", "criteria": {"second_option": "", "first": None}}, 0, True),
        ("native-letter-labels", "hello", {"type": "choice", "instructions": "Pick", "criteria": {" C ": None, "A": "alpha"}}, 1, False),
        ("combining-marks-are-not-letter-labels", "hello", {"type": "choice", "instructions": "Pick", "criteria": {"\u0345": None, "\u3099": None}}, 0, False),
        ("python-label-whitespace", "hello", {"type": "choice", "instructions": "Pick", "criteria": {"\u001cC\u001f": None, "A": None}}, 0, False),
        ("expected-score", {"x": 1}, {"type": "score", "instructions": "Rate", "criteria": ["low", "middle", "high"]}, 0, False),
        ("float-json", {"small": 1e-5, "big": 1e20, "negative_zero": -0.0, "integer_float": 1.0}, {"type": "noul", "instructions": "Ready?"}, 0, False),
    ]
    records = []
    for name, state, value, images, tie in fixtures:
        question = module.as_question(value)
        logits = [0.0 if tie else i / 32.0 for i in range(tokenizer.tokenizer.get_vocab_size())]
        text = module.render(tokenizer, state, question, bos="<|startoftext|>", images="<image>" * images)
        records.append({"name": name, "state": state, "question": value, "images": images, "text": text,
                        "input_ids": tokenizer.encode(text), "groups": module.readout_ids(tokenizer, question),
                        "logits": logits, "probabilities": module.readout(tokenizer, question, logits)})
    output = {"schema": "candle.d1_policy_goldens.v1", "reference_revision": "051bcc464b01b9f92942b364d9586b0ef5912432",
              "prompt_sha256": hashlib.sha256(raw).hexdigest(), "tokenizers": __version__,
              "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(), "cases": records}
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "cases": len(records), "model_forwards": 0}))


if __name__ == "__main__":
    main()
