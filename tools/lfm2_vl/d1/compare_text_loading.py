"""Compare the frozen t01 text/hybrid loading matrix; performs zero forwards."""
import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from verify_run import strict_json, verify_answer

T01_SHA256 = "81373f835b09408b464c32a84dbebc07491991732b3915268dd41eb17b291f8d"
LOCK_SHA256 = "d279213149d17ff2e02afa2a5115b35cfba881b8c60a333d4d3a937e62b1c265"
PROMPT_SHA256 = "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31"


def official_inputs(case, reference, tokenizer, pins):
    import importlib.util
    import sys
    from tokenizers import Tokenizer, __version__
    if __version__ != "0.22.2" or hashlib.sha256(reference.read_bytes()).hexdigest() != PROMPT_SHA256:
        raise ValueError("pinned prompt source/runtime differs")
    pin = next(p for p in pins if p["file"] == "tokenizer.json")
    payload = tokenizer.read_bytes()
    if len(payload) != pin["bytes"] or hashlib.sha256(payload).hexdigest() != pin["sha256"]:
        raise ValueError("pinned tokenizer differs")
    spec = importlib.util.spec_from_file_location("d1_text_loading_official", reference)
    prompt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prompt
    spec.loader.exec_module(prompt)
    class Adapter:
        def __init__(self):
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
        def encode(self, text, add_special_tokens=False):
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    adapter = Adapter()
    plans, inputs = [], []
    for name, value in case["request"]["questions"].items():
        question = prompt.as_question(value)
        text = prompt.render(adapter, case["request"]["state"], question, bos="<|startoftext|>", images="")
        plans.append(dict(name=name, text=text, groups=prompt.readout_ids(adapter, question)))
        inputs.append(dict(text=text, input_ids=adapter.encode(text)))
    return plans, inputs


def read(path):
    return strict_json(path.read_text(encoding="utf-8"))


def exact_counts(record, expected, label):
    for name, count in expected.items():
        value = record.get(name)
        if type(value) is not int or value != count:
            raise ValueError(f"{label} {name} must be the integer {count}")


def answer_error(left, right):
    if isinstance(left, dict):
        if not isinstance(right, dict) or list(left) != list(right):
            raise ValueError("answer keys/order differ")
        return max((answer_error(left[k], right[k]) for k in left), default=0.)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if isinstance(right, bool) or not isinstance(right, (int, float)) or not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("invalid answer number")
        return abs(left - right)
    if left != right:
        raise ValueError("answer selection, type or legend differs")
    return 0.


def compare(root, requests, lock, reference, tokenizer, include_cuda=False):
    payload = requests.read_bytes()
    if hashlib.sha256(payload).hexdigest() != T01_SHA256:
        raise ValueError("frozen t01 input identity differs")
    case = strict_json(payload.decode("utf-8"))
    if case["request_id"] != "t01" or case["images"] or len(case["request"]["questions"]) != 3:
        raise ValueError("t01 must contain three text-only questions")
    if hashlib.sha256(lock.read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError("pinned lock identity changed")
    manifest = read(lock)
    policy = manifest["comparison_policy"]
    if policy["cpu_cuda_last_logits_max_abs"] != 0.02 or policy["readout_goldens_abs"] != 1e-6:
        raise ValueError("frozen comparison bounds changed")
    expected = official_inputs(case, reference, tokenizer, manifest["files"])
    return compare_roots(root, case, expected, manifest["files"], include_cuda)


def compare_roots(root, case, expected_inputs, pins, include_cuda=False):
    reference = None
    maximum = 0.
    readout_maximum = 0.
    reports = []
    for device in (["cpu", "cuda"] if include_cuda else ["cpu"]):
        for path in ["hybrid", "text-only"]:
            for mode in ["dense-f32", "q8-rows"]:
                run = root / f"{device}-{path}-{mode}"
                admission = read(run / "admission.json")
                names = ["d1-3B-Q8_0.gguf", "mmproj-d1-3B-Q8_0.gguf", "tokenizer.json", "processor_config.json", "chat_template.jinja"] if path == "hybrid" else ["d1-3B-Q8_0.gguf", "tokenizer.json", "chat_template.jinja"]
                artifacts = admission["artifacts"]
                exact_counts(admission, dict(cases=1, context=4096), "artifact admission")
                if admission["requests_sha256"] != T01_SHA256 or admission["device"] != device or len(artifacts) != len(names):
                    raise ValueError("artifact admission/input binding differs")
                for name, artifact in zip(names, artifacts):
                    pin = next(p for p in pins if p["file"] == name)
                    exact_counts(artifact, {"bytes": pin["bytes"]}, "artifact")
                    if artifact["path"].replace("\\", "/").rsplit("/", 1)[-1] != name or artifact["sha256"] != pin["sha256"]:
                        raise ValueError("artifact identity differs from pinned manifest")
                report = read(run / "report.json")
                exact_counts(report, dict(planned_cases=1, planned_language_forwards=3,
                                         language_forwards=3, output_tokens=0), "run work")
                if report["device"] != device or report["state"] != "complete" or len(report["results"]) != 1 or report["unattempted_cases"]:
                    raise ValueError("run disposition differs")
                row = report["results"][0]
                response = row["response"]
                execution = response["execution"]
                exact_counts(response, dict(planned_questions=3, unattempted_questions=0), "case work")
                if row["request_id"] != "t01" or row["status"] != "complete" or row["had_image"] is not False:
                    raise ValueError("case disposition differs")
                usage = response["usage"]
                exact_counts(usage, dict(language_forwards=3, vision_forwards=0, output_tokens=0), "case work")
                exact_counts(execution, dict(text_quantized_linears=167,
                    mmproj_quantized_tensors=137 if path == "hybrid" else 0,
                    cpu_q8_matmuls=501 if device == "cpu" else 0,
                    cuda_f32_q8=501 if device == "cuda" else 0,
                    cuda_mmvq=0, cuda_mmq=0, cuda_other_q8=0), "native dispatch")
                if execution["text_device"] != device:
                    raise ValueError("native language storage/dispatch differs")
                if execution["vision_device"] != (device if path == "hybrid" else "none") or len(execution["source_float_mmproj_linears"]) != (27 if path == "hybrid" else 0):
                    raise ValueError("vision inventory differs")
                loaded = read(run / "loaded.json")
                if loaded["embedding_storage"]["mode"] != mode or loaded["consumed_files"] != [a["path"] for a in artifacts[:-1]]:
                    raise ValueError("loaded inputs/storage differ")
                traces = run / "case-000"
                plans = read(traces / "plans.json")["plans"]
                if len(plans) != 3:
                    raise ValueError("plan count differs")
                inputs = [read(traces / f"question-{q}.json") for q in range(3)]
                if plans != expected_inputs[0] or inputs != expected_inputs[1]:
                    raise ValueError("pinned reference prompt, aliases or token IDs differ")
                exact_counts(usage, {"input_tokens": sum(len(trace["input_ids"]) for trace in inputs)}, "input token total")
                names = list(case["request"]["questions"])
                if list(response["answers"]) != names:
                    raise ValueError("answer question order differs")
                for name, plan, trace in zip(names, plans, inputs):
                    if plan["name"] != name or plan["text"] != trace["text"] or not trace["input_ids"] or any(type(token) is not int or not 0 <= token < 128000 for token in trace["input_ids"]):
                        raise ValueError("invalid prompt/token trace")
                    if not plan["groups"] or any(not group or any(type(token) is not int or not 0 <= token < 128000 for token in group) for group in plan["groups"]):
                        raise ValueError("invalid token alias group")
                logits = [load_file(str(traces / f"logits-{q}.safetensors"))["logits"] for q in range(3)]
                if any(v.shape != (128000,) or v.dtype != np.float32 or not np.isfinite(v).all() for v in logits):
                    raise ValueError("invalid finite F32 vocabulary traces")
                if reference is None:
                    reference = (plans, inputs, logits, response["answers"], usage)
                if plans != reference[0] or inputs != reference[1] or usage != reference[4]:
                    raise ValueError("prompts, token IDs, aliases, ordering or usage differ")
                for q, (name, question) in enumerate(case["request"]["questions"].items()):
                    value = logits[q]
                    expected = reference[2][q]
                    error = float(np.max(np.abs(value.astype(np.float64) - expected)))
                    maximum = max(maximum, error)
                    if device == "cpu" and not np.array_equal(value.view(np.uint32), expected.view(np.uint32)):
                        raise ValueError("CPU logits are not bit exact")
                    if error > 0.02:
                        raise ValueError("CUDA logits exceed frozen bound")
                    pooled = [max(float(value[token]) for token in group) for group in plans[q]["groups"]]
                    exps = [math.exp(v - max(pooled)) for v in pooled]
                    probabilities = [v / sum(exps) for v in exps]
                    readout_maximum = max(readout_maximum, verify_answer(question, response["answers"][name], probabilities))
                    difference = answer_error(reference[3][name], response["answers"][name])
                    readout_maximum = max(readout_maximum, difference)
                    if difference > 1e-6 or (device == "cpu" and response["answers"][name] != reference[3][name]):
                        raise ValueError("answer/readout differs")
                reports.append(hashlib.sha256((run / "report.json").read_bytes()).hexdigest())
    return dict(schema="candle.d1_text_loading_parity.v1", passed=True,
                requests=len(reports), language_forwards=len(reports)*3, vision_forwards=0,
                output_tokens=0, max_logit_error=maximum, max_readout_error=readout_maximum,
                model_forwards_in_verifier=0, report_sha256=reports)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ["root", "requests", "lock", "reference", "tokenizer", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--include-cuda", action="store_true")
    args = parser.parse_args()
    result = compare(args.root, args.requests, args.lock, args.reference, args.tokenizer, args.include_cuda)
    import json
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
