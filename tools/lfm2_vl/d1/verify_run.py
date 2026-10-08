"""Offline d1 prompt/alias/readout verification and independent study scoring."""
from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key: {name}")
            result[name] = value
        return result
    def nonfinite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=nonfinite)


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def verify_answer(question, answer, probabilities, bound=1e-6):
    kind = question["type"]
    fields = {"noul": {"type", "noul"},
              "choice": {"type", "choice", "confidence", "probabilities"},
              "score": {"type", "score", "confidence", "probabilities", "legend"}}
    if not isinstance(answer, dict) or set(answer) != fields[kind] or answer["type"] != kind:
        raise ValueError("typed answer fields or discriminator differ")
    keys = (["yes", "no"] if kind == "noul" else list(question["criteria"]) if kind == "choice"
            else [str(i) for i in range(len(question["criteria"]))])
    if kind == "noul":
        yes = finite_number(answer["noul"], "noul")
        observed = [yes, 1. - yes]
    else:
        if not isinstance(answer["probabilities"], dict) or list(answer["probabilities"]) != keys:
            raise ValueError("answer probability keys/order differ")
        observed = list(answer["probabilities"].values())
    expected = list(probabilities)
    if len(expected) != len(keys):
        raise ValueError("reference readout cardinality differs")
    for values in [observed, expected]:
        if any(not 0. <= finite_number(value, "probability") <= 1. for value in values):
            raise ValueError("probability outside 0..1")
        if abs(sum(values) - 1.) > bound:
            raise ValueError("probabilities are not normalized")
    error = max(abs(a - b) for a, b in zip(observed, expected))
    if kind != "noul":
        # Derive output fields from the verified returned probabilities. This
        # keeps allowed numeric readout tolerance separate from selection ties.
        best = max(range(len(observed)), key=observed.__getitem__)
        confidence = finite_number(answer["confidence"], "confidence")
        error = max(error, abs(confidence - observed[best]))
        if kind == "choice":
            if answer["choice"] != keys[best]:
                raise ValueError("choice does not select the first maximum in reference order")
        else:
            score = finite_number(answer["score"], "score")
            error = max(error, abs(score - sum(i * p for i, p in enumerate(observed))))
            legend = answer["legend"]
            if not isinstance(legend, dict) or list(legend) != keys or list(legend.values()) != question["criteria"]:
                raise ValueError("score legend differs from the ordered rubric")
    if error > bound:
        raise ValueError(f"readout or derived answer differs from the frozen {bound} bound: {error}")
    return error


def logits(path: Path) -> list[float]:
    data = path.read_bytes()
    length = struct.unpack_from("<Q", data)[0]
    header = strict_json(data[8:8 + length])
    tensor = header["logits"]
    if tensor["dtype"] != "F32" or len(tensor["shape"]) != 1:
        raise ValueError("expected an F32 vocabulary vector")
    start, end = tensor["data_offsets"]
    values = array.array("f", data[8 + length + start:8 + length + end])
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != tensor["shape"][0] or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid last logits")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    for name in ("run", "freeze", "reference", "tokenizer", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    from tokenizers import Tokenizer, __version__
    if __version__ != "0.22.2":
        raise ValueError("reference tokenizers must be 0.22.2")
    source = args.reference.read_bytes()
    if hashlib.sha256(source).hexdigest() != "a20b5f52e41a6d29e694bb8c861a5a4803ce3eb69364edad79511ccf8edb5c31":
        raise ValueError("reference source changed")
    spec = importlib.util.spec_from_file_location("d1_official_prompt", args.reference)
    prompt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = prompt
    spec.loader.exec_module(prompt)

    class Adapter:
        def __init__(self):
            self.tokenizer = Tokenizer.from_file(str(args.tokenizer))
        def encode(self, text, add_special_tokens=False):
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    tokenizer = Adapter()
    cases = [strict_json(line) for line in (args.freeze / "study.jsonl").read_text("utf-8").splitlines()]
    oracle = strict_json((args.freeze / "oracle.json").read_text("utf-8"))["cases"]
    report = strict_json((args.run / "report.json").read_text("utf-8"))
    if len(cases) != 20 or len(report["results"]) != 20 or report["state"] != "complete" or report["language_forwards"] != 60 or report["output_tokens"] != 0 or report["unattempted_cases"]:
        raise ValueError("study denominator/work counts differ from the frozen plan")
    quality = {"choice_correct": 0, "boolean_correct": 0, "score_in_range": 0}
    maximum_error = 0.0
    rows = []
    text_token_checks = 0
    for index, (case, result) in enumerate(zip(cases, report["results"])):
        if case["request_id"] != result["request_id"] or result["status"] != "complete":
            raise ValueError("case identity or disposition differs")
        root = args.run / f"case-{index:03}"
        plans = strict_json((root / "plans.json").read_text("utf-8"))["plans"]
        usage = result["response"]["usage"]
        if len(case["request"]["questions"]) != 3 or len(plans) != 3 or result["response"]["planned_questions"] != 3 or usage["language_forwards"] != 3 or usage["output_tokens"] != 0 or usage["vision_forwards"] != int(bool(case["images"])) or result["response"]["unattempted_questions"] != 0:
            raise ValueError("case question denominator/work counts differ")
        answers = result["response"]["answers"]
        if list(answers) != list(case["request"]["questions"]):
            raise ValueError("answer question keys/order differ")
        for i, ((name, value), plan) in enumerate(zip(case["request"]["questions"].items(), plans)):
            question = prompt.as_question(value)
            expected = prompt.render(tokenizer, case["request"]["state"], question, bos="<|startoftext|>", images="<image>" * len(case["images"]))
            trace = strict_json((root / f"question-{i}.json").read_text("utf-8"))
            if plan["name"] != name or expected != plan["text"] or expected != trace["text"] or prompt.readout_ids(tokenizer, question) != plan["groups"]:
                raise ValueError(f"prompt/alias mismatch at {case['request_id']}/{name}")
            if not case["images"]:
                if tokenizer.encode(expected) != trace["input_ids"]:
                    raise ValueError("text token mismatch")
                text_token_checks += 1
            probabilities = prompt.readout(tokenizer, question, logits(root / f"logits-{i}.safetensors"))
            answer = answers[name]
            error = verify_answer(value, answer, probabilities)
            maximum_error = max(maximum_error, error)
        target = oracle[case["request_id"]]
        correct = {"choice_correct": answers["inspection"]["choice"] == target["choice"],
                   "boolean_correct": (answers["sufficient"]["noul"] >= 0.5) == target["sufficient"],
                   "score_in_range": target["score_range"][0] <= answers["clarity"]["score"] <= target["score_range"][1]}
        for key, value in correct.items():
            quality[key] += value
        rows.append({"id": case["request_id"], **correct})
    output = {"schema": "candle.d1_offline_verification.v1", "device": report["device"], "cases": 20,
              "language_forwards": 60, "vision_forwards": sum(row["response"]["usage"]["vision_forwards"] for row in report["results"]),
              "reference_prompt_and_alias_checks": 60, "reference_text_token_checks": text_token_checks,
              "typed_answer_checks": 60,
              "readout_max_abs": maximum_error, "readout_bound": 1e-6, "quality": quality, "quality_denominator": 20,
              "accuracy_threshold": None, "rows": rows, "model_forwards_in_verifier": 0,
              "limitations": ["Replays supplied Candle logits; it is not an independent model-logit oracle.", "Image-expanded token and feature parity require the model/processor oracle."],
              "run_report_sha256": hashlib.sha256((args.run / "report.json").read_bytes()).hexdigest()}
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("rows", "limitations")}))


if __name__ == "__main__":
    main()
