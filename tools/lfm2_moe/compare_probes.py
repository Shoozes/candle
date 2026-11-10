"""Compare complete teacher-forced records against the frozen qualification case."""
import argparse
import json
import math
from pathlib import Path


def read_result(path):
    record = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            fields = {"input_token_ids", "probe_token_ids", "probe_top1", "probe_logits"}
            if isinstance(value, dict) and fields <= value.keys():
                if record is not None:
                    raise ValueError("expected exactly one terminal probe record")
                record = value
    if record is None:
        raise ValueError("expected exactly one terminal probe record")
    return record


def token_ids(value, label):
    if not isinstance(value, list) or not value or any(
        type(item) is not int or not 0 <= item <= 0xFFFFFFFF for item in value
    ):
        raise ValueError(f"{label} requires nonempty U32 token IDs")
    return value


def select_case(inputs, name, native_revision):
    if (inputs.get("schema") != "candle.lfm2-moe.qualification-inputs.v1"
            or inputs.get("model_revision") != native_revision):
        raise ValueError("qualification inputs have the wrong schema or model revision")
    matches = [case for case in inputs["cases"] if case.get("name") == name]
    if len(matches) != 1:
        raise ValueError("expected exactly one frozen qualification case")
    return matches[0]


def compare(reference, candidate, case, probe_ids, reference_revision):
    prompt = token_ids(case["prompt_ids"], "frozen prompt")
    forced = token_ids(case["forced_ids"], "frozen continuation")
    ids = token_ids(probe_ids, "frozen logit probes")
    positions = case["probe_positions"]
    if (len(forced) > 32 or len(ids) > 64 or not isinstance(positions, list)
            or any(type(position) is not int for position in positions)
            or positions != list(range(len(forced)))):
        raise ValueError("frozen probe positions or bounds are invalid")
    if reference.get("reference_revision") != reference_revision:
        raise ValueError("reference revision differs from the source lock")
    if (candidate.get("teacher_forced") is not True
            or token_ids(candidate.get("tokens"), "candidate continuation") != forced):
        raise ValueError("candidate did not consume the frozen teacher-forced continuation")
    for record in (reference, candidate):
        if (token_ids(record.get("input_token_ids"), "prompt") != prompt
                or token_ids(record.get("probe_token_ids"), "logit probes") != ids):
            raise ValueError("probe inputs differ from the frozen qualification case")
        if len(token_ids(record.get("probe_top1"), "next-token selections")) != len(forced):
            raise ValueError("missing or extra teacher-forced probe positions")
        rows = record.get("probe_logits")
        if not isinstance(rows, list) or len(rows) != len(forced):
            raise ValueError("missing probe logit positions")
        for row in rows:
            if not isinstance(row, list) or len(row) != len(ids):
                raise ValueError("probe logit shape mismatch")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
                raise ValueError("nonfinite or nonnumeric probe logit")
    if candidate["probe_top1"] != reference["probe_top1"]:
        raise ValueError("teacher-forced next-token selections differ")
    differences = [abs(x - y) for left, right in zip(reference["probe_logits"], candidate["probe_logits"])
                   for x, y in zip(left, right)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("nonfinite probe logit difference")
    return dict(status="passed", case=case["name"], positions=len(forced), logits=len(differences),
                reference_revision=reference_revision, exact_top1=True,
                max_absolute_logit_difference=max(differences),
                rms_logit_difference=math.hypot(*differences) / math.sqrt(len(differences)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    lock = json.loads((root / "docs/lfm2-moe/source-lock.json").read_text(encoding="utf-8"))
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    case = select_case(inputs, args.case, lock["native_revision"])
    print(json.dumps(compare(read_result(args.reference), read_result(args.candidate), case,
                             inputs["logit_probe_ids"], lock["llamacpp_revision"]), allow_nan=False))


if __name__ == "__main__":
    main()
