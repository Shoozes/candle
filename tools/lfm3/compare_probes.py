"""Compare offline LFM3 text probe reports against a frozen case."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def reject_constant(word):
    raise ValueError(f"invalid JSON constant: {word}")


def read_object(path):
    data = path.read_bytes()
    obj = json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return obj, hashlib.sha256(data).hexdigest()


def ids(values, count, vocab, label):
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(type(value) is not int or not 0 <= value < vocab for value in values)
    ):
        raise ValueError(f"{label}: expected {count} vocabulary IDs")
    return values


def count_list(values):
    return len(values) if isinstance(values, list) else 0


def check_case(case):
    vocab, context = case["vocab_size"], case["context"]
    if type(vocab) is not int or vocab <= 0 or type(context) is not int or not 1 <= context <= 4096:
        raise ValueError("invalid vocabulary or context")
    for key, limit in (("tokens", context), ("probes", 64)):
        count = count_list(case[key])
        if not 1 <= count <= limit:
            raise ValueError(f"invalid {key} length")
        ids(case[key], count, vocab, key)
    for side in ("reference", "candidate"):
        settings = case[side]
        if not isinstance(settings, dict):
            raise ValueError(f"{side}: expected settings object")
        chunk, device = settings["chunk"], settings["device"]
        if type(chunk) is not int or not 1 <= chunk <= min(128, context):
            raise ValueError(f"{side}: invalid expected chunk")
        if not isinstance(device, str) or not device:
            raise ValueError(f"{side}: expected device label")
    tolerance = case["max_abs"]
    if type(tolerance) not in (float, int) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid max_abs tolerance")


def validate_report(report, case, side):
    rows, cols, vocab = len(case["tokens"]), len(case["probes"]), case["vocab_size"]
    if (
        ids(report["tokens"], rows, vocab, side + " tokens") != case["tokens"]
        or ids(report["probes"], cols, vocab, side + " probes") != case["probes"]
        or report["dtype"] != "f32"
        or report["device"] != case[side]["device"]
        or type(report["context"]) is not int
        or report["context"] != case["context"]
        or type(report["chunk"]) is not int
        or report["chunk"] != case[side]["chunk"]
    ):
        raise ValueError(f"{side}: report differs from frozen inputs or settings")

    top = ids(report["top_ids"], rows, vocab, side + " top IDs")
    logits = report["probe_logits"]
    if not isinstance(logits, list) or len(logits) != rows:
        raise ValueError(f"{side}: expected {rows} logit rows")
    for index, row in enumerate(logits):
        if not isinstance(row, list) or len(row) != cols or any(
            type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 3.4028235e38
            for value in row
        ):
            raise ValueError(f"{side}: expected {cols} finite F32 logits at position {index}")
    return top, logits


def compare(reference, candidate, case):
    check_case(case)
    ref_top, ref_logits = validate_report(reference, case, "reference")
    got_top, got_logits = validate_report(candidate, case, "candidate")
    for pos, (expected, actual) in enumerate(zip(ref_top, got_top, strict=True)):
        if expected != actual:
            raise ValueError(f"top ID mismatch at position {pos}: reference {expected}, candidate {actual}")
    peak, squared, count = 0.0, 0.0, 0
    for pos, (expected, actual) in enumerate(zip(ref_logits, got_logits, strict=True)):
        for column, (a, b) in enumerate(zip(expected, actual, strict=True)):
            diff = abs(a - b)
            if diff > case["max_abs"]:
                raise ValueError(f"logit mismatch at position {pos}, probe {case['probes'][column]}: {diff:g}")
            peak = max(peak, diff)
            squared += diff * diff
            count += 1
    return {"positions": len(ref_top), "probe_values": count, "max_abs": peak, "rms": math.sqrt(squared / count)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        case, case_sha = read_object(args.case)
        reference, reference_sha = read_object(args.reference)
        candidate, candidate_sha = read_object(args.candidate)
        result = compare(reference, candidate, case)
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps({**result, "case_sha256": case_sha, "reference_sha256": reference_sha,
                      "candidate_sha256": candidate_sha}, allow_nan=False))


if __name__ == "__main__":
    main()
