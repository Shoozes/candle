"""Freeze the corrected retained study for a new Candle run; never run inference."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    frozen = json.loads(raw)
    plan = frozen["plan"]
    if len(plan["cases"]) != 20 or list(plan["questions"]) != ["sufficient", "inspection", "clarity"]:
        raise ValueError("expected the corrected 20-case, three-question study")
    questions = {}
    for name, question in plan["questions"].items():
        kind = question["type"]
        if kind == "boolean":
            questions[name] = {"type": "noul", "instructions": question["instructions"]}
        elif kind == "choice":
            questions[name] = {"type": "choice", "instructions": question["instructions"], "criteria": {
                label: option["description"] for label, option in question["options"].items()}}
        elif kind == "score":
            questions[name] = {"type": "score", "instructions": question["instructions"], "criteria": question["rubric"]}
        else:
            raise ValueError(f"unsupported study question {kind}")
    out = args.out.resolve()
    out.mkdir(exist_ok=False)
    images = {}
    identities = []
    for key, payload in frozen["image_payloads"].items():
        prefix = "data:image/png;base64,"
        if not payload["data"].startswith(prefix):
            raise ValueError("expected prepared PNG payload")
        data = base64.b64decode(payload["data"][len(prefix):], validate=True)
        path = out / f"{key}.png"
        write(path, data)
        images[key] = str(path)
        identities.append({"path": str(path), "bytes": len(data), "sha256": digest(data), "width": payload["width"], "height": payload["height"]})
    cases = [{"request_id": case["id"], "request": {"state": case.get("state"), "questions": questions},
              "images": [images[case["id"]]] if case["arm"] == "image" else []} for case in plan["cases"]]
    if len({case["request_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    jsonl = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases).encode("utf-8")
    write(out / "study.jsonl", jsonl)
    qualifications = [cases[0], next(case for case in cases if case["images"])]
    write(out / "qualification.jsonl", "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in qualifications).encode("utf-8"))
    write(out / "oracle.json", (json.dumps(frozen["oracle"], indent=2) + "\n").encode("utf-8"))
    lock = args.lock.read_bytes()
    write(out / "reference-lock.json", lock)
    receipt = {"schema": "candle.d1_study_freeze.v1", "source": str(args.frozen.resolve()), "source_sha256": digest(raw),
               "source_plan_sha256": frozen["plan_sha256"], "source_oracle_sha256": frozen["oracle_sha256"],
               "requests_sha256": digest(jsonl), "reference_lock_sha256": digest(lock), "images": identities,
               "device_order": ["cpu", "cuda_after_cpu_parity"], "cases_per_device": 20, "language_forwards_per_device": 60,
               "qualification_forwards_per_device": 6, "retries": 0, "output_tokens": 0,
               "calibration": "identity", "sequential": True, "study_timeout_seconds": 1800,
               "qualification_timeout_seconds": 300, "max_job_memory_bytes": 17179869184,
               "accuracy_threshold": None, "prior_42_request_allowance": "closed; not reused"}
    write(out / "freeze.json", (json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"out": str(out), "cases": len(cases), "language_forwards": 60, "images": len(images)}))


if __name__ == "__main__":
    main()
