"""Compare source-bound CPU/CUDA d1 study traces; performs zero model forwards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ["cpu", "cuda", "lock", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cases", type=int, choices=[2, 20], default=20)
    args = parser.parse_args()
    policy = json.loads(args.lock.read_text("utf-8"))["comparison_policy"]
    reports = [json.loads((root / "report.json").read_text("utf-8")) for root in [args.cpu, args.cuda]]
    for device, report in zip(["cpu", "cuda"], reports):
        if report["device"] != device or report["state"] != "complete" or len(report["results"]) != args.cases or report["language_forwards"] != args.cases * 3 or report["output_tokens"] != 0:
            raise ValueError("study device/denominator/work counts differ")
    maximum = 0.
    minimum_cosine = 1.
    rows = []
    feature_checks = 0
    for index, (cpu, cuda) in enumerate(zip(reports[0]["results"], reports[1]["results"])):
        if cpu["request_id"] != cuda["request_id"]:
            raise ValueError("case identities differ")
        for device, row in [("cpu", cpu), ("cuda", cuda)]:
            response = row["response"]
            storage = response["execution"]
            usage = response["usage"]
            if row["status"] != "complete" or usage["language_forwards"] != 3 or usage["vision_forwards"] != int(row["had_image"]) or usage["output_tokens"] != 0 or response["unattempted_questions"]:
                raise ValueError("case disposition/counts differ")
            expected = 3 * 167 + int(row["had_image"]) * 137
            if storage["text_quantized_linears"] != 167 or storage["mmproj_quantized_tensors"] != 137 or len(storage["source_float_mmproj_linears"]) != 27:
                raise ValueError("retained native storage mix differs")
            if storage["text_device"] != device or storage["vision_device"] != device or storage["cpu_q8_matmuls" if device == "cpu" else "cuda_f32_q8"] != expected:
                raise ValueError("native F32/Q8 dispatch differs from the admitted inventory")
        roots = [root / f"case-{index:03}" for root in [args.cpu, args.cuda]]
        errors = []
        for question in range(3):
            inputs = [json.loads((root / f"question-{question}.json").read_text("utf-8")) for root in roots]
            if inputs[0] != inputs[1]:
                raise ValueError("CPU/CUDA prompt or input IDs differ")
            values = [load_file(str(root / f"logits-{question}.safetensors"))["logits"] for root in roots]
            if values[0].shape != values[1].shape or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in values):
                raise ValueError("invalid vocabulary traces")
            error = float(np.max(np.abs(values[0].astype(np.float64) - values[1])))
            maximum = max(maximum, error)
            errors.append(error)
        if (roots[0] / "features.safetensors").exists():
            features = [load_file(str(root / "features.safetensors"))["features"] for root in roots]
            if features[0].shape != features[1].shape or any(not np.isfinite(v).all() for v in features):
                raise ValueError("invalid projected feature traces")
            a, b = [v.astype(np.float64).ravel() for v in features]
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if not np.isfinite(cosine):
                raise ValueError("projected feature cosine is not finite")
            minimum_cosine = min(minimum_cosine, cosine)
            feature_checks += 1
        rows.append(dict(id=cpu["request_id"], logit_errors=errors))
    passed = maximum <= policy["cpu_cuda_last_logits_max_abs"] and feature_checks > 0 and minimum_cosine >= policy["q8_projected_features_min_cosine"]
    result = dict(schema="candle.d1_cpu_cuda_parity.v1", passed=passed, last_logit_checks=args.cases * 3,
                  max_last_logit_error=maximum, last_logit_bound=policy["cpu_cuda_last_logits_max_abs"],
                  projected_feature_checks=feature_checks, minimum_feature_cosine=minimum_cosine,
                  feature_cosine_bound=policy["q8_projected_features_min_cosine"],
                  model_forwards_in_verifier=0, rows=rows,
                  run_report_sha256=[hashlib.sha256((root / "report.json").read_bytes()).hexdigest() for root in [args.cpu, args.cuda]])
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    if not passed:
        raise ValueError("CPU/CUDA parity exceeds the frozen bounds")


if __name__ == "__main__":
    main()
