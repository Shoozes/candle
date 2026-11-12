import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from compare_probes import compare, read_result, select_case

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "tests/fixtures/lfm2_moe_tiny/qualification_inputs.json"
LOCK = ROOT / "docs/lfm2-moe/source-lock.json"


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.inputs = json.loads(INPUTS.read_text(encoding="utf-8"))
        self.lock = json.loads(LOCK.read_text(encoding="utf-8"))
        self.case = self.inputs["cases"][0]
        self.probes = self.inputs["logit_probe_ids"]
        self.reference, self.candidate = self.records(self.case)

    def records(self, case):
        count = len(case["forced_ids"])
        reference = dict(reference_revision=self.lock["llamacpp_revision"],
                         teacher_forced=True, tokens=case["forced_ids"].copy(),
                         input_token_ids=case["prompt_ids"].copy(),
                         probe_token_ids=self.probes.copy(), probe_top1=list(range(20, 20 + count)),
                         probe_logits=[[float(i + j) for j in range(len(self.probes))] for i in range(count)])
        candidate = copy.deepcopy(reference)
        candidate.pop("reference_revision")
        return reference, candidate

    def check(self, reference=None, candidate=None, case=None):
        return compare(self.reference if reference is None else reference,
                       self.candidate if candidate is None else candidate,
                       self.case if case is None else case, self.probes,
                       self.lock["llamacpp_revision"])

    def test_both_frozen_cases_require_all_eight_positions(self):
        for case in self.inputs["cases"]:
            with self.subTest(case=case["name"]):
                reference, candidate = self.records(case)
                candidate["probe_logits"][1][1] += 0.25
                result = self.check(reference, candidate, case)
                self.assertEqual(result["positions"], 8)
                self.assertEqual(result["case"], case["name"])
                self.assertEqual(result["max_absolute_logit_difference"], 0.25)

    def test_matching_truncated_or_extra_runs_do_not_pass(self):
        for count in [1, 7, 9]:
            reference, candidate = copy.deepcopy(self.reference), copy.deepcopy(self.candidate)
            for record in [reference, candidate]:
                record["probe_top1"] = [20] * count
                record["probe_logits"] = [[1.] * len(self.probes) for _ in range(count)]
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.check(reference, candidate)

    def test_matching_wrong_prompt_and_probe_inputs_do_not_pass(self):
        for field in ["input_token_ids", "probe_token_ids"]:
            reference, candidate = copy.deepcopy(self.reference), copy.deepcopy(self.candidate)
            reference[field][0] += 1
            candidate[field][0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(reference, candidate)

    def test_reference_must_match_the_pinned_revision(self):
        for revision in [None, "wrong-reference", "0" * 40]:
            reference = dict(self.reference, reference_revision=revision)
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.check(reference=reference)

    def test_candidate_must_use_the_frozen_forced_continuation(self):
        for field, value in [("teacher_forced", False), ("teacher_forced", 1),
                             ("tokens", [1]), ("tokens", [True] * 8)]:
            candidate = dict(self.candidate, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check(candidate=candidate)

    def test_reference_must_report_the_full_frozen_continuation(self):
        wrong = self.case["forced_ids"].copy()
        wrong[0] += 1
        for field, value, expected in [
            ("teacher_forced", None, "reference did not report a teacher-forced continuation"),
            ("tokens", None, "reference continuation requires nonempty U32 token IDs"),
            ("tokens", wrong, "reference continuation differs from the frozen qualification case"),
        ]:
            reference = dict(self.reference)
            if value is None:
                reference.pop(field)
            else:
                reference[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, expected):
                self.check(reference=reference)

    def test_next_token_shape_type_and_finite_logits_are_required(self):
        bad_values = [("probe_top1", [True] * 8),
                      ("probe_logits", [[1.]]), ("input_token_ids", []),
                      ("probe_token_ids", [False] * len(self.probes))]
        for field, value in bad_values:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check(candidate=dict(self.candidate, **{field: value}))
        for value in [float("nan"), float("inf"), True, "1"]:
            candidate = copy.deepcopy(self.candidate)
            candidate["probe_logits"][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(candidate=candidate)

    def test_top1_mismatch_reports_the_first_difference(self):
        self.candidate["probe_top1"][2:4] = [90, 91]
        with self.assertRaisesRegex(ValueError, r"zero-based position 2 \(reference=22, candidate=90\)"):
            self.check()

    def test_frozen_positions_are_exact_and_integral(self):
        for positions in [[0], [0] * 8, [False] + list(range(1, 8))]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                self.check(case=dict(self.case, probe_positions=positions))

    def test_nonfinite_difference_cannot_be_reported_as_passing(self):
        reference, candidate = copy.deepcopy(self.reference), copy.deepcopy(self.candidate)
        reference["probe_logits"][0][0] = 1e308
        candidate["probe_logits"][0][0] = -1e308
        with self.assertRaises(ValueError):
            self.check(reference, candidate)

    def test_input_schema_revision_and_case_are_checked(self):
        self.assertEqual(select_case(self.inputs, self.case["name"], self.lock["native_revision"]), self.case)
        for field, value in [("schema", "wrong"), ("model_revision", "wrong")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_case(dict(self.inputs, **{field: value}), self.case["name"], self.lock["native_revision"])
        with self.assertRaises(ValueError):
            select_case(self.inputs, "unknown", self.lock["native_revision"])
        duplicate = dict(self.inputs, cases=[self.case, self.case])
        with self.assertRaises(ValueError):
            select_case(duplicate, self.case["name"], self.lock["native_revision"])

    def test_terminal_record_is_unique_among_runtime_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "probe.log"
            encoded = json.dumps(self.reference)
            path.write_text("runtime diagnostics\n" + encoded + "\n", encoding="utf-8")
            self.assertEqual(read_result(path), self.reference)
            path.write_text(encoded + "\n" + encoded + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_result(path)
            path.write_text("runtime failed before output\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_result(path)

    def test_invalid_utf8_in_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "probe.log"
            valid = json.dumps(self.reference).encode("utf-8")
            path.write_bytes(b'{"probe_logits":"\xc4..."}\n' + valid + b"\n")
            with self.assertRaisesRegex(ValueError, r"probe\.log:1: invalid UTF-8 in JSON probe line"):
                read_result(path)

    def test_cli_requires_frozen_inputs_and_reports_the_bound_case(self):
        with tempfile.TemporaryDirectory() as root:
            reference, candidate = Path(root) / "reference.json", Path(root) / "candidate.json"
            reference.write_bytes(
                b'llama_model_loader: tokenizer.ggml.tokens = ["\xc4\x8a\xc4...\r\n'
                + json.dumps(self.reference).encode("utf-8") + b"\r\n"
            )
            candidate.write_text(json.dumps(self.candidate), encoding="utf-8")
            command = [sys.executable, "-B", str(ROOT / "tools/lfm2_moe/compare_probes.py"),
                       "--reference", str(reference), "--candidate", str(candidate)]
            missing = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(missing.returncode, 0)
            command += ["--inputs", str(INPUTS), "--case", self.case["name"]]
            success = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout)["positions"], 8)
            reference.write_text(json.dumps(dict(self.reference, reference_revision="wrong")), encoding="utf-8")
            rejected = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertNotIn('"status": "passed"', rejected.stdout)


if __name__ == "__main__":
    unittest.main()
