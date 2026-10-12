"""Authored malformed-receipt controls for the text-only packet comparator."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from compare_text_loading import compare_roots, T01_SHA256


class TextLoading(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="candle-d1-comparison-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = {"request_id": "t01", "images": [], "request": {"questions": {
            name: {"type": "noul"} for name in ["one", "two", "three"]}}}
        self.pins = [dict(file=name,bytes=1,sha256="00"*32) for name in ["d1-3B-Q8_0.gguf", "mmproj-d1-3B-Q8_0.gguf", "tokenizer.json", "processor_config.json", "chat_template.jinja"]]
        self.expected = ([dict(name=name,text="authored prompt",groups=[[0],[1]]) for name in self.case["request"]["questions"]],
                         [dict(text="authored prompt",input_ids=[2]) for _ in range(3)])
        for device in ["cpu", "cuda"]:
            for path in ["hybrid", "text-only"]:
                for mode in ["dense-f32", "q8-rows"]:
                    run = self.root / f"{device}-{path}-{mode}"
                    traces = run / "case-000"
                    traces.mkdir(parents=True)
                    execution = dict(text_device=device, text_quantized_linears=167,
                        vision_device=device if path == "hybrid" else "none",
                        mmproj_quantized_tensors=137 if path == "hybrid" else 0,
                        source_float_mmproj_linears=["float"]*27 if path == "hybrid" else [],
                        cpu_q8_matmuls=501 if device == "cpu" else 0,
                        cuda_f32_q8=501 if device == "cuda" else 0,
                        cuda_mmvq=0,cuda_mmq=0,cuda_other_q8=0)
                    response = dict(execution=execution, planned_questions=3, unattempted_questions=0,
                        usage=dict(language_forwards=3, vision_forwards=0, output_tokens=0, input_tokens=3),
                        answers={name: dict(type="noul", noul=.5) for name in self.case["request"]["questions"]})
                    self.write(run / "report.json", dict(device=device, state="complete", planned_cases=1,
                        planned_language_forwards=3, language_forwards=3, output_tokens=0, unattempted_cases=[],
                        results=[dict(request_id="t01",status="complete",had_image=False,response=response)]))
                    artifacts = [dict(path=p["file"],bytes=p["bytes"],sha256=p["sha256"]) for p in self.pins if path=="hybrid" or p["file"] not in ["mmproj-d1-3B-Q8_0.gguf","processor_config.json"]]
                    self.write(run / "admission.json",dict(artifacts=artifacts,requests_sha256=T01_SHA256,cases=1,device=device,context=4096))
                    self.write(run / "loaded.json", dict(embedding_storage=dict(mode=mode), consumed_files=[a["path"] for a in artifacts[:-1]]))
                    plans = []
                    for q, name in enumerate(self.case["request"]["questions"]):
                        plans.append(dict(name=name,text="authored prompt",groups=[[0],[1]]))
                        self.write(traces / f"question-{q}.json", dict(text="authored prompt",input_ids=[2]))
                        save_file({"logits":np.zeros(128000,dtype=np.float32)},str(traces / f"logits-{q}.safetensors"))
                    self.write(traces / "plans.json",dict(plans=plans))

    def write(self, path, value):
        path.write_text(json.dumps(value),encoding="utf-8")

    def target(self, device="cpu"):
        return self.root / f"{device}-text-only-q8-rows"

    def compare(self, include_cuda=False):
        return compare_roots(self.root,self.case,self.expected,self.pins,include_cuda)

    def test_complete_cpu_then_cross_device(self):
        self.assertEqual(self.compare()["language_forwards"],12)
        result = self.compare(True)
        self.assertEqual(result["language_forwards"],24)
        self.assertEqual(result["model_forwards_in_verifier"],0)

    def test_wrong_counts_and_vision_inventory_rejected(self):
        path=self.target()/"report.json"
        for field in ["vision_forwards","language_forwards","output_tokens"]:
            report=json.loads(path.read_text())
            report["results"][0]["response"]["usage"][field]+=1
            self.write(path,report)
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.compare()
            report["results"][0]["response"]["usage"][field]-=1
            self.write(path,report)
        report["results"][0]["response"]["execution"]["vision_device"]="cpu"
        self.write(path,report)
        with self.assertRaises(ValueError): self.compare()

    def test_nonfinite_cpu_difference_and_cuda_bounds_rejected(self):
        for device,value in [("cpu",float("nan")),("cpu",1e-7),("cuda",.03)]:
            path=self.target(device)/"case-000"/"logits-0.safetensors"
            logits=np.zeros(128000,dtype=np.float32); logits[10]=value
            save_file({"logits":logits},str(path))
            with self.subTest(device=device,value=value),self.assertRaises(ValueError):
                self.compare(device=="cuda")
            save_file({"logits":np.zeros(128000,dtype=np.float32)},str(path))

    def test_changed_trace_and_alias_bounds_rejected(self):
        trace=self.target()/"case-000"/"question-0.json"
        self.write(trace,dict(text="changed",input_ids=[2]))
        with self.assertRaises(ValueError): self.compare()
        self.write(trace,dict(text="authored prompt",input_ids=[2]))
        path=self.target()/"case-000"/"plans.json"
        plans=json.loads(path.read_text()); plans["plans"][0]["groups"]=[[-1],[1]]
        self.write(path,plans)
        with self.assertRaises(ValueError): self.compare()

    def test_common_mode_wrong_prompt_and_artifact_hash_rejected(self):
        for run in self.root.iterdir():
            path=run/"case-000"/"plans.json"
            plans=json.loads(path.read_text()); plans["plans"][0]["text"]="common regression"
            self.write(path,plans)
            self.write(run/"case-000"/"question-0.json",dict(text="common regression",input_ids=[2]))
        with self.assertRaisesRegex(ValueError,"pinned reference"): self.compare()
        for run in self.root.iterdir():
            self.write(run/"case-000"/"plans.json",dict(plans=self.expected[0]))
            self.write(run/"case-000"/"question-0.json",self.expected[1][0])
        path=self.target()/"admission.json"
        admission=json.loads(path.read_text()); admission["artifacts"][0]["sha256"]="ff"*32
        self.write(path,admission)
        with self.assertRaisesRegex(ValueError,"artifact identity"): self.compare()

    def test_common_mode_input_token_total_rejected(self):
        for run in self.root.iterdir():
            path=run/"report.json"
            report=json.loads(path.read_text())
            report["results"][0]["response"]["usage"]["input_tokens"]=0
            self.write(path,report)
        with self.assertRaisesRegex(ValueError,"input token"): self.compare()

    def test_nonnumeric_work_counts_and_inactive_dispatch_rejected(self):
        path=self.target()/"report.json"
        original=json.loads(path.read_text())
        for value in [False, 0.0, None, "0"]:
            report=json.loads(json.dumps(original))
            report["results"][0]["response"]["usage"]["output_tokens"]=value
            self.write(path,report)
            with self.subTest(value=value),self.assertRaises(ValueError): self.compare()
        report=json.loads(json.dumps(original))
        report["results"][0]["response"]["execution"]["cuda_f32_q8"]=1
        self.write(path,report)
        with self.assertRaises(ValueError): self.compare()


if __name__ == "__main__":
    unittest.main()
