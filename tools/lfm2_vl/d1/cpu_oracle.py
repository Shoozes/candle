"""Independent pinned Transformers CPU oracle using retained GGUF operands only."""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import time

from oracle_weights import decode, directory, text_name, vision_name


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def hash_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def retained(path):
    if path.is_symlink() or not path.is_file() or getattr(path.stat(), "st_file_attributes", 0) & 0x400:
        raise ValueError("reference inputs must be regular, non-reparse files")
    if os.name == "nt":
        import msvcrt
        from ctypes import wintypes
        create = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        handle = create(str(path.resolve()), 0x80000000, 1, None, 3, 0x00200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        stream = os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb")
    else:
        stream = path.open("rb")
    try:
        yield stream
    finally:
        stream.close()


def load_model(config_path, sources, *, preflight_only=False):
    import numpy as np
    import torch
    from transformers import Lfm2VlConfig, Lfm2VlForConditionalGeneration
    config = Lfm2VlConfig.from_dict(json.loads(config_path.read_text("utf-8")))
    for part in [config, config.text_config, config.vision_config]:
        part._attn_implementation = "eager"
    with torch.device("meta"):
        model = Lfm2VlForConditionalGeneration(config)
    assigned = set()
    inventory = []
    plans = []
    # Preflight the whole directory before allocating any dequantized weights.
    for stream, mapper in sources:
        tensors = directory(stream)
        for name, info in tensors.items():
            target = mapper(name)
            if target in assigned:
                raise ValueError(f"duplicate reference parameter mapping: {target}")
            expected = model.get_parameter(target)
            shape = info["shape"]
            if name == "v.patch_embd.weight":
                shape = (shape[0], int(np.prod(shape[1:])))
            elif name.endswith("shortconv.conv.weight") and len(shape) == 2:
                shape = (shape[0], 1, shape[1])
            if shape != tuple(expected.shape):
                raise ValueError(f"reference shape mismatch: {name} {info['shape']} -> {target} {tuple(expected.shape)}")
            assigned.add(target)
        plans.append((stream, mapper, tensors))
    if preflight_only:
        return model, [{"source": name, "target": mapper(name), "shape": list(info["shape"])}
                       for _, mapper, tensors in plans for name, info in tensors.items()]
    for stream, mapper, tensors in plans:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for name, info in tensors.items():
                target = mapper(name)
                expected = model.get_parameter(target)
                values = decode(mapped, info)
                if name == "v.patch_embd.weight":
                    values = np.ascontiguousarray(values.transpose(0, 2, 3, 1)).reshape(expected.shape)
                elif name.endswith("shortconv.conv.weight") and values.ndim == 2:
                    values = values[:, None, :]
                if tuple(values.shape) != tuple(expected.shape) or not np.isfinite(values).all():
                    raise ValueError(f"reference operand mismatch: {name} -> {target}")
                parent, field = target.rsplit(".", 1)
                model.get_submodule(parent).register_parameter(field, torch.nn.Parameter(torch.from_numpy(values), requires_grad=False))
                assigned.add(target)
                inventory.append({"source": name, "target": target, "dtype": info["kind"], "shape": list(values.shape)})
    model.tie_weights()
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Lfm2RotaryEmbedding":
            parent, field = name.rsplit(".", 1)
            model.get_submodule(parent).__setattr__(field, type(module)(module.config, device="cpu"))
    missing = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    meta_buffers = [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if missing or meta_buffers:
        raise ValueError(f"reference graph remains incomplete: parameters={missing}, buffers={meta_buffers}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("reference graph requires F32 parameters")
    return model.eval(), inventory


def evaluate(args, work):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
    from manifest import require_reference_environment
    require_reference_environment()
    import torch
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from transformers import Lfm2VlImageProcessor, Lfm2VlProcessor, PreTrainedTokenizerFast
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = [json.loads(line) for line in args.requests.read_text("utf-8").splitlines()]
    if len(cases) != 2 or any(len(case["request"]["questions"]) != 3 for case in cases):
        raise ValueError("oracle qualification requires two frozen cases/three questions")
    lock = json.loads(args.lock.read_text("utf-8"))
    pins = {pin["file"]: pin for pin in lock["files"]}
    paths = [args.model, args.mmproj, args.tokenizer, args.processor]
    identities = []
    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        handles = []
        for path in paths:
            stream = stack.enter_context(retained(path))
            pin = pins[path.name]
            if os.fstat(stream.fileno()).st_size != pin["bytes"] or hash_stream(stream) != pin["sha256"]:
                raise ValueError(f"retained reference input differs: {path.name}")
            handles.append(stream)
            identities.append({"path": str(path.resolve()), "bytes": pin["bytes"], "sha256": pin["sha256"]})
        work["inputs"] = identities
        config_pin = next(pin for pin in lock["reference_files"] if pin["file"] == "config.json")
        if hashlib.sha256(args.config.read_bytes()).hexdigest() != config_pin["sha256"]:
            raise ValueError("reference config differs from pinned d1 config")
        model, inventory = load_model(args.config, [(handles[0], text_name), (handles[1], vision_name)])
        write(args.out / "admission.json", {"inputs": identities, "inventory": inventory, "dtype": "f32", "device": "cpu", "load_seconds": time.perf_counter() - started})
        raw_tokenizer = Tokenizer.from_file(str(args.tokenizer))
        config = model.config
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
            bos_token=raw_tokenizer.id_to_token(config.bos_token_id),
            eos_token=raw_tokenizer.id_to_token(config.eos_token_id),
            pad_token=raw_tokenizer.id_to_token(config.pad_token_id))
        image_processor = Lfm2VlImageProcessor(**json.loads(args.processor.read_text("utf-8"))["image_processor"])
        processor = Lfm2VlProcessor(image_processor=image_processor, tokenizer=tokenizer)
        rows = []
        max_logit_error = 0.
        minimum_feature_cosine = 1.
        with torch.inference_mode():
            for index, case in enumerate(cases):
                root = args.out / f"case-{index:03}"
                root.mkdir()
                candidate_root = args.candidate / f"case-{index:03}"
                plans = json.loads((candidate_root / "plans.json").read_text("utf-8"))["plans"]
                images = [Image.open(path).convert("RGB") for path in case["images"]]
                if any(image.width * image.height > 1024 * 1024 for image in images):
                    raise ValueError("frozen qualification image exceeds reference pixel cap")
                features = None
                batch = None
                if images:
                    batch = processor(text=plans[0]["text"], images=images, return_tensors="pt")
                    processed = {"pixels": batch["pixel_values"], "mask": batch["pixel_attention_mask"], "spatial": batch["spatial_shapes"]}
                    observed = load_file(str(candidate_root / "processor.safetensors"))
                    for name, tensor in processed.items():
                        if tuple(tensor.shape) != tuple(observed[name].shape) or not torch.allclose(tensor.float(), observed[name].float(), rtol=0., atol=2e-5 if name == "pixels" else 0.):
                            raise ValueError(f"processor tensor mismatch {name}")
                    save_file({key: value.contiguous() for key, value in processed.items()}, str(root / "processor.safetensors"))
                    work["vision_forwards"] += 1
                    features = torch.cat(model.get_image_features(pixel_values=batch["pixel_values"], pixel_attention_mask=batch["pixel_attention_mask"], spatial_shapes=batch["spatial_shapes"]).pooler_output, dim=0)
                    save_file({"features": features.contiguous()}, str(root / "features.safetensors"))
                    observed = load_file(str(candidate_root / "features.safetensors"))["features"]
                    if tuple(features.shape) != tuple(observed.shape):
                        raise ValueError("projected feature shape mismatch")
                    cosine = torch.nn.functional.cosine_similarity(features.flatten().double(), observed.flatten().double(), dim=0).item()
                    minimum_feature_cosine = min(minimum_feature_cosine, cosine)
                errors = []
                for i, plan in enumerate(plans):
                    prepared = processor(text=plan["text"], images=images, return_tensors="pt") if images else tokenizer(plan["text"], add_special_tokens=False, return_tensors="pt")
                    ids = prepared["input_ids"]
                    candidate = json.loads((candidate_root / f"question-{i}.json").read_text("utf-8"))
                    if ids[0].tolist() != candidate["input_ids"]:
                        raise ValueError("independent expanded token sequence differs")
                    embeddings = model.get_input_embeddings()(ids)
                    if features is not None:
                        mask = (ids == config.image_token_id).unsqueeze(-1).expand_as(embeddings)
                        if mask.sum().item() != features.numel():
                            raise ValueError("independent image slots/features disagree")
                        embeddings = embeddings.masked_scatter(mask, features)
                    work["language_forwards"] += 1
                    language = model.model.language_model(inputs_embeds=embeddings, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                    values = model.lm_head(language.last_hidden_state[:, -1:])[0, 0].float()
                    save_file({"logits": values.contiguous()}, str(root / f"logits-{i}.safetensors"))
                    observed = load_file(str(candidate_root / f"logits-{i}.safetensors"))["logits"]
                    error = (values - observed).abs().max().item()
                    max_logit_error = max(max_logit_error, error)
                    errors.append(error)
                    print(json.dumps({"id": case["request_id"], "question": plan["name"], "last_logits_max_abs": error}), flush=True)
                rows.append({"id": case["request_id"], "language_forwards": 3, "vision_forwards": int(bool(images)), "logit_errors": errors})
        for stream, path, pin in zip(handles, paths, [pins[path.name] for path in paths]):
            if hash_stream(stream) != pin["sha256"]:
                raise ValueError(f"reference source changed after execution: {path.name}")
        work["inputs_reverified_after_execution"] = True
        passed = max_logit_error <= 0.02 and minimum_feature_cosine >= 0.9999
        write(args.out / "report.json", {"schema": "candle.d1_cpu_oracle.v1", "passed": passed, "language_forwards": work["language_forwards"],
            "vision_forwards": work["vision_forwards"], "max_logit_error": max_logit_error, "last_logit_bound": 0.02,
            "minimum_feature_cosine": minimum_feature_cosine, "feature_cosine_bound": 0.9999, "rows": rows,
            "reference": "pinned Transformers CPU F32 with dequantized retained GGUF operands", "elapsed_seconds": time.perf_counter() - started})
        if not passed:
            raise ValueError("independent CPU parity exceeds the frozen comparison bounds")


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ["model", "mmproj", "tokenizer", "processor", "config", "lock", "requests", "candidate", "out"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    work = dict(schema="candle.d1_cpu_oracle_work.v1", language_forwards=0, vision_forwards=0,
                output_tokens=0, inputs_reverified_after_execution=False, status="running",
                python=sys.version, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                operand_reader_sha256=hashlib.sha256(Path(__file__).with_name("oracle_weights.py").read_bytes()).hexdigest())
    started = time.perf_counter()
    try:
        evaluate(args, work)
        work["status"] = "complete"
    except Exception as error:
        work["status"] = "failed"
        work["error"] = str(error)
        raise
    finally:
        work["elapsed_seconds"] = time.perf_counter() - started
        write(args.out / "work.json", work)


if __name__ == "__main__":
    main()
