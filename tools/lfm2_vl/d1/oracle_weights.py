"""Read retained Q8 GGUF operands into the independent CPU reference graph.

No weight file is written. Source floating matrices keep their original values.
"""
from __future__ import annotations

import math
import re
import struct


def directory(stream):
    def read(size):
        data = stream.read(size)
        if len(data) != size or stream.tell() > 32 * 1024 * 1024:
            raise ValueError("truncated/oversized GGUF header")
        return data
    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]
    def string():
        size = number("Q")
        if size > 1024 * 1024:
            raise ValueError("oversized GGUF string")
        return read(size).decode("utf-8")
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def value(kind, depth=0):
        if depth > 8:
            raise ValueError("nested GGUF metadata")
        if kind in scalar:
            return number(scalar[kind])
        if kind == 8:
            return string()
        if kind == 9:
            element, count = number("I"), number("Q")
            if count > 1024 * 1024:
                raise ValueError("oversized GGUF array")
            for _ in range(count):
                value(element, depth + 1)
            return None
        raise ValueError(f"unsupported GGUF metadata type {kind}")
    stream.seek(0)
    if read(4) != b"GGUF" or number("I") not in (2, 3):
        raise ValueError("expected GGUF v2/v3")
    count, metadata_count = number("Q"), number("Q")
    if not 0 < count <= 4096 or metadata_count > 8192:
        raise ValueError("GGUF directory exceeds bounds")
    alignment = 32
    for _ in range(metadata_count):
        name, kind = string(), number("I")
        result = value(kind)
        if name == "general.alignment":
            alignment = result
    if not isinstance(alignment, int) or not 0 < alignment <= 4096:
        raise ValueError("invalid GGUF alignment")
    tensors = {}
    for _ in range(count):
        name, rank = string(), number("I")
        if name in tensors or not 0 < rank <= 4:
            raise ValueError("duplicate/rank-invalid GGUF tensor")
        shape = tuple(reversed([number("Q") for _ in range(rank)]))
        if any(not 0 < dim <= 1024 * 1024 for dim in shape):
            raise ValueError("GGUF dimension outside bounds")
        kind, offset = number("I"), number("Q")
        elements = math.prod(shape)
        if kind == 8:
            if elements % 32:
                raise ValueError("Q8 tensor has an incomplete block")
            size = elements // 32 * 34
        elif kind in (0, 1, 30):
            size = elements * (4 if kind == 0 else 2)
        else:
            raise ValueError(f"reference only admits F32/F16/BF16/Q8_0, got {kind}")
        tensors[name] = {"shape": shape, "kind": kind, "relative": offset, "size": size}
    start = (stream.tell() + alignment - 1) // alignment * alignment
    stream.seek(0, 2)
    length = stream.tell()
    end = start
    for tensor in sorted(tensors.values(), key=lambda tensor: tensor["relative"]):
        tensor["offset"] = start + tensor["relative"]
        if tensor["offset"] < end or tensor["offset"] + tensor["size"] > length:
            raise ValueError("overlapping/out-of-file GGUF tensor")
        end = tensor["offset"] + tensor["size"]
    return tensors


def decode(mapped, info):
    import numpy as np
    elements = math.prod(info["shape"])
    kind, offset = info["kind"], info["offset"]
    if kind == 8:
        dtype = np.dtype([("scale", "<f2"), ("codes", "i1", (32,))])
        blocks = np.frombuffer(mapped, dtype=dtype, count=elements // 32, offset=offset)
        values = blocks["codes"].astype(np.float32)
        values *= blocks["scale"].astype(np.float32)[:, None]
    elif kind == 30:
        values = (np.frombuffer(mapped, dtype="<u2", count=elements, offset=offset).astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.frombuffer(mapped, dtype="<f4" if kind == 0 else "<f2", count=elements, offset=offset).astype(np.float32, copy=True)
    return values.reshape(info["shape"])


def text_name(name):
    root = "model.language_model."
    direct = {"token_embd.weight": root + "embed_tokens.weight", "output_norm.weight": root + "embedding_norm.weight", "token_embd_norm.weight": root + "embedding_norm.weight", "output.weight": "lm_head.weight"}
    if name in direct:
        return direct[name]
    match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"unknown text tensor {name}")
    layer, tail = match.groups()
    mappings = {"attn_norm.weight": "operator_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
                "ffn_gate.weight": "feed_forward.w1.weight", "ffn_down.weight": "feed_forward.w2.weight", "ffn_up.weight": "feed_forward.w3.weight",
                "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.out_proj.weight", "attn_q_norm.weight": "self_attn.q_layernorm.weight", "attn_k_norm.weight": "self_attn.k_layernorm.weight"}
    if tail.startswith("shortconv."):
        tail = "conv." + tail.removeprefix("shortconv.")
    else:
        tail = mappings[tail]
    return f"{root}layers.{layer}.{tail}"


def vision_name(name):
    root = "model.vision_tower."
    direct = {"v.patch_embd": "embeddings.patch_embedding", "v.position_embd": "embeddings.position_embedding", "v.post_ln": "post_layernorm"}
    base, suffix = name.rsplit(".", 1)
    if base in direct:
        return root + direct[base] + "." + suffix
    if base in {"mm.1", "mm.2", "mm.input_norm"}:
        mapped = {"mm.1": "linear_1", "mm.2": "linear_2", "mm.input_norm": "layer_norm"}[base]
        return f"model.multi_modal_projector.{mapped}.{suffix}"
    match = re.fullmatch(r"v\.blk\.(\d+)\.(.+)", base)
    if not match:
        raise ValueError(f"unknown vision tensor {name}")
    layer, tail = match.groups()
    mappings = {"ln1": "layer_norm1", "ln2": "layer_norm2", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
                "attn_v": "self_attn.v_proj", "attn_out": "self_attn.out_proj", "ffn_up": "mlp.fc1", "ffn_down": "mlp.fc2"}
    return f"{root}encoder.layers.{layer}.{mappings[tail]}.{suffix}"
