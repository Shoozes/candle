"""Export model-free fixtures from the pinned, local Transformers source tree."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REVISION = "ddb849abe009d1089e6c691bfc897f27211c663c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_source.resolve()))
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM
    import importlib.metadata

    if transformers.__version__ != "5.12.1":
        raise ValueError("Expected the complete pinned Transformers 5.12.1 source")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for label, hidden, intermediate, layers, experts, topk, vocab in [
        ("native", 16, 24, 4, 4, 2, 32),
        ("quantized", 256, 256, 2, 2, 1, 8),
    ]:
        torch.manual_seed(371)
        cfg = Lfm2MoeConfig(
            vocab_size=vocab, hidden_size=hidden, intermediate_size=intermediate,
            moe_intermediate_size=intermediate, num_hidden_layers=layers,
            num_attention_heads=2 if hidden == 16 else 4,
            num_key_value_heads=1, num_dense_layers=1, num_experts=experts,
            num_experts_per_tok=topk, max_position_embeddings=64,
            layer_types=["conv", "full_attention"] * (layers // 2),
            conv_L_cache=3, conv_bias=False, tie_word_embeddings=True,
            use_expert_bias=True, norm_topk_prob=True, routed_scaling_factor=1.,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.},
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
            attn_implementation="eager",
        )
        cfg._experts_implementation = "eager"
        model = Lfm2MoeForCausalLM(cfg).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.copy_(torch.ones_like(parameter) + torch.randn_like(parameter) * 0.02)
                else:
                    parameter.copy_(torch.randn_like(parameter) * (0.12 if hidden == 16 else 0.02))
            for layer in model.model.layers:
                if hasattr(layer.feed_forward, "expert_bias"):
                    layer.feed_forward.expert_bias.copy_(torch.linspace(-0.07, 0.09, experts))

        tensors = {}
        for name, value in model.state_dict().items():
            if name == "lm_head.weight":
                continue
            if name.endswith("experts.gate_up_proj"):
                prefix = name.removesuffix("gate_up_proj")
                for expert in range(experts):
                    tensors[f"{prefix}{expert}.w1.weight"] = value[expert, :intermediate].clone()
                    tensors[f"{prefix}{expert}.w3.weight"] = value[expert, intermediate:].clone()
            elif name.endswith("experts.down_proj"):
                prefix = name.removesuffix("down_proj")
                for expert in range(experts):
                    tensors[f"{prefix}{expert}.w2.weight"] = value[expert].clone()
            else:
                tensors[name] = value.clone().contiguous()

        ids = torch.tensor([[1, 4, 3, 5]], dtype=torch.long)
        decode = torch.tensor([[6]], dtype=torch.long)
        with torch.no_grad():
            full = model(ids, use_cache=False, output_hidden_states=True)
            pref = model(ids, use_cache=True)
            dec = model(decode, past_key_values=pref.past_key_values, use_cache=True)
            tensors["oracle.ids"] = ids.to(torch.int64)
            tensors["oracle.decode_ids"] = decode.to(torch.int64)
            tensors["oracle.hidden"] = full.hidden_states[-1].clone()
            tensors["oracle.logits"] = full.logits.clone()
            tensors["oracle.decode_logits"] = dec.logits.clone()
            router = model.model.layers[1].feed_forward
            raw = torch.linspace(-2., 2., 3 * experts).reshape(3, experts)
            tensors["oracle.router_logits"] = raw
            tensors["oracle.router_bias"] = router.expert_bias.clone()
            for bias in [False, True]:
                for norm in [False, True]:
                    router.use_expert_bias = bias
                    router.norm_topk_prob = norm
                    router.routed_scaling_factor = 1.7
                    selected, weights = router.route_tokens_to_experts(raw)
                    key = f"oracle.route_{int(bias)}_{int(norm)}"
                    tensors[key + ".ids"] = selected.to(torch.int64)
                    tensors[key + ".weights"] = weights

        config_path = args.output / f"{label}.json"
        config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8", newline="\n")
        tensor_path = args.output / f"{label}.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tensor_path))
        for path in [config_path, tensor_path]:
            data = path.read_bytes()
            inventory.append({"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    reference_files = []
    for name in ["configuration_lfm2_moe.py", "modeling_lfm2_moe.py"]:
        path = args.reference_source / "transformers/models/lfm2_moe" / name
        reference_files.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "schema": "candle.lfm2-moe.fixture.v1", "kind": "tiny-random-no-production-weights",
        "transformers_revision": REVISION, "reference_files": reference_files,
        "environment": {n: importlib.metadata.version(n) for n in ["torch", "safetensors", "tokenizers", "huggingface-hub"]},
        "transformers_loaded_version": transformers.__version__, "device": "cpu", "dtype": "float32",
        "attention_implementation": "eager", "experts_implementation": "eager",
        "seed": 371, "files": inventory, "normalization": "packed Transformers experts exported to published per-expert w1/w2/w3 names",
        "tolerance": {"atol": 1e-5, "rtol": 1e-4},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "passed", "files": len(inventory), "tensor_bytes": sum(f["bytes"] for f in inventory), "reference": REVISION}))


if __name__ == "__main__":
    main()
