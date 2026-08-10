# LFM2-VL Tiny Fixture

No fixture tensors are present in Bootstrap Phase.

The later reference-harness phase may add a small deterministic model, inputs, metadata, and golden tensors here. Those files must preserve the real operation classes needed for LFM2-VL parity while remaining suitable for fast CPU verification.

Do not place production model weights, Hugging Face caches, access tokens, or generated runtime output in this directory. Production files remain outside Git, and local reference outputs use ignored paths.
