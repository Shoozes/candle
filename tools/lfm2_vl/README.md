# LFM2-VL Tools

This directory is reserved for reference and fixture tooling described by the LFM2.5-VL execution plan. Bootstrap Phase adds documentation only; it does not add a model downloader, reference exporter, runtime dependency, or production checkpoint.

Future tools must keep these boundaries:

- Configuration-only inspection must not download weights.
- Production downloads must be explicit, recorded, and kept outside Git.
- Tiny deterministic fixtures may be committed under `tests/fixtures/lfm2_vl_tiny/`.
- Local caches and reference outputs belong to ignored paths.
- Reference revisions, package versions, image hashes, dtype, device, and seed must be recorded with generated outputs.

The first tool implementation task is the reference-source lock and fixture-harness phase, after the Linux-home baseline replay is recorded as accepted.
