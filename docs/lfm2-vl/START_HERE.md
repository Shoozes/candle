# Candle 0.11 LFM2.5-VL/MMProj: Blank Folder to First Proven Checkpoint

These instructions are the execution companion to the full specification already written. The specification defines what we are building. This document defines how we start, how Codex should work, which files it may touch, what each phase must prove, and when we stop before moving forward.

The working decision is:

```text
Base: Hugging Face Candle 0.11.0
Repository type: direct Candle fork
Primary environment: WSL2
Initial backend: CPU F32
First model target: LFM2.5-VL-450M
First weight format: native safetensors
Second format: quantized GGUF text plus dense split mmproj
Third format: direct GGUF text plus GGUF mmproj
```

Do not begin by creating a generic VLM framework. Do not begin with CUDA. Do not begin with GGUF mmproj. Do not ask Codex to implement the entire specification in one turn.

---

# 1. Prepare the WSL2 development environment

Use WSL2 rather than a Windows-mounted source tree. Keep the repository under the Linux home directory, such as `~/code/candle-lfm2-vl`, rather than under `/mnt/c` or `/mnt/d`. OpenAI’s current Codex guidance recommends this layout for better filesystem behavior and fewer permission or symlink problems. ([OpenAI Developers][1])

From PowerShell:

```powershell
wsl
```

From the WSL shell:

```bash
mkdir -p ~/code
cd ~/code
```

Install basic build dependencies:

```bash
sudo apt update

sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    ninja-build \
    git \
    curl \
    python3 \
    python3-venv \
    python3-pip
```

Inspect the existing toolchain before changing it:

```bash
git --version
rustup show
rustc --version
cargo --version
python3 --version
cmake --version
ninja --version
```

Do not install or enable Candle CUDA yet. The baseline and initial parity work must compile and run without CUDA, NVIDIA SDKs, or GPU libraries.

## Storage gate

Our Windows system drive has been tight in previous diagnostics. Keep source and normal Rust builds inside WSL, but do not begin production-model downloads until storage has been checked.

In WSL:

```bash
df -h ~
df -h /mnt/d 2>/dev/null || true
```

In PowerShell:

```powershell
Get-PSDrive C,D | Select-Object Name,Used,Free
```

Before downloading both production checkpoints and keeping multiple reference formats, reserve substantial free space. Large Hugging Face caches can be placed on `D:` while the source remains in WSL:

```bash
mkdir -p /mnt/d/ai-cache/huggingface
mkdir -p /mnt/d/ai-models/lfm2-vl

export HF_HOME=/mnt/d/ai-cache/huggingface
export CANDLE_LFM2_VL_MODEL_DIR=/mnt/d/ai-models/lfm2-vl
```

Put those exports in a project-local, ignored environment file later. Do not commit tokens or credentials.

---

# 2. Install and start Codex

The current Codex CLI installer for Linux and WSL is:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Verify it:

```bash
codex --version
which codex
```

Start Codex once and sign in using our ChatGPT account:

```bash
codex
```

The current CLI provides `/status`, `/permissions`, `/model`, `/init`, and `/review`. OpenAI also recommends Git checkpoints before and after focused tasks. ([OpenAI Developers][2])

Exit Codex after authentication. We will create the repository and its instructions before starting the first actual coding session.

---

# 3. Create the Candle 0.11 working repository

Create a direct checkout of Candle 0.11.0 in the blank folder:

```bash
mkdir -p ~/code/candle-lfm2-vl
cd ~/code/candle-lfm2-vl

git clone \
    --branch 0.11.0 \
    --single-branch \
    https://github.com/huggingface/candle.git \
    .
```

Create our working branch:

```bash
git switch -c feat/lfm2-vl-mmproj
```

Rename the upstream remote:

```bash
git remote rename origin upstream
```

Mark the untouched baseline:

```bash
git tag -a lfm2-vl-baseline-candle-0.11.0 \
    -m "Untouched Candle 0.11.0 baseline for LFM2.5-VL development"
```

Verify:

```bash
git remote -v
git branch --show-current
git describe --tags --always
git status --short
```

Expected state:

```text
branch: feat/lfm2-vl-mmproj
working tree: clean
upstream: huggingface/candle
baseline tag: lfm2-vl-baseline-candle-0.11.0
```

When our GitHub fork exists, add it as `origin`:

```bash
git remote add origin git@github.com:<our-account>/candle.git
```

Do not push yet. Establish the local baseline and documentation first.

---

# 4. Add the specification and execution document

Create the project documentation directory:

```bash
mkdir -p docs/lfm2-vl
```

Save the previous full specification as:

```text
docs/lfm2-vl/SPEC.md
```

Save this execution document as:

```text
docs/lfm2-vl/START_HERE.md
```

The source tree should now contain:

```text
docs/
  lfm2-vl/
    SPEC.md
    START_HERE.md
```

Do not place the full specification inside `AGENTS.md`. Codex project instruction files have a default combined size limit of 32 KiB, and project-root instructions should remain compact and operational. Codex reads root and nested `AGENTS.md` files at session start, with more local instructions taking precedence. ([OpenAI Developers][3])

---

# 5. Create the root `AGENTS.md`

First inspect whether the upstream checkout already contains one:

```bash
ls -la AGENTS.md AGENTS.override.md 2>/dev/null || true
```

If Candle 0.11 already contains an `AGENTS.md`, preserve its instructions and append the project-specific section below. Do not replace upstream instructions blindly.

Create or update:

## File: `AGENTS.md`

````markdown
# Project Instructions

## Mission

Extend Candle 0.11.0 with correct, tested, config-driven LFM2.5-VL support, including:

- LFM2.5 text-model compatibility.
- Embedding-driven LFM2 prefill.
- SigLIP2 NaFlex packed-patch vision encoding.
- LFM2-VL pixel-unshuffle multimodal projection.
- Image placeholder expansion and embedding replacement.
- Native safetensors loading.
- Quantized GGUF text with split dense mmproj.
- Direct llama.cpp-compatible GGUF mmproj loading.
- CPU and CUDA verification after CPU parity is proven.

The complete technical specification is:

`docs/lfm2-vl/SPEC.md`

The execution sequence is:

`docs/lfm2-vl/START_HERE.md`

Read both before planning or editing.

## Authority Order

When sources disagree, use this order:

1. The pinned official Hugging Face Transformers implementation.
2. The pinned official LiquidAI model and processor files.
3. Golden tensor fixtures generated from the pinned reference environment.
4. Existing Candle 0.11 architecture and compatibility requirements.
5. mistral.rs as the primary Rust implementation reference.
6. llama.cpp as the GGUF, preprocessing, and independent parity reference.
7. MLX-VLM and Transformers.js as secondary independent references.
8. The written specification.

Document every material conflict in `docs/lfm2-vl/DECISIONS.md`.

## Baseline

- Base revision: Candle 0.11.0.
- Working branch: `feat/lfm2-vl-mmproj`.
- First checkpoint: `LiquidAI/LFM2.5-VL-450M`.
- Second checkpoint: `LiquidAI/LFM2.5-VL-1.6B`.
- First backend: CPU F32.
- CUDA is optional until CPU parity passes.
- Native safetensors precede all GGUF work.
- Text-only LFM2 compatibility must remain intact.

## Required Execution Order

Work only in this sequence:

1. Bootstrap and baseline verification.
2. Reference-source lock and fixture harness.
3. LFM2 text configuration and `forward_embeds`.
4. SigLIP2 NaFlex from preprocessed tensors.
5. Pixel unshuffle, projector, and composite native model.
6. Rust image processor and prompt expansion.
7. Quantized text plus split dense mmproj.
8. Direct GGUF mmproj loading.
9. Quantized mmproj execution.
10. CUDA optimization and broader stabilization.

Do not implement later phases early.

## Engineering Rules

- Prefer the smallest correct change over broad architectural rewrites.
- Do not create a generic VLM framework before LFM2-VL works.
- Do not hardcode checkpoint names, hidden widths, token counts, layer counts, or image dimensions.
- Derive model behavior from normalized configuration.
- Use checked arithmetic for external dimensions and allocations.
- Return actionable errors for malformed images, configs, weights, and token spans.
- Do not silently truncate, pad, duplicate, or discard image features to force a count match.
- Do not silently fall back to text-only behavior.
- Do not use `unwrap`, `expect`, or unchecked indexing on external input paths.
- Preserve current Candle public behavior unless an explicit compatibility change is documented.
- Keep CPU builds functional without CUDA, TensorRT, or NVIDIA libraries.
- Add optional accelerator behavior behind existing or explicit feature gates.
- Do not add production dependencies without documenting why they are required.
- No placeholder implementations, fake output, skeleton production paths, or unproven success claims.
- Generated captions are not proof. Component tensor parity is required.
- Keep comments short and aligned with actual behavior.
- Keep source files focused. Split modules when responsibilities become distinct.
- Do not modify unrelated Candle architectures.
- Do not reformat unrelated code.

## Source and Licensing Rules

- Treat external implementations primarily as references.
- Prefer a fresh Candle-native implementation over copying large code blocks.
- When code is directly adapted, preserve required copyright and license notices.
- Record source repository, file, commit, and license in `docs/lfm2-vl/SOURCES.md`.
- Pin external revisions before using them as parity authorities.
- Never use an unpinned moving branch as a golden reference.

## Workflow Before Editing

Before every task:

1. Read `AGENTS.md`.
2. Read `docs/lfm2-vl/SPEC.md`.
3. Read `docs/lfm2-vl/STATUS.md`.
4. Inspect the relevant current Candle code.
5. State the task boundary and expected files.
6. Run the narrowest existing verification command that establishes the starting state.
7. Identify the exact acceptance gate.

When a safe ambiguity exists, choose the simplest path consistent with the specification and record the decision. Do not stop for cosmetic or naming ambiguity.

## Workflow After Editing

After every task:

1. Run `cargo fmt --all -- --check`.
2. Run targeted tests for the changed module.
3. Run targeted `cargo check` for affected crates and examples.
4. Run `git diff --check`.
5. Inspect the complete diff.
6. Update `docs/lfm2-vl/STATUS.md`.
7. Update `docs/lfm2-vl/DECISIONS.md` when architecture or compatibility decisions changed.
8. Report exact commands, pass/fail status, blockers, and remaining work.

Do not report a test as passing unless it was executed in the current task.

## Verification Policy

Use focused verification during development and broader verification at phase gates.

Minimum Rust checks:

```bash
cargo fmt --all -- --check
cargo check --locked -p candle-core
cargo check --locked -p candle-nn
cargo check --locked -p candle-transformers
git diff --check
````

At relevant gates also check:

```bash
cargo check --locked -p candle-examples --example lfm2
cargo check --locked -p candle-examples --example quantized-lfm2
```

Do not hide pre-existing failures. Record them separately from failures caused by the current change.

## Model and Fixture Rules

* Do not commit production model weights.
* Do not commit Hugging Face caches.
* Do not download full production checkpoints during bootstrap.
* Every production download must record repository, revision, filename, size, and hash.
* Tiny deterministic fixtures belong under `tests/fixtures/lfm2_vl_tiny/`.
* Generated runtime output belongs under ignored `artifacts/`.
* Reference manifests must record package versions, model revision, processor revision, dtype, device, seed, and source image hash.

## Git Rules

* Never use destructive Git commands unless explicitly instructed.
* Do not reset, clean, force-push, rewrite, or discard unrelated work.
* Do not commit unless the current task explicitly requests a commit.
* Keep commits limited to one proven responsibility.
* Do not open a pull request during early implementation phases.
* Create a checkpoint commit after every green phase gate.
* Review staged files before committing.

## Codex Task Scope

One Codex task should prove one focused result.

Good task:

* Normalize LFM2.5 feed-forward dimensions and add tests.

Bad task:

* Implement all LFM2-VL and GGUF support.

Subagents may perform read-only source comparison or test planning. Do not allow multiple agents to edit overlapping files concurrently.

## Status Handoff

`docs/lfm2-vl/STATUS.md` must always state:

* Current phase.
* Current baseline commit.
* Last green verification.
* Files currently under active work.
* Proven behavior.
* Known failures.
* Blockers.
* Exact next task.

The next Codex session must be able to continue from this file without reconstructing project history from chat logs.

````

Commit no code yet.

---

# 6. Start the first Codex session

From the repository root:

```bash
cd ~/code/candle-lfm2-vl
codex
````

Inside Codex, inspect the session:

```text
/status
```

Set the model:

```text
/model
```

Use the strongest available coding model with high reasoning for architecture work. Medium reasoning is sufficient for mechanical documentation and script tasks.

Set permissions:

```text
/permissions
```

Recommended initial policy:

```text
Read repository: allowed
Write inside repository: allowed
Run local build and test commands: allowed
Network access: ask
Write outside repository: ask
Destructive commands: deny
```

Do not run `/init`; we already created a more exact `AGENTS.md`.

---

# 7. First Codex prompt: bootstrap only

Paste this as the first task:

```text
Read these files completely before doing anything:

- AGENTS.md
- docs/lfm2-vl/SPEC.md
- docs/lfm2-vl/START_HERE.md

This is Bootstrap Phase only.

Do not implement or modify LFM2, SigLIP2, LFM2-VL, projector, tokenizer, image processor, GGUF, or Candle runtime code.

Goals:

1. Inspect the Candle 0.11.0 workspace and confirm actual package and example names.
2. Create the minimal LFM2-VL project-control structure.
3. Add reproducible CPU-only baseline verification.
4. Record the untouched baseline state.
5. Leave the repository ready for the source-lock phase.

Create:

- docs/lfm2-vl/STATUS.md
- docs/lfm2-vl/DECISIONS.md
- docs/lfm2-vl/SOURCES.md
- docs/lfm2-vl/PARITY.md
- docs/lfm2-vl/TENSOR_MAP.md
- docs/lfm2-vl/FAILURE_LOG.md
- scripts/lfm2-vl/env-report.sh
- scripts/lfm2-vl/verify-baseline.sh
- tools/lfm2_vl/README.md
- tests/fixtures/lfm2_vl_tiny/README.md

Add only necessary project-specific ignore patterns to the existing .gitignore:

- .venv/
- artifacts/
- downloads/
- models/
- local Hugging Face caches
- local reference outputs

Do not ignore committed tiny fixture files generally.

The baseline verification script must:

- use `set -euo pipefail`
- run from the repository root regardless of its invocation directory
- run `cargo fmt --all -- --check`
- run locked CPU-only checks for candle-core, candle-nn, and candle-transformers
- check the existing lfm2 example
- check the existing quantized-lfm2 example
- run `git diff --check`
- not download models
- not enable CUDA

The environment report must not dump secrets or the entire environment. It should record:

- UTC timestamp
- OS and kernel
- WSL distribution
- Git version
- current commit
- branch
- Rust compiler and Cargo versions
- Python version
- CMake and Ninja versions
- optional NVIDIA driver summary when available
- disk free-space summary

Create initial decisions covering:

- direct Candle 0.11.0 fork
- WSL2-first development
- CPU F32 parity before CUDA
- 450M before 1.6B
- native safetensors before GGUF
- production model files excluded from Git

Run the baseline verification script.

Update STATUS.md with exact command results.

Do not commit.

At completion report:

- files created or changed
- commands run
- exact pass/fail results
- any pre-existing baseline failure
- whether the repository is ready for source locking
```

Codex should finish this task without modifying Candle source code.

---

# 8. Review the bootstrap result

Inside Codex:

```text
/review
```

The current Codex review workflow can inspect uncommitted changes without modifying the working tree. ([OpenAI Developers][2])

Outside Codex, inspect manually:

```bash
git status --short
git diff --stat
git diff -- .gitignore AGENTS.md docs/lfm2-vl scripts/lfm2-vl tools/lfm2_vl tests/fixtures/lfm2_vl_tiny
```

Run the scripts ourselves:

```bash
bash scripts/lfm2-vl/env-report.sh
bash scripts/lfm2-vl/verify-baseline.sh
```

Confirm that no Candle source file changed:

```bash
git diff --name-only | grep -E \
    '^(candle-core|candle-nn|candle-transformers|candle-examples)/' \
    && echo "Unexpected source change" \
    || echo "Bootstrap source scope is clean"
```

Commit the bootstrap:

```bash
git add \
    AGENTS.md \
    .gitignore \
    docs/lfm2-vl \
    scripts/lfm2-vl \
    tools/lfm2_vl \
    tests/fixtures/lfm2_vl_tiny

git diff --cached --stat
git diff --cached --check

git commit -m "chore: establish LFM2.5-VL implementation baseline"
```

Tag it:

```bash
git tag -a lfm2-vl-phase-0-bootstrap \
    -m "Candle 0.11 LFM2.5-VL bootstrap and baseline verification"
```

Push only after the commit has been reviewed:

```bash
git push -u origin feat/lfm2-vl-mmproj
git push origin lfm2-vl-phase-0-bootstrap
```

---

# 9. Source-lock phase

This phase records exactly which external revisions define correct behavior. It changes documentation and reference metadata only.

Start a new Codex session with live search available:

```bash
codex --search
```

The current CLI supports `--search` when work depends on current external documentation or source behavior. ([OpenAI Developers][2])

Paste:

```text
Read:

- AGENTS.md
- docs/lfm2-vl/SPEC.md
- docs/lfm2-vl/START_HERE.md
- docs/lfm2-vl/STATUS.md

Perform Source Lock Phase only.

Do not change Candle Rust source code.
Do not download full model weights.
Do not add runtime dependencies.

Inspect and pin authoritative revisions for:

1. Hugging Face Transformers:
   - LFM2 configuration and modeling
   - LFM2-VL configuration, modeling, processing, and image processing
   - SigLIP2 configuration and modeling

2. LiquidAI:
   - LFM2.5-VL-450M config
   - LFM2.5-VL-450M processor config
   - LFM2.5-VL-1.6B config
   - LFM2.5-VL-1.6B processor config
   - tokenizer and special-token files

3. mistral.rs:
   - LFM2 text model
   - LFM2-VL model
   - LFM2-VL vision encoder
   - LFM2-VL processor
   - relevant license

4. llama.cpp:
   - initial LFM2-VL support
   - LFM2-VL parity fixes
   - current LFM2 GGUF conversion
   - current SigLIP/LFM2 mmproj loader
   - GGUF metadata and tensor names

5. Secondary references:
   - MLX-VLM LFM2-VL
   - Transformers.js LFM2-VL

Update:

- docs/lfm2-vl/SOURCES.md
- docs/lfm2-vl/TENSOR_MAP.md
- docs/lfm2-vl/DECISIONS.md
- docs/lfm2-vl/STATUS.md

Create:

- tools/lfm2_vl/reference-lock.json
- docs/lfm2-vl/LICENSE_NOTES.md

For every pinned source record:

- repository
- exact commit or model revision
- path
- purpose
- authority level
- license
- whether implementation code may be adapted directly or only used as reference

TENSOR_MAP.md must compare at least:

- Hugging Face native names
- Candle target names
- llama.cpp GGUF names
- expected tensor shapes for 450M
- expected tensor shapes for 1.6B
- required orientation transforms

Do not include large copied source blocks.

Run documentation and JSON validation.
Run the baseline verification script.
Do not commit.

Report exact pins, unresolved conflicts, and the next required reference-harness task.
```

The most useful Rust donor remains mistral.rs, which already contains Candle-based LFM2-VL model, vision, projector, and processor modules. It should be treated as a porting reference, not imported wholesale because it is coupled to mistral.rs-specific loaders, caches, SDPA, quantization, and pipeline traits.

Review, then commit:

```bash
git add docs/lfm2-vl tools/lfm2_vl/reference-lock.json
git diff --cached --check
git commit -m "docs: lock LFM2.5-VL reference sources"
```

---

# 10. Build the reference harness before production Rust changes

The reference harness must be able to operate in three modes:

```text
config-only
tiny-random
production
```

`config-only` reads configs and validates normalized dimensions without downloading model weights.

`tiny-random` constructs a miniature architecture with deterministic random weights and exports committed golden fixtures.

`production` downloads and loads an actual model only when explicitly invoked.

Start a new Codex task:

```text
Read AGENTS.md, SPEC.md, START_HERE.md, STATUS.md, SOURCES.md, and reference-lock.json.

Implement Reference Harness Phase only.

Do not modify Candle Rust source code.

Create under tools/lfm2_vl/reference/:

- README.md
- requirements-reference.in
- requirements-reference.txt
- export_fixtures.py
- inspect_config.py
- manifest.py
- tensor_dump.py
- test_reference_tools.py

Requirements:

1. Pin exact Python package versions.
2. Record the pinned Transformers commit/version and model revisions.
3. Provide a config-only mode that does not download model weights.
4. Provide a deterministic tiny-random mode.
5. Keep production downloads opt-in.
6. Never store access tokens.
7. Export tensors with safetensors.
8. Export metadata with stable sorted JSON.
9. Record image hash, package versions, model revision, processor revision, device, dtype, and seed.
10. Reject an existing output directory unless `--overwrite` is supplied.

The tiny model must preserve the real operation classes:

- packed linear patch embedding
- resized learned position embeddings
- bidirectional masked attention
- post LayerNorm
- factor-2 pixel unshuffle
- optional projector LayerNorm
- two projector linear layers
- LFM2 image-placeholder replacement
- LFM2 attention and short-convolution layers

Use small deterministic dimensions suitable for CPU CI.

Add a local setup script or documented commands using `.venv`.

Do not generate or commit production checkpoint tensors.
Only tiny fixtures may become committed test assets.

Add smoke tests for config-only and tiny-random modes.

Run:
- Python tests
- baseline Rust verification
- JSON and safetensors fixture validation

Update STATUS.md.

Do not commit.

Report the exact reference environment and generated tiny fixture inventory.
```

After review:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel
python -m pip install -r tools/lfm2_vl/reference/requirements-reference.txt

python -m pytest tools/lfm2_vl/reference
```

Commit:

```bash
git add tools/lfm2_vl/reference tests/fixtures/lfm2_vl_tiny docs/lfm2-vl
git diff --cached --check
git commit -m "test: add deterministic LFM2-VL reference harness"
```

---

# 11. First actual implementation phase: repair text-only LFM2

This is the first production-code phase. It must remain text-only.

The acceptance target is:

```text
450M normalized FFN width = 4608
1.6B normalized FFN width = 8192
existing Candle LFM2 behavior remains intact
dense embedding-driven prefill works
quantized embedding-driven prefill works
```

Paste:

```text
Read all root instructions and LFM2-VL documents.

Implement only the LFM2 Text Compatibility Phase from the specification.

Do not add SigLIP2.
Do not add LFM2-VL.
Do not add image processing.
Do not add mmproj loading.
Do not add CUDA-specific behavior.

Required work:

1. Add exact tests for current LFM2.5 configuration normalization.
2. Prove the 450M effective feed-forward width is 4608.
3. Prove the 1.6B effective feed-forward width is 8192.
4. Parse current and legacy configuration aliases:
   - intermediate_size
   - block_ff_dim
   - block_auto_adjust_ff_dim
   - block_ffn_dim_multiplier
   - block_multiple_of
   - conv_L_cache / conv_l_cache
   - tie_word_embeddings / tie_embedding
   - rope_theta / rope_parameters
5. Correct effective FFN computation.
6. Split model-root construction so nested `model.language_model` weights can be loaded without adding the standalone `model` prefix twice.
7. Add dense APIs:
   - embed_tokens
   - forward_hidden or equivalent internal hidden-state path
   - project_logits
   - forward_embeds
8. Preserve the existing token-ID forward method.
9. Add equivalent quantized embedding-driven forwarding without changing current GGUF text behavior.
10. Add cache-clear coverage.
11. Keep all current examples compiling.

Tests must cover:

- standalone legacy LFM2 config
- LFM2.5-VL-450M text config
- LFM2.5-VL-1.6B text config
- tied output weights
- explicit lm_head
- nested model-root loading
- token-ID forward versus embedding-driven forward equivalence
- prefill followed by one-token decode
- cache reset determinism

Run:

- cargo fmt
- targeted LFM2 tests
- candle-transformers check
- lfm2 example check
- quantized-lfm2 example check
- baseline verification script
- git diff --check

Update STATUS.md and DECISIONS.md.

Do not commit.

Report:

- files changed
- old versus new FFN calculations
- compatibility behavior
- exact tests run
- remaining text-only gaps
```

Review closely. This task touches the foundation used by all later multimodal work.

Recommended commit:

```bash
git add candle-transformers candle-examples docs/lfm2-vl
git diff --cached --check
git commit -m "fix(lfm2): support current LFM2.5 configs and input embeddings"
```

Tag the first real milestone:

```bash
git tag -a lfm2-vl-phase-1-text \
    -m "LFM2.5 text configuration and embedding-forward parity"
```

## First mandatory stop point

Stop here and assess before adding vision.

The repository is ready for SigLIP2 only when:

```text
baseline script passes
450M FFN test passes at 4608
1.6B FFN test passes at 8192
legacy LFM2 example still compiles
quantized LFM2 example still compiles
token-ID and embedding-driven forwarding agree
incremental decode works
cache reset works
STATUS.md contains no unresolved text-model blocker
```

Do not continue merely because the code compiles.

---

# 12. SigLIP2 NaFlex phase

This phase accepts already-preprocessed packed tensors. Raw image processing remains out of scope.

Paste:

```text
Implement only the SigLIP2 NaFlex Tensor Phase.

Use the committed tiny reference fixture as the parity authority.

Do not implement raw-image resize, tiling, normalization, or patchification.
Do not implement LFM2-VL composition.
Do not implement GGUF.
Do not optimize CUDA.

Create a separate SigLIP2 module rather than overloading fixed-grid siglip.rs with architecture branches.

Required behavior:

- packed patch input
- linear patch embedding with bias
- learned square positional table
- per-crop positional interpolation
- bilinear interpolation
- align_corners=false semantics
- antialiasing during downscale
- F32 interpolation path initially
- per-shape positional cache
- bidirectional padding mask
- F32 attention score and softmax path
- encoder residual blocks
- post LayerNorm
- no pooling head for LFM2-VL
- return padded final hidden states

Add tests for:

- patch projection
- position interpolation
- wide and tall grids
- downscale and upscale interpolation
- padding-mask isolation
- each tiny encoder-layer checkpoint
- final valid-patch output
- deterministic repeated execution
- malformed spatial shapes
- non-square base positional table rejection
- max-patch overflow

Run all targeted tests and baseline checks.

Update STATUS.md and PARITY.md.

Do not commit.

Report component-level maximum absolute error and cosine similarity against the fixture.
```

Commit only after component parity passes:

```bash
git commit -am "feat(siglip2): add packed-patch NaFlex vision encoder"
```

---

# 13. Projector and native composite phase

This phase still receives preprocessed tensors. It does not accept raw images yet.

Paste:

```text
Implement only the LFM2-VL Projector and Native Composite Phase.

Use:
- proven dense LFM2 embedding forwarding
- proven SigLIP2 packed-tensor vision output
- committed tiny fixtures

Do not implement raw image processing.
Do not implement GGUF.
Do not add quantized vision execution.
Do not add CUDA-specific optimization.

Required behavior:

1. Parse LFM2-VL top-level config dynamically.
2. Validate all model dimension relationships.
3. Implement exact factor-N pixel unshuffle.
4. Preserve official channel ordering.
5. Add optional projector LayerNorm.
6. Add projector linear_1.
7. Add exact configured GELU.
8. Add projector linear_2.
9. Unpad each crop using pixel_attention_mask.
10. Reshape each valid crop using spatial_shapes.
11. Project each crop.
12. Flatten projected features in official order.
13. Concatenate crops in processor order.
14. Embed the text prompt.
15. Validate contiguous image-token spans.
16. Validate feature count equals placeholder count.
17. Replace only image-token embeddings.
18. Leave image start/end, row/column, and thumbnail tokens unchanged.
19. Run one multimodal prefill.
20. Continue ordinary cached text decoding without rerunning vision.
21. Expose an EncodedImages result suitable for later image caching.

Tests must prove:

- pixel unshuffle exact-value parity
- optional LayerNorm behavior
- projector stage parity
- crop flatten order
- multiple-crop concatenation order
- single-image span insertion
- multiple-image span insertion
- mismatch failure
- vision runs once
- prefill and incremental decode
- cache reset

Run targeted Rust tests, baseline verification, and fixture comparison.

Update STATUS.md, PARITY.md, and DECISIONS.md.

Do not commit.

Report stage-by-stage numerical parity.
```

Recommended commit:

```bash
git commit -am "feat(lfm2-vl): add native vision projection and multimodal prefill"
```

---

# 14. Rust processor and prompt phase

Only begin after tensor-level native parity is green.

Paste:

```text
Implement only the Rust LFM2-VL Processor and Prompt Phase.

Do not change proven model math unless a golden fixture demonstrates a defect.
Do not implement GGUF.
Do not implement CUDA optimization.

Create a small processor module or crate as defined by the specification.

Required behavior:

- parse processor_config.json
- explicit override precedence
- RGB conversion
- rescale and normalize
- smart resize
- factor-aligned dimensions
- large-image threshold
- tile-grid enumeration
- aspect-ratio selection
- reference tie-break behavior
- row-major crop order
- optional thumbnail
- exact patchification order
- fixed patch padding
- pixel attention mask
- spatial shape metadata
- image and crop metadata
- projected token counts from one canonical function
- special-token lookup through the tokenizer
- image start and end tokens
- row and column tokens
- thumbnail token
- image placeholder repetition
- exact image-span recording
- multiple images
- validation before model execution

Tests must include:

- square
- wide
- tall
- very wide
- very tall
- odd source dimensions
- grayscale
- RGBA
- small upscaled image
- large tiled image
- tiled image with thumbnail
- multiple images
- sentinel/image count mismatch
- missing special token
- projected-token mismatch

Compare all processor outputs against pinned reference fixtures.

Update STATUS.md and PARITY.md.

Do not commit.

Report exact metadata parity, not only generated text.
```

Recommended commit:

```bash
git commit -am "feat(lfm2-vl): add native image and prompt processing"
```

At this point native end-to-end CPU F32 is the first full product proof.

---

# 15. Hybrid and GGUF phases

Do not combine these into one task.

## Hybrid prompt

```text
Implement only quantized GGUF text plus split dense safetensors mmproj.

Do not parse GGUF mmproj yet.
Do not quantize the vision tower.
Do not optimize CUDA.

Create the versioned split-mmproj manifest and exporter defined in the specification.

Required proof:

- dense unified model and split dense mmproj produce equivalent image features
- quantized text accepts projected image embeddings
- model-pair hidden sizes are validated before inference
- tokenizer and image token IDs are validated
- vision may run on a distinct device
- only projected features cross to the text device
- no production weights are committed

Update STATUS.md and PARITY.md.
```

Commit:

```bash
git commit -am "feat(lfm2-vl): support quantized text with split dense mmproj"
```

## Direct GGUF mmproj prompt

```text
Implement only direct llama.cpp-compatible LFM2-VL GGUF mmproj loading.

First implement a compatibility path that dequantizes GGUF mmproj tensors into the proven dense model.

Do not immediately introduce quantized vision operators.

Required work:

- metadata parser
- projector-type validation
- tensor-name normalization
- patch-embedding layout reversal
- vision configuration reconstruction
- processor metadata reconstruction
- optional projector LayerNorm
- linear and bias loading
- pair validation
- malformed-file errors
- tensor inventory diagnostics

Compare native safetensors image features against dequantized GGUF mmproj image features.

Only after that parity gate passes, add quantized linear execution behind a focused abstraction.

Start with Q8_0.
Do not add lower-bit vision support before Q8_0 parity is documented.

Update STATUS.md, PARITY.md, TENSOR_MAP.md, and DECISIONS.md.
```

Commit in two parts:

```text
feat(lfm2-vl): load llama.cpp-compatible GGUF mmproj
feat(lfm2-vl): execute Q8 LFM2-VL vision projections
```

---

# 16. Codex operating procedure for every later task

Use one focused session per task.

At the start:

```bash
codex
```

Inside:

```text
/status
```

Then explicitly instruct Codex to read:

```text
AGENTS.md
docs/lfm2-vl/SPEC.md
docs/lfm2-vl/STATUS.md
docs/lfm2-vl/DECISIONS.md
```

Use `codex resume` only when continuing the same focused task. Start a new session after a phase checkpoint so Codex reloads the current instructions and status. The current CLI supports saved-session resumption. ([OpenAI Developers][2])

Before committing:

```text
/review
```

Then manually inspect:

```bash
git status --short
git diff --stat
git diff
git diff --check
```

Never accept this as a result:

```text
"It compiles, so the model probably works."
```

Require:

```text
exact test command
actual exit status
fixture comparison
numerical tolerance
known mismatch
files changed
remaining blocker
```

---

# 17. Commit and checkpoint strategy

Use one commit per proven responsibility:

```text
chore: establish LFM2.5-VL implementation baseline
docs: lock LFM2.5-VL reference sources
test: add deterministic LFM2-VL reference harness
fix(lfm2): support current LFM2.5 configs and input embeddings
feat(siglip2): add packed-patch NaFlex vision encoder
feat(lfm2-vl): add native vision projection and multimodal prefill
feat(lfm2-vl): add native image and prompt processing
feat(lfm2-vl): support quantized text with split dense mmproj
feat(lfm2-vl): load llama.cpp-compatible GGUF mmproj
feat(lfm2-vl): execute Q8 LFM2-VL vision projections
perf(lfm2-vl): cache positional and projected image embeddings
```

Recommended tags:

```text
lfm2-vl-phase-0-bootstrap
lfm2-vl-phase-0-reference
lfm2-vl-phase-1-text
lfm2-vl-phase-2-siglip2
lfm2-vl-phase-3-native-composite
lfm2-vl-phase-4-native-e2e
lfm2-vl-phase-5-hybrid
lfm2-vl-phase-6-gguf
lfm2-vl-phase-7-q8
```

Do not tag a phase with known acceptance failures.

---

# 18. `STATUS.md` format

Codex should maintain this exact compact shape:

## File: `docs/lfm2-vl/STATUS.md`

```markdown
# LFM2.5-VL Status

## Baseline

- Upstream: Hugging Face Candle
- Base version: 0.11.0
- Working branch: feat/lfm2-vl-mmproj
- Baseline tag: lfm2-vl-baseline-candle-0.11.0
- Current commit: <commit>

## Current Phase

- Phase: <phase>
- Task: <focused task>
- Scope: <files or modules>
- Status: not started | active | blocked | green

## Last Green Verification

- Date:
- Environment:
- Commands:
- Results:

## Proven

- <only behavior demonstrated by current tests>

## Known Failures

- <failure>
  - Reproduction:
  - Expected:
  - Actual:
  - Suspected layer:

## Blockers

- None

or:

- <specific blocker and required resolution>

## Active Files

- <path>

## Reference Pins

- Transformers:
- LiquidAI 450M:
- LiquidAI 1.6B:
- mistral.rs:
- llama.cpp:

## Next Task

<one exact next task only>
```

This prevents Codex sessions from repeatedly rediscovering the project.

---

# 19. `DECISIONS.md` format

## File: `docs/lfm2-vl/DECISIONS.md`

```markdown
# LFM2.5-VL Decisions

## D-0001: Direct Candle Fork

Status: Accepted

Decision:
Work directly from Candle 0.11.0 rather than building a wrapper around an unmodified dependency.

Why:
The implementation requires changes to LFM2 construction, embedding forwarding, model registration, examples, and quantized loading.

Consequences:
The repository retains upstream Candle history and should keep unrelated diffs minimal.

## D-0002: CPU F32 Before CUDA

Status: Accepted

Decision:
All component parity must pass on CPU F32 before CUDA-specific work.

Why:
This separates model and preprocessing defects from accelerator precision and kernel defects.

Consequences:
Initial performance is not an acceptance criterion.

## D-0003: 450M Before 1.6B

Status: Accepted

Decision:
Use LFM2.5-VL-450M as the first production checkpoint.

Why:
Its feed-forward dimensions expose the current Candle normalization defect that the 1.6B dimensions can accidentally hide.

Consequences:
The 1.6B checkpoint remains a required second compatibility test.
```

Add later decisions rather than rewriting old history.

---

# 20. Full verification cadence

During focused development:

```bash
cargo fmt --all -- --check

cargo test --locked -p candle-transformers <targeted-filter>

cargo check --locked -p candle-transformers

git diff --check
```

At a phase gate:

```bash
bash scripts/lfm2-vl/verify-baseline.sh

cargo test --locked -p candle-transformers

cargo check --locked -p candle-examples --example lfm2

cargo check --locked -p candle-examples --example quantized-lfm2
```

After CUDA becomes relevant, add separate commands rather than changing the default verifier:

```bash
cargo check \
    --locked \
    --features cuda \
    -p candle-transformers
```

CPU verification remains mandatory even after CUDA works.

Do not make full workspace Clippy an early blocker if untouched Candle 0.11 code has toolchain-dependent warnings. First record the baseline. Then run Clippy on affected crates and distinguish existing issues from introduced ones.

---

# 21. What Codex must not do

Reject any Codex plan that attempts to:

```text
Implement the whole specification in one turn.
Start from latest Candle instead of 0.11.0.
Replace LFM2 with llama.cpp calls.
Use llama.cpp as an invisible sidecar.
Begin with GGUF mmproj.
Begin with CUDA.
Hardcode the 450M config.
Hardcode factor 2 throughout generic APIs.
Add a generic VLM runtime before LFM2-VL parity.
Judge success from a plausible caption.
Skip the tiny fixture.
Commit downloaded model files.
Modify unrelated architectures.
Silently change public Candle behavior.
Add broad dependencies before proving necessity.
Copy mistral.rs wholesale.
```

Codex should choose solutions, but only within the current phase boundary.

---

# 22. The first development objective

The first objective is not “image captioning works.”

The first objective is:

```text
Candle 0.11 baseline is preserved.
Reference revisions are pinned.
The reference harness is deterministic.
The 450M text config normalizes to 4608.
The 1.6B text config normalizes to 8192.
Dense LFM2 accepts precomputed embeddings.
Quantized LFM2 accepts precomputed embeddings.
Text-only prefill and decode remain correct.
```

Once that is green, we have a stable foundation for SigLIP2 and mmproj support.

## TL;DR

Clone Candle 0.11.0 directly into a WSL2 home-directory repository, tag the untouched baseline, add the specification, create the supplied root `AGENTS.md`, and let Codex perform only the documentation and baseline bootstrap first. Then lock source revisions, build the deterministic fixture harness, and repair text-only LFM2 before touching vision. Stop after the text gate and verify 450M `4608`, 1.6B `8192`, embedding-driven prefill, incremental decode, and legacy compatibility. Only then begin SigLIP2.

[1]: https://developers.openai.com/codex/windows/wsl "WSL | ChatGPT Learn"
[2]: https://developers.openai.com/codex/cli "Codex CLI | ChatGPT Learn"
[3]: https://developers.openai.com/codex/agent-configuration/agents-md "Custom instructions with AGENTS.md | ChatGPT Learn"
