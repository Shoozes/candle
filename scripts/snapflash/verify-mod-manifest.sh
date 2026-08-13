#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BASELINE="${1:-6f74e7c390c717f8fd34f23ce02aceb058173370}"
MANIFEST="${REPO_ROOT}/docs/snapflash/MOD_MANIFEST.md"
cd -- "$REPO_ROOT"

if ! git cat-file -e "${BASELINE}^{commit}" 2>/dev/null; then
    printf 'error: baseline commit is unavailable: %s\n' "$BASELINE" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    printf 'error: SnapFlash-derived manifest is missing: %s\n' "$MANIFEST" >&2
    exit 2
fi

TEMP_DIR="$(mktemp -d)"
case "$TEMP_DIR" in
    /tmp/*) ;;
    *)
        printf 'error: refusing unexpected temporary directory: %s\n' "$TEMP_DIR" >&2
        exit 2
        ;;
esac
trap 'rm -rf -- "$TEMP_DIR"' EXIT

REPO_PATHS="${TEMP_DIR}/repo-paths.txt"
MANIFEST_PATHS="${TEMP_DIR}/manifest-paths.txt"
STALE_PATHS="${TEMP_DIR}/stale-paths.txt"

{
    git diff --name-only --diff-filter=ACDMRTUXB "$BASELINE" --
    git ls-files --others --exclude-standard
} | LC_ALL=C sort -u >"$REPO_PATHS"

sed -n \
    -e 's/^| `\([^`]*\)` |.*$/\1/p' \
    -e 's/^- `\([^`]*\)`$/\1/p' \
    "$MANIFEST" | LC_ALL=C sort -u >"$MANIFEST_PATHS"

if grep -E '^(Cargo\.lock|\.tools/|\.venv/|artifacts/|downloads/|models/|target/)|(^|/)__pycache__/' "$MANIFEST_PATHS"; then
    printf 'error: SnapFlash-derived manifest contains a prohibited local/runtime path\n' >&2
    exit 1
fi

comm -23 "$MANIFEST_PATHS" "$REPO_PATHS" >"$STALE_PATHS"
if [[ -s "$STALE_PATHS" ]]; then
    printf 'error: SnapFlash-derived manifest paths absent from the baseline-to-current delta:\n' >&2
    sed 's/^/  - /' "$STALE_PATHS" >&2
    exit 1
fi

required_paths=(
    candle-transformers/Cargo.toml
    candle-transformers/src/models/stable_diffusion/embeddings.rs
    candle-transformers/src/models/stable_diffusion/lora.rs
    candle-transformers/src/models/stable_diffusion/mutable.rs
    candle-transformers/src/models/stable_diffusion/mod.rs
    candle-transformers/src/models/stable_diffusion/unet_2d.rs
    candle-transformers/tests/stable_diffusion_mutable_tests.rs
)
for path in "${required_paths[@]}"; do
    if ! grep -Fxq "$path" "$MANIFEST_PATHS"; then
        printf 'error: SnapFlash-derived manifest omits required path: %s\n' "$path" >&2
        exit 1
    fi
done

if grep -Eiq 'snapflash|edgesymbio' \
    candle-transformers/src/models/stable_diffusion/lora.rs \
    candle-transformers/src/models/stable_diffusion/mutable.rs; then
    printf 'error: application-specific name leaked into Candle LoRA source\n' >&2
    exit 1
fi
if ! grep -Fq 'pub mod lora;' candle-transformers/src/models/stable_diffusion/mod.rs || \
   ! grep -Fq 'pub mod mutable;' candle-transformers/src/models/stable_diffusion/mod.rs; then
    printf 'error: Candle Stable Diffusion module does not export both LoRA modules\n' >&2
    exit 1
fi

modified_count=0
added_count=0
while IFS= read -r path; do
    if git cat-file -e "${BASELINE}:${path}" 2>/dev/null; then
        modified_count=$((modified_count + 1))
    else
        added_count=$((added_count + 1))
    fi
done <"$MANIFEST_PATHS"

if [[ "$modified_count" -ne 5 || "$added_count" -ne 7 ]]; then
    printf 'error: expected SnapFlash-derived overlay counts 5 modified/7 added; found %s/%s\n' \
        "$modified_count" "$added_count" >&2
    exit 1
fi

printf 'snapflash-mod-manifest baseline=%s total=%s fork_modified=%s mod_added=%s\n' \
    "$BASELINE" "$((modified_count + added_count))" "$modified_count" "$added_count"
printf 'snapflash-mod-manifest: passed\n'
