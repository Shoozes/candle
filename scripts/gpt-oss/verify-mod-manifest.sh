#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BASELINE="${1:-7c2e89295dad4aeebc6ef7a92c255360b6957c2c}"
MANIFEST="${REPO_ROOT}/docs/gpt-oss/MOD_MANIFEST.md"
cd -- "$REPO_ROOT"

if ! git cat-file -e "${BASELINE}^{commit}" 2>/dev/null; then
    printf 'error: baseline commit is unavailable: %s\n' "$BASELINE" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    printf 'error: GPT-OSS experimental manifest is missing: %s\n' "$MANIFEST" >&2
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
} | tr -d '\r' | sort -u >"$REPO_PATHS"

sed -n \
    -e 's/\r$//' \
    -e 's/^| `\([^`]*\)` |.*$/\1/p' \
    -e 's/^- `\([^`]*\)`$/\1/p' \
    "$MANIFEST" | tr -d '\r' | sort -u >"$MANIFEST_PATHS"

if grep -E '^(.tools/|\.venv/|artifacts/|downloads/|models/|target/)|(^|/)__pycache__/' "$MANIFEST_PATHS"; then
    printf 'error: GPT-OSS manifest contains a prohibited local/runtime path\n' >&2
    exit 1
fi

comm -23 "$MANIFEST_PATHS" "$REPO_PATHS" >"$STALE_PATHS"
if [[ -s "$STALE_PATHS" ]]; then
    printf 'error: GPT-OSS manifest paths absent from the baseline-to-current delta:\n' >&2
    sed 's/^/  - /' "$STALE_PATHS" >&2
    exit 1
fi

required_paths=(
    candle-transformers/src/models/gpt_oss/checkpoint.rs
    candle-transformers/src/models/gpt_oss/config.rs
    candle-transformers/src/models/gpt_oss/mod.rs
    candle-transformers/src/models/gpt_oss/model.rs
    candle-transformers/src/models/gpt_oss/mxfp4.rs
)
for path in "${required_paths[@]}"; do
    if ! grep -Fxq "$path" "$MANIFEST_PATHS"; then
        printf 'error: GPT-OSS manifest omits required path: %s\n' "$path" >&2
        exit 1
    fi
done

if grep -Eiq 'edgesymbio|snapflash' candle-transformers/src/models/gpt_oss/*.rs; then
    printf 'error: application-specific name leaked into GPT-OSS Candle source\n' >&2
    exit 1
fi

printf 'gpt-oss-mod-manifest baseline=%s paths=%s\n' \
    "$BASELINE" "$(wc -l <"$MANIFEST_PATHS" | tr -d ' ')"
printf 'gpt-oss-mod-manifest: passed\n'

