#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
BASELINE="${1:-6f74e7c390c717f8fd34f23ce02aceb058173370}"
MANIFEST="${REPO_ROOT}/docs/lfm2-vl/MOD_MANIFEST.md"
cd -- "$REPO_ROOT"

if ! git cat-file -e "${BASELINE}^{commit}" 2>/dev/null; then
    printf 'error: baseline commit is unavailable: %s\n' "$BASELINE" >&2
    exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
    printf 'error: mod manifest is missing: %s\n' "$MANIFEST" >&2
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

CURRENT_PATHS="${TEMP_DIR}/current-paths.txt"
MANIFEST_PATHS="${TEMP_DIR}/manifest-paths.txt"
MISSING_PATHS="${TEMP_DIR}/missing-paths.txt"
STALE_PATHS="${TEMP_DIR}/stale-paths.txt"

{
    git diff --name-only --diff-filter=ACDMRTUXB "$BASELINE" --
    git ls-files --others --exclude-standard
} | LC_ALL=C sort -u >"$CURRENT_PATHS"

sed -n \
    -e 's/^| `\([^`]*\)` |.*$/\1/p' \
    -e 's/^- `\([^`]*\)`$/\1/p' \
    "$MANIFEST" | LC_ALL=C sort -u >"$MANIFEST_PATHS"

if grep -E '^(Cargo\.lock|\.tools/|\.venv/|artifacts/|downloads/|models/|target/)|(^|/)__pycache__/' "$CURRENT_PATHS"; then
    printf 'error: current publication delta contains a prohibited local/runtime path\n' >&2
    exit 1
fi

comm -23 "$CURRENT_PATHS" "$MANIFEST_PATHS" >"$MISSING_PATHS"
comm -13 "$CURRENT_PATHS" "$MANIFEST_PATHS" >"$STALE_PATHS"
if [[ -s "$MISSING_PATHS" ]]; then
    printf 'error: changed paths missing from MOD_MANIFEST.md:\n' >&2
    sed 's/^/  - /' "$MISSING_PATHS" >&2
    exit 1
fi
if [[ -s "$STALE_PATHS" ]]; then
    printf 'error: MOD_MANIFEST.md paths absent from the baseline-to-current delta:\n' >&2
    sed 's/^/  - /' "$STALE_PATHS" >&2
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
done <"$CURRENT_PATHS"

if [[ "$modified_count" -ne 9 ]]; then
    printf 'error: expected exactly 9 fork-origin modifications, found %s\n' "$modified_count" >&2
    exit 1
fi

total_count=$((modified_count + added_count))
printf 'mod-manifest baseline=%s total=%s fork_modified=%s mod_added=%s\n' \
    "$BASELINE" "$total_count" "$modified_count" "$added_count"
printf 'mod-manifest: passed\n'
