#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
BASELINE="${1:-6f74e7c390c717f8fd34f23ce02aceb058173370}"
REGISTRY="${REPO_ROOT}/docs/FORK_OVERLAYS.md"
MANIFESTS=(
    "docs/lfm2-vl/MOD_MANIFEST.md"
    "docs/snapflash/MOD_MANIFEST.md"
)
cd -- "$REPO_ROOT"

if ! git cat-file -e "${BASELINE}^{commit}" 2>/dev/null; then
    printf 'error: baseline commit is unavailable: %s\n' "$BASELINE" >&2
    exit 2
fi
if [[ ! -f "$REGISTRY" ]]; then
    printf 'error: fork overlay registry is missing: %s\n' "$REGISTRY" >&2
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
ALL_MANIFEST_PATHS="${TEMP_DIR}/all-manifest-paths.txt"
UNION_PATHS="${TEMP_DIR}/union-paths.txt"
DUPLICATE_PATHS="${TEMP_DIR}/duplicate-paths.txt"
SHARED_PATHS="${TEMP_DIR}/shared-paths.txt"
UNDECLARED_DUPLICATES="${TEMP_DIR}/undeclared-duplicates.txt"
MISSING_PATHS="${TEMP_DIR}/missing-paths.txt"
STALE_PATHS="${TEMP_DIR}/stale-paths.txt"

{
    git diff --name-only --diff-filter=ACDMRTUXB "$BASELINE" --
    git ls-files --others --exclude-standard
} | LC_ALL=C sort -u >"$REPO_PATHS"

: >"$ALL_MANIFEST_PATHS"
for manifest in "${MANIFESTS[@]}"; do
    if [[ ! -f "$manifest" ]]; then
        printf 'error: registered overlay manifest is missing: %s\n' "$manifest" >&2
        exit 2
    fi
    sed -n \
        -e 's/^| `\([^`]*\)` |.*$/\1/p' \
        -e 's/^- `\([^`]*\)`$/\1/p' \
        "$manifest" | LC_ALL=C sort -u >>"$ALL_MANIFEST_PATHS"
done

LC_ALL=C sort -u "$ALL_MANIFEST_PATHS" >"$UNION_PATHS"
LC_ALL=C sort "$ALL_MANIFEST_PATHS" | uniq -d >"$DUPLICATE_PATHS"
sed -n '/<!-- shared-paths:start -->/,/<!-- shared-paths:end -->/ {
    s/^- `\([^`]*\)`$/\1/p
}' "$REGISTRY" | LC_ALL=C sort -u >"$SHARED_PATHS"

comm -23 "$DUPLICATE_PATHS" "$SHARED_PATHS" >"$UNDECLARED_DUPLICATES"
if [[ -s "$UNDECLARED_DUPLICATES" ]]; then
    printf 'error: overlay manifests overlap outside the shared-path registry:\n' >&2
    sed 's/^/  - /' "$UNDECLARED_DUPLICATES" >&2
    exit 1
fi

if grep -E '^(\.tools/|\.venv/|artifacts/|downloads/|models/|target/)|(^|/)__pycache__/' "$UNION_PATHS"; then
    printf 'error: overlay manifests contain a prohibited local/runtime path\n' >&2
    exit 1
fi

comm -23 "$REPO_PATHS" "$UNION_PATHS" >"$MISSING_PATHS"
comm -13 "$REPO_PATHS" "$UNION_PATHS" >"$STALE_PATHS"
if [[ -s "$MISSING_PATHS" ]]; then
    printf 'error: changed paths are absent from every overlay manifest:\n' >&2
    sed 's/^/  - /' "$MISSING_PATHS" >&2
    exit 1
fi
if [[ -s "$STALE_PATHS" ]]; then
    printf 'error: overlay manifest paths are absent from the baseline-to-current delta:\n' >&2
    sed 's/^/  - /' "$STALE_PATHS" >&2
    exit 1
fi

printf 'fork-overlays baseline=%s paths=%s overlays=%s shared=%s\n' \
    "$BASELINE" \
    "$(wc -l <"$UNION_PATHS" | tr -d ' ')" \
    "${#MANIFESTS[@]}" \
    "$(wc -l <"$SHARED_PATHS" | tr -d ' ')"
printf 'fork-overlays: passed\n'
