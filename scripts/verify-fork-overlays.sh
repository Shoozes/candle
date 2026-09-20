#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${FORK_OVERLAYS_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
DEFAULT_ROLLING_BASELINE="d830e03078a29d39a1aacd741620475eb33b7609"
BASELINE_KIND="rolling"
BASELINE=""
REGISTRY="${REPO_ROOT}/docs/FORK_OVERLAYS.md"
MANIFESTS=(
    "docs/lfm2-vl/MOD_MANIFEST.md"
    "docs/snapflash/MOD_MANIFEST.md"
    "docs/gpt-oss/MOD_MANIFEST.md"
)

usage() {
    cat <<'EOF'
Usage:
  scripts/verify-fork-overlays.sh [ROLLING_BASELINE]
  scripts/verify-fork-overlays.sh --rolling-baseline COMMIT
  scripts/verify-fork-overlays.sh --upstream-baseline COMMIT

Rolling mode requires every computed baseline-to-candidate path to have an
owner and permits historic registered paths unchanged since that checkpoint.
Upstream mode also requires every current manifest path in the upstream delta.
EOF
}

while (($# > 0)); do
    case "$1" in
        --rolling-baseline|--upstream-baseline)
            if (($# < 2)); then
                printf 'error: %s requires a commit-ish\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == "--upstream-baseline" ]]; then
                BASELINE_KIND="upstream"
            else
                BASELINE_KIND="rolling"
            fi
            BASELINE="$2"
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        --*) printf 'error: unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -n "$BASELINE" ]]; then
                printf 'error: multiple baselines supplied\n' >&2
                exit 2
            fi
            BASELINE_KIND="rolling"
            BASELINE="$1"
            shift
            ;;
    esac
done

if [[ -z "$BASELINE" ]]; then
    BASELINE="$DEFAULT_ROLLING_BASELINE"
fi

GIT=(git -c core.autocrlf=true -c core.safecrlf=false)
cd -- "$REPO_ROOT"

if [[ ! -d .git ]]; then
    printf 'error: repository root is not a Git worktree: %s\n' "$REPO_ROOT" >&2
    exit 2
fi
if ! "${GIT[@]}" rev-parse --verify --quiet HEAD^{commit} >/dev/null; then
    printf 'error: candidate worktree has no commit at HEAD\n' >&2
    exit 2
fi
if ! BASELINE_COMMIT="$("${GIT[@]}" rev-parse --verify --quiet "${BASELINE}^{commit}")"; then
    printf 'error: baseline is not a valid commit-ish: %s\n' "$BASELINE" >&2
    exit 2
fi
HEAD_COMMIT="$("${GIT[@]}" rev-parse --verify --quiet HEAD^{commit})"
if ! "${GIT[@]}" merge-base --is-ancestor "$BASELINE_COMMIT" "$HEAD_COMMIT"; then
    printf 'error: baseline %s is not an ancestor of candidate HEAD %s\n' \
        "$BASELINE_COMMIT" "$HEAD_COMMIT" >&2
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

CHANGED_RAW="${TEMP_DIR}/changed.raw"
DIRTY_RAW="${TEMP_DIR}/dirty.raw"
DELETED_RAW="${TEMP_DIR}/deleted.raw"
CHANGED_PATHS="${TEMP_DIR}/changed.paths"
DIRTY_PATHS="${TEMP_DIR}/dirty.paths"
DELETED_PATHS="${TEMP_DIR}/deleted.paths"
CURRENT_MANIFEST_PATHS="${TEMP_DIR}/current-manifest.paths"
BASELINE_MANIFEST_PATHS="${TEMP_DIR}/baseline-manifest.paths"
OWNER_PATHS="${TEMP_DIR}/owner.paths"
UNION_PATHS="${TEMP_DIR}/union-paths.txt"
DUPLICATE_PATHS="${TEMP_DIR}/duplicate-paths.txt"
SHARED_PATHS="${TEMP_DIR}/shared-paths.txt"
UNDECLARED_DUPLICATES="${TEMP_DIR}/undeclared-duplicates.txt"
MISSING_PATHS="${TEMP_DIR}/missing-paths.txt"
STALE_PATHS="${TEMP_DIR}/stale.paths"

: >"$CHANGED_RAW"
: >"$DIRTY_RAW"
: >"$DELETED_RAW"
: >"$BASELINE_MANIFEST_PATHS"

append_status_paths() {
    local changed_output="$1" deleted_output="$2" status first second
    while IFS= read -r -d '' status; do
        if [[ "$status" == R* || "$status" == C* ]]; then
            IFS= read -r -d '' first || { printf 'error: malformed NUL Git rename record\n' >&2; exit 2; }
            IFS= read -r -d '' second || { printf 'error: malformed NUL Git rename record\n' >&2; exit 2; }
            printf '%s\0%s\0' "$first" "$second" >>"$changed_output"
            [[ "$status" == R* ]] && printf '%s\0' "$first" >>"$deleted_output"
        else
            IFS= read -r -d '' first || { printf 'error: malformed NUL Git status record\n' >&2; exit 2; }
            printf '%s\0' "$first" >>"$changed_output"
            [[ "$status" == D* ]] && printf '%s\0' "$first" >>"$deleted_output"
        fi
    done <"$3"
    return 0
}

collect_diff() {
    local changed_output="$1" deleted_output="$2" raw_output="$3"
    shift 3
    if ! "${GIT[@]}" diff --name-status -z --find-renames --diff-filter=ACDMRTUXB "$@" -- >"$raw_output"; then
        printf 'error: Git status inventory failed for %s\n' "$*" >&2
        exit 2
    fi
    append_status_paths "$changed_output" "$deleted_output" "$raw_output"
}

collect_diff "$CHANGED_RAW" "$DELETED_RAW" "${TEMP_DIR}/committed.raw" "$BASELINE_COMMIT" "$HEAD_COMMIT"
collect_diff "$CHANGED_RAW" "$DELETED_RAW" "${TEMP_DIR}/staged.raw" --cached HEAD
collect_diff "$CHANGED_RAW" "$DELETED_RAW" "${TEMP_DIR}/unstaged.raw"
if ! "${GIT[@]}" ls-files --others --exclude-standard -z >"${TEMP_DIR}/untracked.raw"; then
    printf 'error: Git untracked-path inventory failed\n' >&2
    exit 2
fi
while IFS= read -r -d '' path; do
    printf '%s\0' "$path" >>"$CHANGED_RAW"
    printf '%s\0' "$path" >>"$DIRTY_RAW"
done <"${TEMP_DIR}/untracked.raw"
[[ -s "${TEMP_DIR}/staged.raw" ]] && append_status_paths "$DIRTY_RAW" /dev/null "${TEMP_DIR}/staged.raw"
[[ -s "${TEMP_DIR}/unstaged.raw" ]] && append_status_paths "$DIRTY_RAW" /dev/null "${TEMP_DIR}/unstaged.raw"
sort -z -u "$CHANGED_RAW" >"$CHANGED_PATHS"
sort -z -u "$DIRTY_RAW" >"$DIRTY_PATHS"
sort -z -u "$DELETED_RAW" >"$DELETED_PATHS"

parse_manifest_stream() {
    sed -n \
        -e 's/\r$//' \
        -e 's/^| `\([^`]*\)` |.*$/\1/p' \
        -e 's/^- `\([^`]*\)`$/\1/p' |
        while IFS= read -r path; do
            printf '%s\0' "$path"
        done
}

for manifest in "${MANIFESTS[@]}"; do
    if [[ ! -f "$manifest" ]]; then
        printf 'error: registered overlay manifest is missing: %s\n' "$manifest" >&2
        exit 2
    fi
    parse_manifest_stream <"$manifest" >>"$CURRENT_MANIFEST_PATHS"
    if "${GIT[@]}" cat-file -e "${BASELINE_COMMIT}:${manifest}" 2>/dev/null; then
        "${GIT[@]}" show "${BASELINE_COMMIT}:${manifest}" |
            parse_manifest_stream >>"$BASELINE_MANIFEST_PATHS"
    fi
done

LC_ALL=C sort -z -u "$CURRENT_MANIFEST_PATHS" >"$UNION_PATHS"
cat "$CURRENT_MANIFEST_PATHS" "$BASELINE_MANIFEST_PATHS" |
    LC_ALL=C sort -z -u >"$OWNER_PATHS"
LC_ALL=C sort -z "$CURRENT_MANIFEST_PATHS" | uniq -z -d >"$DUPLICATE_PATHS"
sed -n '/<!-- shared-paths:start -->/,/<!-- shared-paths:end -->/ {
    s/\r$//
    s/^- `\([^`]*\)`$/\1/p
}' "$REGISTRY" |
    while IFS= read -r path; do
        printf '%s\0' "$path"
    done |
    LC_ALL=C sort -z -u >"$SHARED_PATHS"

comm -z -23 "$DUPLICATE_PATHS" "$SHARED_PATHS" >"$UNDECLARED_DUPLICATES"
if [[ -s "$UNDECLARED_DUPLICATES" ]]; then
    printf 'error: overlay manifests overlap outside the shared-path registry:\n' >&2
    while IFS= read -r -d '' path; do
        printf '  - %s\n' "$path" >&2
    done <"$UNDECLARED_DUPLICATES"
    exit 1
fi

while IFS= read -r -d '' path; do
    case "$path" in
        .tools/.secrets/*|.venv/*|artifacts/*|downloads/*|models/*|target/*|__pycache__/*|*/__pycache__/*)
            printf 'error: overlay manifests contain a prohibited local/runtime path: %s\n' "$path" >&2
            exit 1
            ;;
    esac
done <"$UNION_PATHS"

comm -z -23 "$CHANGED_PATHS" "$OWNER_PATHS" >"$MISSING_PATHS"
if [[ -s "$MISSING_PATHS" ]]; then
    printf 'error: changed paths are absent from every overlay manifest:\n' >&2
    while IFS= read -r -d '' path; do
        printf '  - %s\n' "$path" >&2
    done <"$MISSING_PATHS"
    exit 1
fi

if [[ "$BASELINE_KIND" == "upstream" ]]; then
    comm -z -23 "$UNION_PATHS" "$CHANGED_PATHS" >"$STALE_PATHS"
    if [[ -s "$STALE_PATHS" ]]; then
        printf 'error: current overlay manifest paths are absent from the upstream delta:\n' >&2
        while IFS= read -r -d '' path; do
            printf '  - %s\n' "$path" >&2
        done <"$STALE_PATHS"
        exit 1
    fi
fi

while IFS= read -r -d '' path; do
    if [[ ! -e "$path" ]]; then
        if ! grep -Fzx -- "$path" "$DELETED_PATHS" >/dev/null; then
            printf 'error: overlay manifest path is missing from the checkout: %s\n' "$path" >&2
            exit 1
        fi
    fi
done <"$UNION_PATHS"

count_paths() {
    local count=0 path
    while IFS= read -r -d '' path; do
        ((count += 1))
    done <"$1"
    printf '%s' "$count"
}

printf 'fork-overlays baseline=%s paths=%s overlays=%s shared=%s\n' \
    "$BASELINE" \
    "$(count_paths "$UNION_PATHS")" \
    "${#MANIFESTS[@]}" \
    "$(count_paths "$SHARED_PATHS")"
printf 'fork-overlays baseline-kind=%s candidate-head=%s dirty-paths=%s\n' \
    "$BASELINE_KIND" \
    "$HEAD_COMMIT" \
    "$(count_paths "$DIRTY_PATHS")"
printf 'fork-overlays: passed\n'
