#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[@]}")" && pwd -P)"
VERIFIER="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)/verify-fork-overlays.sh"
TEMP_ROOT="$(mktemp -d)"
case "$TEMP_ROOT" in
    /tmp/*) ;;
    *)
        printf 'error: refusing unexpected temporary directory: %s\n' "$TEMP_ROOT" >&2
        exit 2
        ;;
esac
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

write_lines() {
    local path="$1"
    shift
    mkdir -p -- "$(dirname -- "$path")"
    printf '%s\n' "$@" >"$path"
}

new_repo() {
    local name="$1"
    local root="${TEMP_ROOT}/${name}"
    mkdir -p -- "$root"
    git init -q -b main "$root"
    git -C "$root" config user.name "Candle Overlay Test"
    git -C "$root" config user.email "candle-overlay-test@invalid.local"
    git -C "$root" config core.autocrlf false

    write_lines "$root/docs/FORK_OVERLAYS.md" \
        "# Test fork overlay registry" \
        "<!-- shared-paths:start -->" \
        '- `shared.txt`' \
        "<!-- shared-paths:end -->"
    write_lines "$root/docs/lfm2-vl/MOD_MANIFEST.md" \
        "# Test LFM2-VL manifest" \
        '- `docs/lfm2-vl/MOD_MANIFEST.md`' \
        '- `owned.txt`' \
        '- `staged.txt`' \
        '- `shared.txt`'
    write_lines "$root/docs/snapflash/MOD_MANIFEST.md" \
        "# Test SnapFlash manifest" \
        '- `docs/snapflash/MOD_MANIFEST.md`' \
        '- `shared.txt`'
    write_lines "$root/docs/gpt-oss/MOD_MANIFEST.md" \
        "# Test GPT-OSS manifest" \
        '- `docs/gpt-oss/MOD_MANIFEST.md`'
    write_lines "$root/owned.txt" "base owned"
    write_lines "$root/staged.txt" "base staged"
    write_lines "$root/shared.txt" "base shared"

    git -C "$root" add -- .
    git -C "$root" commit -q -m "test: baseline"
    printf '%s\n' "$root"
}

run_capture() {
    local root="$1"
    shift
    if LAST_OUTPUT="$(FORK_OVERLAYS_REPO_ROOT="$root" bash "$VERIFIER" "$@" 2>&1)"; then
        LAST_STATUS=0
    else
        LAST_STATUS="$?"
    fi
}

assert_status() {
    local label="$1"
    local expected="$2"
    if [[ "$LAST_STATUS" -ne "$expected" ]]; then
        printf 'FAIL %s: expected exit %s, got %s\n%s\n' \
            "$label" "$expected" "$LAST_STATUS" "$LAST_OUTPUT" >&2
        exit 1
    fi
}

assert_contains() {
    local label="$1"
    local needle="$2"
    if ! grep -Fq -- "$needle" <<<"$LAST_OUTPUT"; then
        printf 'FAIL %s: output did not contain %s\n%s\n' \
            "$label" "$needle" "$LAST_OUTPUT" >&2
        exit 1
    fi
}

allowed_root="$(new_repo allowed-dirty)"
allowed_base="$(git -C "$allowed_root" rev-parse HEAD)"
write_lines "$allowed_root/owned.txt" "dirty owned"
write_lines "$allowed_root/staged.txt" "staged change"
git -C "$allowed_root" add -- staged.txt
write_lines "$allowed_root/new file.txt" "untracked with spaces"
printf '%s\n' '- `new file.txt`' >>"$allowed_root/docs/lfm2-vl/MOD_MANIFEST.md"
run_capture "$allowed_root" --rolling-baseline "$allowed_base"
assert_status "allowed dirty candidate" 0
assert_contains "allowed dirty candidate" "fork-overlays: passed"
assert_contains "allowed dirty candidate" "dirty-paths="
printf 'verify-fork-overlays case=allowed-dirty passed\n'

unowned_root="$(new_repo unowned)"
unowned_base="$(git -C "$unowned_root" rev-parse HEAD)"
write_lines "$unowned_root/unowned.txt" "not registered"
run_capture "$unowned_root" --rolling-baseline "$unowned_base"
assert_status "unowned path" 1
assert_contains "unowned path" "unowned.txt"
printf 'verify-fork-overlays case=unowned passed\n'

rename_root="$(new_repo rename-delete)"
write_lines "$rename_root/rename-source.txt" "rename source"
write_lines "$rename_root/deleted.txt" "to delete"
printf '%s\n' \
    '- `rename-source.txt`' \
    '- `deleted.txt`' >>"$rename_root/docs/lfm2-vl/MOD_MANIFEST.md"
git -C "$rename_root" add -- .
git -C "$rename_root" commit -q -m "test: register rename and deletion"
rename_base="$(git -C "$rename_root" rev-parse HEAD)"
git -C "$rename_root" mv -- rename-source.txt rename-target.txt
git -C "$rename_root" rm -q -- deleted.txt
printf '%s\n' '- `rename-target.txt`' >>"$rename_root/docs/lfm2-vl/MOD_MANIFEST.md"
run_capture "$rename_root" --rolling-baseline "$rename_base"
assert_status "rename and deletion" 0
assert_contains "rename and deletion" "fork-overlays: passed"
printf 'verify-fork-overlays case=rename-deletion passed\n'

baseline_root="$(new_repo baseline-errors)"
baseline_commit="$(git -C "$baseline_root" rev-parse HEAD)"
run_capture "$baseline_root" --rolling-baseline not-a-commit
assert_status "invalid baseline" 2
assert_contains "invalid baseline" "baseline is not a valid commit-ish"
run_capture "$baseline_root" --rolling-baseline deadbeefdeadbeefdeadbeefdeadbeefdeadbeef
assert_status "missing baseline" 2
assert_contains "missing baseline" "baseline is not a valid commit-ish"
printf 'verify-fork-overlays case=invalid-missing-baseline passed\n'

run_capture "$baseline_root" --rolling-baseline "$baseline_commit"
assert_status "rolling historic paths" 0
assert_contains "rolling historic paths" "baseline-kind=rolling"
printf 'verify-fork-overlays case=rolling-historic-paths passed\n'

run_capture "$baseline_root" --upstream-baseline "$baseline_commit"
assert_status "upstream stale-path check" 1
assert_contains "upstream stale-path check" "current overlay manifest paths are absent from the upstream delta"
printf 'verify-fork-overlays case=upstream-stale-paths passed\n'

printf 'verify-fork-overlays regression: 7 checks passed\n'
