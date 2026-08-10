#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd -- "$REPO_ROOT"

first_line() {
    local output
    if output="$("$@" 2>&1)"; then
        printf '%s\n' "${output%%$'\n'*}"
    else
        printf '%s\n' "unavailable"
    fi
}

optional_first_line() {
    local label="$1"
    shift
    if command -v "$1" >/dev/null 2>&1; then
        printf '%s: %s\n' "$label" "$(first_line "$@")"
    else
        printf '%s: missing\n' "$label"
    fi
}

os_name="unknown"
if [[ -r /etc/os-release ]]; then
    os_name="$(. /etc/os-release; printf '%s' "${PRETTY_NAME:-unknown}")"
fi

linux_home=""
if linux_home="$(getent passwd "$(id -un)" | awk -F: 'NR == 1 { print $6 }')"; then
    :
fi
if [[ -z "$linux_home" ]]; then
    linux_home="${HOME:-.}"
fi

printf 'repo-root: %s\n' "$REPO_ROOT"
printf 'timestamp-utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'os: %s\n' "$os_name"
printf 'kernel: %s\n' "$(uname -srmo)"
printf 'wsl-distribution: %s\n' "${WSL_DISTRO_NAME:-unknown}"

optional_first_line git git --version
if git rev-parse --git-dir >/dev/null 2>&1; then
    printf 'commit: %s\n' "$(git rev-parse HEAD)"
    if branch="$(git symbolic-ref --short -q HEAD)"; then
        printf 'branch: %s\n' "$branch"
    else
        printf 'branch: detached-head\n'
    fi
else
    printf 'commit: unavailable\n'
    printf 'branch: unavailable\n'
fi

optional_first_line rustc rustc --version
optional_first_line cargo cargo --version
optional_first_line python3 python3 --version
optional_first_line cmake cmake --version
optional_first_line ninja ninja --version

printf 'nvidia:\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia_summary="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>&1)"; then
        printf '%s\n' "$nvidia_summary"
    else
        printf 'unavailable\n'
    fi
else
    printf 'unavailable\n'
fi

printf 'disk-free:\n'
for path in "$linux_home" /mnt/c /mnt/d; do
    if [[ -e "$path" ]]; then
        disk_line="$(df -h -- "$path" 2>/dev/null | tail -n 1)" || disk_line=""
        if [[ -n "$disk_line" ]]; then
            printf '%s: %s\n' "$path" "$disk_line"
        else
            printf '%s: unavailable\n' "$path"
        fi
    else
        printf '%s: unavailable\n' "$path"
    fi
done
