#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

zig build check-array-merge-alias -Dcpu=native
checker="./zig-out/bin/check_array_merge_alias"
if [[ ! -x "$checker" && -x "${checker}.exe" ]]; then checker="${checker}.exe"; fi
if [[ ! -x "$checker" ]]; then
    printf 'array merge alias checker not found: %s\n' "$checker" >&2
    exit 1
fi

panic_cases=(same_a same_b head tail inside)
control_cases=(adjacent separate)
for operation in union difference; do
    for case in "${panic_cases[@]}"; do
        output="$(mktemp /tmp/rawr-array-merge-alias.XXXXXX)"
        if bash -c 'set +e; "$1" "$2" "$3"; code=$?; exit "$code"' \
            _ "$checker" "$operation" "$case" >"$output" 2>&1
        then
            printf 'expected alias assertion: operation=%s case=%s\n' "$operation" "$case" >&2
            rm -f -- "$output"
            exit 1
        fi
        if ! grep -q 'array merge output aliases input' "$output"; then
            printf 'wrong failure for operation=%s case=%s\n' "$operation" "$case" >&2
            cat "$output" >&2
            rm -f -- "$output"
            exit 1
        fi
        rm -f -- "$output"
    done
    for case in "${control_cases[@]}"; do
        "$checker" "$operation" "$case"
    done
done

printf 'array merge alias guard: OK (10 panic cases, 4 controls)\n'
