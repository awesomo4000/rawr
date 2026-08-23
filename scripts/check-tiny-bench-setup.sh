#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

zig build bench-tiny-setup -Dcpu=native

worker="./zig-out/bin/bench_tiny_setup"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then
    worker="${worker}.exe"
fi
if [[ ! -x "$worker" ]]; then
    printf 'tiny setup worker not found: %s\n' "$worker" >&2
    exit 1
fi

first="$(mktemp /tmp/rawr-tiny-hashes-1.XXXXXX)"
second="$(mktemp /tmp/rawr-tiny-hashes-2.XXXXXX)"
trap 'rm -f "$first" "$second"' EXIT

"$worker" hashes >"$first" 2>&1
"$worker" hashes >"$second" 2>&1
if ! cmp -s "$first" "$second"; then
    printf 'tiny fixture hashes are not deterministic across fresh processes\n' >&2
    diff -u "$first" "$second" || true
    exit 1
fi

"$worker" mutation_interleaved
"$worker" mutation_sequential
"$worker" mutation_structural
"$worker" check
printf 'tiny setup fresh-process checks: OK\n'
