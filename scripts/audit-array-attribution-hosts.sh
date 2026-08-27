#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

if (( $# != 2 )); then
    printf 'usage: %s <first-process.tsv> <second-process.tsv>\n' "$0" >&2
    exit 2
fi

first="$1"
second="$2"
tmp_dir="$(mktemp -d /tmp/rawr-array-attribution-audit.XXXXXX)"
cleanup() {
    case "$tmp_dir" in
        /tmp/rawr-array-attribution-audit.*) rm -rf -- "$tmp_dir" ;;
        *) printf 'refusing to remove unexpected audit directory: %s\n' "$tmp_dir" >&2 ;;
    esac
}
trap cleanup EXIT INT TERM

for file in "$first" "$second"; do
    if [[ ! -r "$file" ]]; then
        printf 'process artifact not readable: %s\n' "$file" >&2
        exit 1
    fi
    awk -v expected_runs=5 -v expected_tuples=16 -v expected_processes=80 \
        -f scripts/validate-array-attribution-results.awk "$file" >/dev/null
done

normalize() {
    local input="$1"
    local output="$2"
    awk -F '\t' 'BEGIN { OFS = FS }
        {
            # Timing and A3 dispatch are host-specific. Every other field is
            # semantic output, workload accounting, or an arm-meaning guard.
            print $1, $2, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, \
                $16, $17, $18, $19, $20, $21, $22
        }
    ' "$input" | sort -u >"$output"
    if [[ "$(wc -l <"$output" | tr -d '[:space:]')" != 16 ]]; then
        printf 'expected 16 normalized tuples from %s\n' "$input" >&2
        exit 1
    fi
}

normalize "$first" "$tmp_dir/first.tsv"
normalize "$second" "$tmp_dir/second.tsv"
if ! cmp -s "$tmp_dir/first.tsv" "$tmp_dir/second.tsv"; then
    printf 'array-attribution cross-host semantic mismatch\n' >&2
    diff -u "$tmp_dir/first.tsv" "$tmp_dir/second.tsv" >&2 || true
    exit 1
fi

printf 'array-attribution cross-host audit: OK (16 tuples, 80 processes per host)\n'
