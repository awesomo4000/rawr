#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

if (( $# != 2 )); then
    printf 'usage: %s <first-artifact-prefix> <second-artifact-prefix>\n' "$0" >&2
    exit 2
fi

first_prefix="$1"
second_prefix="$2"
tmp_dir="$(mktemp -d /tmp/rawr-realdata-cross-host.XXXXXX)"
cleanup() {
    case "$tmp_dir" in
        /tmp/rawr-realdata-cross-host.*) rm -rf -- "$tmp_dir" ;;
        *) printf 'refusing to remove unexpected audit directory: %s\n' "$tmp_dir" >&2 ;;
    esac
}
trap cleanup EXIT INT TERM

validate_prefix() {
    local prefix="$1"
    local process_rows="${prefix}-process.tsv"
    local manifest="${prefix}-manifest.tsv"
    local header="${prefix}-header.txt"

    for file in "$process_rows" "$manifest" "$header"; do
        if [[ ! -r "$file" ]]; then
            printf 'required artifact not readable: %s\n' "$file" >&2
            exit 1
        fi
    done

    awk -v expected_runs=5 -v expected_tuples=42 -v expected_processes=210 \
        -f scripts/validate-realdata-results.awk "$process_rows" >/dev/null

    local row_count tuple_count
    row_count="$(awk -F '\t' '$1 == "ROW" { count++ } END { print count + 0 }' "$manifest")"
    tuple_count="$(awk -F '\t' '$1 == "TUPLE" { count++ } END { print count + 0 }' "$manifest")"
    if [[ "$row_count" != 21 || "$tuple_count" != 42 ]]; then
        printf 'invalid manifest counts for %s: rows=%s tuples=%s\n' \
            "$prefix" "$row_count" "$tuple_count" >&2
        exit 1
    fi

    grep -q '^# allocator-pairing: rawr=smp_allocator, CRoaring=default-libc$' "$header"
    grep -q '^# protocol: 1 warmup cycle, 7 timed cycles, process median$' "$header"
}

normalize_semantics() {
    local input="$1"
    local output="$2"
    awk -F '\t' 'BEGIN { OFS = FS }
        {
            # Exclude timing and serialized byte counts. The remaining fields are
            # exactly the cross-host semantic and source-layout audit surface.
            print $1, $2, $3, $4, $6, $7, $8, $9, $10, $11
        }
    ' "$input" | sort -u >"$output"
    if [[ "$(wc -l <"$output" | tr -d '[:space:]')" != 42 ]]; then
        printf 'expected 42 normalized semantic tuples from %s\n' "$input" >&2
        exit 1
    fi
}

validate_prefix "$first_prefix"
validate_prefix "$second_prefix"

if ! cmp -s "${first_prefix}-manifest.tsv" "${second_prefix}-manifest.tsv"; then
    printf 'cross-host manifest mismatch\n' >&2
    diff -u "${first_prefix}-manifest.tsv" "${second_prefix}-manifest.tsv" >&2 || true
    exit 1
fi

normalize_semantics "${first_prefix}-process.tsv" "$tmp_dir/first.tsv"
normalize_semantics "${second_prefix}-process.tsv" "$tmp_dir/second.tsv"
if ! cmp -s "$tmp_dir/first.tsv" "$tmp_dir/second.tsv"; then
    printf 'cross-host corpus, digest, or container-layout mismatch\n' >&2
    diff -u "$tmp_dir/first.tsv" "$tmp_dir/second.tsv" >&2 || true
    exit 1
fi

printf 'real-data cross-host audit: OK (42 tuples, 210 processes per host)\n'
