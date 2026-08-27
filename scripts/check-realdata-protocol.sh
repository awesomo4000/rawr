#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

tmp_dir="$(mktemp -d /tmp/rawr-realdata-protocol.XXXXXX)"
cleanup() {
    case "$tmp_dir" in
        /tmp/rawr-realdata-protocol.*) rm -rf -- "$tmp_dir" ;;
        *) printf 'refusing to remove unexpected protocol directory: %s\n' "$tmp_dir" >&2 ;;
    esac
}
trap cleanup EXIT INT TERM

baseline="$tmp_dir/baseline.tsv"
: >"$baseline"
for implementation in rawr croaring; do
    for run in 1 2 3 4 5; do
        printf 'fixture\tpair-and\t%s\t199\t%d\t0x1111\t0x2222\t100\t10\t2\t0\t0\n' \
            "$implementation" "$run" >>"$baseline"
    done
done

validate() {
    awk -v expected_runs=5 -v expected_tuples=2 -v expected_processes=10 \
        -f scripts/validate-realdata-results.awk "$1"
}

validate "$baseline" >/dev/null

expect_failure() {
    local name="$1"
    local expected="$2"
    local input="$3"
    local log="$tmp_dir/${name}.log"
    if validate "$input" >"$log" 2>&1; then
        printf 'seeded protocol violation passed: %s\n' "$name" >&2
        exit 1
    fi
    if ! grep -q "$expected" "$log"; then
        printf 'wrong guard fired for %s; expected %s\n' "$name" "$expected" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'caught %s: %s\n' "$name" "$expected"
}

awk -F '\t' 'BEGIN { OFS=FS } $3 == "croaring" { $6="0x3333" } { print }' \
    "$baseline" >"$tmp_dir/cross-digest.tsv"
expect_failure cross-digest DigestCrossImplementationMismatch "$tmp_dir/cross-digest.tsv"

awk -F '\t' 'BEGIN { OFS=FS } NR == 1 { $6="0x3333" } { print }' \
    "$baseline" >"$tmp_dir/repeat-digest.tsv"
expect_failure repeat-digest DigestRepeatMismatch "$tmp_dir/repeat-digest.tsv"

awk -F '\t' 'BEGIN { OFS=FS } NR == 1 { $7="0x4444" } { print }' \
    "$baseline" >"$tmp_dir/fingerprint.tsv"
expect_failure fingerprint CorpusFingerprintMismatch "$tmp_dir/fingerprint.tsv"

awk -F '\t' 'BEGIN { OFS=FS } $3 == "croaring" { $8=101 } { print }' \
    "$baseline" >"$tmp_dir/cardinality.tsv"
expect_failure cardinality SourceCardinalityMismatch "$tmp_dir/cardinality.tsv"

awk -F '\t' 'BEGIN { OFS=FS } NR == 1 { $9=11 } { print }' \
    "$baseline" >"$tmp_dir/histogram.tsv"
expect_failure histogram HistogramRepeatMismatch "$tmp_dir/histogram.tsv"

sed '$d' "$baseline" >"$tmp_dir/process-count.tsv"
expect_failure process-count ProcessCountMismatch "$tmp_dir/process-count.tsv"

printf 'real-data protocol controls: OK\n'
