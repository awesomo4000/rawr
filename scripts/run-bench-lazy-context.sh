#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/lazy-context-${stamp}"
rows="${prefix}-rows.tsv"
summary="${prefix}-summary.txt"

zig build bench-lazy-or-attribution bench-compare -Dcpu=native
: >"$rows"

run_focused() {
    local run="$1"
    local output="${prefix}-focused-run${run}.txt"
    ./zig-out/bin/bench_lazy_or_attribution >"$output" 2>&1
    awk -F '\t' '$1 == "RESULT" && $2 == "full" {
        print "focused\t2x9\t" $3 "\t" $4
    }' "$output" >>"$rows"
}

run_context() {
    local context="$1"
    local protocol="$2"
    local run="$3"
    local output="${prefix}-${context}-${protocol}-run${run}.txt"
    ./zig-out/bin/bench_croaring \
        "--lazy-context=${context}" "--protocol=${protocol}" >"$output" 2>&1
    awk -F '\t' '$1 == "CONTEXT_RESULT" {
        print $2 "\t" $3 "\t" $4 "\t" $5
    }' "$output" >>"$rows"
}

for run in $(seq 1 "$runs"); do
    printf 'run %s/%s focused\n' "$run" "$runs"
    run_focused "$run"
    for condition in \
        'target-only 2x9' \
        'target-only 3x21' \
        'full-init-first 3x21' \
        'full-init-last 3x21' \
        'allocator-prime 3x21' \
        'cache-prime 3x21'
    do
        set -- $condition
        printf 'run %s/%s %s %s\n' "$run" "$runs" "$1" "$2"
        run_context "$1" "$2" "$run"
    done
done

{
    printf 'Lazy-OR broad-context matrix: %s independent processes\n' "$runs"
    printf '==========================================================\n\n'
    grep '^#' "${prefix}-focused-run1.txt"
    printf '\n%-18s %-7s %-12s %12s %12s %12s\n' condition protocol variant 'median ms' 'min ms' 'max ms'
    sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$rows" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            printf "%-18s %-7s %-12s %12.3f %12.3f %12.3f\n", \
                condition, protocol, variant, middle / 1000000, values[1] / 1000000, values[count] / 1000000
        }
        {
            key = $1 SUBSEP $2 SUBSEP $3
            if (count != 0 && key != previous) {
                emit()
                delete values
                count = 0
            }
            condition = $1
            protocol = $2
            variant = $3
            previous = key
            values[++count] = $4
        }
        END { emit() }
    '
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
