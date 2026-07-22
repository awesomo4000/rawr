#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/parity-isolated-${stamp}"
rows="${prefix}-rows.tsv"
summary="${prefix}-summary.txt"

zig build bench-parity-isolated -Dcpu=native
: >"$rows"

for run in $(seq 1 "$runs"); do
    output="${prefix}-run${run}.txt"
    printf 'run %s/%s\n' "$run" "$runs"
    ./zig-out/bin/bench_parity_isolated >"$output" 2>&1
    awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 }' "$output" >>"$rows"
done

{
    printf 'Isolated parity board: %s independent processes\n' "$runs"
    printf '================================================\n\n'
    grep '^#' "${prefix}-run1.txt"
    grep '^VALIDATION' "${prefix}-run1.txt"
    printf '\n%-24s %-12s %12s %12s %12s\n' target variant 'median ms' 'min ms' 'max ms'
    sort -t $'\t' -k1,1 -k2,2 -k3,3n "$rows" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            printf "%-24s %-12s %12.3f %12.3f %12.3f\n", \
                target, variant, middle / 1000000, values[1] / 1000000, values[count] / 1000000
        }
        {
            key = $1 SUBSEP $2
            if (count != 0 && key != previous) {
                emit()
                delete values
                count = 0
            }
            target = $1
            variant = $2
            previous = key
            values[++count] = $3
        }
        END { emit() }
    '
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
