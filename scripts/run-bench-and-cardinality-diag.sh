#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/and-cardinality-diag-${stamp}"
full_rows="${prefix}-full-rows.tsv"
kernel_rows="${prefix}-kernel-rows.tsv"
summary="${prefix}-summary.txt"

zig build bench-and-cardinality-diag -Dcpu=native
: >"$full_rows"
: >"$kernel_rows"

for run in $(seq 1 "$runs"); do
    output="${prefix}-run${run}.txt"
    printf 'run %s/%s\n' "$run" "$runs"
    ./zig-out/bin/bench_and_cardinality_diag >"$output" 2>&1
    awk -F '\t' '$1 == "FULL_RESULT" { print $2 "\t" $3 }' "$output" >>"$full_rows"
    awk -F '\t' '$1 == "KERNEL_RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 }' "$output" >>"$kernel_rows"
done

{
    printf 'Skewed andCardinality diagnosis: %s independent processes\n' "$runs"
    printf '===========================================================\n\n'
    grep '^#' "${prefix}-run1.txt"

    printf '\nFull API: original 32x4096 all-hit corpus\n'
    printf '%-12s %12s %12s %12s\n' variant 'median ms' 'min ms' 'max ms'
    sort -t $'\t' -k1,1 -k2,2n "$full_rows" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            printf "%-12s %12.3f %12.3f %12.3f\n", \
                variant, middle / 1000000, values[1] / 1000000, values[count] / 1000000
        }
        {
            if (count != 0 && $1 != variant) {
                emit()
                delete values
                count = 0
            }
            variant = $1
            values[++count] = $2
        }
        END { emit() }
    '

    printf '\nDirect kernels: median process medians in ns/container\n'
    printf '%-11s %-10s %-20s %12s %12s %12s\n' case distribution kernel median min max
    sort -t $'\t' -k1,1V -k2,2 -k3,3 -k4,4n "$kernel_rows" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            printf "%-11s %-10s %-20s %12.3f %12.3f %12.3f\n", \
                case_name, distribution, kernel, middle / 1000, values[1] / 1000, values[count] / 1000
        }
        {
            key = $1 SUBSEP $2 SUBSEP $3
            if (count != 0 && key != previous) {
                emit()
                delete values
                count = 0
            }
            case_name = $1
            distribution = $2
            kernel = $3
            previous = key
            values[++count] = $4
        }
        END { emit() }
    '
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
