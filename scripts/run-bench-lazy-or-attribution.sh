#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/lazy-or-attribution-${stamp}"
rows="${prefix}-rows.tsv"
summary="${prefix}-summary.txt"

zig build bench-lazy-or-attribution -Dcpu=native
: >"$rows"

for run in $(seq 1 "$runs"); do
    output="${prefix}-run${run}.txt"
    printf 'run %s/%s\n' "$run" "$runs"
    ./zig-out/bin/bench_lazy_or_attribution >"$output" 2>&1
    awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 }' "$output" >>"$rows"
done

{
    printf 'Lazy-OR construction attribution: %s independent processes\n' "$runs"
    printf '============================================================\n\n'
    grep '^#' "${prefix}-run1.txt"
    printf '\n'
    grep -E '^(COUNT|VALIDATION)' "${prefix}-run1.txt"
    printf '\n%-20s %-12s %12s %12s %12s\n' component variant 'median ms' 'min ms' 'max ms'
    sort -t $'\t' -k1,1 -k2,2 -k3,3n "$rows" | awk -F '\t' '
        function emit(    mid) {
            if (count == 0) return
            mid = values[int((count + 1) / 2)]
            printf "%-20s %-12s %12.3f %12.3f %12.3f\n", component, variant, mid / 1000000, values[1] / 1000000, values[count] / 1000000
        }
        {
            key = $1 SUBSEP $2
            if (count != 0 && key != previous) {
                emit()
                delete values
                count = 0
            }
            component = $1
            variant = $2
            previous = key
            values[++count] = $3
        }
        END { emit() }
    '
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
