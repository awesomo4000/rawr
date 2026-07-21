#!/bin/bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

zig build bench-transient -Dcpu=native
mkdir -p misc

timestamp="$(date +%Y%m%d-%H%M%S)"
prefix="misc/bench-transient-arena-${timestamp}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rawr-transient-arena.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

for run in 1 2 3 4 5; do
    echo "run ${run}/5"
    ./zig-out/bin/bench_transient_arena >"${prefix}-run${run}.txt" 2>&1
    awk -F '\t' '$1 == "RESULT" { print }' "${prefix}-run${run}.txt" >"${tmp_dir}/run${run}.tsv"
done

summary="${prefix}-summary.txt"
{
    sed -n '1,/Placeholder pipeline/p' "${prefix}-run1.txt"
    echo
    echo "Five-process aggregate"
    echo "======================"
    awk -F '\t' '
        function sort_values(values, count,    i, j, value) {
            for (i = 2; i <= count; i++) {
                value = values[i]
                j = i - 1
                while (j >= 1 && values[j] > value) {
                    values[j + 1] = values[j]
                    j--
                }
                values[j + 1] = value
            }
        }
        {
            key = $2 SUBSEP $3 SUBSEP $4
            if (!(key in seen)) {
                seen[key] = 1
                order[++group_count] = key
                experiment[key] = $2
                variant[key] = $3
                phase[key] = $4
            }
            count[key]++
            elapsed[key, count[key]] = $5 + 0
            for (field = 6; field <= 17; field++) metric[key, field, count[key]] = $field + 0
        }
        END {
            printf "%-13s %-11s %-12s %12s %12s %12s %12s %12s %10s %12s %12s\n", \
                "experiment", "variant", "phase", "median(ns)", "min", "p25", "p75", "max", "alloc", "requested", "peak-class"
            for (group = 1; group <= group_count; group++) {
                key = order[group]
                n = count[key]
                delete values
                for (i = 1; i <= n; i++) values[i] = elapsed[key, i]
                sort_values(values, n)
                median = values[int((n + 1) / 2)]
                p25 = values[int((n - 1) * 0.25) + 1]
                p75 = values[int((n - 1) * 0.75) + 1]

                delete alloc_values
                delete requested_values
                delete peak_values
                for (i = 1; i <= n; i++) {
                    alloc_values[i] = metric[key, 6, i]
                    requested_values[i] = metric[key, 10, i]
                    peak_values[i] = metric[key, 13, i]
                }
                sort_values(alloc_values, n)
                sort_values(requested_values, n)
                sort_values(peak_values, n)

                printf "%-13s %-11s %-12s %12d %12d %12d %12d %12d %10d %12d %12d\n", \
                    experiment[key], variant[key], phase[key], median, values[1], p25, p75, values[n], \
                    alloc_values[int((n + 1) / 2)], requested_values[int((n + 1) / 2)], peak_values[int((n + 1) / 2)]
                printf "AGGREGATE\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", \
                    experiment[key], variant[key], phase[key], median, values[1], p25, p75, values[n], \
                    alloc_values[int((n + 1) / 2)], requested_values[int((n + 1) / 2)], peak_values[int((n + 1) / 2)]
            }
        }
    ' "${tmp_dir}"/run*.tsv
} | tee "$summary"

echo
echo "Saved runs and summary under: ${prefix}-*.txt"
