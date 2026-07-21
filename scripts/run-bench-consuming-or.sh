#!/bin/bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

zig build bench-consuming-or -Dcpu=native
mkdir -p misc

timestamp="$(date +%Y%m%d-%H%M%S)"
prefix="misc/bench-consuming-or-${timestamp}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rawr-consuming-or.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

for run in 1 2 3 4 5; do
    echo "run ${run}/5"
    ./zig-out/bin/bench_consuming_or >"${prefix}-run${run}.txt" 2>&1
    awk -F '\t' '$1 == "RESULT" { print }' "${prefix}-run${run}.txt" >"${tmp_dir}/run${run}.tsv"
done

summary="${prefix}-summary.txt"
{
    sed -n '1,/validation: passed/p' "${prefix}-run1.txt"
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
                workload[key] = $2
                unmatched[key] = $3
                variant[key] = $4
            }
            count[key]++
            union_ns[key, count[key]] = $5 + 0
            lifecycle_ns[key, count[key]] = $6 + 0
        }
        END {
            printf "%-10s %8s %-10s %14s %14s %14s %14s %14s %14s %10s\n", \
                "workload", "unmatched", "variant", "union median", "union min", "union max", \
                "life median", "life min", "life max", "ratio"
            for (group = 1; group <= group_count; group++) {
                key = order[group]
                n = count[key]
                delete u
                delete l
                for (i = 1; i <= n; i++) {
                    u[i] = union_ns[key, i]
                    l[i] = lifecycle_ns[key, i]
                }
                sort_values(u, n)
                sort_values(l, n)
                median_index = int((n + 1) / 2)
                union_median[key] = u[median_index]
                lifecycle_median[key] = l[median_index]
                printf "%-10s %7d%% %-10s %14d %14d %14d %14d %14d %14d", \
                    workload[key], unmatched[key], variant[key], union_median[key], u[1], u[n], \
                    lifecycle_median[key], l[1], l[n]
                if (variant[key] == "consuming") {
                    baseline_key = workload[key] SUBSEP unmatched[key] SUBSEP "baseline"
                    printf " %9.3fx", union_median[key] / union_median[baseline_key]
                }
                printf "\n"
            }
        }
    ' "${tmp_dir}"/run*.tsv
    echo
    echo "Allocation attribution (deterministic; run 1)"
    echo "============================================="
    awk -F '\t' '$1 == "ALLOC" {
        printf "%-10s %7d%% %-10s total=%-5d index=%-5d matched=%-5d clones=%-5d moved=%-5d\n", \
            $2, $3, $4, $5, $6, $7, $8, $9
    }' "${prefix}-run1.txt"
} | tee "$summary"

echo
echo "Saved runs and summary under: ${prefix}-*.txt"
