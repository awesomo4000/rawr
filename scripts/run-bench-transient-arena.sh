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
    sed -n '1,/Phase A experiments/p' "${prefix}-run1.txt"
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
            for (group = 1; group <= group_count; group++) {
                key = order[group]
                n = count[key]
                delete values
                delete peak_values
                for (i = 1; i <= n; i++) {
                    values[i] = elapsed[key, i]
                    peak_values[i] = metric[key, 13, i]
                }
                sort_values(values, n)
                sort_values(peak_values, n)
                medians[key] = values[int((n + 1) / 2)]
                peak_medians[key] = peak_values[int((n + 1) / 2)]
            }

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

            print ""
            print "Combined ratios and gates"
            print "-------------------------"
            split("sparse-2way sparse-nway", sparse_experiments, " ")
            split("arena fba", transient_variants, " ")
            for (experiment_index = 1; experiment_index <= 2; experiment_index++) {
                experiment_name = sparse_experiments[experiment_index]
                baseline_key = experiment_name SUBSEP "baseline" SUBSEP "combined"
                croaring_key = experiment_name SUBSEP "croaring" SUBSEP "combined"
                for (variant_index = 1; variant_index <= 2; variant_index++) {
                    variant_name = transient_variants[variant_index]
                    transient_key = experiment_name SUBSEP variant_name SUBSEP "combined"
                    improvement = medians[transient_key] / medians[baseline_key]
                    gate = medians[transient_key] / medians[croaring_key]
                    memory = peak_medians[transient_key] / peak_medians[baseline_key]
                    speed_status = (gate <= 1.10 && (experiment_name == "sparse-2way" || improvement < 1.0)) ? "PASS" : "NO-GO"
                    memory_status = memory <= 1.10 ? "PASS" : "NO-GO"
                    printf "%s %-5s transient/baseline=%.3fx transient/croaring=%.3fx speed=%s peak/baseline=%.3fx memory=%s\n", \
                        experiment_name, variant_name, improvement, gate, speed_status, memory, memory_status
                }
            }

            dense_baseline_key = "dense-nway" SUBSEP "baseline" SUBSEP "combined"
            for (variant_index = 1; variant_index <= 2; variant_index++) {
                variant_name = transient_variants[variant_index]
                dense_key = "dense-nway" SUBSEP variant_name SUBSEP "combined"
                dense_ratio = medians[dense_key] / medians[dense_baseline_key]
                memory = peak_medians[dense_key] / peak_medians[dense_baseline_key]
                printf "dense-nway  %-5s transient/baseline=%.3fx peak/baseline=%.3fx\n", \
                    variant_name, dense_ratio, memory
            }

            split("sparse-nway dense-nway", nway_experiments, " ")
            for (experiment_index = 1; experiment_index <= 2; experiment_index++) {
                experiment_name = nway_experiments[experiment_index]
                baseline_key = experiment_name SUBSEP "baseline" SUBSEP "combined"
                production_key = experiment_name SUBSEP "production" SUBSEP "combined"
                printf "%s replica/production=%.3fx\n", \
                    experiment_name, medians[baseline_key] / medians[production_key]
            }
        }
    ' "${tmp_dir}"/run*.tsv
} | tee "$summary"

echo
echo "Saved runs and summary under: ${prefix}-*.txt"
