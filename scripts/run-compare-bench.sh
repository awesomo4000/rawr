#!/bin/bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

zig build bench-compare -Dcpu=native
mkdir -p misc

timestamp="$(date +%Y%m%d-%H%M%S)"
prefix="misc/bench-croaring-${timestamp}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rawr-bench-croaring.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

for run in 1 2 3 4 5; do
    echo "run ${run}/5"
    ./zig-out/bin/bench_croaring >"${prefix}-run${run}.txt" 2>&1
    awk -F '\t' '$1 == "RESULT" { print }' "${prefix}-run${run}.txt" >"${tmp_dir}/run${run}.tsv"
done

summary="${prefix}-summary.txt"
{
    sed -n '1,/Initializing test data.../p' "${prefix}-run1.txt"
    echo
    echo "Five-process aggregate"
    echo "======================"
    echo "Times are median milliseconds [minimum, maximum] across process runs."
    echo
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

        function timing_text(median, minimum, maximum) {
            return sprintf("%.3f [%.3f,%.3f]", median / 1000000, minimum / 1000000, maximum / 1000000)
        }

        {
            operation = $2
            if (!(operation in seen)) {
                seen[operation] = 1
                order[++operation_count] = operation
            }

            sample = ++count[operation]
            smp[operation, sample] = $3 + 0
            if ($4 != "N/A") {
                c_alloc[operation, sample] = $4 + 0
                c_count[operation]++
            }
            croaring[operation, sample] = $5 + 0
        }

        END {
            printf "%-40s %24s %24s %24s %9s %9s %9s\n", \
                "Operation", "rawr smp", "rawr c", "CRoaring", "smp/CR", "c/CR", "c/smp"
            printf "%-40s %24s %24s %24s %9s %9s %9s\n", \
                "----------------------------------------", "------------------------", "------------------------", \
                "------------------------", "---------", "---------", "---------"

            for (operation_index = 1; operation_index <= operation_count; operation_index++) {
                operation = order[operation_index]
                samples = count[operation]
                if (samples != 5) {
                    printf "error: %s has %d samples, expected 5\n", operation, samples > "/dev/stderr"
                    failed = 1
                    continue
                }

                delete values
                for (sample = 1; sample <= samples; sample++) values[sample] = smp[operation, sample]
                sort_values(values, samples)
                smp_min = values[1]
                smp_median = values[3]
                smp_max = values[5]
                smp_text = timing_text(smp_median, smp_min, smp_max)

                delete values
                for (sample = 1; sample <= samples; sample++) values[sample] = croaring[operation, sample]
                sort_values(values, samples)
                cr_min = values[1]
                cr_median = values[3]
                cr_max = values[5]
                cr_text = timing_text(cr_median, cr_min, cr_max)
                smp_cr = cr_median > 0 ? smp_median / cr_median : 0

                if (c_count[operation] == 0) {
                    printf "%-40s %24s %24s %24s %8.3fx %9s %9s\n", \
                        operation, smp_text, "N/A", cr_text, smp_cr, "N/A", "N/A"
                    printf "AGGREGATE\t%s\t%d\tN/A\t%d\t%.6f\tN/A\tN/A\t%d\t%d\tN/A\tN/A\t%d\t%d\n", \
                        operation, smp_median, cr_median, smp_cr, smp_min, smp_max, cr_min, cr_max
                    continue
                }

                if (c_count[operation] != samples) {
                    printf "error: %s has %d c allocator samples, expected %d\n", \
                        operation, c_count[operation], samples > "/dev/stderr"
                    failed = 1
                    continue
                }

                delete values
                for (sample = 1; sample <= samples; sample++) values[sample] = c_alloc[operation, sample]
                sort_values(values, samples)
                c_min = values[1]
                c_median = values[3]
                c_max = values[5]
                c_text = timing_text(c_median, c_min, c_max)
                c_cr = cr_median > 0 ? c_median / cr_median : 0
                c_smp = smp_median > 0 ? c_median / smp_median : 0

                printf "%-40s %24s %24s %24s %8.3fx %8.3fx %8.3fx\n", \
                    operation, smp_text, c_text, cr_text, smp_cr, c_cr, c_smp
                printf "AGGREGATE\t%s\t%d\t%d\t%d\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\t%d\t%d\t%d\n", \
                    operation, smp_median, c_median, cr_median, smp_cr, c_cr, c_smp, \
                    smp_min, smp_max, c_min, c_max, cr_min, cr_max
            }

            if (failed) exit 1
        }
    ' "${tmp_dir}"/run*.tsv
} | tee "$summary"

echo ""
echo "Saved runs and summary under: ${prefix}-*.txt"
