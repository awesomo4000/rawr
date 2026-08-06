#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p misc

runs="${RUNS:-5}"
case "$(uname -m)" in
    arm64|aarch64) gate_host="m4" ;;
    *) gate_host="zen4" ;;
esac
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/lazy-or-attribution-${stamp}"
rows="${prefix}-rows.tsv"
aggregate="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"

if (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be odd and at least 5 (got %s)\n' "$runs" >&2
    exit 2
fi

zig build bench-lazy-or-attribution -Dcpu=native
: >"$rows"

for run in $(seq 1 "$runs"); do
    output="${prefix}-run${run}.txt"
    printf 'run %s/%s\n' "$run" "$runs"
    ./zig-out/bin/bench_lazy_or_attribution >"$output" 2>&1
    awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 }' "$output" >>"$rows"
done

sort -t $'\t' -k1,1 -k2,2 -k3,3n "$rows" | awk -F '\t' '
    function emit(    mid) {
        if (count == 0) return
        mid = values[int((count + 1) / 2)]
        print component "\t" variant "\t" mid "\t" values[1] "\t" values[count]
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
' >"$aggregate"

{
    printf 'Lazy-OR construction attribution: %s independent processes\n' "$runs"
    printf '============================================================\n\n'
    grep '^#' "${prefix}-run1.txt"
    printf '\n'
    grep -E '^(COUNT|MATERIALIZATION|LIFECYCLE|ACCOUNT|ACCOUNT_DELTA|TEARDOWN|VALIDATION)' "${prefix}-run1.txt"
    printf '\n%-36s %-12s %12s %12s %12s\n' component variant 'median ms' 'min ms' 'max ms'
    awk -F '\t' '{ printf "%-36s %-12s %12.3f %12.3f %12.3f\n", $1, $2, $3 / 1000000, $4 / 1000000, $5 / 1000000 }' "$aggregate"

    printf '\nStop-gate decision\n'
    printf '%s\n' '------------------'
    awk -F '\t' -v gate_host="$gate_host" '
        function cell(component, variant) { return component SUBSEP variant }
        function pass(value, limit) { return value <= limit ? "PASS" : "FAIL" }
        function overlap(a_min, a_max, b_min, b_max) { return a_min <= b_max && b_min <= a_max ? "overlap" : "separate" }
        {
            key = cell($1, $2)
            median[key] = $3
            low[key] = $4
            high[key] = $5
        }
        END {
            sparse_construct = cell("prototype-sparse-construction", "headered")
            sparse_construct_candidate = cell("prototype-sparse-construction", "headerless")
            sparse_combined = cell("prototype-sparse-combined", "headered")
            sparse_combined_candidate = cell("prototype-sparse-combined", "headerless")

            construct_delta = median[sparse_construct] - median[sparse_construct_candidate]
            combined_delta = median[sparse_combined] - median[sparse_combined_candidate]
            construct_ratio = median[sparse_construct_candidate] / median[sparse_construct]
            combined_ratio = median[sparse_combined_candidate] / median[sparse_combined]
            if (gate_host == "m4") {
                construct_projection = 5746000 - construct_delta
                combined_projection = 14612000 - combined_delta
                printf "sparse construction: headered %.3f ms, headerless %.3f ms, delta %.3f ms, canonical projection %.3f ms <= 3.802 ms: %s (%s ranges)\n", median[sparse_construct] / 1000000, median[sparse_construct_candidate] / 1000000, construct_delta / 1000000, construct_projection / 1000000, pass(construct_projection, 3802000), overlap(low[sparse_construct], high[sparse_construct], low[sparse_construct_candidate], high[sparse_construct_candidate])
                printf "sparse combined:     headered %.3f ms, headerless %.3f ms, delta %.3f ms, canonical projection %.3f ms <= 13.643 ms: %s (%s ranges)\n", median[sparse_combined] / 1000000, median[sparse_combined_candidate] / 1000000, combined_delta / 1000000, combined_projection / 1000000, pass(combined_projection, 13643000), overlap(low[sparse_combined], high[sparse_combined], low[sparse_combined_candidate], high[sparse_combined_candidate])
            } else {
                printf "sparse construction candidate/baseline %.3fx <= 1.05x: %s (%s ranges)\n", construct_ratio, pass(construct_ratio, 1.05), overlap(low[sparse_construct], high[sparse_construct], low[sparse_construct_candidate], high[sparse_construct_candidate])
                printf "sparse combined     candidate/baseline %.3fx <= 1.05x: %s (%s ranges)\n", combined_ratio, pass(combined_ratio, 1.05), overlap(low[sparse_combined], high[sparse_combined], low[sparse_combined_candidate], high[sparse_combined_candidate])
            }

            split("construction repair combined", phases, " ")
            dense_ok = 1
            for (i = 1; i <= 3; i++) {
                baseline = cell("prototype-dense-" phases[i], "headered")
                candidate = cell("prototype-dense-" phases[i], "headerless")
                ratio = median[candidate] / median[baseline]
                if (ratio > 1.05) dense_ok = 0
                printf "dense %-12s candidate/baseline %.3fx <= 1.05x: %s (%s ranges)\n", phases[i], ratio, pass(ratio, 1.05), overlap(low[baseline], high[baseline], low[candidate], high[candidate])
            }
            if (gate_host == "m4") {
                decision = (construct_projection <= 3802000 && combined_projection <= 13643000 && dense_ok) ? "GO" : "NO-GO"
                printf "decision: %s to 35-01 (both sparse hard gates and all dense controls must pass)\n", decision
            } else {
                decision = (construct_ratio <= 1.05 && combined_ratio <= 1.05 && dense_ok) ? "PASS" : "FAIL"
                printf "Zen 4 no-regression control: %s (all sparse and dense candidate/baseline ratios must be <= 1.05x)\n", decision
            }
        }
    ' "$aggregate"
} | tee "$summary"

printf '\nsaved aggregate: %s\nsaved summary: %s\n' "$aggregate" "$summary"
