#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-or-many-fusion -Doptimize=ReleaseFast -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_or_many_fusion"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'orMany fusion worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/or-many-fusion-${stamp}"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

base_cases=(
    attribution-array
    attribution-bitset
    attribution-run
    cell1-baseline
    cell2-first-bitset-seed
    cell3-word-major
    cell4-seed-word-major
    cell5-bitset-ceiling-baseline
    cell5-bitset-ceiling-word-major
    cell5-bitset-ceiling-seed-word-major
    full-rawr
    full-croaring
)

run_case() {
    local selected="$1" run output result_line result_case result_batch result_median
    run=1
    while (( run <= runs )); do
        output="${prefix}-${selected}-run${run}.txt"
        printf 'run %s/%s case=%s\n' "$run" "$runs" "$selected"
        "$worker" "--case=${selected}" >"$output" 2>&1
        if [[ "$(awk -F '\t' '$1 == "VALIDATION" { n++ } END { print n + 0 }' "$output")" != 1 ]] ||
           [[ "$(awk -F '\t' '$1 == "RESULT" { n++ } END { print n + 0 }' "$output")" != 1 ]]; then
            printf 'invalid worker protocol in %s\n' "$output" >&2
            exit 1
        fi
        result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $6 "\t" $7 }' "$output")"
        IFS=$'\t' read -r result_case result_batch result_median <<<"$result_line"
        if [[ "$result_case" != "$selected" || "$result_batch" != 128 || ! "$result_median" =~ ^[1-9][0-9]*$ ]]; then
            printf 'invalid RESULT in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        printf '%s\t%s\n' "$result_case" "$result_median" >>"$process_rows"
        ((run++))
    done
}

aggregate() {
    sort -t $'\t' -k1,1 -k2,2n "$process_rows" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            print "AGG\t" selected "\t" middle "\t" values[1] "\t" values[count]
        }
        {
            if (count != 0 && $1 != selected) {
                emit()
                delete values
                count = 0
            }
            selected = $1
            values[++count] = $2
        }
        END { emit() }
    ' >"$aggregate_rows"
}

for selected in "${base_cases[@]}"; do run_case "$selected"; done
aggregate

projection="$(awk -F '\t' '
    $1 == "AGG" { median[$2] = $3 }
    END {
        attribution_total = median["attribution-array"] + median["attribution-bitset"] + median["attribution-run"]
        bitset_share = median["attribution-bitset"] / attribution_total
        strategy = median["cell4-seed-word-major"] < median["cell3-word-major"] ? "seed-word-major" : "word-major"
        ceiling_candidate = strategy == "seed-word-major" ? median["cell5-bitset-ceiling-seed-word-major"] : median["cell5-bitset-ceiling-word-major"]
        improvement = 1.0 - ceiling_candidate / median["cell5-bitset-ceiling-baseline"]
        if (improvement < 0) improvement = 0
        projected = median["full-rawr"] * (1.0 - bitset_share * improvement) / median["full-croaring"]
        printf "%s %.9f %.9f %.9f\n", strategy, bitset_share, improvement, projected
    }
' "$aggregate_rows")"
read -r strategy bitset_share ceiling_improvement projected_ratio <<<"$projection"

candidate_case="full-candidate-${strategy}"
if awk -v ratio="$projected_ratio" 'BEGIN { exit !(ratio <= 1.10) }'; then
    run_case "$candidate_case"
    aggregate
    decision="projection-go"
else
    decision="projection-no-go"
fi

{
    printf 'orMany word-major fusion diagnostic\n'
    printf '===================================\n'
    printf 'Processes per case: %s\n' "$runs"
    cat "$header_file"
    printf '\n%-42s %16s %25s\n' case 'median ns/batch' 'process range'
    awk -F '\t' '$1 == "AGG" {
        printf "%-42s %16.0f [%10.0f,%10.0f]\n", $2, $3, $4, $5
    }' "$aggregate_rows"
    printf '\nAttribution bitset share: %.4f\n' "$bitset_share"
    printf 'Ceiling improvement (%s): %.4f\n' "$strategy" "$ceiling_improvement"
    printf 'Projected full-row ratio: %.4fx (%s)\n' "$projected_ratio" "$decision"
    if [[ "$decision" == projection-go ]]; then
        awk -F '\t' -v candidate="$candidate_case" '
            $1 == "AGG" { median[$2] = $3 }
            END { printf "Direct candidate ratio: %.4fx\n", median[candidate] / median["full-croaring"] }
        ' "$aggregate_rows"
    else
        printf 'Direct candidate: skipped by projection gate\n'
    fi
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
