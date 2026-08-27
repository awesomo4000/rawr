#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

./scripts/fetch-realdata.sh wikileaks-noquotes

build_args=(bench-array-attribution -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_array_attribution"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'array-attribution worker not found: %s\n' "$worker" >&2
    exit 1
fi

output_dir="${REALDATA_OUTPUT_DIR:-misc}"
mkdir -p "$output_dir"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="${output_dir}/array-attribution-${stamp}"
manifest_file="${prefix}-manifest.tsv"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
validation_file="${prefix}-validation.txt"
summary="${prefix}-summary.txt"

"$worker" --list >"$manifest_file" 2>&1
"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

row_count="$(awk -F '\t' '$1 == "ROW" { count++ } END { print count + 0 }' "$manifest_file")"
tuple_count="$(awk -F '\t' '$1 == "TUPLE" { count++ } END { print count + 0 }' "$manifest_file")"
if [[ "$row_count" != 2 || "$tuple_count" != 16 ]]; then
    printf 'expected 2 rows and 16 tuples, got %s rows and %s tuples\n' \
        "$row_count" "$tuple_count" >&2
    exit 1
fi

printf 'Array attribution: %s tuples, %s independent processes each\n' "$tuple_count" "$runs"
while IFS=$'\t' read -r kind operation arm; do
    [[ "$kind" == "TUPLE" ]] || continue
    run=1
    while (( run <= runs )); do
        output="${prefix}-${operation}-${arm}-run${run}.txt"
        printf 'run %s/%s operation=%s arm=%s\n' "$run" "$runs" "$operation" "$arm"
        "$worker" "--operation=${operation}" "--arm=${arm}" >"$output" 2>&1
        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 ]]; then
            printf 'expected one RESULT from %s, got %s\n' "$output" "$result_count" >&2
            exit 1
        fi
        result_line="$(awk -F '\t' '$1 == "RESULT" {
            for (field = 2; field <= NF; field++) {
                printf "%s%s", $field, (field == NF ? ORS : OFS)
            }
        }' OFS=$'\t' "$output")"
        IFS=$'\t' read -r result_operation result_arm _ <<<"$result_line"
        if [[ "$result_operation" != "$operation" || "$result_arm" != "$arm" ]]; then
            printf 'RESULT does not match requested tuple in %s\n' "$output" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run+=1))
    done
done <"$manifest_file"

expected_processes=$((tuple_count * runs))
awk -v expected_runs="$runs" -v expected_tuples="$tuple_count" \
    -v expected_processes="$expected_processes" \
    -f scripts/validate-array-attribution-results.awk "$process_rows" >"$validation_file"

tab=$'\t'
sort -t "$tab" -k1,1 -k2,2 -k3,3n "$process_rows" | awk -F '\t' '
    function emit(    middle, field) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        printf "AGG\t%s\t%s\t%s\t%s\t%s", operation, arm, middle, values[1], values[count]
        for (field = 4; field <= 22; field++) printf "\t%s", static[field]
        printf "\n"
    }
    {
        key = $1 SUBSEP $2
        if (count != 0 && key != previous) {
            emit()
            delete values
            delete static
            count = 0
        }
        operation = $1
        arm = $2
        for (field = 4; field <= 22; field++) static[field] = $field
        previous = key
        values[++count] = $3
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Array OR/ANDNOT attribution\n'
    printf '===========================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    cat "$validation_file"
    printf '\n%-13s %-25s %23s %12s %12s %-14s\n' \
        operation arm 'cycle ms [min,max]' 'ns/pair' 'ns/input' branch
    printf '%-13s %-25s %23s %12s %12s %-14s\n' \
        '-------------' '-------------------------' '-----------------------' \
        '------------' '------------' '--------------'
    awk -F '\t' '$1 == "AGG" {
        pairs = $9
        inputs = $10
        printf "%-13s %-25s %7.3f [%7.3f,%7.3f] %12.2f %12.2f %-14s\n", \
            $2, $3, $4 / 1000000.0, $5 / 1000000.0, $6 / 1000000.0, \
            (pairs == 0 ? 0 : $4 / pairs), (inputs == 0 ? 0 : $4 / inputs), $18
    }' "$aggregate_rows"

    printf '\nPair accounting and input-size distribution\n'
    awk -F '\t' '$1 == "AGG" {
        key = $2
        if (seen[key]++) next
        printf "%s pairs=%s inputs=%s bitset-path=%s matched-other=%s unmatched-left=%s unmatched-right=%s sizes=[%s,%s,%s,%s,%s]\n", \
            $2, $9, $10, $11, $12, $13, $14, $21, $22, $23, $24, $25
    }' "$aggregate_rows"

    printf '\nAttribution verdicts\n'
    awk -F '\t' '
        $1 == "AGG" {
            op = $2
            arm = $3
            median[op, arm] = $4 + 0
            minimum[op, arm] = $5 + 0
            maximum[op, arm] = $6 + 0
            conversions[op, arm] = $15 + 0
        }
        END {
            operations[1] = "pair-or"
            operations[2] = "pair-andnot"
            for (op_index = 1; op_index <= 2; op_index++) {
                op = operations[op_index]
                e1 = "e1-rawr-endtoend"; e2 = "e2-croaring-endtoend"
                a1 = "a1-rawr-scalar"; a2 = "a2-croaring-scalar"
                a3 = "a3-croaring-production"; b1 = "b1-rawr-production"
                b2 = "b2-croaring-production"; b3 = "b3-rawr-no-normalize"
                nmed = median[op,b1] - median[op,b2]
                dmed = median[op,e1] - median[op,e2]
                nmin = minimum[op,b1] - maximum[op,b2]
                nmax = maximum[op,b1] - minimum[op,b2]
                dmin = minimum[op,e1] - maximum[op,e2]
                dmax = maximum[op,e1] - minimum[op,e2]
                share = (dmed == 0 ? 0 : nmed / dmed)
                verdict = ""
                smin = 0; smax = 0
                if (dmin <= 0) {
                    verdict = "UNDEFINED"
                } else if (nmax <= 0) {
                    verdict = "FAIL"
                } else if (nmin < 0 && nmax > 0) {
                    verdict = "INCONCLUSIVE"
                } else {
                    smin = nmin / dmax
                    smax = nmax / dmin
                    verdict = (smin >= 0.70 ? "PASS" : (smax < 0.70 ? "FAIL" : "INCONCLUSIVE"))
                }
                normalization = median[op,b1] - median[op,b3]
                rawr_alloc = median[op,b3] - median[op,a1]
                scalar = median[op,a1] - median[op,a2]
                avx = median[op,a2] - median[op,a3]
                cr_alloc = median[op,b2] - median[op,a3]
                printf "%s endtoend_delta=%.3fms matched_delta=%.3fms share=%.3f interval=[%.3f,%.3f] verdict=%s\n", \
                    op, dmed / 1000000.0, nmed / 1000000.0, share, smin, smax, verdict
                printf "  terms_ms normalization=%+.3f rawr_alloc_assembly=%+.3f scalar=%+.3f avx2=%+.3f croaring_alloc_assembly=%+.3f conversions=%d\n", \
                    normalization / 1000000.0, rawr_alloc / 1000000.0, scalar / 1000000.0, \
                    avx / 1000000.0, cr_alloc / 1000000.0, conversions[op,b1]
            }
        }
    ' "$aggregate_rows"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
printf 'raw process rows: %s\n' "$process_rows"
