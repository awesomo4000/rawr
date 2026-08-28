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

datasets=(uscensus2000 census1881 wikileaks-noquotes)
for dataset in "${datasets[@]}"; do
    ./scripts/fetch-realdata.sh "$dataset"
done

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
if [[ "$row_count" != 6 || "$tuple_count" != 30 ]]; then
    printf 'expected 6 rows and 30 tuples, got %s rows and %s tuples\n' \
        "$row_count" "$tuple_count" >&2
    exit 1
fi

printf 'Array scalar candidates: %s tuples, %s independent processes each\n' "$tuple_count" "$runs"
while IFS=$'\t' read -r kind dataset operation arm; do
    [[ "$kind" == "TUPLE" ]] || continue
    run=1
    while (( run <= runs )); do
        output="${prefix}-${dataset}-${operation}-${arm}-run${run}.txt"
        printf 'run %s/%s dataset=%s operation=%s arm=%s\n' \
            "$run" "$runs" "$dataset" "$operation" "$arm"
        "$worker" "--dataset=${dataset}" "--operation=${operation}" "--arm=${arm}" \
            >"$output" 2>&1
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
        IFS=$'\t' read -r result_dataset result_operation result_arm _ <<<"$result_line"
        if [[ "$result_dataset" != "$dataset" || "$result_operation" != "$operation" ||
              "$result_arm" != "$arm" ]]; then
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
sort -t "$tab" -k1,1 -k2,2 -k3,3 -k4,4n "$process_rows" | awk -F '\t' '
    function emit(    middle, field) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        printf "AGG\t%s\t%s\t%s\t%s\t%s\t%s", dataset, operation, arm, middle, values[1], values[count]
        for (field = 5; field <= 40; field++) printf "\t%s", static[field]
        printf "\n"
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            delete static
            count = 0
        }
        dataset = $1
        operation = $2
        arm = $3
        for (field = 5; field <= 40; field++) static[field] = $field
        previous = key
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Array OR/ANDNOT scalar candidates\n'
    printf '=================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    cat "$validation_file"
    printf '\n%-20s %-13s %-25s %23s %12s %12s\n' \
        dataset operation arm 'cycle ms [min,max]' 'ns/pair' 'ns/input'
    printf '%-20s %-13s %-25s %23s %12s %12s\n' \
        '--------------------' '-------------' '-------------------------' \
        '-----------------------' '------------' '------------'
    awk -F '\t' '$1 == "AGG" {
        pairs = $10
        inputs = $11
        printf "%-20s %-13s %-25s %7.3f [%7.3f,%7.3f] %12.2f %12.2f\n", \
            $2, $3, $4, $5 / 1000000.0, $6 / 1000000.0, $7 / 1000000.0, \
            (pairs == 0 ? 0 : $5 / pairs), (inputs == 0 ? 0 : $5 / inputs)
    }' "$aggregate_rows"

    printf '\nUntimed merge diagnostics\n'
    awk -F '\t' '$1 == "AGG" {
        key = $2 SUBSEP $3
        if (seen[key]++) next
        tail_share = ($27 == 0 ? 0 : $28 / $27)
        printf "%s %s pairs=%s outputs=%s tail=%s share=%.4f tail-pairs=%s tail=[%s,%s,%s,%s,%s] decisions=[L:%s,R:%s,E:%s] streaks=%s [%s,%s,%s,%s,%s]\n", \
            $2, $3, $10, $27, $28, tail_share, $29, $30, $31, $32, $33, $34, \
            $35, $36, $37, $38, $39, $40, $41, $42, $43
    }' "$aggregate_rows"

    printf '\nTarget recovery verdicts (wikileaks-noquotes)\n'
    awk -F '\t' '
        function report(op, candidate, threshold, label,    dmed,nmed,dmin,dmax,nmin,nmax,s,smin,smax,interval,verdict) {
            dmed = median[op,a1] - median[op,a2]
            nmed = median[op,a1] - median[op,candidate]
            dmin = minimum[op,a1] - maximum[op,a2]
            dmax = maximum[op,a1] - minimum[op,a2]
            nmin = minimum[op,a1] - maximum[op,candidate]
            nmax = maximum[op,a1] - minimum[op,candidate]
            s = (dmed == 0 ? 0 : nmed / dmed)
            interval = "N/A"
            if (dmax <= 0) verdict = "NO_GAP"
            else if (dmin <= 0 && dmax > 0) verdict = "INCONCLUSIVE_GAP"
            else if (nmax <= 0) verdict = "NO_GO"
            else if (nmin < 0 && nmax > 0) verdict = "INCONCLUSIVE"
            else {
                smin = nmin / dmax
                smax = nmax / dmin
                interval = sprintf("[%.3f,%.3f]", smin, smax)
                verdict = (smin >= threshold ? "GO" : (smax < threshold ? "NO_GO" : "INCONCLUSIVE"))
            }
            if (label == "completeness") {
                if (verdict == "GO") verdict = "EXPLAINS_TERM"
                else if (verdict == "NO_GO") verdict = "RESIDUAL_REMAINS"
            }
            printf "%s %-23s %s=%.3f interval=%s verdict=%s\n", \
                op, candidate, label, s, interval, verdict
        }
        $1 == "AGG" && $2 == "wikileaks-noquotes" {
            median[$3,$4] = $5 + 0
            minimum[$3,$4] = $6 + 0
            maximum[$3,$4] = $7 + 0
        }
        END {
            a1 = "a1-rawr-scalar"; a2 = "a2-croaring-scalar"
            candidates[1] = "c1-bulk-tail"
            candidates[2] = "c2-branchy"
            candidates[3] = "c3-branchy-bulk-tail"
            operations[1] = "pair-or"; operations[2] = "pair-andnot"
            for (o = 1; o <= 2; o++) {
                for (c = 1; c <= 3; c++) report(operations[o], candidates[c], 0.50, "recovery")
                report(operations[o], "c3-branchy-bulk-tail", 1.0, "completeness")
            }
        }
    ' "$aggregate_rows"

    printf '\nControl-corpus regression checks\n'
    awk -F '\t' '
        $1 == "AGG" {
            minimum[$2,$3,$4] = $6 + 0
            maximum[$2,$3,$4] = $7 + 0
        }
        END {
            datasets[1] = "uscensus2000"; datasets[2] = "census1881"
            operations[1] = "pair-or"; operations[2] = "pair-andnot"
            candidates[1] = "c1-bulk-tail"
            candidates[2] = "c2-branchy"
            candidates[3] = "c3-branchy-bulk-tail"
            for (d = 1; d <= 2; d++) for (o = 1; o <= 2; o++) for (c = 1; c <= 3; c++) {
                dataset = datasets[d]; op = operations[o]; candidate = candidates[c]
                regression = minimum[dataset,op,candidate] > maximum[dataset,op,"a1-rawr-scalar"]
                printf "%s %s %-23s %s\n", dataset, op, candidate, (regression ? "REGRESSION" : "OK")
            }
        }
    ' "$aggregate_rows"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
printf 'raw process rows: %s\n' "$process_rows"
