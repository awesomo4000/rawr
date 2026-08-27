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

./scripts/fetch-realdata.sh

build_args=(bench-realdata -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_realdata"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'real-data worker not found: %s\n' "$worker" >&2
    exit 1
fi

output_dir="${REALDATA_OUTPUT_DIR:-misc}"
mkdir -p "$output_dir"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="${output_dir}/realdata-bench-${stamp}"
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
if [[ "$row_count" != 21 || "$tuple_count" != 42 ]]; then
    printf 'expected 21 rows and 42 tuples, got %s rows and %s tuples\n' \
        "$row_count" "$tuple_count" >&2
    exit 1
fi

printf 'Real-data comparison: %s rows, %s tuples, %s independent processes each\n' \
    "$row_count" "$tuple_count" "$runs"

while IFS=$'\t' read -r kind dataset operation implementation denominator; do
    [[ "$kind" == "TUPLE" ]] || continue
    if ! [[ "$denominator" =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid denominator for %s/%s/%s: %s\n' \
            "$dataset" "$operation" "$implementation" "$denominator" >&2
        exit 1
    fi
    run=1
    while (( run <= runs )); do
        output="${prefix}-${dataset}-${operation}-${implementation}-run${run}.txt"
        printf 'run %s/%s dataset=%s operation=%s implementation=%s\n' \
            "$run" "$runs" "$dataset" "$operation" "$implementation"
        "$worker" \
            "--dataset=${dataset}" \
            "--operation=${operation}" \
            "--implementation=${implementation}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 ]]; then
            printf 'expected one RESULT from %s, got %s\n' "$output" "$result_count" >&2
            exit 1
        fi
        result_line="$(awk -F '\t' '$1 == "RESULT" {
            print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8 "\t" \
                $9 "\t" $10 "\t" $11 "\t" $12 "\t" $13
        }' "$output")"
        IFS=$'\t' read -r result_dataset result_operation result_impl result_denominator _ \
            <<<"$result_line"
        if [[ "$result_dataset" != "$dataset" || "$result_operation" != "$operation" || \
              "$result_impl" != "$implementation" || "$result_denominator" != "$denominator" ]]; then
            printf 'RESULT does not match requested tuple in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run+=1))
    done
done <"$manifest_file"

expected_processes=$((tuple_count * runs))
awk -v expected_runs="$runs" -v expected_tuples="$tuple_count" \
    -v expected_processes="$expected_processes" \
    -f scripts/validate-realdata-results.awk "$process_rows" >"$validation_file"

tab=$'\t'
sort -t "$tab" -k1,1 -k2,2 -k3,3 -k5,5n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" dataset "\t" operation "\t" implementation "\t" denominator \
            "\t" middle "\t" values[1] "\t" values[count] "\t" digest "\t" fingerprint \
            "\t" cardinality "\t" arrays "\t" bitsets "\t" runs "\t" bytes
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        dataset = $1
        operation = $2
        implementation = $3
        denominator = $4
        digest = $6
        fingerprint = $7
        cardinality = $8
        arrays = $9
        bitsets = $10
        runs = $11
        bytes = $12
        previous = key
        values[++count] = $5
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Rawr vs CRoaring real-data comparison\n'
    printf '======================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    cat "$validation_file"
    printf '\nSource metadata (computed after timing)\n'
    awk -F '\t' '
        $1 == "AGG" {
            key = $2 SUBSEP $4
            if (seen[key]++) next
            printf "%-22s %-9s fingerprint=%s cardinality=%s containers=array:%s bitset:%s run:%s\n", \
                $2, $4, $10, $11, $12, $13, $14
        }
    ' "$aggregate_rows"
    printf '\n%-22s %-23s %8s %25s %25s %8s %12s %12s\n' \
        dataset operation ops/cycle 'rawr cycle ms [min,max]' 'CR cycle ms [min,max]' ratio \
        'rawr ns/op' 'CR ns/op'
    printf '%-22s %-23s %8s %25s %25s %8s %12s %12s\n' \
        '----------------------' '-----------------------' '--------' \
        '-------------------------' '-------------------------' '--------' '------------' '------------'
    awk -F '\t' '
        NR == FNR {
            if ($1 == "ROW") display[$2 SUBSEP $3] = $4
            next
        }
        $1 == "AGG" {
            key = $2 SUBSEP $3
            impl = $4
            denominator[key] = $5
            median[key, impl] = $6
            minimum[key, impl] = $7
            maximum[key, impl] = $8
            seen[key] = 1
        }
        END {
            for (key in seen) {
                split(key, parts, SUBSEP)
                rawr_ms = median[key, "rawr"] / 1000000.0
                cr_ms = median[key, "croaring"] / 1000000.0
                ratio = cr_ms == 0 ? 0 : rawr_ms / cr_ms
                printf "%-22s %-23s %8d %8.3f [%7.3f,%7.3f] %8.3f [%7.3f,%7.3f] %7.3fx %12.1f %12.1f\n", \
                    parts[1], display[key], denominator[key], rawr_ms, \
                    minimum[key, "rawr"] / 1000000.0, maximum[key, "rawr"] / 1000000.0, \
                    cr_ms, minimum[key, "croaring"] / 1000000.0, \
                    maximum[key, "croaring"] / 1000000.0, ratio, \
                    median[key, "rawr"] / denominator[key], \
                    median[key, "croaring"] / denominator[key]
            }
        }
    ' "$manifest_file" "$aggregate_rows" | sort
    printf '\nSerialized byte totals (reported, not required equal)\n'
    awk -F '\t' '$1 == "AGG" && $3 == "serialize-deserialize" {
        printf "%-22s %-9s %s\n", $2, $4, $15
    }' "$aggregate_rows"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
printf 'raw process rows: %s\n' "$process_rows"
