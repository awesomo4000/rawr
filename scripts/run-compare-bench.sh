#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

case "${1:-}" in
    ""|--parity|--parity-pilot)
        if [[ -n "${1:-}" ]]; then shift; fi
        ;;
    --dashboard)
        shift
        if (( $# != 0 )); then
            printf 'usage: %s [--dashboard]\n' "$0" >&2
            exit 2
        fi
        zig build bench-compare
        mkdir -p misc

        outfile="misc/bench-croaring-$(date +%Y%m%d-%H%M%S).txt"
        printf 'Screening dashboard only; use the default runner for parity decisions.\n\n'
        ./zig-out/bin/bench_croaring 2>&1 | tee "$outfile"
        printf '\nSaved to: %s\n' "$outfile"
        exit 0
        ;;
    *) printf 'usage: %s [--dashboard]\n' "$0" >&2; exit 2 ;;
esac
if (( $# != 0 )); then
    printf 'usage: %s [--dashboard]\n' "$0" >&2
    exit 2
fi

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-parity-worker -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_parity_worker"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then
    worker="${worker}.exe"
fi
if [[ ! -x "$worker" ]]; then
    printf 'parity worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/parity-${stamp}"
manifest_file="${prefix}-manifest.tsv"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"

"$worker" --list >"$manifest_file" 2>&1
"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

tuple_count="$(awk -F '\t' '$1 == "TUPLE" { count++ } END { print count + 0 }' "$manifest_file")"
row_count="$(awk -F '\t' '$1 == "ROW" { count++ } END { print count + 0 }' "$manifest_file")"
if [[ "$row_count" != 38 ]]; then
    printf 'expected 38 manifest rows, got %s\n' "$row_count" >&2
    exit 1
fi
printf 'Accurate parity table: %s rows, %s tuples, %s independent processes each\n' \
    "$row_count" "$tuple_count" "$runs"

while IFS=$'\t' read -r kind row implementation allocator reference_row reference_impl reference_allocator manifest_batch; do
    [[ "$kind" == "TUPLE" ]] || continue
    if ! [[ "$manifest_batch" =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid manifest batch for %s/%s/%s: %s\n' \
            "$row" "$implementation" "$allocator" "$manifest_batch" >&2
        exit 1
    fi
    run=1
    while (( run <= runs )); do
        output="${prefix}-${row}-${implementation}-${allocator}-run${run}.txt"
        printf 'run %s/%s row=%s implementation=%s allocator=%s\n' \
            "$run" "$runs" "$row" "$implementation" "$allocator"
        "$worker" \
            "--row=${row}" \
            "--implementation=${implementation}" \
            "--allocator=${allocator}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 ]]; then
            printf 'expected one RESULT from %s, got %s\n' "$output" "$result_count" >&2
            exit 1
        fi
        result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$output")"
        IFS=$'\t' read -r result_row result_impl result_allocator result_unit result_batch result_median \
            <<<"$result_line"
        if [[ "$result_row" != "$row" || "$result_impl" != "$implementation" || \
              "$result_allocator" != "$allocator" || -z "$result_unit" || \
              ! "$result_batch" =~ ^[1-9][0-9]*$ || ! "$result_median" =~ ^[0-9]+$ ]]; then
            printf 'invalid RESULT tuple in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        if [[ "$result_batch" != "$manifest_batch" ]]; then
            printf 'RESULT batch does not match manifest in %s: %s != %s\n' \
                "$output" "$result_batch" "$manifest_batch" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run++))
    done
done < <(awk -F '\t' '$1 == "TUPLE"' "$manifest_file")

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k6,6n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" row "\t" implementation "\t" allocator "\t" unit "\t" batch \
            "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        row = $1
        implementation = $2
        allocator = $3
        unit = $4
        batch = $5
        previous = key
        values[++count] = $6
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Accurate Rawr vs CRoaring parity table\n'
    printf '======================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    printf '\n%-28s %-8s %12s %25s %12s %25s %8s\n' \
        operation variant unit 'rawr median [min,max]' unit 'CR median [min,max]' ratio
    printf '%-28s %-8s %12s %25s %12s %25s %8s\n' \
        '----------------------------' '--------' '------------' '-------------------------' \
        '------------' '-------------------------' '--------'
    awk -F '\t' '
        function normalized(ns, batch, unit) {
            return unit == "ms" ? ns / 1000000.0 : ns / batch
        }
        NR == FNR {
            if ($1 != "AGG") next
            key = $2 SUBSEP $3 SUBSEP $4
            unit[key] = $5
            batch[key] = $6
            median[key] = $7
            minimum[key] = $8
            maximum[key] = $9
            next
        }
        $1 == "ROW" {
            display[$2] = $3
            next
        }
        $1 == "TUPLE" && $3 == "rawr" {
            rawr_key = $2 SUBSEP $3 SUBSEP $4
            cr_key = $5 SUBSEP $6 SUBSEP $7
            rawr_median = normalized(median[rawr_key], batch[rawr_key], unit[rawr_key])
            rawr_min = normalized(minimum[rawr_key], batch[rawr_key], unit[rawr_key])
            rawr_max = normalized(maximum[rawr_key], batch[rawr_key], unit[rawr_key])
            cr_median = normalized(median[cr_key], batch[cr_key], unit[cr_key])
            cr_min = normalized(minimum[cr_key], batch[cr_key], unit[cr_key])
            cr_max = normalized(maximum[cr_key], batch[cr_key], unit[cr_key])
            ratio = cr_median == 0 ? 0 : rawr_median / cr_median
            variant = $4 == "none" ? "default" : $4
            printf "%-28s %-8s %12s %8.3f [%7.3f,%7.3f] %12s %8.3f [%7.3f,%7.3f] %9.4gx\n", \
                display[$2], variant, unit[rawr_key], rawr_median, rawr_min, rawr_max, \
                unit[cr_key], cr_median, cr_min, cr_max, ratio
        }
    ' "$aggregate_rows" "$manifest_file"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
