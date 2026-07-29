#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-serialize-diag -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_serialize_diag"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then
    worker="${worker}.exe"
fi
if [[ ! -x "$worker" ]]; then
    printf 'serialize diagnostic worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/serialize-diag-${stamp}"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
cells=(temp-writer direct-writer temp-direct direct-direct croaring)

"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

for cell in "${cells[@]}"; do
    run=1
    while (( run <= runs )); do
        output="${prefix}-${cell}-run${run}.txt"
        printf 'run %s/%s cell=%s\n' "$run" "$runs" "$cell"
        "$worker" "--cell=${cell}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        validation_count="$(awk -F '\t' '$1 == "VALIDATION" { count++ } END { print count + 0 }' "$output")"
        alloc_count="$(awk -F '\t' '$1 == "ALLOC" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 || "$validation_count" != 1 || "$alloc_count" != 1 ]]; then
            printf 'invalid worker protocol in %s\n' "$output" >&2
            exit 1
        fi

        result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 }' "$output")"
        IFS=$'\t' read -r result_cell median_ns <<<"$result_line"
        if [[ "$result_cell" != "$cell" || ! "$median_ns" =~ ^[1-9][0-9]*$ ]]; then
            printf 'invalid RESULT in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run++))
    done
done

sort -t $'\t' -k1,1 -k2,2n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" cell "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        if (count != 0 && $1 != cell) {
            emit()
            delete values
            count = 0
        }
        cell = $1
        values[++count] = $2
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Serialize fixed-buffer diagnosis: %s independent processes per cell\n' "$runs"
    printf '=======================================================================\n'
    cat "$header_file"
    printf '\n%-16s %14s %14s %14s %12s\n' cell 'median ms' 'min ms' 'max ms' 'vs CRoaring'
    awk -F '\t' '
        $1 == "AGG" { median[$2] = $3; low[$2] = $4; high[$2] = $5 }
        END {
            order[1] = "temp-writer"
            order[2] = "direct-writer"
            order[3] = "temp-direct"
            order[4] = "direct-direct"
            order[5] = "croaring"
            for (i = 1; i <= 5; i++) {
                cell = order[i]
                printf "%-16s %14.3f %14.3f %14.3f %11.3fx\n", cell, median[cell] / 1000000, low[cell] / 1000000, high[cell] / 1000000, median[cell] / median["croaring"]
            }
            printf "\nFactorial comparisons (median ratios)\n"
            printf "remove temps with Writer: %.3fx\n", median["direct-writer"] / median["temp-writer"]
            printf "bypass Writer with temps: %.3fx\n", median["temp-direct"] / median["temp-writer"]
            printf "remove temps with direct output: %.3fx\n", median["direct-direct"] / median["temp-direct"]
            printf "bypass Writer with direct construction: %.3fx\n", median["direct-direct"] / median["direct-writer"]
        }
    ' "$aggregate_rows"

    printf '\nAllocation structure\n'
    for cell in "${cells[@]}"; do
        awk -F '\t' '$1 == "ALLOC" { print; exit }' "${prefix}-${cell}-run1.txt"
    done
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
