#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    build_args=(bench-remove-range-copy -Dcpu=native)
    case "${CROARING_AVX512:-0}" in
        0) ;;
        1) build_args+=(-Dcroaring-avx512=true) ;;
        *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
    esac
    zig build "${build_args[@]}"
fi

worker="./zig-out/bin/bench_remove_range_copy"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'removeRangeCopy diagnostic worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/remove-range-copy-${stamp}"
header_file="${prefix}-header.txt"
process_file="${prefix}-process.tsv"
aggregate_file="${prefix}-aggregate.tsv"
summary_file="${prefix}-summary.txt"

"$worker" --header >"$header_file" 2>&1
: >"$process_file"

run_tuple() {
    local cell="$1" implementation="$2" run output
    run=1
    while (( run <= runs )); do
        output="${prefix}-${cell}-${implementation}-run${run}.txt"
        printf 'run %s/%s cell=%s implementation=%s\n' "$run" "$runs" "$cell" "$implementation"
        "$worker" "--cell=${cell}" "--implementation=${implementation}" >"$output" 2>&1
        if [[ "$(awk -F '\t' '$1 == "RESULT" { n++ } END { print n + 0 }' "$output")" != 1 ]] ||
           [[ "$(awk -F '\t' '$1 == "VALIDATION" { n++ } END { print n + 0 }' "$output")" != 1 ]] ||
           [[ "$(awk -F '\t' '$1 == "SHAPE" { n++ } END { print n + 0 }' "$output")" != 1 ]]; then
            printf 'invalid worker protocol: %s\n' "$output" >&2
            exit 1
        fi
        awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 }' \
            "$output" >>"$process_file"
        ((run++))
    done
}

run_tuple baseline rawr
run_tuple fused-default rawr
run_tuple fused-presized rawr
run_tuple baseline croaring

sort -t $'\t' -k1,1 -k2,2 -k5,5n "$process_file" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" cell "\t" implementation "\t" unit "\t" batch \
            "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        tuple = $1 FS $2
        if (count != 0 && tuple != previous) {
            emit()
            delete values
            count = 0
        }
        cell = $1; implementation = $2; unit = $3; batch = $4
        previous = tuple
        values[++count] = $5
    }
    END { emit() }
' >"$aggregate_file"

{
    printf 'removeRangeCopy fused-construction diagnostic\n'
    printf '=============================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    printf '\n%-18s %-10s %26s %12s\n' cell impl 'ns/op median [min,max]' 'vs CRoaring'
    awk -F '\t' '
        $1 == "AGG" {
            key = $2 SUBSEP $3
            median[key] = $6 / $5
            low[key] = $7 / $5
            high[key] = $8 / $5
        }
        END {
            cr = median["baseline" SUBSEP "croaring"]
            cells[1] = "baseline"; cells[2] = "fused-default"; cells[3] = "fused-presized"
            printf "%-18s %-10s %8.3f [%8.3f,%8.3f] %11s\n", "baseline", "croaring", cr,
                low["baseline" SUBSEP "croaring"], high["baseline" SUBSEP "croaring"], "1.000x"
            for (i = 1; i <= 3; i++) {
                cell = cells[i]
                key = cell SUBSEP "rawr"
                printf "%-18s %-10s %8.3f [%8.3f,%8.3f] %10.3fx\n", cell, "rawr", median[key],
                    low[key], high[key], median[key] / cr
            }
        }
    ' "$aggregate_file"

    printf '\nAllocation accounting (container constructions, alloc calls, construction frees, requested bytes, teardown frees)\n'
    for cell in baseline fused-default fused-presized; do
        awk -F '\t' '$1 == "ALLOC" { print; exit }' "${prefix}-${cell}-rawr-run1.txt"
    done
} | tee "$summary_file"

printf '\nsaved summary: %s\n' "$summary_file"
