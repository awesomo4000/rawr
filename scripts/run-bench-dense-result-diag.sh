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
    build_args=(bench-dense-result-diag -Dcpu=native)
    case "${CROARING_AVX512:-0}" in
        0) ;;
        1) build_args+=(-Dcroaring-avx512=true) ;;
        *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
    esac
    zig build "${build_args[@]}"
fi

worker="./zig-out/bin/bench_dense_result_diag"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'dense-result diagnostic worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/dense-result-diag-${stamp}"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
cells=(baseline a b c a-c b-c)
phases=(full construction timer_control)

"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

run_tuple() {
    local op="$1" cell="$2" implementation="$3" phase="$4" run output
    run=1
    while (( run <= runs )); do
        output="${prefix}-${op}-${cell}-${implementation}-${phase}-run${run}.txt"
        printf 'run %s/%s op=%s cell=%s implementation=%s phase=%s\n' \
            "$run" "$runs" "$op" "$cell" "$implementation" "$phase"
        "$worker" "--op=${op}" "--cell=${cell}" "--implementation=${implementation}" "--phase=${phase}" >"$output" 2>&1

        if [[ "$(awk -F '\t' '$1 == "RESULT" { n++ } END { print n + 0 }' "$output")" != 1 ]] ||
           [[ "$(awk -F '\t' '$1 == "VALIDATION" { n++ } END { print n + 0 }' "$output")" != 1 ]] ||
           [[ "$(awk -F '\t' '$1 == "SHAPE" { n++ } END { print n + 0 }' "$output")" != 1 ]]; then
            printf 'invalid worker protocol: %s\n' "$output" >&2
            exit 1
        fi
        awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$output" >>"$process_rows"
        ((run++))
    done
}

for op in band bor; do
    for cell in "${cells[@]}"; do
        for phase in "${phases[@]}"; do
            run_tuple "$op" "$cell" rawr "$phase"
        done
    done
    for phase in "${phases[@]}"; do
        run_tuple "$op" baseline croaring "$phase"
    done
done

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4 -k6,6n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" op "\t" cell "\t" implementation "\t" phase "\t" batch "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        tuple = $1 FS $2 FS $3 FS $4
        if (count != 0 && tuple != previous) {
            emit()
            delete values
            count = 0
        }
        op = $1; cell = $2; implementation = $3; phase = $4; batch = $5
        previous = tuple
        values[++count] = $6
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Dense result-construction diagnosis: %s independent processes per tuple\n' "$runs"
    printf '============================================================================\n'
    cat "$header_file"
    printf '\n%-5s %-10s %14s %14s %14s %14s %12s\n' op cell 'construct ns/op' 'teardown ns/op' 'full ns/op' 'full range' 'vs CRoaring'
    awk -F '\t' '
        $1 == "AGG" {
            key = $2 SUBSEP $3 SUBSEP $4
            if ($5 == "full") {
                full[key] = $7 / $6
                low[key] = $8 / $6
                high[key] = $9 / $6
            } else if ($5 == "construction") {
                construct[key] = $7 / $6
            } else {
                control[key] = $7 / $6
            }
        }
        END {
            ops[1] = "band"; ops[2] = "bor"
            cells[1] = "baseline"; cells[2] = "a"; cells[3] = "b"
            cells[4] = "c"; cells[5] = "a-c"; cells[6] = "b-c"
            for (oi = 1; oi <= 2; oi++) {
                op = ops[oi]
                cr_key = op SUBSEP "baseline" SUBSEP "croaring"
                cr_construction = construct[cr_key] - control[cr_key]
                printf "%-5s %-10s %14.3f %14.3f %14.3f %6.1f..%-6.1f %11s\n", op, "CRoaring", cr_construction, full[cr_key] - cr_construction, full[cr_key], low[cr_key], high[cr_key], "1.000x"
                for (ci = 1; ci <= 6; ci++) {
                    cell = cells[ci]
                    key = op SUBSEP cell SUBSEP "rawr"
                    adjusted = construct[key] - control[key]
                    printf "%-5s %-10s %14.3f %14.3f %14.3f %6.1f..%-6.1f %10.3fx\n", op, cell, adjusted, full[key] - adjusted, full[key], low[key], high[key], full[key] / full[cr_key]
                }
            }
        }
    ' "$aggregate_rows"

    printf '\nPersistent allocator and FBA accounting (one untimed result per rawr cell)\n'
    for op in band bor; do
        for cell in "${cells[@]}"; do
            awk -F '\t' '$1 == "ALLOC" { print; exit }' "${prefix}-${op}-${cell}-rawr-full-run1.txt"
        done
    done
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
