#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p misc

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-parity-worker bench-lazy-residency -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

parity_worker="./zig-out/bin/bench_parity_worker"
diag_worker="./zig-out/bin/bench_lazy_or_residency"
if [[ ! -x "$parity_worker" && -x "${parity_worker}.exe" ]]; then parity_worker="${parity_worker}.exe"; fi
if [[ ! -x "$diag_worker" && -x "${diag_worker}.exe" ]]; then diag_worker="${diag_worker}.exe"; fi
if [[ ! -x "$parity_worker" || ! -x "$diag_worker" ]]; then
    printf 'lazy residency workers not found\n' >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/lazy-residency-${stamp}"
process_times="${prefix}-process-times.tsv"
process_faults="${prefix}-process-faults.tsv"
process_reuse="${prefix}-process-reuse.tsv"
aggregate_times="${prefix}-aggregate-times.tsv"
aggregate_faults="${prefix}-aggregate-faults.tsv"
aggregate_reuse="${prefix}-aggregate-reuse.tsv"
summary="${prefix}-summary.txt"
header="${prefix}-header.txt"
: >"$process_times"
: >"$process_faults"
: >"$process_reuse"

run_tuple() {
    local cell="$1" implementation="$2" run="$3" output
    output="${prefix}-${cell}-${implementation}-run${run}.txt"
    printf 'run %s/%s cell=%s implementation=%s\n' "$run" "$runs" "$cell" "$implementation"

    if [[ "$cell" == A0 ]]; then
        local allocator
        allocator=smp
        [[ "$implementation" == croaring ]] && allocator=libc
        "$parity_worker" \
            --row=lazy-or-construction \
            "--implementation=${implementation}" \
            "--allocator=${allocator}" >"$output" 2>&1
        local result_count median
        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        [[ "$result_count" == 1 ]] || { printf 'invalid A0 output: %s\n' "$output" >&2; exit 1; }
        median="$(awk -F '\t' '$1 == "RESULT" { print $7 }' "$output")"
        [[ "$median" =~ ^[0-9]+$ ]] || { printf 'invalid A0 result: %s\n' "$output" >&2; exit 1; }
        printf 'A0\t%s\t%s\n' "$implementation" "$median" >>"$process_times"
        return
    fi

    "$diag_worker" "--cell=${cell}" "--implementation=${implementation}" >"$output" 2>&1
    local result_count validation_count
    result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
    validation_count="$(awk -F '\t' '$1 == "VALIDATION" && $4 == "ok" { count++ } END { print count + 0 }' "$output")"
    if [[ "$result_count" != 1 || "$validation_count" != 1 ]]; then
        printf 'invalid diagnostic output: %s\n' "$output" >&2
        exit 1
    fi
    awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 }' "$output" >>"$process_times"
    awk -F '\t' '$1 == "FAULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8 "\t" $9 "\t" $10 "\t" $11 }' "$output" >>"$process_faults"
    awk -F '\t' '$1 == "REUSE" { fraction = $5 == 0 ? 0 : $6 / $5; print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" fraction }' "$output" >>"$process_reuse"
    if [[ "$cell" == C0 && "$implementation" == rawr && "$run" == 1 ]]; then
        grep '^#' "$output" >"$header"
    fi
}

for implementation in rawr croaring; do
    for cell in A0 C0 C1 C2 C3 C4; do
        for run in $(seq 1 "$runs"); do
            run_tuple "$cell" "$implementation" "$run"
        done
    done
done

sort -t $'\t' -k1,1 -k2,2 -k3,3n "$process_times" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" cell "\t" implementation "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        cell = $1
        implementation = $2
        previous = key
        values[++count] = $3
    }
    END { emit() }
' >"$aggregate_times"

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4 -k7,7n "$process_faults" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "FAULT_AGG\t" cell "\t" implementation "\t" phase "\t" metric "\t" source \
            "\t" valid "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
            valid = 1
        }
        if (count == 0) valid = 1
        cell = $1
        implementation = $2
        phase = $3
        metric = $4
        source = $5
        valid = valid && $6
        previous = key
        values[++count] = $7
    }
    END { emit() }
' >"$aggregate_faults"

if [[ -s "$process_reuse" ]]; then
    sort -t $'\t' -k1,1 -k2,2 -k6,6n "$process_reuse" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = fractions[int((count + 1) / 2)]
            print "REUSE_AGG\t" cell "\t" implementation "\t" middle "\t" fractions[1] "\t" fractions[count] \
                "\t" prepass_pages "\t" production_pages
        }
        {
            key = $1 SUBSEP $2
            if (count != 0 && key != previous) {
                emit()
                delete fractions
                count = 0
            }
            cell = $1
            implementation = $2
            prepass_pages = $3
            production_pages = $4
            previous = key
            fractions[++count] = $6
        }
        END { emit() }
    ' >"$aggregate_reuse"
else
    : >"$aggregate_reuse"
fi

{
    printf 'Lazy-OR page-residency diagnosis\n'
    printf '================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header"

    printf '\n%-5s %-9s %12s %12s %12s\n' cell impl 'median ms' 'min ms' 'max ms'
    awk -F '\t' '$1 == "AGG" { printf "%-5s %-9s %12.3f %12.3f %12.3f\n", $2, $3, $4 / 1000000, $5 / 1000000, $6 / 1000000 }' "$aggregate_times"

    printf '\nFault deltas (five-process median of process medians)\n'
    printf '%-5s %-9s %-10s %-8s %12s %12s %12s %-32s\n' cell impl phase metric median min max source
    awk -F '\t' '$1 == "FAULT_AGG" { printf "%-5s %-9s %-10s %-8s %12d %12d %12d %-32s\n", $2, $3, $4, $5, $8, $9, $10, $6 }' "$aggregate_faults"

    printf '\nPage-reuse proof\n'
    printf '%-5s %-9s %12s %12s %12s\n' cell impl median min max
    awk -F '\t' '$1 == "REUSE_AGG" { printf "%-5s %-9s %11.2f%% %11.2f%% %11.2f%%\n", $2, $3, $4 * 100, $5 * 100, $6 * 100 }' "$aggregate_reuse"

    printf '\nPre-registered contrasts and gates\n'
    printf '%s\n' '----------------------------------'
    awk -F '\t' '
        FILENAME == ARGV[1] && $1 == "AGG" {
            key = $2 SUBSEP $3
            median[key] = $4
            low[key] = $5
            high[key] = $6
            next
        }
        FILENAME == ARGV[2] && $1 == "FAULT_AGG" && $4 == "operation" && $5 == "primary" {
            key = $2 SUBSEP $3
            fault_valid[key] = $7
            fault[key] = $8
            next
        }
        FILENAME == ARGV[2] && $1 == "FAULT_AGG" && $4 == "prepass" && $5 == "primary" {
            key = $2 SUBSEP $3
            prepass_fault[key] = $8
            next
        }
        function K(cell, impl) { return cell SUBSEP impl }
        function abs(value) { return value < 0 ? -value : value }
        function overlap(a, b) { return low[a] <= high[b] && low[b] <= high[a] }
        function pass(value) { return value ? "PASS" : "FAIL" }
        END {
            for (n = 1; n <= 2; n++) {
                impl = n == 1 ? "rawr" : "croaring"
                a0 = K("A0", impl)
                c0 = K("C0", impl)
                anchor_diff = abs(median[a0] - median[c0]) / median[a0]
                printf "A0/C0 %-9s: median delta %.2f%% <= 5%% and ranges overlap: %s\n", impl, anchor_diff * 100, pass(anchor_diff <= 0.05 && overlap(a0, c0))
            }

            c0 = K("C0", "rawr")
            c1 = K("C1", "rawr")
            c2 = K("C2", "rawr")
            c3 = K("C3", "rawr")
            c4 = K("C4", "rawr")
            printf "rawr C3-C4 residency/cold:  %.3f ms (gate >= 0.500 ms, ranges separate): %s\n", (median[c3] - median[c4]) / 1000000, pass(median[c3] - median[c4] >= 500000 && !overlap(c3, c4))
            printf "rawr C1-C2 residency/warm:  %.3f ms\n", (median[c1] - median[c2]) / 1000000
            printf "rawr C1-C3 cache/unconditioned: %.3f ms\n", (median[c1] - median[c3]) / 1000000
            printf "rawr C2-C4 cache/conditioned:   %.3f ms\n", (median[c2] - median[c4]) / 1000000
            printf "rawr C0-C1 bookkeeping:        %.3f ms\n", (median[c0] - median[c1]) / 1000000
            if (fault[c3] == 0) {
                printf "rawr C3-C4 primary faults: C3 is zero; >=50%% reduction: FAIL\n"
            } else {
                reduction = (fault[c3] - fault[c4]) / fault[c3]
                printf "rawr C3-C4 primary faults: %.2f%% reduction >= 50%%: %s\n", reduction * 100, pass(fault_valid[c3] && fault_valid[c4] && reduction >= 0.50)
            }
            printf "rawr prepass primary faults: C1=%d C2=%d C3=%d C4=%d\n", prepass_fault[c1], prepass_fault[c2], prepass_fault[c3], prepass_fault[c4]

            cr_c0 = K("C0", "croaring")
            cr_low = median[cr_c0]
            cr_high = median[cr_c0]
            range_low = low[cr_c0]
            range_high = high[cr_c0]
            split("C1 C2 C3 C4", cells, " ")
            for (i = 1; i <= 4; i++) {
                key = K(cells[i], "croaring")
                if (median[key] < cr_low) cr_low = median[key]
                if (median[key] > cr_high) cr_high = median[key]
                if (low[key] > range_low) range_low = low[key]
                if (high[key] < range_high) range_high = high[key]
            }
            cr_move = (cr_high - cr_low) / median[cr_c0]
            printf "CRoaring C0-C4 movement: %.2f%% <= 2%% and common range overlap: %s\n", cr_move * 100, pass(cr_move <= 0.02 && range_low <= range_high)
            for (i = 1; i <= 4; i++) {
                key = K(cells[i], "croaring")
                cell_move = abs(median[key] - median[cr_c0]) / median[cr_c0]
                printf "  CRoaring %s vs C0: %.2f%% <= 2%% and ranges overlap: %s\n", cells[i], cell_move * 100, pass(cell_move <= 0.02 && overlap(cr_c0, key))
            }
        }
    ' "$aggregate_times" "$aggregate_faults"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
