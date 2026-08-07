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

build_args=(bench-parity-worker bench-lazy-allocator -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

parity_worker="./zig-out/bin/bench_parity_worker"
diag_worker="./zig-out/bin/bench_lazy_or_allocator"
if [[ ! -x "$parity_worker" && -x "${parity_worker}.exe" ]]; then parity_worker="${parity_worker}.exe"; fi
if [[ ! -x "$diag_worker" && -x "${diag_worker}.exe" ]]; then diag_worker="${diag_worker}.exe"; fi
if [[ ! -x "$parity_worker" || ! -x "$diag_worker" ]]; then
    printf 'lazy allocator workers not found\n' >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/lazy-allocator-${stamp}"
process_times="${prefix}-process-times.tsv"
process_addresses="${prefix}-process-addresses.tsv"
aggregate_times="${prefix}-aggregate-times.tsv"
aggregate_addresses="${prefix}-aggregate-addresses.tsv"
summary="${prefix}-summary.txt"
header="${prefix}-header.txt"
: >"$process_times"
: >"$process_addresses"

run_a0() {
    local implementation="$1" allocator="$2" run="$3" output result_count median
    output="${prefix}-A0-${implementation}-${allocator}-run${run}.txt"
    printf 'run %s/%s cell=A0 implementation=%s allocator=%s\n' \
        "$run" "$runs" "$implementation" "$allocator"
    "$parity_worker" \
        --row=lazy-or-construction \
        "--implementation=${implementation}" \
        "--allocator=${allocator}" >"$output" 2>&1
    result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
    [[ "$result_count" == 1 ]] || { printf 'invalid A0 output: %s\n' "$output" >&2; exit 1; }
    median="$(awk -F '\t' '$1 == "RESULT" { print $7 }' "$output")"
    [[ "$median" =~ ^[0-9]+$ ]] || { printf 'invalid A0 result: %s\n' "$output" >&2; exit 1; }
    printf 'A0\t%s\t%s\t%s\n' "$implementation" "$allocator" "$median" >>"$process_times"
}

run_diag() {
    local cell="$1" allocator="$2" run="$3" output result_count validation_count
    output="${prefix}-${cell}-rawr-${allocator}-run${run}.txt"
    printf 'run %s/%s cell=%s implementation=rawr allocator=%s\n' \
        "$run" "$runs" "$cell" "$allocator"
    "$diag_worker" "--cell=${cell}" "--allocator=${allocator}" >"$output" 2>&1
    result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
    validation_count="$(awk -F '\t' '$1 == "VALIDATION" && $4 == "ok" { count++ } END { print count + 0 }' "$output")"
    if [[ "$result_count" != 1 || "$validation_count" != 1 ]]; then
        printf 'invalid diagnostic output: %s\n' "$output" >&2
        exit 1
    fi
    awk -F '\t' '$1 == "RESULT" { print $2 "\trawr\t" $3 "\t" $4 }' "$output" >>"$process_times"
    awk -F '\t' '
        $1 == "ADDRESS" {
            print $2 "\t" $3 "\tspan\t" $4
            print $2 "\t" $3 "\tpages\t" $5
            print $2 "\t" $3 "\tstraddling\t" $6
            print $2 "\t" $3 "\tcontiguous\t" $7
            print $2 "\t" $3 "\tmonotonic\t" $8
            print $2 "\t" $3 "\tstride_median\t" $9
            print $2 "\t" $3 "\tstride_min\t" $10
            print $2 "\t" $3 "\tstride_max\t" $11
        }
    ' "$output" >>"$process_addresses"
    if [[ "$cell" == C0 && "$allocator" == smp && "$run" == 1 ]]; then
        grep '^#' "$output" >"$header"
    fi
}

for run in $(seq 1 "$runs"); do
    run_a0 rawr smp "$run"
    run_a0 rawr libc "$run"
    run_a0 croaring libc "$run"
done

for allocator in smp libc; do
    for cell in C0 P1 P2 P3; do
        for run in $(seq 1 "$runs"); do
            run_diag "$cell" "$allocator" "$run"
        done
    done
done

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$process_times" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" cell "\t" implementation "\t" allocator "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        cell = $1
        implementation = $2
        allocator = $3
        previous = key
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_times"

if [[ -s "$process_addresses" ]]; then
    sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$process_addresses" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            print "ADDRESS_AGG\t" cell "\t" allocator "\t" metric "\t" middle "\t" values[1] "\t" values[count]
        }
        {
            key = $1 SUBSEP $2 SUBSEP $3
            if (count != 0 && key != previous) {
                emit()
                delete values
                count = 0
            }
            cell = $1
            allocator = $2
            metric = $3
            previous = key
            values[++count] = $4
        }
        END { emit() }
    ' >"$aggregate_addresses"
else
    : >"$aggregate_addresses"
fi

{
    printf 'Lazy-OR allocator cost attribution\n'
    printf '==================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header"

    printf '\n%-4s %-9s %-5s %12s %12s %12s\n' cell impl alloc 'median ms' 'min ms' 'max ms'
    awk -F '\t' '$1 == "AGG" { printf "%-4s %-9s %-5s %12.3f %12.3f %12.3f\n", $2, $3, $4, $5 / 1000000, $6 / 1000000, $7 / 1000000 }' "$aggregate_times"

    printf '\nAddress-layout statistics (five-process medians)\n'
    printf '%-4s %-5s %-16s %16s %16s %16s\n' cell alloc metric median min max
    awk -F '\t' '$1 == "ADDRESS_AGG" { printf "%-4s %-5s %-16s %16d %16d %16d\n", $2, $3, $4, $5, $6, $7 }' "$aggregate_addresses"

    printf '\nPre-registered contrasts and gates\n'
    printf '%s\n' '----------------------------------'
    awk -F '\t' '
        $1 == "AGG" {
            key = $2 SUBSEP $3 SUBSEP $4
            median[key] = $5
            low[key] = $6
            high[key] = $7
        }
        function K(cell, impl, alloc) { return cell SUBSEP impl SUBSEP alloc }
        function abs(value) { return value < 0 ? -value : value }
        function overlap(a, b) { return low[a] <= high[b] && low[b] <= high[a] }
        function pass(value) { return value ? "PASS" : "FAIL" }
        END {
            for (n = 1; n <= 2; n++) {
                alloc = n == 1 ? "smp" : "libc"
                a0 = K("A0", "rawr", alloc)
                c0 = K("C0", "rawr", alloc)
                delta = abs(median[a0] - median[c0]) / median[a0]
                printf "A0/C0 rawr/%-4s: median delta %.2f%% <= 5%% and ranges overlap: %s\n", alloc, delta * 100, pass(delta <= 0.05 && overlap(a0, c0))
            }

            smp = K("A0", "rawr", "smp")
            libc = K("A0", "rawr", "libc")
            cr = K("A0", "croaring", "libc")
            recovery = median[smp] - median[libc]
            total_gap = median[smp] - median[cr]
            if (total_gap > 0 && recovery > 0) {
                printf "canonical allocator recovery: %.3f ms / %.3f ms = %.2f%%\n", recovery / 1000000, total_gap / 1000000, 100 * recovery / total_gap
            } else {
                printf "canonical allocator recovery: N/A (rawr/SMP does not have a positive CRoaring gap on this host)\n"
            }

            p1s = K("P1", "rawr", "smp")
            p1c = K("P1", "rawr", "libc")
            p2s = K("P2", "rawr", "smp")
            p2c = K("P2", "rawr", "libc")
            zs_low = low[p2s] - high[p1s]
            zs_high = high[p2s] - low[p1s]
            zc_low = low[p2c] - high[p1c]
            zc_high = high[p2c] - low[p1c]
            alloc_sep = low[p1s] > high[p1c] || low[p1c] > high[p1s]
            zero_sep = zs_low > zc_high || zc_low > zs_high
            allocation_direction = median[p1s] > median[p1c] ? "slower" : "faster"
            zero_direction = (median[p2s] - median[p1s]) > (median[p2c] - median[p1c]) ? "slower" : "faster"
            printf "P1 allocation-only ranges separate: %s (SMP %s)\n", pass(alloc_sep), allocation_direction
            printf "P2-P1 SMP:  %.3f ms [%.3f, %.3f]\n", (median[p2s] - median[p1s]) / 1000000, zs_low / 1000000, zs_high / 1000000
            printf "P2-P1 libc: %.3f ms [%.3f, %.3f]\n", (median[p2c] - median[p1c]) / 1000000, zc_low / 1000000, zc_high / 1000000
            printf "P2-P1 intervals separate: %s (SMP %s)\n", pass(zero_sep), zero_direction

            call_overhead = alloc_sep && median[p1s] > median[p1c]
            layout = zero_sep && zs_low > zc_high
            verdict = call_overhead && layout ? "both" : (call_overhead ? "per-call allocator overhead" : (layout ? "allocator-induced layout" : "inconclusive"))
            a0s = K("A0", "rawr", "smp")
            c0s = K("C0", "rawr", "smp")
            a0c = K("A0", "rawr", "libc")
            c0c = K("C0", "rawr", "libc")
            anchors_valid = abs(median[a0s] - median[c0s]) / median[a0s] <= 0.05 && overlap(a0s, c0s) && abs(median[a0c] - median[c0c]) / median[a0c] <= 0.05 && overlap(a0c, c0c)
            if (!anchors_valid) verdict = "inconclusive (A0/C0 gate failed)"
            printf "Phase 2 corroborating verdict: %s\n", verdict
        }
    ' "$aggregate_times"
} | tee "$summary"

printf '\nSaved to: %s\n' "$summary"
