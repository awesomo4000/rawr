#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p misc

runs="${RUNS:-5}"
scope="${BENCH_SCOPE:-full}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi
if [[ "$scope" != "full" && "$scope" != "smoke" ]]; then
    printf 'BENCH_SCOPE must be full or smoke\n' >&2
    exit 2
fi

zig build bench-address-sorted -Dcpu=native

worker="./zig-out/bin/bench_address_sorted"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'address-sorted worker not found\n' >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/address-sorted-${stamp}"
process_rows="${prefix}-process.tsv"
aggregate="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
header="${prefix}-header.txt"
: >"$process_rows"
"$worker" --header >"$header" 2>&1

run_tuple() {
    phase="$1"
    allocator="$2"
    strategy="$3"
    scratch="$4"
    noise="$5"
    lifecycle="$6"
    count="$7"

    printf 'run phase=%s allocator=%s strategy=%s scratch=%s noise=%s lifecycle=%s count=%s processes=%s\n' \
        "$phase" "$allocator" "$strategy" "$scratch" "$noise" "$lifecycle" "$count" "$runs"
    run=1
    while (( run <= runs )); do
        output="${prefix}-${phase}-${allocator}-${strategy}-${scratch}-${noise}-${lifecycle}-${count}-run${run}.txt"
        "$worker" \
            "--phase=${phase}" \
            "--allocator=${allocator}" \
            "--strategy=${strategy}" \
            "--scratch=${scratch}" \
            "--noise=${noise}" \
            "--lifecycle=${lifecycle}" \
            "--count=${count}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        validation_count="$(awk -F '\t' '$1 == "VALIDATION" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 || "$validation_count" != 1 ]]; then
            printf 'invalid worker output: %s\n' "$output" >&2
            exit 1
        fi

        awk -F '\t' '
            $1 == "RESULT" && $2 == "repair" {
                key = $2 OFS $3 OFS $4 OFS $5 OFS $6 OFS $7 OFS $8
                print key OFS "total" OFS $9
                print key OFS "cardinality" OFS $12
                print key OFS "peak_rss" OFS $15
            }
            $1 == "RESULT" && $2 == "teardown" {
                key = $2 OFS $3 OFS $4 OFS $5 OFS $6 OFS $7 OFS $8
                print key OFS "teardown" OFS $9
                print key OFS "refill" OFS $12
                print key OFS "traversal" OFS $15
                print key OFS "combined" OFS $18
                print key OFS "teardown_and_combined" OFS $21
                print key OFS "noise" OFS $24
                print key OFS "peak_rss" OFS $27
            }
            $1 == "RESULT" && $2 == "teardown-control" {
                key = $2 OFS $3 OFS $4 OFS $5 OFS $6 OFS $7 OFS $8
                print key OFS "teardown" OFS $9
                print key OFS "peak_rss" OFS $12
            }
        ' OFS='\t' "$output" >>"$process_rows"
        run=$((run + 1))
    done
}

if [[ "$scope" == "smoke" ]]; then
    run_tuple repair smp unsorted none none steady 8
    run_tuple repair smp payload_asc cold none steady 8
    run_tuple repair croaring unsorted none none steady 8
    run_tuple teardown smp payload_desc none shared first_cycle 8
    run_tuple teardown_control smp payload_asc none none steady 8
else
    # Repair: size-gate sweep with the honest cold and reusable payload sorts.
    for count in 8 64 256 1024 4096 8192 16364; do
        for allocator in smp libc; do
            run_tuple repair "$allocator" unsorted none none steady "$count"
            run_tuple repair "$allocator" payload_asc cold none steady "$count"
            run_tuple repair "$allocator" payload_asc reused none steady "$count"
        done
        run_tuple repair croaring unsorted none none steady "$count"
    done

    # Full-corpus header-key controls. Payload-key arms above are the candidates.
    for allocator in smp libc; do
        run_tuple repair "$allocator" header_asc cold none steady 16364
        run_tuple repair "$allocator" header_asc reused none steady 16364
    done

    # Teardown size sweep without allocator noise. The full count is covered by
    # the complete three-stage matrix below.
    for count in 8 64 256 1024 4096 8192; do
        for allocator in smp libc; do
            for strategy in unsorted payload_asc payload_desc; do
                run_tuple teardown "$allocator" "$strategy" none none steady "$count"
            done
        done
    done

    # Full teardown experiment: first-cycle and arm-specific steady state, with
    # and without the pinned shared-allocator noise workload.
    for allocator in smp libc; do
        for strategy in unsorted header_asc payload_asc payload_desc; do
            for noise in none shared; do
                for lifecycle in first_cycle steady; do
                    run_tuple teardown "$allocator" "$strategy" none "$noise" "$lifecycle" 16364
                done
            done
        done
    done
    for noise in none shared; do
        for lifecycle in first_cycle steady; do
            run_tuple teardown croaring unsorted none "$noise" "$lifecycle" 16364
        done
    done

    # Canonical clone-corpus control: eight run containers must remain below any
    # plausible sort threshold.
    for allocator in smp libc; do
        for strategy in unsorted header_asc payload_asc payload_desc; do
            run_tuple teardown_control "$allocator" "$strategy" none none steady 8
        done
    done
fi

sort -t $'\t' \
    -k1,1 -k2,2 -k3,3 -k4,4 -k5,5 -k6,6 -k7,7n -k8,8 -k9,9n \
    "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" phase "\t" allocator "\t" strategy "\t" scratch "\t" noise "\t" lifecycle "\t" containers "\t" metric "\t" middle "\t" values[1] "\t" values[count] "\t" count
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP $5 SUBSEP $6 SUBSEP $7 SUBSEP $8
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        phase = $1
        allocator = $2
        strategy = $3
        scratch = $4
        noise = $5
        lifecycle = $6
        containers = $7
        metric = $8
        previous = key
        values[++count] = $9
    }
    END { emit() }
' >"$aggregate"

{
    printf 'Address-sorted repair and teardown diagnosis\n'
    printf '=============================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header"
    printf '\nAll values below are median [min,max] across fresh processes.\n'
    printf '%-17s %-8s %-12s %-7s %-7s %-11s %7s %-12s %11s %11s %11s\n' \
        phase alloc strategy scratch noise lifecycle count metric median_ms min_ms max_ms
    awk -F '\t' '
        $1 == "AGG" && $9 != "peak_rss" {
            printf "%-17s %-8s %-12s %-7s %-7s %-11s %7d %-12s %11.3f %11.3f %11.3f\n", $2, $3, $4, $5, $6, $7, $8, $9, $10 / 1000000, $11 / 1000000, $12 / 1000000
        }
    ' "$aggregate"
    printf '\nPeak RSS by tuple is retained in: %s\n' "$aggregate"
    printf 'Interpretation must use range separation; overlapping ranges are inconclusive.\n'
    printf 'No parity-closure classification is made by this diagnostic.\n'
} | tee "$summary"

printf '\nSaved aggregate: %s\nSaved summary: %s\n' "$aggregate" "$summary"
