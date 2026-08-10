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
if [[ "$scope" != "smoke" && "$scope" != "canonical" && "$scope" != "noise" && "$scope" != "sweep" && "$scope" != "full" ]]; then
    printf 'BENCH_SCOPE must be smoke, canonical, noise, sweep, or full\n' >&2
    exit 2
fi

zig build bench-free-order -Dcpu=native

worker="./zig-out/bin/bench_free_order"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'free-order worker not found\n' >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/free-order-${stamp}"
process_rows="${prefix}-process.tsv"
aggregate="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
header="${prefix}-header.txt"
: >"$process_rows"
"$worker" --header >"$header" 2>&1

run_tuple() {
    corpus="$1"
    allocator="$2"
    strategy="$3"
    noise="$4"
    demotions="$5"

    printf 'run corpus=%s allocator=%s strategy=%s noise=%s demotions=%s processes=%s\n' \
        "$corpus" "$allocator" "$strategy" "$noise" "$demotions" "$runs"
    run=1
    while (( run <= runs )); do
        output="${prefix}-${corpus}-${allocator}-${strategy}-${noise}-${demotions}-run${run}.txt"
        "$worker" \
            "--corpus=${corpus}" \
            "--allocator=${allocator}" \
            "--strategy=${strategy}" \
            "--noise=${noise}" \
            "--demotions=${demotions}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        validation_count="$(awk -F '\t' '$1 == "VALIDATION" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 || "$validation_count" != 1 ]]; then
            printf 'invalid worker output: %s\n' "$output" >&2
            sed -n '1,120p' "$output" >&2
            exit 1
        fi

        awk -F '\t' '
            $1 == "RESULT" {
                key = $2 OFS $3 OFS $4 OFS $5 OFS $6
                print key OFS "construction" OFS $7
                print key OFS "repair" OFS $8
                print key OFS "scratch" OFS $9
                print key OFS "reorder" OFS $10
                print key OFS "demote_free" OFS $11
                print key OFS "teardown" OFS $12
                print key OFS "full" OFS $13
                print key OFS "peak_rss" OFS $14
                print key OFS "demoted" OFS $15
                print key OFS "arrays" OFS $16
                print key OFS "bitsets" OFS $17
                print key OFS "runs" OFS $18
                print key OFS "travel_ppm" OFS $19
                print key OFS "page_local_ppm" OFS $20
                print key OFS "descending_ppm" OFS $21
                print key OFS "fallback" OFS $22
            }
        ' OFS='\t' "$output" >>"$process_rows"
        run=$((run + 1))
    done
}

strategies="interleaved key reverse bucket radix pdq block"
sweep_strategies="${SWEEP_STRATEGIES:-$strategies}"

if [[ "$scope" == "smoke" ]]; then
    run_tuple sweep smp interleaved none 64
    run_tuple sweep smp reverse none 64
    run_tuple sweep smp bucket none 64
    run_tuple canonical smp interleaved none 16364
    run_tuple canonical smp reverse none 16364
elif [[ "$scope" == "canonical" ]]; then
    for allocator in smp libc; do
        for strategy in $strategies; do
            run_tuple canonical "$allocator" "$strategy" none 16364
        done
    done
    run_tuple canonical croaring interleaved none 16364
elif [[ "$scope" == "noise" ]]; then
    for allocator in smp libc; do
        for strategy in $strategies; do
            run_tuple canonical "$allocator" "$strategy" shared 16364
        done
    done
    run_tuple canonical croaring interleaved shared 16364
elif [[ "$scope" == "sweep" ]]; then
    for demotions in 8 64 256 1024 4096 8192 16364; do
        for allocator in smp libc; do
            for strategy in $sweep_strategies; do
                run_tuple sweep "$allocator" "$strategy" none "$demotions"
            done
        done
    done
else
    for allocator in smp libc; do
        for strategy in $strategies; do
            for noise in none shared; do
                run_tuple canonical "$allocator" "$strategy" "$noise" 16364
            done
        done
    done
    for noise in none shared; do
        run_tuple canonical croaring interleaved "$noise" 16364
    done

    for demotions in 8 64 256 1024 4096 8192 16364; do
        for allocator in smp libc; do
            for strategy in $sweep_strategies; do
                run_tuple sweep "$allocator" "$strategy" none "$demotions"
            done
        done
    done
fi

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4 -k5,5n -k6,6 -k7,7n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" corpus "\t" allocator "\t" strategy "\t" noise "\t" demotions "\t" metric "\t" middle "\t" values[1] "\t" values[count] "\t" count
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4 SUBSEP $5 SUBSEP $6
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        corpus = $1
        allocator = $2
        strategy = $3
        noise = $4
        demotions = $5
        metric = $6
        previous = key
        values[++count] = $7
    }
    END { emit() }
' >"$aggregate"

{
    printf 'Deferred demote-free ordering diagnosis\n'
    printf '======================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header"
    printf '\nTiming values are median [min,max] across fresh-process medians.\n'
    printf '%-10s %-8s %-12s %-7s %9s %-13s %11s %11s %11s\n' \
        corpus alloc strategy noise demotions metric median_ms min_ms max_ms
    awk -F '\t' '
        $1 == "AGG" && ($7 == "construction" || $7 == "repair" || $7 == "scratch" ||
                         $7 == "reorder" || $7 == "demote_free" || $7 == "teardown" || $7 == "full") {
            printf "%-10s %-8s %-12s %-7s %9d %-13s %11.3f %11.3f %11.3f\n", $2, $3, $4, $5, $6, $7, $8 / 1000000, $9 / 1000000, $10 / 1000000
        }
    ' "$aggregate"
    printf '\nCounts, RSS, order quality, and fallbacks are retained in: %s\n' "$aggregate"
    printf 'Interpretation must use range separation; overlapping ranges are inconclusive.\n'
    printf 'No production path or default parity row is changed by this diagnostic.\n'
} | tee "$summary"

printf '\nSaved aggregate: %s\nSaved summary: %s\n' "$aggregate" "$summary"
