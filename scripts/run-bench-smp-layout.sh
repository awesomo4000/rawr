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

zig build bench-smp-layout -Dcpu=native

worker="./zig-out/bin/bench_smp_layout"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'SMP layout worker not found\n' >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/smp-layout-${stamp}"
process_times="${prefix}-process-times.tsv"
process_addresses="${prefix}-process-addresses.tsv"
aggregate_times="${prefix}-aggregate-times.tsv"
aggregate_addresses="${prefix}-aggregate-addresses.tsv"
summary="${prefix}-summary.txt"
header="${prefix}-header.txt"
: >"$process_times"
: >"$process_addresses"
"$worker" --header >"$header" 2>&1

legacy_cells=(
    alloc_words
    alloc_header_words
    interleaved_words
    interleaved_header_words
    zero_order_words
    zero_order_header_words
    zero_sorted_words
    zero_sorted_header_words
    sort_zero_words
    sort_zero_header_words
)

spec45_cells=(
    scattered_interleaved
    batched_unsorted
    batched_sorted
    chunked_256k
    chunked_1m
    chunked_4m
)

case "${BENCH_SCOPE:-all}" in
    all) cells=("${legacy_cells[@]}" "${spec45_cells[@]}") ;;
    legacy) cells=("${legacy_cells[@]}") ;;
    spec45) cells=("${spec45_cells[@]}") ;;
    *)
        printf 'BENCH_SCOPE must be all, legacy, or spec45\n' >&2
        exit 2
        ;;
esac

for allocator in smp libc; do
    for cell in "${cells[@]}"; do
        printf 'run allocator=%s cell=%s processes=%s\n' "$allocator" "$cell" "$runs"
        for run in $(seq 1 "$runs"); do
            output="${prefix}-${allocator}-${cell}-run${run}.txt"
            "$worker" "--allocator=${allocator}" "--cell=${cell}" >"$output" 2>&1

            result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
            address_count="$(awk -F '\t' '$1 == "ADDRESS" { count++ } END { print count + 0 }' "$output")"
            validation_count="$(awk -F '\t' '$1 == "VALIDATION" && $4 == "ok" { count++ } END { print count + 0 }' "$output")"
            if [[ "$result_count" != 1 || "$address_count" != 1 || "$validation_count" != 1 ]]; then
                printf 'invalid worker output: %s\n' "$output" >&2
                exit 1
            fi

            awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 }' "$output" >>"$process_times"
            awk -F '\t' '
                $1 == "ADDRESS" {
                    print $2 "\t" $3 "\tspan\t" $4
                    print $2 "\t" $3 "\tmedian_stride\t" $5
                    print $2 "\t" $3 "\tadjacent_pairs\t" $6
                    print $2 "\t" $3 "\tmonotonic_pairs\t" $7
                }
            ' "$output" >>"$process_addresses"
        done
    done
done

sort -t $'\t' -k1,1 -k2,2 -k3,3n "$process_times" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" allocator "\t" cell "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        allocator = $1
        cell = $2
        previous = key
        values[++count] = $3
    }
    END { emit() }
' >"$aggregate_times"

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$process_addresses" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "ADDRESS_AGG\t" allocator "\t" cell "\t" metric "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        allocator = $1
        cell = $2
        metric = $3
        previous = key
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_addresses"

{
    printf 'Standalone SMP allocator address-order diagnosis\n'
    printf '================================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header"

    printf '\n%-5s %-29s %11s %11s %11s\n' alloc cell 'median ms' 'min ms' 'max ms'
    awk -F '\t' '$1 == "AGG" { printf "%-5s %-29s %11.3f %11.3f %11.3f\n", $2, $3, $4 / 1000000, $5 / 1000000, $6 / 1000000 }' "$aggregate_times"

    if [[ "${BENCH_SCOPE:-all}" != spec45 ]]; then
        printf '\nAddress-order controls\n'
        printf '%s\n' '----------------------'
        awk -F '\t' '
            $1 == "AGG" { median[$2 SUBSEP $3] = $4 }
            function value(allocator, cell) { return median[allocator SUBSEP cell] }
            function ms(value) { return value / 1000000 }
            END {
                for (n = 1; n <= 2; n++) {
                    allocator = n == 1 ? "smp" : "libc"
                    words_recovery = value(allocator, "zero_order_words") - value(allocator, "zero_sorted_words")
                    header_recovery = value(allocator, "zero_order_header_words") - value(allocator, "zero_sorted_header_words")
                    header_penalty = value(allocator, "zero_order_header_words") - value(allocator, "zero_order_words")
                    printf "%s allocation-order -> address-order zero recovery: words %.3f ms, header+words %.3f ms\n", allocator, ms(words_recovery), ms(header_recovery)
                    printf "%s interleaved 16-byte header zeroing penalty: %.3f ms\n", allocator, ms(header_penalty)
                    printf "%s sort+zero total: words %.3f ms, header+words %.3f ms\n", allocator, ms(value(allocator, "sort_zero_words")), ms(value(allocator, "sort_zero_header_words"))
                }
            }
        ' "$aggregate_times"
    fi

    if [[ "${BENCH_SCOPE:-all}" != legacy ]]; then
        printf '\nSpec 45 chunked-allocation screen\n'
        printf '%s\n' '---------------------------------'
        awk -F '\t' '
            $1 == "AGG" { median[$2 SUBSEP $3] = $4; low[$2 SUBSEP $3] = $5; high[$2 SUBSEP $3] = $6 }
            function value(table, allocator, cell) { return table[allocator SUBSEP cell] }
            function ms(value) { return value / 1000000 }
            END {
                for (n = 1; n <= 2; n++) {
                    allocator = n == 1 ? "smp" : "libc"
                    available = value(median, allocator, "batched_unsorted") - value(median, allocator, "batched_sorted")
                    control_clear = value(high, allocator, "batched_sorted") < value(low, allocator, "batched_unsorted")
                    printf "%s available %.3f ms, control-ranges-clear=%s\n", allocator, ms(available), control_clear ? "yes" : "no"
                    best = 0
                    selected = ""
                    for (c = 1; c <= 3; c++) {
                        cell = c == 1 ? "chunked_256k" : (c == 2 ? "chunked_1m" : "chunked_4m")
                        if (best == 0 || value(median, allocator, cell) < best) best = value(median, allocator, cell)
                        recovered = value(median, allocator, "scattered_interleaved") - value(median, allocator, cell)
                        share = available == 0 ? 0 : recovered / available
                        candidate_faster = value(high, allocator, cell) < value(low, allocator, "scattered_interleaved")
                        ranges_disjoint = candidate_faster || value(high, allocator, "scattered_interleaved") < value(low, allocator, cell)
                        screen = available > 0 && recovered >= 0.50 * available && candidate_faster
                        decision = allocator == "smp" ? (screen ? "GO" : "NO-GO") : "report-only"
                        printf "%s %-12s recovered %.3f ms, share %.1f%%, ranges-disjoint=%s, candidate-faster=%s, decision=%s\n", allocator, cell, ms(recovered), share * 100, ranges_disjoint ? "yes" : "no", candidate_faster ? "yes" : "no", decision
                    }
                    if (allocator == "smp") {
                        for (c = 1; c <= 3; c++) {
                            cell = c == 1 ? "chunked_256k" : (c == 2 ? "chunked_1m" : "chunked_4m")
                            if (selected == "" && value(median, allocator, cell) <= best * 1.05) selected = cell
                        }
                        printf "%s host-local smallest within 5%% of best: %s\n", allocator, selected
                    }
                }
            }
        ' "$aggregate_times"
    fi

    printf '\nAddress statistics (allocation order)\n'
    printf '%-5s %-29s %-16s %16s %16s %16s\n' alloc cell metric median min max
    awk -F '\t' '$1 == "ADDRESS_AGG" { printf "%-5s %-29s %-16s %16d %16d %16d\n", $2, $3, $4, $5, $6, $7 }' "$aggregate_addresses"

    printf '\nInterpretation\n'
    printf '%s\n' '- zero_sorted sorts outside timing and isolates traversal order; it is diagnostic only.'
    printf '%s\n' '- sort_zero includes sorting and reports the full cost; it is not a production proposal.'
    printf '%s\n' '- This executable imports no rawr or CRoaring code.'
    if [[ "${BENCH_SCOPE:-all}" != legacy ]]; then
        printf '%s\n' '- Spec 45 cells time allocation; retained result teardown remains outside timing.'
    fi
} | tee "$summary"

printf '\nSaved to: %s\n' "$summary"
