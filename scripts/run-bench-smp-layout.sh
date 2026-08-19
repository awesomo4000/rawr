#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p misc

runs="${RUNS:-5}"
scope="${BENCH_SCOPE:-all}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    zig build bench-smp-layout -Dcpu=native
fi

worker="${BENCH_WORKER:-./zig-out/bin/bench_smp_layout}"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'SMP layout worker not found\n' >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/smp-layout-${stamp}"
process_times="${prefix}-process-times.tsv"
process_addresses="${prefix}-process-addresses.tsv"
process_components="${prefix}-process-components.tsv"
aggregate_times="${prefix}-aggregate-times.tsv"
aggregate_addresses="${prefix}-aggregate-addresses.tsv"
aggregate_components="${prefix}-aggregate-components.tsv"
summary="${prefix}-summary.txt"
header="${prefix}-header.txt"
: >"$process_times"
: >"$process_addresses"
: >"$process_components"
"$worker" --header >"$header" 2>&1

diagnostic_cells=(
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
construction_cells=(
    construction_interleaved
    construction_batched_unsorted
    construction_batched_sorted
)

case "$scope" in
    all) cells=("${diagnostic_cells[@]}" "${construction_cells[@]}") ;;
    diagnostic) cells=("${diagnostic_cells[@]}") ;;
    construction) cells=("${construction_cells[@]}") ;;
    *)
        printf 'BENCH_SCOPE must be all, diagnostic, or construction\n' >&2
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
            component_count="$(awk -F '\t' '$1 == "COMPONENT" { count++ } END { print count + 0 }' "$output")"
            validation_count="$(awk -F '\t' '$1 == "VALIDATION" && $4 == "ok" { count++ } END { print count + 0 }' "$output")"
            if [[ "$result_count" != 1 || "$address_count" != 1 || "$component_count" != 3 || "$validation_count" != 1 ]]; then
                printf 'invalid worker output: %s\n' "$output" >&2
                exit 1
            fi

            awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 }' "$output" >>"$process_times"
            awk -F '\t' '$1 == "COMPONENT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$output" >>"$process_components"
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

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$process_components" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "COMPONENT_AGG\t" allocator "\t" cell "\t" component "\t" middle "\t" values[1] "\t" values[count]
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
        component = $3
        previous = key
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_components"

{
    printf 'Standalone SMP allocator address-order diagnosis\n'
    printf '================================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    printf 'Scope: %s\n' "$scope"
    cat "$header"

    printf '\n%-5s %-29s %11s %11s %11s\n' alloc cell 'median ms' 'min ms' 'max ms'
    awk -F '\t' '$1 == "AGG" { printf "%-5s %-29s %11.3f %11.3f %11.3f\n", $2, $3, $4 / 1000000, $5 / 1000000, $6 / 1000000 }' "$aggregate_times"

    if [[ "$scope" == "all" || "$scope" == "diagnostic" ]]; then
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

    printf '\nAddress statistics (allocation order)\n'
    printf '%-5s %-29s %-16s %16s %16s %16s\n' alloc cell metric median min max
    awk -F '\t' '$1 == "ADDRESS_AGG" { printf "%-5s %-29s %-16s %16d %16d %16d\n", $2, $3, $4, $5, $6, $7 }' "$aggregate_addresses"

    if [[ "$scope" == "all" || "$scope" == "construction" ]]; then
        printf '\nConstruction feasibility\n'
        printf '%s\n' '------------------------'
        awk -F '\t' '
            FNR == NR && $1 == "AGG" {
                median[$2 SUBSEP $3] = $4
                minimum[$2 SUBSEP $3] = $5
                maximum[$2 SUBSEP $3] = $6
                next
            }
            $1 == "COMPONENT_AGG" {
                component[$2 SUBSEP $3 SUBSEP $4] = $5
            }
            function ms(value) { return value / 1000000 }
            END {
                for (n = 1; n <= 2; n++) {
                    allocator = n == 1 ? "smp" : "libc"
                    baseline = median[allocator SUBSEP "construction_interleaved"]
                    unsorted = median[allocator SUBSEP "construction_batched_unsorted"]
                    sorted = median[allocator SUBSEP "construction_batched_sorted"]
                    ordering = unsorted - sorted
                    batching = baseline - unsorted
                    movement = unsorted == 0 ? 0 : (sorted - unsorted) * 100 / unsorted
                    zero_recovery = component[allocator SUBSEP "construction_batched_unsorted" SUBSEP "zero"] - component[allocator SUBSEP "construction_batched_sorted" SUBSEP "zero"]
                    printf "%s batching effect (baseline - unsorted): %.3f ms\n", allocator, ms(batching)
                    printf "%s ordering recovery (unsorted - sorted): %.3f ms (%+.2f%% sorted movement)\n", allocator, ms(ordering), movement
                    printf "%s sorted full range: [%.3f, %.3f] ms; unsorted full range: [%.3f, %.3f] ms\n", allocator,
                        ms(minimum[allocator SUBSEP "construction_batched_sorted"]),
                        ms(maximum[allocator SUBSEP "construction_batched_sorted"]),
                        ms(minimum[allocator SUBSEP "construction_batched_unsorted"]),
                        ms(maximum[allocator SUBSEP "construction_batched_unsorted"])
                    printf "%s sorted components: prepass %.3f ms, sort %.3f ms\n", allocator,
                        ms(component[allocator SUBSEP "construction_batched_sorted" SUBSEP "prepass"]),
                        ms(component[allocator SUBSEP "construction_batched_sorted" SUBSEP "sort"])
                    printf "%s zero traversal: sorted %.3f ms, unsorted %.3f ms, recovery %.3f ms\n", allocator,
                        ms(component[allocator SUBSEP "construction_batched_sorted" SUBSEP "zero"]),
                        ms(component[allocator SUBSEP "construction_batched_unsorted" SUBSEP "zero"]),
                        ms(zero_recovery)
                }
            }
        ' "$aggregate_times" "$aggregate_components"
    fi

    printf '\nInterpretation\n'
    printf '%s\n' '- zero_sorted sorts outside timing and isolates traversal order; it is diagnostic only.'
    printf '%s\n' '- sort_zero includes sorting and reports the full cost; it is not a production proposal.'
    printf '%s\n' '- construction_batched_sorted includes prepass, scratch, allocation, pdq sort, zeroing, and scratch release.'
    printf '%s\n' '- This executable imports no rawr or CRoaring code.'
} | tee "$summary"

printf '\nSaved to: %s\n' "$summary"
