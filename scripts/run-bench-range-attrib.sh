#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

zig build bench-range-attrib -Dcpu=native
worker="./zig-out/bin/bench_range_attrib"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then
    worker="${worker}.exe"
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/range-attrib-${stamp}"
header_file="${prefix}-header.txt"
inventory_file="${prefix}-inventory.tsv"
process_file="${prefix}-process.tsv"
aggregate_file="${prefix}-aggregate.tsv"
summary_file="${prefix}-summary.txt"

"$worker" --header >"$header_file" 2>&1
: >"$inventory_file"
: >"$process_file"

for variant in "rawr smp" "rawr libc" "croaring libc"; do
    read -r implementation allocator <<<"$variant"
    "$worker" --inventory "--implementation=${implementation}" "--allocator=${allocator}" \
        >>"$inventory_file" 2>&1
done

for condition in timer-control clone-body remove-body clone-remove-body; do
    for variant in "rawr smp" "rawr libc" "croaring libc"; do
        read -r implementation allocator <<<"$variant"
        run=1
        while (( run <= runs )); do
            output="${prefix}-${condition}-${implementation}-${allocator}-run${run}.txt"
            printf 'run %s/%s condition=%s implementation=%s allocator=%s\n' \
                "$run" "$runs" "$condition" "$implementation" "$allocator"
            "$worker" "--condition=${condition}" "--implementation=${implementation}" \
                "--allocator=${allocator}" >"$output" 2>&1
            result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
            if [[ "$result_count" != 1 ]]; then
                printf 'expected one RESULT from %s, got %s\n' "$output" "$result_count" >&2
                exit 1
            fi
            awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' \
                "$output" >>"$process_file"
            ((run++))
        done
    done
done

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k6,6n "$process_file" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" condition "\t" implementation "\t" allocator "\t" unit "\t" batch \
            "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        condition = $1
        implementation = $2
        allocator = $3
        unit = $4
        batch = $5
        previous = key
        values[++count] = $6
    }
    END { emit() }
' >"$aggregate_file"

{
    printf 'Clone/removeRange attribution diagnostic\n'
    printf '========================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    printf '\nInventory (containers arrays bitsets runs allocations requested-bytes copied-bytes)\n'
    awk -F '\t' '$1 == "INVENTORY" { printf "%-9s %-5s %4s %4s %4s %4s %5s %8s %6s\n", $2, $3, $4, $5, $6, $7, $8, $9, $10 }' \
        "$inventory_file"
    printf '\n%-18s %-9s %-5s %24s\n' condition implementation alloc 'ns/op median [min,max]'
    printf '%-18s %-9s %-5s %24s\n' '------------------' '---------' '-----' '------------------------'
    awk -F '\t' '$1 == "AGG" {
        printf "%-18s %-9s %-5s %8.3f [%7.3f,%7.3f]\n", $2, $3, $4, $7/$6, $8/$6, $9/$6
    }' "$aggregate_file"
} | tee "$summary_file"

printf '\nsaved summary: %s\n' "$summary_file"
