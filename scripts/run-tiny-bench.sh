#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-tiny-worker bench-tiny-setup -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_tiny_worker"
setup="./zig-out/bin/bench_tiny_setup"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$setup" && -x "${setup}.exe" ]]; then setup="${setup}.exe"; fi
if [[ ! -x "$worker" || ! -x "$setup" ]]; then
    printf 'tiny benchmark executables not found\n' >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/tiny-bench-${stamp}"
manifest_file="${prefix}-manifest.tsv"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
accounting_rows="${prefix}-accounting.tsv"
summary="${prefix}-summary.txt"

"$worker" --list >"$manifest_file" 2>&1
"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

row_count="$(awk -F '\t' '$1 == "ROW" { count++ } END { print count + 0 }' "$manifest_file")"
tuple_count="$(awk -F '\t' '$1 == "TUPLE" { count++ } END { print count + 0 }' "$manifest_file")"
if [[ "$row_count" != 36 || "$tuple_count" != 180 ]]; then
    printf 'expected 36 rows and 180 tuples, got %s rows and %s tuples\n' "$row_count" "$tuple_count" >&2
    exit 1
fi

shape_filter=" ${TINY_SHAPES:-localized spread one-per-container} "
card_filter=" ${TINY_CARDINALITIES:-0 1 2 4 6 8 12 16 20 32 64 128} "
selected=0
while IFS=$'\t' read -r kind shape cardinality implementation allocator manifest_batch; do
    [[ "$kind" == "TUPLE" ]] || continue
    [[ "$shape_filter" == *" $shape "* ]] || continue
    [[ "$card_filter" == *" $cardinality "* ]] || continue
    if ! [[ "$manifest_batch" =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid manifest batch for %s/%s/%s/%s\n' "$shape" "$cardinality" "$implementation" "$allocator" >&2
        exit 1
    fi
    ((selected+=1))
    run=1
    while (( run <= runs )); do
        output="${prefix}-${shape}-${cardinality}-${implementation}-${allocator}-run${run}.txt"
        printf 'run %s/%s shape=%s card=%s implementation=%s allocator=%s\n' \
            "$run" "$runs" "$shape" "$cardinality" "$implementation" "$allocator"
        "$worker" \
            "--shape=${shape}" \
            "--cardinality=${cardinality}" \
            "--implementation=${implementation}" \
            "--allocator=${allocator}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 ]]; then
            printf 'expected one RESULT from %s, got %s\n' "$output" "$result_count" >&2
            exit 1
        fi
        result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$output")"
        IFS=$'\t' read -r result_shape result_card result_impl result_allocator result_batch result_median \
            <<<"$result_line"
        if [[ "$result_shape" != "$shape" || "$result_card" != "$cardinality" || \
              "$result_impl" != "$implementation" || "$result_allocator" != "$allocator" || \
              "$result_batch" != "$manifest_batch" || ! "$result_median" =~ ^[0-9]+$ ]]; then
            printf 'invalid RESULT tuple in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run+=1))
    done
done <"$manifest_file"

if (( selected == 0 )); then
    printf 'filters selected no timing tuples\n' >&2
    exit 2
fi

tab=$'\t'
LC_ALL=C sort -t "$tab" -k1,1 -k2,2n -k3,3 -k4,4 -k6,6n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" shape "\t" cardinality "\t" implementation "\t" allocator "\t" batch \
            "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        shape = $1
        cardinality = $2
        implementation = $3
        allocator = $4
        batch = $5
        previous = key
        values[++count] = $6
    }
    END { emit() }
' >"$aggregate_rows"

# Accounting runs in its own process and only after all timing cells.
"$setup" accounting >"$accounting_rows" 2>&1

{
    printf 'Tiny bitmap lifecycle sweep\n'
    printf '===========================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    printf '\n'
    awk -f scripts/summarize-tiny-bench.awk "$aggregate_rows" "$accounting_rows"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
printf 'raw process rows: %s\n' "$process_rows"
printf 'allocation accounting: %s\n' "$accounting_rows"
