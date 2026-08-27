#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

sweep_prefix="${SWEEP_PREFIX:-}"
if [[ -z "$sweep_prefix" ]]; then
    printf 'SWEEP_PREFIX must name an accepted 48-01 result prefix\n' >&2
    printf 'example: SWEEP_PREFIX=misc/tiny-bench-20260823-084027\n' >&2
    exit 2
fi
if [[ ! -f "${sweep_prefix}-aggregate.tsv" || ! -f "${sweep_prefix}-accounting.tsv" ]]; then
    printf 'missing sweep aggregate/accounting files for prefix: %s\n' "$sweep_prefix" >&2
    exit 2
fi

build_args=(bench-tiny-mixed-worker bench-tiny-setup -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_tiny_mixed_worker"
setup="./zig-out/bin/bench_tiny_setup"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$setup" && -x "${setup}.exe" ]]; then setup="${setup}.exe"; fi
if [[ ! -x "$worker" || ! -x "$setup" ]]; then
    printf 'tiny mixed benchmark executables not found\n' >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/tiny-mixed-bench-${stamp}"
check_file="${prefix}-fixture-check.txt"
manifest_file="${prefix}-manifest.tsv"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
accounting_rows="${prefix}-accounting.tsv"
summary="${prefix}-summary.txt"

# The full corpus hash is asserted in this separate process before timing.
"$setup" check >"$check_file" 2>&1
"$worker" --list >"$manifest_file" 2>&1
"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

meta_count="$(awk -F '\t' '$1 == "MIXED_META" { count++ } END { print count + 0 }' "$manifest_file")"
cell_count="$(awk -F '\t' '$1 == "MIXED_CELL" { count++ } END { print count + 0 }' "$manifest_file")"
tuple_count="$(awk -F '\t' '$1 == "MIXED_TUPLE" { count++ } END { print count + 0 }' "$manifest_file")"
manifest_total="$(awk -F '\t' '$1 == "MIXED_META" { print $2 }' "$manifest_file")"
band_total="$(awk -F '\t' '$1 == "MIXED_CELL" && $2 == "band" { total += $4 } END { print total + 0 }' "$manifest_file")"
zero_count="$(awk -F '\t' '$1 == "MIXED_CELL" && $3 == "0" { print $4 }' "$manifest_file")"
if [[ "$meta_count" != 1 || "$cell_count" != 8 || "$tuple_count" != 17 || \
      "$manifest_total" != 100000 || "$band_total" != 100000 || "$zero_count" != 0 ]]; then
    printf 'invalid mixed manifest: meta=%s cells=%s tuples=%s total=%s bands=%s zero=%s\n' \
        "$meta_count" "$cell_count" "$tuple_count" "$manifest_total" "$band_total" "$zero_count" >&2
    exit 1
fi

selected=0
while IFS=$'\t' read -r kind cell band implementation allocator manifest_batch; do
    [[ "$kind" == "MIXED_TUPLE" ]] || continue
    if ! [[ "$manifest_batch" =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid manifest batch for %s/%s/%s/%s\n' "$cell" "$band" "$implementation" "$allocator" >&2
        exit 1
    fi
    ((selected+=1))
    run=1
    while (( run <= runs )); do
        safe_band="${band//+/-plus}"
        output="${prefix}-${cell}-${safe_band}-${implementation}-${allocator}-run${run}.txt"
        printf 'run %s/%s cell=%s band=%s implementation=%s allocator=%s\n' \
            "$run" "$runs" "$cell" "$band" "$implementation" "$allocator"
        command=(
            "$worker"
            "--cell=${cell}"
            "--implementation=${implementation}"
            "--allocator=${allocator}"
        )
        if [[ "$cell" == "band" ]]; then command+=("--band=${band}"); fi
        "${command[@]}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "MIXED_RESULT" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 ]]; then
            printf 'expected one MIXED_RESULT from %s, got %s\n' "$output" "$result_count" >&2
            exit 1
        fi
        result_line="$(awk -F '\t' '$1 == "MIXED_RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$output")"
        IFS=$'\t' read -r result_cell result_band result_impl result_allocator result_batch result_median \
            <<<"$result_line"
        if [[ "$result_cell" != "$cell" || "$result_band" != "$band" || \
              "$result_impl" != "$implementation" || "$result_allocator" != "$allocator" || \
              "$result_batch" != "$manifest_batch" || ! "$result_median" =~ ^[0-9]+$ ]]; then
            printf 'invalid MIXED_RESULT tuple in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run+=1))
    done
done <"$manifest_file"

if (( selected != 17 )); then
    printf 'expected 17 selected timing tuples, got %s\n' "$selected" >&2
    exit 1
fi

tab=$'\t'
LC_ALL=C sort -t "$tab" -k1,1 -k2,2 -k3,3 -k4,4 -k6,6n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "MIXED_AGG\t" cell "\t" band "\t" implementation "\t" allocator "\t" batch \
            "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3 SUBSEP $4
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        cell = $1
        band = $2
        implementation = $3
        allocator = $4
        batch = $5
        previous = key
        values[++count] = $6
    }
    END { emit() }
' >"$aggregate_rows"

process_count="$(wc -l <"$process_rows" | tr -d ' ')"
aggregate_count="$(wc -l <"$aggregate_rows" | tr -d ' ')"
if [[ "$process_count" != $((selected * runs)) || "$aggregate_count" != "$selected" ]]; then
    printf 'unexpected result counts: process=%s aggregate=%s\n' "$process_count" "$aggregate_count" >&2
    exit 1
fi

# Accounting is isolated from every timed process and runs only after timing.
"$setup" mixed_accounting >"$accounting_rows" 2>&1

{
    printf 'Tiny bitmap mixed-corpus measurement\n'
    printf '====================================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    printf '\n'
    awk -f scripts/summarize-tiny-mixed-bench.awk \
        "$aggregate_rows" \
        "$accounting_rows" \
        "${sweep_prefix}-aggregate.tsv" \
        "${sweep_prefix}-accounting.tsv"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
printf 'raw process rows: %s\n' "$process_rows"
printf 'allocation accounting: %s\n' "$accounting_rows"
