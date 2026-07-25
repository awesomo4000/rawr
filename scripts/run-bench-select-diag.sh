#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-select-diag -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_select_diag"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then
    worker="${worker}.exe"
fi
if [[ ! -x "$worker" ]]; then
    printf 'select diagnostic worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/select-diag-${stamp}"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
paths=(rawr-inline rawr-noinline croaring-zig croaring-c checksum-baseline rawr-skip rawr-intra)

"$worker" --header >"$header_file" 2>&1
: >"$process_rows"

for path in "${paths[@]}"; do
    run=1
    while (( run <= runs )); do
        output="${prefix}-${path}-run${run}.txt"
        printf 'run %s/%s path=%s\n' "$run" "$runs" "$path"
        "$worker" "--path=${path}" >"$output" 2>&1

        result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
        validation_count="$(awk -F '\t' '$1 == "VALIDATION" { count++ } END { print count + 0 }' "$output")"
        corpus_count="$(awk -F '\t' '$1 == "CORPUS" { count++ } END { print count + 0 }' "$output")"
        ranks_count="$(awk -F '\t' '$1 == "RANKS" { count++ } END { print count + 0 }' "$output")"
        if [[ "$result_count" != 1 || "$validation_count" != 1 || \
              "$corpus_count" != 1 || "$ranks_count" != 1 ]]; then
            printf 'invalid worker protocol in %s\n' "$output" >&2
            exit 1
        fi

        result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 }' "$output")"
        IFS=$'\t' read -r result_path count sum median_ns <<<"$result_line"
        if [[ "$result_path" != "$path" || ! "$count" =~ ^[1-9][0-9]*$ || \
              ! "$sum" =~ ^[0-9]+$ || ! "$median_ns" =~ ^[1-9][0-9]*$ ]]; then
            printf 'invalid RESULT in %s: %s\n' "$output" "$result_line" >&2
            exit 1
        fi
        printf '%s\n' "$result_line" >>"$process_rows"
        ((run++))
    done
done

sort -t $'\t' -k1,1 -k4,4n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" path "\t" query_count "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        if (count != 0 && $1 != path) {
            emit()
            delete values
            count = 0
        }
        path = $1
        query_count = $2
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Select call-boundary diagnosis: %s independent processes per path\n' "$runs"
    printf '=====================================================================\n'
    cat "$header_file"
    printf '\n%-20s %14s %14s %14s\n' path 'median ns/query' 'min ns/query' 'max ns/query'
    awk -F '\t' '$1 == "AGG" {
        printf "%-20s %14.3f %14.3f %14.3f\n", $2, $4 / $3, $5 / $3, $6 / $3
    }' "$aggregate_rows"

    printf '\nCall-boundary matrix\n'
    awk -F '\t' '
        $1 == "AGG" { median[$2] = $4 }
        END {
            printf "rawr inline/noinline: %.3fx\n", median["rawr-inline"] / median["rawr-noinline"]
            printf "CRoaring in-C/Zig: %.3fx\n", median["croaring-c"] / median["croaring-zig"]
            printf "inline rawr / Zig-call CRoaring: %.3fx\n", median["rawr-inline"] / median["croaring-zig"]
            printf "public-boundary rawr / CRoaring: %.3fx\n", median["rawr-noinline"] / median["croaring-c"]
        }
    ' "$aggregate_rows"

    printf '\nRawr attribution (baseline-subtracted)\n'
    awk -F '\t' '
        $1 == "AGG" { median[$2] = $4; query_count = $3 }
        END {
            baseline = median["checksum-baseline"] / query_count
            skip = (median["rawr-skip"] - median["checksum-baseline"]) / query_count
            intra = (median["rawr-intra"] - median["checksum-baseline"]) / query_count
            composed = baseline + skip + intra
            full = median["rawr-inline"] / query_count
            printf "loop/checksum baseline: %.3f ns/query\n", baseline
            printf "container skip: %.3f ns/query\n", skip
            printf "intra-container select: %.3f ns/query\n", intra
            printf "named fusion/codegen residual: %.3f ns/query\n", full - composed
        }
    ' "$aggregate_rows"

    printf '\nCorpus (one validated process)\n'
    awk -F '\t' '$1 == "CORPUS" {
        printf "queries=%s rawr[array=%s bitset=%s run=%s] CRoaring[array=%s bitset=%s run=%s]\n", $3, $4, $5, $6, $7, $8, $9
        exit
    }' "${prefix}-rawr-inline-run1.txt"
    awk -F '\t' '$1 == "RANKS" {
        printf "rank min=%s max=%s mean=%.3f\n", $3, $4, $5 / 1000000
        exit
    }' "${prefix}-rawr-inline-run1.txt"
    awk -F '\t' '$1 == "TARGET" {
        printf "container[%s]=%s%s", $3, $4, ($3 == 7 ? "\n" : " ")
    }' "${prefix}-rawr-inline-run1.txt"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
