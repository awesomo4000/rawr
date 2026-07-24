#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-iterate-diag -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_iterate_diag"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then
    worker="${worker}.exe"
fi
if [[ ! -x "$worker" ]]; then
    printf 'iterate diagnostic worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/iterate-diag-${stamp}"
header_file="${prefix}-header.txt"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
paths=(rawr-pull rawr-push croaring-pull croaring-push)

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
        if [[ "$result_count" != 1 || "$validation_count" != 1 || "$corpus_count" != 1 ]]; then
            printf 'invalid worker protocol in %s\n' "$output" >&2
            exit 1
        fi

        result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 }' "$output")"
        IFS=$'\t' read -r result_path cardinality sum median_ns <<<"$result_line"
        if [[ "$result_path" != "$path" || ! "$cardinality" =~ ^[1-9][0-9]*$ || \
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
        print "AGG\t" path "\t" cardinality "\t" sum "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        if (count != 0 && $1 != path) {
            emit()
            delete values
            count = 0
        }
        path = $1
        cardinality = $2
        sum = $3
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_rows"

if [[ "$(awk -F '\t' '$1 == "AGG" { print $3 "\t" $4 }' "$aggregate_rows" | sort -u | wc -l | tr -d ' ')" != 1 ]]; then
    printf 'path cardinality/checksum disagreement\n' >&2
    exit 1
fi

{
    printf 'Iterate four-path diagnosis: %s independent processes per path\n' "$runs"
    printf '=================================================================\n'
    cat "$header_file"
    printf '\n%-16s %14s %14s %14s\n' path 'median ns/value' 'min ns/value' 'max ns/value'
    awk -F '\t' '$1 == "AGG" {
        printf "%-16s %14.3f %14.3f %14.3f\n", $2, $5 / $3, $6 / $3, $7 / $3
    }' "$aggregate_rows"

    printf '\nAttribution\n'
    awk -F '\t' '
        $1 == "AGG" { median[$2] = $5; cardinality = $3 }
        END {
            printf "pull-vs-pull rawr/CRoaring: %.3fx\n", median["rawr-pull"] / median["croaring-pull"]
            printf "push-vs-push rawr/CRoaring: %.3fx (not like-for-like callback work)\n", median["rawr-push"] / median["croaring-push"]
            printf "rawr pull-push API-model delta: %.3f ns/value\n", (median["rawr-pull"] - median["rawr-push"]) / cardinality
            printf "CRoaring pull-push API-model delta: %.3f ns/value\n", (median["croaring-pull"] - median["croaring-push"]) / cardinality
        }
    ' "$aggregate_rows"

    printf '\nContainer mix (one validated process per path)\n'
    for path in "${paths[@]}"; do
        awk -F '\t' -v expected="$path" '$1 == "CORPUS" && $2 == expected {
            printf "%-16s cardinality=%s arrays=%s bitsets=%s runs=%s\n", $2, $3, $4, $5, $6
            exit
        }' "${prefix}-${path}-run1.txt"
    done
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
