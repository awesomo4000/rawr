#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    zig build bench-m4-cluster-diag -Dcpu=native
fi
worker="./zig-out/bin/bench_m4_cluster_diag"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi

paths=(
    dense_and_full dense_and_containers dense_or_full dense_or_containers dense_clone
    range_mask flip_inplace remove_inplace flip_full remove_full
    lazy_or_full lazy_or_accumulate or_many_full or_many_accumulate
    prod_and prod_or prod_lazy_or prod_count
    and_card_w2 and_card_w4 and_card_w8 and_nocard_w2 and_nocard_w4 and_nocard_w8
    or_card_w2 or_card_w4 or_card_w8 or_nocard_w2 or_nocard_w4 or_nocard_w8
    lazy_or_w2 lazy_or_w4 lazy_or_w8 count_w2 count_w4 count_w8
)

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/m4-cluster-diag-${stamp}"
rows="${prefix}-rows.tsv"
summary="${prefix}-summary.txt"
"$worker" --header >"${prefix}-header.txt" 2>&1
: >"$rows"

for path in "${paths[@]}"; do
    for ((run = 1; run <= runs; run++)); do
        output="${prefix}-${path}-run${run}.txt"
        printf 'run %s/%s path=%s\n' "$run" "$runs" "$path"
        "$worker" "--path=${path}" >"$output" 2>&1
        result="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 }' "$output")"
        if [[ "$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")" != 1 ]]; then
            printf 'invalid result protocol: %s\n' "$output" >&2
            exit 1
        fi
        printf '%s\n' "$result" >>"$rows"
    done
done

{
    printf 'M4 cluster component diagnosis: %s independent processes per path\n' "$runs"
    printf '===================================================================\n'
    cat "${prefix}-header.txt"
    printf '\n%-28s %14s %14s %14s\n' path 'median ns/op' 'min ns/op' 'max ns/op'
    sort -t $'\t' -k1,1 -k4,4n "$rows" | awk -F '\t' '
        function emit(    middle) {
            if (count == 0) return
            middle = values[int((count + 1) / 2)]
            printf "%-28s %14.3f %14.3f %14.3f\n", path, middle / batch, values[1] / batch, values[count] / batch
        }
        {
            if (count && $1 != path) { emit(); delete values; count = 0 }
            path = $1; batch = $2; values[++count] = $4
        }
        END { emit() }
    '
    printf '\nUntimed shapes and allocation counts (first process per relevant path)\n'
    for path in dense_and_full dense_or_full flip_full remove_full lazy_or_full or_many_full; do
        awk -F '\t' '$1 == "SHAPE" || $1 == "ALLOC"' "${prefix}-${path}-run1.txt"
    done
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
