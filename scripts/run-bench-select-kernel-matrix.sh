#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

build_args=(bench-select-kernel-matrix -Doptimize=ReleaseFast -Dcpu=native)
case "${CROARING_AVX512:-0}" in
    0) ;;
    1) build_args+=(-Dcroaring-avx512=true) ;;
    *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
esac
zig build "${build_args[@]}"

worker="./zig-out/bin/bench_select_kernel_matrix"
if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
if [[ ! -x "$worker" ]]; then
    printf 'select kernel worker not found: %s\n' "$worker" >&2
    exit 1
fi

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/select-kernel-matrix-${stamp}"
header_file="${prefix}-header.txt"
manifest_file="${prefix}-manifest.tsv"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
disassembly_file="${prefix}-disassembly.txt"
summary="${prefix}-summary.txt"
"$worker" --header >"$header_file" 2>&1
"$worker" --list >"$manifest_file" 2>&1
: >"$process_rows"

while IFS=$'\t' read -r kind corpus path; do
    [[ "$kind" == TUPLE ]] || continue
    if [[ "$corpus" != canonical && "$path" != scalar && "$path" != unroll-2 && "$path" != unroll-4 ]]; then
        continue
    fi
    run=1
    while (( run <= runs )); do
        output="${prefix}-${corpus}-${path}-run${run}.txt"
        printf 'run %s/%s corpus=%s path=%s\n' "$run" "$runs" "$corpus" "$path"
        "$worker" "--corpus=${corpus}" "--path=${path}" >"$output" 2>&1
        if [[ "$(awk -F '\t' '$1 == "VALIDATION" { n++ } END { print n + 0 }' "$output")" != 1 ]] ||
           [[ "$(awk -F '\t' '$1 == "RESULT" { n++ } END { print n + 0 }' "$output")" != 1 ]]; then
            printf 'invalid worker protocol in %s\n' "$output" >&2
            exit 1
        fi
        awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 }' "$output" >>"$process_rows"
        ((run++))
    done
done <"$manifest_file"

sort -t $'\t' -k1,1 -k2,2 -k4,4n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" corpus "\t" path "\t" queries "\t" middle \
            "\t" values[1] "\t" values[count]
    }
    {
        tuple = $1 FS $2
        if (count != 0 && tuple != previous) {
            emit()
            delete values
            count = 0
        }
        corpus = $1
        path = $2
        queries = $3
        previous = tuple
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_rows"

if command -v llvm-objdump >/dev/null 2>&1; then
    llvm-objdump -d -C "$worker" >"$disassembly_file" 2>&1
elif command -v objdump >/dev/null 2>&1; then
    objdump -d -C "$worker" >"$disassembly_file" 2>&1
elif command -v otool >/dev/null 2>&1; then
    otool -tvV "$worker" >"$disassembly_file" 2>&1
else
    printf 'No disassembler found on this host.\n' >"$disassembly_file"
fi

{
    printf 'Select kernel matrix diagnostic\n'
    printf '===============================\n'
    printf 'Processes per tuple: %s\n' "$runs"
    cat "$header_file"
    printf '\nCanonical matrix\n'
    printf '%-18s %14s %25s %12s %12s\n' path 'ns/query' 'process range' 'vs scalar' 'vs CRoaring'
    awk -F '\t' '
        $1 == "AGG" && $2 == "canonical" {
            median[$3] = $5 / $4
            low[$3] = $6 / $4
            high[$3] = $7 / $4
            order[++n] = $3
        }
        END {
            for (i = 1; i <= n; i++) {
                path = order[i]
                printf "%-18s %14.3f [%7.3f,%7.3f] %11.3fx %11.3fx\n", path, \
                    median[path], low[path], high[path], median[path] / median["scalar"], \
                    median[path] / median["croaring"]
            }
        }
    ' "$aggregate_rows"

    printf '\nArchitecture-neutral controls (candidate/scalar; must be <= 1.05x)\n'
    awk -F '\t' '
        $1 == "AGG" && $2 != "canonical" { median[$2, $3] = $5 / $4; corpora[$2] = 1 }
        END {
            for (corpus in corpora) {
                printf "%-14s unroll-2 %.3fx  unroll-4 %.3fx\n", corpus, \
                    median[corpus, "unroll-2"] / median[corpus, "scalar"], \
                    median[corpus, "unroll-4"] / median[corpus, "scalar"]
            }
        }
    ' "$aggregate_rows"

    printf '\nPrefix ceiling metadata (first validated canonical process)\n'
    awk -F '\t' '$1 == "PREFIX" { print; exit }' "${prefix}-canonical-prefix-ceiling-run1.txt"
    printf 'Disassembly artifact: %s\n' "$disassembly_file"
    printf 'Branch counters: best-effort and unavailable in this runner; collect separately where supported.\n'
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
