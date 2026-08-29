#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

usage() {
    printf 'usage: %s OLD_TREE HEAD_TREE\n' "$0" >&2
    printf '  Builds both clean git trees and measures the serialize row in one session.\n' >&2
}

if (( $# != 2 )); then
    usage
    exit 2
fi

old_tree="$(cd "$1" && pwd)"
head_tree="$(cd "$2" && pwd)"
runs="${RUNS:-5}"
zig_bin="${ZIG:-zig}"
output_dir="${OUTPUT_DIR:-$(pwd)/misc}"

if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi
if [[ "$old_tree" == "$head_tree" ]]; then
    printf 'OLD_TREE and HEAD_TREE must be different directories\n' >&2
    exit 2
fi
if [[ ! -x "$zig_bin" ]] && ! command -v "$zig_bin" >/dev/null 2>&1; then
    printf 'Zig executable not found: %s\n' "$zig_bin" >&2
    exit 2
fi

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
    else
        shasum -a 256 "$1" | awk '{ print $1 }'
    fi
}

require_clean_tree() {
    local tree="$1"
    if ! git -C "$tree" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf 'not a git working tree: %s\n' "$tree" >&2
        exit 2
    fi
    if [[ -n "$(git -C "$tree" status --porcelain)" ]]; then
        printf 'benchmark tree is not clean: %s\n' "$tree" >&2
        git -C "$tree" status --short >&2
        exit 2
    fi
}

require_clean_tree "$old_tree"
require_clean_tree "$head_tree"

mkdir -p "$output_dir"
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="${output_dir}/serialize-history-${stamp}"
environment_file="${prefix}-environment.txt"
process_rows="${prefix}-process.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
: >"$process_rows"

{
    printf 'serialize history environment\n'
    printf '=============================\n'
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'uname='; uname -a
    printf 'zig=%s\n' "$($zig_bin version)"
    if command -v ldd >/dev/null 2>&1; then
        printf 'libc='; ldd --version 2>&1 | awk 'NR == 1 { print; exit }'
    else
        printf 'libc=ldd unavailable\n'
    fi
    printf 'runs=%s\n' "$runs"
    printf 'optimize=ReleaseFast\n'
    printf 'cpu=native\n'
    printf 'croaring_avx512=%s\n' "${CROARING_AVX512:-0}"
} >"$environment_file"

build_and_run_tree() {
    local label="$1"
    local tree="$2"
    local commit worker manifest header worker_hash
    local build_args=(bench-parity-worker -Dcpu=native)

    case "${CROARING_AVX512:-0}" in
        0) expected_croaring_avx512=off ;;
        1)
            expected_croaring_avx512=on
            build_args+=(-Dcroaring-avx512=true)
            ;;
        *) printf 'CROARING_AVX512 must be 0 or 1\n' >&2; exit 2 ;;
    esac

    commit="$(git -C "$tree" rev-parse HEAD)"
    printf 'building label=%s commit=%s\n' "$label" "$commit"
    (cd "$tree" && "$zig_bin" build "${build_args[@]}")

    worker="${tree}/zig-out/bin/bench_parity_worker"
    if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
    if [[ ! -x "$worker" ]]; then
        printf 'parity worker not found: %s\n' "$worker" >&2
        exit 1
    fi

    manifest="${prefix}-${label}-manifest.tsv"
    header="${prefix}-${label}-header.txt"
    "$worker" --list >"$manifest" 2>&1
    "$worker" --header >"$header" 2>&1
    if ! grep -Fqx "# croaring-avx512: ${expected_croaring_avx512}" "$header"; then
        printf 'worker CRoaring AVX512 setting does not match CROARING_AVX512=%s in %s\n' \
            "${CROARING_AVX512:-0}" "$tree" >&2
        exit 1
    fi
    worker_hash="$(sha256_file "$worker")"

    {
        printf 'tree_%s=%s\n' "$label" "$tree"
        printf 'commit_%s=%s\n' "$label" "$commit"
        printf 'worker_sha256_%s=%s\n' "$label" "$worker_hash"
    } >>"$environment_file"

    local serialize_rows serialize_tuples
    serialize_rows="$(awk -F '\t' '$1 == "ROW" && $2 == "serialize" { count++ } END { print count + 0 }' "$manifest")"
    serialize_tuples="$(awk -F '\t' '$1 == "TUPLE" && $2 == "serialize" { count++ } END { print count + 0 }' "$manifest")"
    if [[ "$serialize_rows" != 1 || "$serialize_tuples" != 3 ]]; then
        printf 'expected one serialize row and three tuples in %s, got %s and %s\n' \
            "$manifest" "$serialize_rows" "$serialize_tuples" >&2
        exit 1
    fi

    local tuple_count=0
    while IFS=$'\t' read -r kind row implementation allocator _; do
        [[ "$kind" == "TUPLE" && "$row" == "serialize" ]] || continue
        case "${implementation}/${allocator}" in
            rawr/smp|rawr/libc|croaring/libc) ;;
            *)
                printf 'unexpected serialize tuple in %s: %s/%s\n' \
                    "$manifest" "$implementation" "$allocator" >&2
                exit 1
                ;;
        esac
        ((tuple_count+=1))

        local run=1
        while (( run <= runs )); do
            local output result_count result_line result_row result_impl result_allocator
            local result_unit result_batch result_median
            output="${prefix}-${label}-${implementation}-${allocator}-run${run}.txt"
            printf 'run %s/%s label=%s implementation=%s allocator=%s\n' \
                "$run" "$runs" "$label" "$implementation" "$allocator"
            "$worker" \
                --row=serialize \
                "--implementation=${implementation}" \
                "--allocator=${allocator}" >"$output" 2>&1

            result_count="$(awk -F '\t' '$1 == "RESULT" { count++ } END { print count + 0 }' "$output")"
            if [[ "$result_count" != 1 ]]; then
                printf 'expected one RESULT from %s, got %s\n' "$output" "$result_count" >&2
                exit 1
            fi
            result_line="$(awk -F '\t' '$1 == "RESULT" { print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 }' "$output")"
            IFS=$'\t' read -r result_row result_impl result_allocator result_unit result_batch result_median \
                <<<"$result_line"
            if [[ "$result_row" != serialize || "$result_impl" != "$implementation" || \
                  "$result_allocator" != "$allocator" || "$result_unit" != ms || \
                  "$result_batch" != 1 || ! "$result_median" =~ ^[0-9]+$ ]]; then
                printf 'invalid RESULT tuple in %s: %s\n' "$output" "$result_line" >&2
                exit 1
            fi
            printf '%s\t%s\t%s\t%s\n' "$label" "$implementation" "$allocator" "$result_median" \
                >>"$process_rows"
            ((run+=1))
        done
    done <"$manifest"

    if [[ "$tuple_count" != 3 ]]; then
        printf 'expected three recognized serialize tuples in %s, got %s\n' "$manifest" "$tuple_count" >&2
        exit 1
    fi
}

build_and_run_tree historical "$old_tree"
build_and_run_tree head "$head_tree"

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" label "\t" implementation "\t" allocator "\t" middle "\t" values[1] "\t" values[count]
    }
    {
        key = $1 SUBSEP $2 SUBSEP $3
        if (count != 0 && key != previous) {
            emit()
            delete values
            count = 0
        }
        label = $1
        implementation = $2
        allocator = $3
        previous = key
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Serialize history comparison\n'
    printf '============================\n'
    cat "$environment_file"
    printf '\n%-12s %-10s %-8s %10s %23s\n' label implementation allocator 'median ms' 'full range ms'
    printf '%-12s %-10s %-8s %10s %23s\n' ------------ ---------- -------- ---------- -----------------------
    awk -F '\t' '$1 == "AGG" {
        printf "%-12s %-10s %-8s %10.3f [%9.3f,%9.3f]\n", $2, $3, $4, $5 / 1000000.0, $6 / 1000000.0, $7 / 1000000.0
    }' "$aggregate_rows"

    awk -F '\t' '
        $1 == "AGG" {
            key = $2 SUBSEP $3 SUBSEP $4
            median[key] = $5
        }
        END {
            old_smp = median["historical" SUBSEP "rawr" SUBSEP "smp"]
            head_smp = median["head" SUBSEP "rawr" SUBSEP "smp"]
            old_libc = median["historical" SUBSEP "rawr" SUBSEP "libc"]
            head_libc = median["head" SUBSEP "rawr" SUBSEP "libc"]
            old_cr = median["historical" SUBSEP "croaring" SUBSEP "libc"]
            head_cr = median["head" SUBSEP "croaring" SUBSEP "libc"]
            printf "\nHistorical 0.824 ms anchor to current-session historical rawr/SMP: %.3fx\n", old_smp / 824000.0
            printf "Current-session HEAD/historical rawr/SMP: %.3fx\n", head_smp / old_smp
            printf "Current-session HEAD/historical rawr/libc: %.3fx\n", head_libc / old_libc
            printf "Current-session HEAD/historical CRoaring/libc: %.3fx\n", head_cr / old_cr
            printf "Historical current-session rawr/SMP to CRoaring: %.3fx\n", old_smp / old_cr
            printf "HEAD current-session rawr/SMP to CRoaring: %.3fx\n", head_smp / head_cr
        }
    ' "$aggregate_rows"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
