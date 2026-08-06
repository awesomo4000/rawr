#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."

runs="${RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 || runs % 2 == 0 )); then
    printf 'RUNS must be an odd integer >= 5\n' >&2
    exit 2
fi

zig build bench-compact-header-array bench-compact-header-run \
    -Doptimize=ReleaseFast -Dcpu=native

mkdir -p misc
stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/compact-headers-${stamp}"
process_rows="${prefix}-process-rows.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
: >"$process_rows"

for representation in array run; do
    worker="./zig-out/bin/bench_compact_header_${representation}"
    if [[ ! -x "$worker" && -x "${worker}.exe" ]]; then worker="${worker}.exe"; fi
    if [[ ! -x "$worker" ]]; then
        printf 'compact-header worker not found: %s\n' "$worker" >&2
        exit 1
    fi

    run=1
    while (( run <= runs )); do
        output="${prefix}-${representation}-run${run}.txt"
        printf 'run %s/%s representation=%s\n' "$run" "$runs" "$representation"
        "$worker" >"$output" 2>&1
        if [[ "$(awk -F '\t' '$1 == "VALIDATION" { n++ } END { print n + 0 }' "$output")" != 1 ]]; then
            printf 'missing validation marker in %s\n' "$output" >&2
            exit 1
        fi
        awk -F '\t' -v representation="$representation" '
            $1 == "RESULT" {
                print representation "\t" $2 "\t" $3 "\t" $4 "\t" $7 "\t" $8 \
                    "\t" $9 "\t" $10 "\t" $11
                count++
            }
            END { if (count == 0) exit 1 }
        ' "$output" >>"$process_rows"
        ((run++))
    done
done

sort -t $'\t' -k1,1 -k2,2 -k3,3 -k4,4n "$process_rows" | awk -F '\t' '
    function emit(    middle) {
        if (count == 0) return
        middle = values[int((count + 1) / 2)]
        print "AGG\t" representation "\t" variant "\t" cell "\t" middle \
            "\t" values[1] "\t" values[count] "\t" allocs "\t" frees \
            "\t" requested "\t" class_bytes "\t" teardown
    }
    {
        tuple = $1 FS $2 FS $3
        if (count != 0 && tuple != previous) {
            emit()
            delete values
            count = 0
        }
        representation = $1
        variant = $2
        cell = $3
        allocs = $5
        frees = $6
        requested = $7
        class_bytes = $8
        teardown = $9
        previous = tuple
        values[++count] = $4
    }
    END { emit() }
' >"$aggregate_rows"

{
    printf 'Compact separate-header replica diagnostic\n'
    printf '==========================================\n'
    printf 'Processes per representation: %s\n\n' "$runs"
    printf '%-6s %-19s %13s %25s %10s\n' representation cell 'baseline ns' 'compact median [min,max]' ratio
    awk -F '\t' '
        $1 == "AGG" {
            key = $2 SUBSEP $4
            median[key, $3] = $5
            low[key, $3] = $6
            high[key, $3] = $7
            order[++n] = key
        }
        END {
            for (i = 1; i <= n; i++) {
                key = order[i]
                if (seen[key]++) continue
                split(key, parts, SUBSEP)
                baseline = median[key, "baseline"]
                compact = median[key, "compact"]
                if (baseline == 0 || compact == 0) continue
                printf "%-6s %-19s %13.0f %8.0f [%7.0f,%7.0f] %9.3fx\n", \
                    parts[1], parts[2], baseline, compact, low[key, "compact"], \
                    high[key, "compact"], compact / baseline
            }
        }
    ' "$aggregate_rows"
    printf '\nAccounting is retained in %s. Replica cells are attribution only; full-row GO requires the spec-32 three-way candidate.\n' "$aggregate_rows"
} | tee "$summary"

printf '\nsaved summary: %s\n' "$summary"
