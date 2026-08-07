#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p misc

host_os="$(uname -s)"
case "$host_os" in
    Darwin)
        command -v sample >/dev/null 2>&1 || {
            printf 'Darwin /usr/bin/sample is required\n' >&2
            exit 2
        }
        profiler_name="/usr/bin/sample"
        zeroing_names="_platform_memset, _platform_bzero, __bzero, bzero"
        ;;
    Linux)
        perf_bin="${PERF:-}"
        if [[ -z "$perf_bin" ]] && command -v perf >/dev/null 2>&1 && perf --version >/dev/null 2>&1; then
            perf_bin="$(command -v perf)"
        fi
        if [[ -z "$perf_bin" ]]; then
            perf_bin="$(find /usr/lib/linux-tools -type f -name perf 2>/dev/null | sort | tail -1)"
        fi
        if [[ -z "$perf_bin" || ! -x "$perf_bin" ]] || ! "$perf_bin" --version >/dev/null 2>&1; then
            printf 'a working Linux perf binary is required; set PERF=/path/to/perf\n' >&2
            exit 2
        fi
        profiler_name="$perf_bin record -F 999 --call-graph dwarf"
        zeroing_names="compiler_rt.memset, memset, bzero"
        ;;
    *)
        printf 'unsupported profiling host: %s\n' "$host_os" >&2
        exit 2
        ;;
esac

runs="${PROFILE_RUNS:-5}"
if ! [[ "$runs" =~ ^[0-9]+$ ]] || (( runs < 5 )); then
    printf 'PROFILE_RUNS must be an integer >= 5\n' >&2
    exit 2
fi

sample_interval_ms=1
timed_constructions=21
zig build bench-lazy-allocator -Dcpu=native

worker="./zig-out/bin/bench_lazy_or_allocator"
if [[ ! -x "$worker" ]]; then
    printf 'allocator attribution worker not found: %s\n' "$worker" >&2
    exit 1
fi

stamp="$(date -u +%Y%m%d-%H%M%S)"
prefix="misc/lazy-allocator-profile-${stamp}"
process_rows="${prefix}-process.tsv"
aggregate_rows="${prefix}-aggregate.tsv"
summary="${prefix}-summary.txt"
: >"$process_rows"

profile_one() {
    local allocator="$1" run="$2" output trace pid ready_attempts profile_line
    output="${prefix}-${allocator}-run${run}.txt"
    printf 'profile %s/%s allocator=%s\n' "$run" "$runs" "$allocator"

    if [[ "$host_os" == Darwin ]]; then
        trace="${prefix}-${allocator}-run${run}-sample.txt"
        "$worker" --cell=C0 "--allocator=${allocator}" --profile-wait >"$output" 2>&1 &
        pid=$!
        ready_attempts=0
        while ! grep -q '^PROFILE_READY' "$output"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" || true
                printf 'worker exited before profiler rendezvous: %s\n' "$output" >&2
                exit 1
            fi
            ready_attempts=$((ready_attempts + 1))
            if (( ready_attempts > 500 )); then
                kill "$pid" 2>/dev/null || true
                wait "$pid" || true
                printf 'timed out waiting for profiler rendezvous: %s\n' "$output" >&2
                exit 1
            fi
            sleep 0.01
        done

        sample "$pid" 2 "$sample_interval_ms" -mayDie -file "$trace" >/dev/null
        wait "$pid"
        profile_line="$(awk -f tools/parse-darwin-sample.awk "$trace")"
    else
        trace="${prefix}-${allocator}-run${run}-perf.data"
        "$perf_bin" record -F 999 -g --call-graph dwarf,16384 -o "$trace" -- \
            "$worker" --cell=C0 "--allocator=${allocator}" --profile-wait \
            >"$output" 2>&1
        profile_line="$("$perf_bin" script -i "$trace" | awk -f tools/parse-linux-perf.awk)"
    fi

    if [[ "$(awk -F '\t' '$1 == "VALIDATION" && $4 == "ok" { count++ } END { print count + 0 }' "$output")" != 1 ]]; then
        printf 'profile worker validation failed: %s\n' "$output" >&2
        exit 1
    fi
    if ! [[ "$profile_line" =~ ^PROFILE ]]; then
        printf 'profile parse failed: %s\n' "$trace" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "$profile_line" "$allocator" "$run" >>"$process_rows"
}

for allocator in smp libc; do
    for run in $(seq 1 "$runs"); do
        profile_one "$allocator" "$run"
    done
done

awk -F '\t' '
    $1 == "PROFILE" {
        allocator = $10
        process_count[allocator]++
        wrapper[allocator] += $2
        alloc[allocator] += $3
        map[allocator] += $4
        zero[allocator] += $5
        work[allocator] += $6
        other[allocator] += $7
        unsymbolized[allocator] += $8
        symbolized[allocator] += $9
    }
    END {
        for (n = 1; n <= 2; n++) {
            allocator = n == 1 ? "smp" : "libc"
            print "AGG\t" allocator "\t" process_count[allocator] "\t" wrapper[allocator] "\t" alloc[allocator] "\t" map[allocator] "\t" zero[allocator] "\t" work[allocator] "\t" other[allocator] "\t" unsymbolized[allocator] "\t" symbolized[allocator]
        }
    }
' "$process_rows" >"$aggregate_rows"

{
    printf 'Lazy-OR allocator sampling attribution\n'
    printf '======================================\n'
    printf 'Tool: %s\n' "$profiler_name"
    printf 'Sample interval: %s ms\n' "$sample_interval_ms"
    printf 'Fresh processes per allocator: %s\n' "$runs"
    printf 'Timed constructions per process: %s\n' "$timed_constructions"
    printf 'Filter: descendants of bench_croaring.rawr_prof_timed_lazy_or\n'
    printf 'Resolved zeroing variants: %s\n' "$zeroing_names"
    printf '\n%-6s %10s %10s %10s %10s %10s %10s %12s\n' alloc samples allocation mapping zeroing work other unsymbolized
    awk -F '\t' '$1 == "AGG" { printf "%-6s %10d %10d %10d %10d %10d %10d %11.2f%%\n", $2, $4, $5, $6, $7, $8, $9, 100 * $10 / ($10 + $11) }' "$aggregate_rows"

    printf '\nNormalized estimated sampled ms per completed construction\n'
    printf '%-6s %12s %12s %12s %12s %12s %12s\n' alloc total allocation mapping zeroing work other
    awk -F '\t' -v interval="$sample_interval_ms" -v timed="$timed_constructions" '
        $1 == "AGG" {
            denom = $3 * timed
            printf "%-6s %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f\n", $2, interval * $4 / denom, interval * $5 / denom, interval * $6 / denom, interval * $7 / denom, interval * $8 / denom, interval * $9 / denom
        }
    ' "$aggregate_rows"

    printf '\nProfiled SMP-libc delta by mutually exclusive bucket\n'
    awk -F '\t' -v interval="$sample_interval_ms" -v timed="$timed_constructions" '
        $1 == "AGG" {
            denom = $3 * timed
            total[$2] = interval * $4 / denom
            allocation[$2] = interval * $5 / denom
            mapping[$2] = interval * $6 / denom
            zeroing[$2] = interval * $7 / denom
            work[$2] = interval * $8 / denom
            other[$2] = interval * $9 / denom
            unsymbolized_fraction[$2] = $10 / ($10 + $11)
        }
        END {
            delta = total["smp"] - total["libc"]
            call_machinery_delta = allocation["smp"] + mapping["smp"] - allocation["libc"] - mapping["libc"]
            address_work_delta = zeroing["smp"] + work["smp"] - zeroing["libc"] - work["libc"]
            printf "profiled total delta: %.3f ms/construction\n", delta
            printf "%-12s %12s %14s\n", "bucket", "delta ms", "share of delta"
            printf "%-12s %12.3f %13.1f%%\n", "allocation", allocation["smp"] - allocation["libc"], 100 * (allocation["smp"] - allocation["libc"]) / delta
            printf "%-12s %12.3f %13.1f%%\n", "mapping", mapping["smp"] - mapping["libc"], 100 * (mapping["smp"] - mapping["libc"]) / delta
            printf "%-12s %12.3f %13.1f%%\n", "zeroing", zeroing["smp"] - zeroing["libc"], 100 * (zeroing["smp"] - zeroing["libc"]) / delta
            printf "%-12s %12.3f %13.1f%%\n", "work", work["smp"] - work["libc"], 100 * (work["smp"] - work["libc"]) / delta
            printf "%-12s %12.3f %13.1f%%\n", "other", other["smp"] - other["libc"], 100 * (other["smp"] - other["libc"]) / delta
            if (unsymbolized_fraction["smp"] > 0.05 || unsymbolized_fraction["libc"] > 0.05) {
                print "Phase 1 verdict: inconclusive (unsymbolized samples exceed 5%)"
            } else if (address_work_delta > 0 && call_machinery_delta <= 0) {
                print "Phase 1 verdict: allocator-induced memory layout"
            } else if (call_machinery_delta > 0 && address_work_delta <= 0) {
                print "Phase 1 verdict: per-call allocator overhead"
            } else if (call_machinery_delta > 0 && address_work_delta > 0) {
                print "Phase 1 verdict: both"
            } else {
                print "Phase 1 verdict: inconclusive"
            }
        }
    ' "$aggregate_rows"
} | tee "$summary"

printf '\nSaved to: %s\n' "$summary"
