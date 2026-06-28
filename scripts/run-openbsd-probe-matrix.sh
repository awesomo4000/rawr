#!/bin/sh
set +e

PROBE=${PROBE:-./zig-out/bin/openbsd_probe}
LOG_DIR=${LOG_DIR:-misc/openbsd-probe-logs}

CASES=${CASES:-"00 01 02 03 04 05 06 07 08"}
SEQUENCE_CASES=${SEQUENCE_CASES:-"99"}
ALLOCATORS=${ALLOCATORS:-"openbsd_c std_c smp"}
CALLS=${CALLS:-"auto never_inline"}
HARNESSES=${HARNESSES:-"direct noinline_harness"}

VALUES=${VALUES:-1000000}
WARMUP=${WARMUP:-3}
RUNS=${RUNS:-21}
ITERATIONS=${ITERATIONS:-10000}
TAIL_LINES=${TAIL_LINES:-80}
STOP_ON_FAIL=${STOP_ON_FAIL:-1}
BUILD=${BUILD:-1}
RUN_ISOLATED=${RUN_ISOLATED:-1}
RUN_SEQUENCES=${RUN_SEQUENCES:-1}

if [ "${INCLUDE_ALL_CASE:-0}" = "1" ]; then
    CASES="$CASES 99"
fi

mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.txt"
: > "$SUMMARY"

if [ "$BUILD" = "1" ]; then
    echo "building openbsd-probe"
    zig build openbsd-probe
    status=$?
    if [ "$status" -ne 0 ]; then
        echo "build failed: exit=$status"
        exit "$status"
    fi
fi

if [ ! -x "$PROBE" ]; then
    echo "probe not executable: $PROBE"
    exit 1
fi

total=0
failed=0

run_probe() {
    mode=$1
    case_id=$2
    allocator=$3
    call_mode=$4
    harness=$5

    total=$((total + 1))
    label="${mode} case=${case_id} allocator=${allocator} call=${call_mode} harness=${harness}"
    log="$LOG_DIR/${total}_${mode}_${case_id}_${allocator}_${call_mode}_${harness}.log"

    echo "RUN $total $label"
    echo "RUN $total $label" >> "$SUMMARY"

    "$PROBE" \
        --case="$case_id" \
        --allocator="$allocator" \
        --call="$call_mode" \
        --harness="$harness" \
        --values="$VALUES" \
        --warmup="$WARMUP" \
        --runs="$RUNS" \
        --iterations="$ITERATIONS" \
        --no-trace > "$log" 2>&1
    status=$?

    if [ "$status" -eq 0 ]; then
        echo "ok"
        echo "ok" >> "$SUMMARY"
        return 0
    fi

    failed=$((failed + 1))
    echo "FAIL exit=$status $label"
    echo "FAIL exit=$status $label" >> "$SUMMARY"
    echo "log: $log"
    echo "log: $log" >> "$SUMMARY"
    echo "last $TAIL_LINES output lines:"
    tail -n "$TAIL_LINES" "$log"

    if [ "$STOP_ON_FAIL" = "1" ]; then
        echo "stopping after first failure"
        echo "stopping after first failure" >> "$SUMMARY"
        exit "$status"
    fi

    return 0
}

if [ "$RUN_ISOLATED" = "1" ]; then
    for case_id in $CASES; do
        for allocator in $ALLOCATORS; do
            for call_mode in $CALLS; do
                for harness in $HARNESSES; do
                    run_probe "isolated" "$case_id" "$allocator" "$call_mode" "$harness"
                done
            done
        done
    done
fi

if [ "$RUN_SEQUENCES" = "1" ]; then
    for case_id in $SEQUENCE_CASES; do
        for allocator in $ALLOCATORS; do
            for call_mode in $CALLS; do
                for harness in $HARNESSES; do
                    run_probe "sequence" "$case_id" "$allocator" "$call_mode" "$harness"
                done
            done
        done
    done
fi

echo "done total=$total failed=$failed logs=$LOG_DIR"
echo "done total=$total failed=$failed logs=$LOG_DIR" >> "$SUMMARY"

if [ "$failed" -ne 0 ]; then
    exit 1
fi

exit 0
