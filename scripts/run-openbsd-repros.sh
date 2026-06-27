#!/bin/sh
set +e

for exe in ./zig-out/bin/openbsd_repro_*; do
    [ -x "$exe" ] || continue
    case "$exe" in
        *openbsd_repro_2[6-9]_*|*openbsd_repro_3[0-9]_*)
            if [ "${RUN_FULL:-0}" != "1" ]; then
                echo "===== $exe ====="
                echo "skipped full benchmark repro; run with RUN_FULL=1"
                echo
                continue
            fi
            ;;
    esac
    echo "===== $exe ====="
    "$exe"
    status=$?
    echo "exit=$status"
    echo
done
