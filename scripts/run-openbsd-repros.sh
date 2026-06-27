#!/bin/sh
set +e

for exe in ./zig-out/bin/openbsd_repro_*; do
    [ -x "$exe" ] || continue
    echo "===== $exe ====="
    "$exe"
    status=$?
    echo "exit=$status"
    echo
done
