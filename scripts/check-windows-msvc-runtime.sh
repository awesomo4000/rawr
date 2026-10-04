#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
set -eu

zig=${ZIG:-zig}
target=x86_64-windows-msvc
logs=misc/portability-msvc
mkdir -p "$logs"
"$zig" version > "$logs/version.txt"
"$zig" env > "$logs/env.txt"
printf '%s\n' "$target" > "$logs/target.txt"

expect_error() {
    label=$1
    name=$2
    shift 2
    if "$zig" run check_package.zig -- "$zig" "$@" > "$logs/$label.log" 2>&1; then
        echo "expected $name" >&2
        exit 1
    fi
    grep -q "error: $name" "$logs/$label.log"
}
expect_error original-override RunTargetOverride --target "$target"
expect_error msvc-override RunTargetOverride --run-windows-msvc --target "$target"
expect_error build-only MsvcRequiresExecution --run-windows-msvc --build-only
expect_error duplicate DuplicateMsvcMode --run-windows-msvc --run-windows-msvc
expect_error mixed-modes ConflictingExecutionModes --run-windows-msvc --run-linux-musl

mutation=$(mktemp ./check_package_msvc_control_XXXXXX.zig)
trap 'rm -f "$mutation"' EXIT HUP INT TERM
sed 's/=> "x86_64-windows-msvc"/=> "x86_64-windows-gnu"/' check_package.zig > "$mutation"
if "$zig" run "$mutation" -- "$zig" --run-windows-msvc --scratch-suffix msvc-control > "$logs/gnu-control.log" 2>&1; then
    echo "GNU mutation unexpectedly passed" >&2
    exit 1
fi
grep -q ExpectedWindowsMsvcConsumer "$logs/gnu-control.log"
echo "msvc controls passed"

failed=0
if "$zig" run check_package.zig -- "$zig" --run-windows-msvc > "$logs/check-package.log" 2>&1; then
    echo "check-package exit=0"
else
    echo "check-package failed"
    failed=1
fi
for step in test test64 difftest difftest64; do
    if "$zig" build "$step" "-Dtarget=$target" --summary all > "$logs/$step.log" 2>&1; then
        echo "$step exit=0"
    else
        echo "$step failed"
        failed=1
    fi
done
exit "$failed"
