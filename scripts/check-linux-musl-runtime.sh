#!/bin/sh
# SPDX-License-Identifier: MPL-2.0
set -eu

zig=${ZIG:-zig}
case "$(uname -m)" in
    x86_64) target=x86_64-linux-musl ;;
    aarch64) target=aarch64-linux-musl ;;
    *) echo "unsupported host architecture" >&2; exit 2 ;;
esac
[ "$(uname -s)" = Linux ] || exit 2
logs=misc/portability-musl
mkdir -p "$logs"
"$zig" version > "$logs/version.txt"
"$zig" env > "$logs/env.txt"
printf '%s\n' "$target" > "$logs/target.txt"

expect_error() {
    name=$1
    shift
    if "$zig" run check_package.zig -- "$zig" "$@" > "$logs/$name.log" 2>&1; then
        echo "expected $name" >&2
        exit 1
    fi
    grep -q "error: $name" "$logs/$name.log"
}
expect_error RunTargetOverride --target "$target"
expect_error RunTargetOverride --run-linux-musl --target "$target"
expect_error MuslRequiresExecution --run-linux-musl --build-only
expect_error DuplicateMuslMode --run-linux-musl --run-linux-musl

# The disposable checker must live beside its imported package manifest.
mutation=$(mktemp ./check_package_musl_control_XXXXXX.zig)
trap 'rm -f "$mutation"' EXIT HUP INT TERM
sed 's/=> "x86_64-linux-musl"/=> "x86_64-linux-gnu"/; s/=> "aarch64-linux-musl"/=> "aarch64-linux-gnu"/' check_package.zig > "$mutation"
if "$zig" run "$mutation" -- "$zig" --run-linux-musl --scratch-suffix musl-control > "$logs/gnu-control.log" 2>&1; then
    echo "GNU mutation unexpectedly passed" >&2
    exit 1
fi
grep -q ExpectedLinuxMuslConsumer "$logs/gnu-control.log"
echo "musl controls passed"

failed=0
if "$zig" run check_package.zig -- "$zig" --run-linux-musl > "$logs/check-package.log" 2>&1; then
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
