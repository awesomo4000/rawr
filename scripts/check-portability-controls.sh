#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

zig_exe="${ZIG:-$(command -v zig)}"
tmp_root="$(mktemp -d /tmp/rawr-portability-controls.XXXXXX)"
cleanup() {
    case "$tmp_root" in
        /tmp/rawr-portability-controls.*) rm -rf -- "$tmp_root" ;;
        *) printf 'refusing to remove unexpected control directory: %s\n' "$tmp_root" >&2 ;;
    esac
}
trap cleanup EXIT INT TERM

fresh_copy() {
    local name="$1"
    local dest="$tmp_root/$name"
    mkdir -p "$dest"
    gtar -cf - \
        --exclude=.git \
        --exclude=.zig-cache \
        --exclude=zig-out \
        --exclude=misc \
        . | gtar -xf - -C "$dest"
    printf '%s\n' "$dest"
}

expect_success() {
    local name="$1"
    local dir="$2"
    shift 2
    local log="$tmp_root/$name.log"
    if ! (cd "$dir" && "$@") >"$log" 2>&1; then
        printf 'control unexpectedly failed: %s\n' "$name" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'CONTROL name=%s result=pass\n' "$name"
}

expect_failure() {
    local name="$1"
    local expected="$2"
    local dir="$3"
    shift 3
    local log="$tmp_root/$name.log"
    if (cd "$dir" && "$@") >"$log" 2>&1; then
        printf 'seeded portability defect passed: %s\n' "$name" >&2
        exit 1
    fi
    if ! grep -Fq "$expected" "$log"; then
        printf 'wrong failure for %s; expected %s\n' "$name" "$expected" >&2
        cat "$log" >&2
        exit 1
    fi
    printf 'CONTROL name=%s result=caught expected=%s\n' "$name" "$expected"
}

owned_dir="$(fresh_copy owned-bitmap)"
perl -0pi -e \
    's/pub fn cardinality\(self: \*const OwnedBitmap\) u64/pub fn cardinality(self: *const OwnedBitmap) u32/' \
    "$owned_dir/src/bitmap.zig"
expect_failure owned-bitmap-extended "expected type 'u32'" "$owned_dir" \
    "$zig_exe" build check-portability-x86_64-linux-gnu-probe --summary none
perl -0pi -e 's/    try probeOwnedBitmap\(allocator, &left, &right, bytes\);\n//' \
    "$owned_dir/tools/check_32_api.zig"
expect_success owned-bitmap-unextended "$owned_dir" \
    "$zig_exe" build check-portability-x86_64-linux-gnu-probe --summary none

package_dir="$(fresh_copy package-allowlist)"
perl -0pi -e \
    's/(    const lib = b\.addLibrary\(\.\{)/    addBenchmarkPlatformShim(b, lib_mod, target);\n\n$1/' \
    "$package_dir/build.zig"
expect_success package-in-repo "$package_dir" \
    "$zig_exe" build -Dtarget=x86_64-openbsd --summary none
expect_failure package-allowlist "bench_openbsd.c" "$package_dir" \
    "$zig_exe" build check-portability-x86_64-openbsd-package --summary none

option_dir="$(fresh_copy build-option)"
perl -0pi -e \
    's/\n    translate_c\.defineCMacro\("CROARING_COMPILER_SUPPORTS_AVX512", croaring_avx512_value\);//' \
    "$option_dir/build.zig"
expect_success build-option-plain-library "$option_dir" \
    "$zig_exe" build -Dtarget=x86_64-linux-gnu -Dcpu=baseline -Dcroaring-avx512=true --summary none
expect_failure build-option-affected-step "OPTION name=croaring-avx512=true result=fail" "$option_dir" \
    "$zig_exe" build check-portability-options --summary none

dispatch_dir="$(fresh_copy feature-dispatch)"
perl -0pi -e \
    's/pub const has_x86_simd = .*?;\n/pub const has_x86_simd = builtin.cpu.arch == .x86_64;\n/s' \
    "$dispatch_dir/src/array_simd.zig"
expect_failure feature-dispatch "baseline target selected a SIMD array-intersection kernel" \
    "$dispatch_dir" "$zig_exe" build \
    check-portability-x86_64-linux-gnu-baseline-no-avx-probe --summary none

expect_success mid-matrix-reporting "$PWD" \
    "$zig_exe" build check-portability-reporting-control --summary none

reporting_dir="$(fresh_copy reporting-completeness)"
perl -0pi -e \
    's/(    while \(args\.next\(\)\) \|cell\| \{\n)/$1        if (cells == 2) break;\n/' \
    "$reporting_dir/tools/check_portability.zig"
expect_failure mid-matrix-truncated "CellCountMismatch" "$reporting_dir" \
    "$zig_exe" build check-portability-reporting-control --summary none

printf 'portability falsification controls: OK\n'
