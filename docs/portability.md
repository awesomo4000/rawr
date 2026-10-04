<!-- SPDX-License-Identifier: MPL-2.0 -->

# Portability evidence

Results recorded on 10/02/2026 through 10/04/2026 use Zig 0.16.0 and production source at `851ea48`.
Remote runs used isolated source snapshots at `48135e4`, with identical production
sources, build configuration, and package checker. Additional aarch64 hosts used
`ee63c47`; FreeBSD/x86_64 and NetBSD/x86_64 used `16e2bf5` on 10/03/2026,
also with those files unchanged. The compile matrix covers the enumerated public API
probe and a consumer built from the package's 33-file allowlist. It does not prove
that every public method works at runtime.

The subsequent native Linux GNU and Linux/musl runs used `add5f61` plus the checker and script extension
committed with [47-03](../specs/47-03-linux-musl-runtime.md). Library sources and
the shipped build configuration were unchanged.

Windows/MSVC used `89c422b` plus the checker and script changes committed with
[47-04](../specs/47-04-windows-msvc-runtime.md), with the same library/build files.

Windows/aarch64 testing on 10/04/2026 used `95a187e` plus
[47-05](../specs/47-05-windows-arm-runtime.md)'s checker/driver changes in a
Windows 11 ARM64 UTM VM, OS build 26200. The installed ARM64 Zig 0.16.0 compiler
crashed during several build-runner operations, although direct invocation of
both unit suites passed for both ABIs. Minimal no-I/O executables also compiled
and ran under both ABIs. These facts do not establish the cause of the crashes.
The documented fallback compiler is the official x86_64 Zig 0.16.0 release under
Windows emulation; emitted checkers, consumers and tests explicitly target ARM64.

`verified` requires both unit suites and the allowlist package consumer to execute
successfully. Host-default commands are `zig build test`, `zig build test64`, and
`zig build check-package`. Explicit Linux/musl and Windows/MSVC execution use
the commands below.
`compiles` records cross-compilation without runtime evidence for that ABI.

Tier 2 is independent: `passes` requires both `zig build difftest` and
`zig build difftest64` to run successfully. `gap` identifies unavailable or failing
development tooling; `not-run` means it was not attempted. Neither changes the
library's Tier 1 status.

## Target triples

| Target triple | Tier 1 | Tier 2 | Runtime evidence |
| --- | --- | --- | --- |
| aarch64-openbsd | verified | passes | `aarch64-openbsd.7.9...7.9-none`; OpenBSD 7.9 VM; all five runtime commands passed |
| x86_64-openbsd | verified | passes | `x86_64-openbsd.7.8...7.8-none`; OpenBSD 7.8 VM; all five runtime commands passed |
| aarch64-freebsd | verified | passes | `aarch64-freebsd.15.1...15.1-none`; FreeBSD 15.1-RELEASE-p1 VM; all five runtime commands passed |
| x86_64-freebsd | verified | passes | `x86_64-freebsd.15.0.68...15.0.68-none`; FreeBSD 15.0-RELEASE-p8 on Hyper-V; all five runtime commands passed |
| aarch64-windows-gnu | verified | passes | Explicit `aarch64-windows-gnu` baseline binaries on Windows 11 ARM64 build 26200 under UTM; all five checks passed using the x64-compiler workaround below |
| x86_64-windows-gnu | verified | passes | `x86_64-windows.win11_dt...win11_dt-gnu`; native Windows 11 via Git Bash; all five runtime commands passed |
| aarch64-windows-msvc | verified | passes | Explicit `aarch64-windows-msvc` baseline binaries on the same VM; both unit suites and package consumer passed with the compiler workaround; both differential suites passed after installing Microsoft Build Tools and the Windows SDK |
| x86_64-windows-msvc | verified | passes | Explicit `x86_64-windows-msvc` baseline binaries executed on native Windows 11 via Git Bash; all five runtime commands passed |
| aarch64-linux-gnu | verified | passes | `aarch64-linux.6.18.34...6.18.34-gnu.2.41`; native Raspberry Pi Linux, kernel `6.18.34+rpt-rpi-2712`; all five runtime commands passed |
| x86_64-linux-gnu | verified | passes | Native `x86_64-linux.6.8...6.8-gnu.2.39`, kernel `6.8.0-146-generic`, and WSL2 `x86_64-linux.5.10...6.19-gnu.2.39`, kernel `6.6.87.2-microsoft-standard-WSL2`; all five runtime commands passed in each environment |
| aarch64-linux-musl | verified | passes | Explicit `aarch64-linux-musl` baseline binaries executed on native Raspberry Pi Linux; all five checks passed |
| x86_64-linux-musl | verified | passes | Explicit `x86_64-linux-musl` baseline binaries executed on native Linux 6.8.0-146-generic, glibc 2.39; all five checks passed |
| aarch64-macos | verified | passes | `aarch64-macos.26.7...26.7-none`; macOS 26.7; all five runtime commands passed |
| x86_64-macos | compiles | not-run | Runtime provisioning deferred by owner |
| aarch64-netbsd | verified | passes | `aarch64-netbsd.10.1...10.1-none`; NetBSD 10.1 VM; all five runtime commands passed |
| x86_64-netbsd | verified | passes | `x86_64-netbsd.10.1...10.1-none`; NetBSD 10.1 on Hyper-V; all five runtime commands passed |

Linux/x86_64 GNU now has separate native and WSL2 correctness evidence. These
runs do not perform spec 52 Part A's paired performance measurement. No runtime
claim for an ABI transfers to its sibling. Fifteen target cells are Tier 1
verified, and both differential suites passed for all 15. macOS/x86_64 remains
compile-only by owner choice.

Each of the initial ten tested environments passed 252/254 and 236/238 unit tests
respectively, with two skipped tests in each suite. Each package consumer built
and ran from 33 allowlisted files, and both differential commands exited
successfully. Tests used the default Debug build; the package consumer used
ReleaseSafe and differential executables used ReleaseFast, as defined by the
existing build steps. No runtime command received `-Dtarget`.

The later musl runs explicitly targeted the named ABI. Aarch64 passed 252/254 and
236/238 tests with two skips per suite. Baseline x86_64 passed 250/254 and 234/238
with four skips per suite: its target lacks the AVX/SSSE3 array-intersection path,
so the two x86-specific tests skip alongside the two NEON tests. Both package
consumers and both differential suites passed on each architecture.
The additional native Linux/x86_64 GNU run passed 252/254 and 236/238 tests,
with two skips each, and all three other runtime checks.
Windows/x86_64 MSVC baseline passed 250/254 and 234/238 tests, with four SIMD
skips per suite for the same baseline-feature reason as x86_64 musl. Its
33-file package consumer and both CRoaring differential suites passed. The
host-default Zig target remained GNU; only explicit MSVC commands count here.

Windows/aarch64 GNU and MSVC each passed 252/254 and 236/238 unit tests with two
skips, plus execution of their 33-file package consumers. GNU differential
suites passed initially. MSVC initially failed to find libc headers. On
10/04/2026, installing Build Tools 2022 17.14.37710.0 with ARM64 tools and
Windows SDK 10.0.26100 resolved that prerequisite. Both explicit ARM64 MSVC
differential suites then compiled and ran successfully, with 1,000 randomized
iterations each. The compiler discovered the SDK without custom include paths.
The installation completed with exit 0 after resuming an interruption caused
by the Mac host running out of disk space. The native ARM compiler crashes
remain a separate issue; these reruns retained the x64 compiler workaround.

Local raw logs are under gitignored `misc/portability-runtime/`. Retrieved remote
logs are under `misc/portability-47-openbsd/`, `misc/portability-47-windows/`, and
`misc/portability-47-wsl/`, plus `misc/portability-47-linux-arm/`,
`misc/portability-47-openbsd-arm/`, `misc/portability-47-freebsd-arm/`, and
`misc/portability-47-netbsd-arm/` for the additional aarch64 runs, each with a
nested `misc/portability-runtime/` directory.
The 10/03/2026 Hyper-V logs are under `misc/portability-47-freebsd-x86/` and
`misc/portability-47-netbsd-x86/`, with the same nested directory.
Musl logs are under `misc/portability-47-musl-arm/` and
`misc/portability-47-musl-x86/`, each with nested `misc/portability-musl/`.
Native Linux/x86_64 GNU logs are under `misc/portability-47-linux-x86/`, with
nested `misc/portability-runtime/`.
MSVC logs are under `misc/portability-47-msvc-x86/misc/portability-msvc/`.
ARM Windows logs are under `misc/portability-47-armwin/misc/`, in
`portability-armwin-native/` for the initial build failures and direct unit
controls, and `portability-armwin-fallback/` for the final guarded runs.
`portability-armwin-sdk/` holds the successful post-install MSVC differential logs.
This document retains the results without machine names or user-specific paths.

The OpenBSD consumer path passed both cross-compilation and native execution from
the allowlist without the benchmark shim. FreeBSD/aarch64 and NetBSD/aarch64 also
passed the allowlist consumer and both tiers after source-transfer approval.
FreeBSD/x86_64 and NetBSD/x86_64 subsequently passed the same runtime checks on
Hyper-V. No production or build changes were needed on any of these hosts.

## Feature dispatch

These configurations are compile-only checks. Both the API probe and the
allowlist consumer compile; the probe asserts the write and cardinality kernel
registries contain exactly `dispatch`, `gallop`, and `merge`.

| Target triple | CPU profile | Result | Asserted array-intersection path |
| --- | --- | --- | --- |
| x86_64-linux-gnu | baseline-avx | compiles | Scalar |
| aarch64-linux-gnu | baseline-neon | compiles | Scalar |

The profile suffixes remove AVX and NEON respectively. This assertion applies to
array intersection only. Other code uses portable Zig vectors whose lowering
depends on the compiler and target.

## Guard boundaries

`tools/check_32_api.zig` is shared by `check-32` and `check-portability`. Its exported
root forces analysis of its enumerated calls, including all six `OwnedBitmap`
methods and all nine `RoaringBitmap` producers returning `OwnedBitmap`. Methods
not enumerated by the probe are not implicitly covered.

`scripts/check-portability-controls.sh` seeds an `OwnedBitmap.cardinality` return
type defect. Both guards reject it; both succeed with that defect still present
after the `probeOwnedBitmap` call is removed. This demonstrates the call is required
for those guards to catch that defect. It does not test runtime safety checks.

The probe uses a root-local trapping panic handler to avoid a Zig 0.16.0
`std.debug.SelfInfo.Windows` pointer-alignment compilation error on
`aarch64-windows-msvc`. The unmodified package consumer also compiles for that
target; runtime execution has not been tested there.

Other mutation controls check package allowlisting, the CRoaring AVX512 build
option, baseline kernel selection, and complete reporting after a matrix failure.
The OpenBSD allowlist control attaches the benchmark shim to the library module
in a disposable copy: the full checkout builds and the packaged copy fails on
the omitted `src/bench_openbsd.c`. The actual library module has no such attachment.

## Reproduction

Run `zig build check-portability` for the 16 target triples and two CPU profiles,
and `zig build check-32` for the separate 32-bit matrix. Run
`./scripts/check-portability-controls.sh` for the mutation controls; it requires
GNU tar (`gtar`), Bash, and Perl.

For runtime evidence, record `zig version` and `zig env`, then run the three Tier 1
and two Tier 2 commands above without `-Dtarget`. Retain each command's exit status
and output. A successful compile-only run never upgrades a row to `verified`.

For same-architecture Linux/musl execution, run:

```sh
sh scripts/check-linux-musl-runtime.sh
```

The script explicitly targets `aarch64-linux-musl` or `x86_64-linux-musl` for
both unit suites and both differential suites. It invokes the checker with
`zig run check_package.zig -- zig --run-linux-musl`; set `ZIG` to an absolute
compiler path if needed. That mode executes the allowlist consumer and asserts
Linux/musl in the consumer's compiled target. A deliberate GNU-target mutation
must fail with `ExpectedLinuxMuslConsumer`. Invalid combinations must also fail.

The opt-in mode accepts only Linux hosts with the matching aarch64 or x86_64
architecture. Arbitrary runtime target/CPU overrides still fail. Running static
musl binaries on glibc hosts is musl ABI execution evidence, not evidence of a
musl-based distribution.

For native Windows/x86_64 MSVC execution, use Git Bash, not WSL:

```sh
sh scripts/check-windows-msvc-runtime.sh
```

This explicitly targets `x86_64-windows-msvc` for both unit and differential
suites. `zig run check_package.zig -- zig --run-windows-msvc` executes the
allowlist consumer with a compiled Windows/MSVC assertion. A seeded GNU target
must fail with `ExpectedWindowsMsvcConsumer`. Mixed modes, build-only, duplicate
flags and runtime overrides are rejected. The mode now accepts Windows/x86_64
or Windows/aarch64, selecting the checker's compiled architecture;
it does not permit arbitrary target execution or prove compilation with cl.exe.
Other non-default ABI execution remains a follow-up.

For the ARM Windows VM workaround, run native ARM64 PowerShell in the checkout:

```powershell
./scripts/check-windows-arm-runtime.ps1 -Zig "$HOME/zig-x86_64-windows-0.16.0/zig.exe"
```

The x64 compiler runs under Windows emulation but builds an ARM64 checker and
explicit ARM64 test executables. The checker has separate `--run-windows-gnu`
and `--run-windows-msvc` modes. Generated consumers assert architecture as well
as ABI; controls deliberately swap both ABI directions and select x64 instead
of ARM64, requiring the named assertions to fail. The script returns nonzero
if any suite fails, including a missing SDK, and retains each outcome separately.
