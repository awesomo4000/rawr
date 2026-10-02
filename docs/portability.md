<!-- SPDX-License-Identifier: MPL-2.0 -->

# Portability evidence

Results recorded on 10/02/2026 use Zig 0.16.0 and production source at `851ea48`.
Remote runs used isolated source snapshots at `48135e4`, with identical production
sources, build configuration, and package checker. Additional aarch64 hosts used
`ee63c47`, also with those files unchanged. The compile matrix covers the enumerated public API
probe and a consumer built from the package's 33-file allowlist. It does not prove
that every public method works at runtime.

`verified` requires all three host-default commands to succeed: `zig build test`,
`zig build test64`, and `zig build check-package`. `compiles` records successful
cross-compilation without that runtime evidence. Non-default ABIs remain
`compiles`; the runtime package checker deliberately accepts only the host default.

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
| x86_64-freebsd | compiles | not-run | No runtime host provided |
| aarch64-windows-gnu | compiles | not-run | No runtime host provided |
| x86_64-windows-gnu | verified | passes | `x86_64-windows.win11_dt...win11_dt-gnu`; native Windows 11 via Git Bash; all five runtime commands passed |
| aarch64-windows-msvc | compiles | not-run | No runtime host provided |
| x86_64-windows-msvc | compiles | not-run | Non-default ABI; no targeted runtime run |
| aarch64-linux-gnu | verified | passes | `aarch64-linux.6.18.34...6.18.34-gnu.2.41`; native Raspberry Pi Linux, kernel `6.18.34+rpt-rpi-2712`; all five runtime commands passed |
| x86_64-linux-gnu | verified | passes | `x86_64-linux.5.10...6.19-gnu.2.39`; Ubuntu 24.04 under WSL2, kernel `6.6.87.2-microsoft-standard-WSL2`; all five runtime commands passed |
| aarch64-linux-musl | compiles | not-run | No runtime host provided |
| x86_64-linux-musl | compiles | not-run | Non-default ABI; no targeted runtime run |
| aarch64-macos | verified | passes | `aarch64-macos.26.7...26.7-none`; macOS 26.7; all five runtime commands passed |
| x86_64-macos | compiles | not-run | No runtime host provided |
| aarch64-netbsd | verified | passes | `aarch64-netbsd.10.1...10.1-none`; NetBSD 10.1 VM; all five runtime commands passed |
| x86_64-netbsd | compiles | not-run | Previously provided host unavailable: jump-host name resolution failed |

The Linux/x86_64 environment available for this run is WSL2. Native Linux evidence
remains separate; spec 52 Part A can add it when its required runtime checks pass.
The native Linux/aarch64 run does not upgrade the x86_64 cell. No runtime claim for
an ABI transfers to its sibling.

Each of the eight tested environments passed 252/254 and 236/238 unit tests
respectively, with two skipped tests in each suite. Each package consumer built
and ran from 33 allowlisted files, and both differential commands exited
successfully. Tests used the default Debug build; the package consumer used
ReleaseSafe and differential executables used ReleaseFast, as defined by the
existing build steps. No runtime command received `-Dtarget`.

Local raw logs are under gitignored `misc/portability-runtime/`. Retrieved remote
logs are under `misc/portability-47-openbsd/`, `misc/portability-47-windows/`, and
`misc/portability-47-wsl/`, plus `misc/portability-47-linux-arm/`,
`misc/portability-47-openbsd-arm/`, `misc/portability-47-freebsd-arm/`, and
`misc/portability-47-netbsd-arm/` for the additional aarch64 runs, each with a
nested `misc/portability-runtime/` directory.
This document retains the results without machine names or user-specific paths.

The OpenBSD consumer path passed both cross-compilation and native execution from
the allowlist without the benchmark shim. FreeBSD/aarch64 and NetBSD/aarch64 also
passed the allowlist consumer and both tiers after source-transfer approval.
No production or build changes were needed. FreeBSD/x86_64 remains compile-only;
the NetBSD/x86_64 host was unavailable, so its runtime result remains open.

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

Support for running compatible non-default ABIs through the package checker is
a follow-up. This run does not extend that checker or infer execution from
cross-compilation.
