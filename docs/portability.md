<!-- SPDX-License-Identifier: MPL-2.0 -->

# Portability evidence

Results recorded on 10/02/2026 use Zig 0.16.0 and production source at `851ea48`.
The compile matrix covers the enumerated public API
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
| aarch64-openbsd | compiles | not-run | No runtime host provided |
| x86_64-openbsd | compiles | not-run | Host reachable; source-transfer approval pending |
| aarch64-freebsd | compiles | not-run | No runtime host provided |
| x86_64-freebsd | compiles | not-run | No runtime host provided |
| aarch64-windows-gnu | compiles | not-run | No runtime host provided |
| x86_64-windows-gnu | compiles | not-run | Windows host reachable; source-transfer approval pending; default ABI not yet recorded |
| aarch64-windows-msvc | compiles | not-run | No runtime host provided |
| x86_64-windows-msvc | compiles | not-run | Windows host reachable; default ABI not yet recorded; no runtime run |
| aarch64-linux-gnu | compiles | not-run | No runtime host provided |
| x86_64-linux-gnu | compiles | not-run | WSL2 available; source-transfer approval pending; resolved Zig target not yet recorded |
| aarch64-linux-musl | compiles | not-run | No runtime host provided |
| x86_64-linux-musl | compiles | not-run | Non-default ABI; no targeted runtime run |
| aarch64-macos | verified | passes | `aarch64-macos.26.7...26.7-none`; macOS 26.7; all five runtime commands passed |
| x86_64-macos | compiles | not-run | No runtime host provided |
| aarch64-netbsd | compiles | not-run | No runtime host provided |
| x86_64-netbsd | compiles | not-run | Previously provided host unavailable: jump-host name resolution failed |

The Linux/x86_64 environment available for this run is WSL2. Native Linux evidence
remains separate; spec 52 Part A can add it when its required runtime checks pass.
No runtime claim for an ABI transfers to its sibling.

The local unit suites passed 252/254 and 236/238 tests respectively, with two
skipped tests in each. The package consumer built and ran from 33 allowlisted
files. Both differential commands exited successfully. Local raw logs are under
gitignored `misc/portability-runtime/`; this document retains the results without
machine names or user-specific paths.

OpenBSD and Windows/WSL2 testing has not started: source transfer to isolated
remote directories requires additional approval in the execution environment.
The OpenBSD consumer path passed cross-compilation in `47-01`; its runtime result
is still open. FreeBSD has no provided runtime host for this run.

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
