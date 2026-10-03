<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-04: Windows x86_64 MSVC runtime

Owner requested native Windows MSVC testing on 10/03/2026. Parent:
[47](47-portability-matrix.md). Use explicit Git Bash, not the WSL launcher.
Preserve the existing dirty checkout by testing an isolated source snapshot.

## Outcome, 10/03/2026

All five checks passed using Zig 0.16.0 on native Windows 11 via the explicit
Git Bash executable, not WSL. Source snapshot `89c422b` plus this chunk's checker
and script changes; no library or shipped build changes. The existing remote
checkout and its uncommitted files were left untouched.

The host default was `x86_64-windows.win11_dt...win11_dt-gnu`. Every measured
cell instead used explicit `x86_64-windows-msvc` with the baseline CPU.
Unit suites passed 250/254 and 234/238 tests, four skips each for the unavailable
x86 SIMD and NEON test paths. The package consumer executed from 33 allowlisted
files with its Windows/MSVC assertion, and both differential suites passed.
No additional SDK installation was required during this work.

Original and MSVC-specific target overrides failed with `RunTargetOverride`.
Build-only, duplicate flags and mixed modes failed with their named errors.
The GNU-target mutation failed with `ExpectedWindowsMsvcConsumer`, while the
actual MSVC consumer succeeded. macOS rejected execution mode with
`MsvcRequiresWindowsHost`; its original package check still passed. The
non-x86_64 Windows restriction was inspected, not executed on another host.
Formatting, shell syntax, `check-docs` and diff checks passed.

Logs: `misc/portability-47-msvc-x86/misc/portability-msvc/`. Both tiers now pass
for this cell; the evidence table has 13 verified cells. This verifies Zig's
MSVC ABI target and C-backed tooling, not compilation using Microsoft's cl.exe.

## Scope and controls

Run `test`, `test64`, `difftest` and `difftest64` with
`-Dtarget=x86_64-windows-msvc`. Add `--run-windows-msvc` to the package checker,
restricted to native Windows/x86_64. Keep target/CPU overrides rejected, reject
build-only and mixed execution modes, and assert Windows/MSVC in the compiled
consumer. A disposable mutation selecting GNU instead must fail that assertion.
The original host-default path remains unchanged. No library changes.

Record Tier 1 and Tier 2 independently. A missing SDK or CRoaring compile failure
is not a passing differential suite. Investigate failures before attributing them
to rawr. This run establishes MSVC ABI coverage, not compilation by cl.exe.

## Acceptance

- [x] Named invalid-mode controls fail, including a GNU-consumer mutation.
- [x] Explicit MSVC package consumer and both unit suites execute successfully.
- [x] Both differential suite outcomes recorded, including tooling gaps if any.
- [x] Evidence, README and spec outcome committed with the checker changes.
