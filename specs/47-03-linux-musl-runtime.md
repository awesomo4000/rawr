<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-03: Native Linux and musl runtime evidence

Parent: [47](47-portability-matrix.md). Owner requested Linux/aarch64 musl
and Linux/x86_64 musl coverage on the available Linux hosts on 10/03/2026.
Windows/aarch64 and macOS/x86_64 remain compile-only by owner choice.

## Outcome, 10/03/2026

Complete. Both musl targets passed all five checks on real Linux hosts, without
an emulator. Native Linux/x86_64 GNU independently passed all five checks.
Source was `add5f61` plus this chunk's checker/script changes; library sources,
package allowlist and shipped build configuration were unchanged.

Native GNU resolved to `x86_64-linux.6.8...6.8-gnu.2.39`, kernel
`6.8.0-146-generic`. Musl targets were explicitly `x86_64-linux-musl` and
`aarch64-linux-musl`, using baseline CPUs on the supplied native Linux hosts.
Native GNU and aarch64 musl passed 252/254 and 236/238 unit tests, two skips each.
Baseline x86_64 musl passed 250/254 and 234/238, four skips each: its absent
AVX/SSSE3 path also skips the two x86-specific SIMD tests. Every package consumer
executed from 33 allowlisted files and both differential suites passed per cell.

Both hosts rejected the original override, musl plus an explicit override,
musl plus build-only, and duplicate musl flags with the named errors. The seeded
GNU consumer failed with `ExpectedLinuxMuslConsumer` on both hosts; the genuine
musl consumer then executed successfully. macOS rejected this mode with
`MuslRequiresLinuxHost`; its original host-default package check still passed.
The unsupported-architecture rejection was inspected, not executed on another
Linux architecture. Formatting, shell syntax, `check-docs` and diff checks passed.

Logs and exact targets are recorded in [the evidence](../docs/portability.md).
Twelve of sixteen target cells now have runtime evidence. This is correctness
coverage only and does not complete spec 52 Part A.

## Scope

Run the five host-default checks on native Linux/x86_64. Run `test`, `test64`,
`difftest`, and `difftest64` with explicit same-architecture Linux/musl targets
on both Linux hosts. No library changes or performance conclusions.

Extend the allowlist package checker with `--run-linux-musl`, selecting only
the executing Linux host's x86_64 or aarch64 architecture and baseline CPU.
Reject non-Linux hosts, other architectures, explicit target/CPU overrides,
and build-only combined with this mode. Preserve `RunTargetOverride` otherwise.
Invoke the checker directly with `zig run check_package.zig -- <zig-path>`;
the existing host-default build step is unchanged.

The generated consumer must assert its compiled target is Linux/musl when this
mode is selected. Seed a GNU target in a disposable checker copy and require
that assertion to fail. This prevents host-default execution posing as musl.
Record target triples, source revision, commands and logs. Static musl binaries
executed on glibc Linux establish that ABI's coverage, not a musl distribution.
Native x86_64 correctness runs do not complete spec 52's performance experiment.

## Acceptance

- [x] Original override guard and new invalid combinations reject as named.
- [x] A deliberately GNU-built musl consumer fails its target assertion.
- [x] Both Linux/musl cells run all five checks, with no emulator.
- [x] Native Linux/x86_64 host-default commands run independently.
- [x] Evidence tables, README and outcome committed together; no ABI promotion
      from another cell and no performance claim.
