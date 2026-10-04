<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-05: Windows ARM runtime

Owner requested both Windows/aarch64 ABI cells on the supplied Windows 11 UTM
VM on 10/04/2026. Parent: [47](47-portability-matrix.md).

## Outcome, 10/04/2026

### SDK follow-up

Owner authorized installing the missing Microsoft components. Build Tools 2022
17.14.37710.0, ARM64 tools and Windows SDK 10.0.26100 installed successfully
with exit 0 after resuming a host-disk-full interruption. The existing 2019
installation was left untouched. Both `zig build difftest` and
`zig build difftest64`, explicitly targeting `aarch64-windows-msvc`, then built
and ran successfully with the same x64 Zig 0.16.0 compiler workaround and source
snapshot as below. Each ran 1,000 randomized iterations. No custom SDK paths,
library changes or build changes were needed. Logs are retained under
`misc/portability-47-armwin/misc/portability-armwin-sdk/`.

This closes the Tier 2 missing-header gap. All 15 runtime-tested target cells
now pass both differential suites. The native ARM compiler issue remains open;
macOS/x86_64 remains compile-only by owner choice.

### Initial run

Complete with a Tier 2 gap. Both ARM64 ABI cells passed the 252/254 and 236/238
unit suites with two skips each and executed 33-file allowlist consumers.
GNU also passed both differential suites. MSVC differential builds failed in
translate-c with `LibCStdLibHeaderNotFound`; no differential test executed in
that ABI. No SDK was installed and no library or shipped build changes were made.

The host-default ARM Zig target was
`aarch64-windows.win11_dt...win11_dt-gnu`; final tests explicitly targeted
`aarch64-windows-gnu` or `aarch64-windows-msvc`, baseline CPU. Windows was build
26200 in UTM. Source: `95a187e` plus this chunk's checker/driver changes.

The initial Git Bash streaming archive transfer failed and left missing files;
those runs are invalid and excluded. SFTP plus native Windows tar completed the
transfer. Native ARM Zig then crashed in build operations with access violations
and exception `0xc00000aa`. Minimal no-I/O executables for both ABIs built and
ran successfully; all four unit suites passed when invoked directly with a
separate cache. This identifies a toolchain/environment problem in the tested
workflow, not its underlying cause. The fallback below passed both Tier 1 cells;
it does not establish that the native ARM compiler build workflow is working.

All eight invalid-mode controls failed with their named errors. Both wrong-ABI
mutations and the wrong-architecture mutation failed with their expected compile
assertions. Genuine consumers then executed successfully. The original macOS
package path passed, and explicit Windows GNU mode rejected macOS. Local format,
shell-syntax, `check-docs` and diff checks passed. The existing x86 shell mutation
was updated for the new switch syntax; it was not re-executed on x86 Windows in
this chunk.

Reproducer: `scripts/check-windows-arm-runtime.ps1 -Zig <compiler-exe>` from
native ARM64 PowerShell. Logs are in `misc/portability-47-armwin/misc/`, under
`portability-armwin-native/` and `portability-armwin-fallback/`. The script
correctly exits nonzero for the MSVC tooling gap rather than calling it a pass.
Evidence and README now report 15 Tier 1 verified cells, one Tier 2 gap, and
macOS/x86_64 still compile-only. The native-compiler crashes and the missing
MSVC headers remain separate open tooling issues.

## Scope

Use an isolated source snapshot and native Windows processes. Explicitly target
`aarch64-windows-gnu` and `aarch64-windows-msvc` for both unit and differential
suites. Verify the host-default GNU package consumer independently. Extend the
MSVC package mode to choose aarch64 on an aarch64 checker host; retain x86_64
support and reject other architectures. No library changes without separately
reviewing a finding. If a shipped-source OS conditional is required, stop and
report per spec 47.

The native ARM compiler crashed during build-runner operations while direct
unit-test invocations passed. A diagnostic fallback uses official x86_64 Zig
0.16.0 under Windows emulation to build ARM64 executables, never x64 tests.
The release ZIP SHA-256 is
`68659eb5f1e4eb1437a722f1dd889c5a322c9954607f5edcf337bc3684a75a7e`.
Source: [official Zig 0.16.0 x86_64 Windows archive](https://ziglang.org/download/0.16.0/zig-x86_64-windows-0.16.0.zip),
verified against the official download index. The existing ARM installation and
PATH were not replaced.
Keep that toolchain qualification explicit in the evidence.

The fallback needs an explicit `--run-windows-gnu` mode too, since the compiler's
host default is x64. Compile the checker itself as ARM64, select targets from
its compiled architecture, and assert both ABI and architecture in the consumer.
Seed both wrong-ABI directions and a wrong-architecture target; all must fail
with the corresponding assertion, not an unrelated error.

Run the MSVC consumer with its existing target assertion. Seed a GNU target and
require `ExpectedWindowsMsvcConsumer`, then run the genuine consumer. Re-exercise
the invalid-mode controls. Use PowerShell with per-process stdout, stderr and
exit codes so Git Bash emulation cannot obscure results. Existing x86_64 shell
controls must still mutate the selected target after the checker extension.

## Acceptance

- [x] Complete source transferred and native aarch64 Zig 0.16.0 confirmed.
- [x] Both explicit ABI unit-suite results and package results recorded.
- [x] Both differential suites recorded independently for each ABI.
- [x] MSVC target assertion and invalid-mode controls exercised.
- [x] Evidence, outcome and tooling committed together; no unsupported runtime
      claim if compiler/tooling blocks a check.
