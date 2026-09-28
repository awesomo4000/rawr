<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-01: Run the 16-target compile matrix

Toplevel: [47-portability-matrix.md](47-portability-matrix.md).
Gated on: [47-00](47-00-portability-machinery.md) complete, **with all five controls exercised**.

> **Outcome — complete.** All 16 target triples and both baseline-feature cells ran in the required
> risk order. Every minimal control, public API probe, and allowlist-only package consumer compiled.
> The two baseline profiles selected exactly the scalar array-intersection registry
> `{ dispatch, gallop, merge }`; this says nothing about portable vector lowering elsewhere.
>
> The first run found one `broken` cell: `aarch64-windows-msvc` compiled its minimal control and package
> consumer but failed the probe in Zig 0.16's `std.debug.SelfInfo.Windows`, where an internal `@ptrCast`
> increased pointer alignment. The reference trace led back to the probe root's `runProbe() catch
> unreachable`, whose panic path pulled Windows stack-trace machinery into a compile-only object. The
> probe now uses `catch @trap()` and the root-local `std.debug.no_panic` handler: it still forces analysis
> of every enumerated rawr call, including ReleaseSafe checks, without requiring platform stack-trace
> machinery. The cell and then the full matrix passed after that tooling-only fix.
> No production source, hot path, or OS-conditional shipped source changed.
>
> Final classification: all 16 target triples **compile**; no cell is described as verified because
> nothing ran on a target host. There were no `not targetable` cells, no package-only failures, and both
> documented `-Dcroaring-avx512` option values compiled through the affected CRoaring-backed step.

Runs what `47-00` built and deals with what falls out. **No runtime hosts, no evidence table, no README** —
those are `47-02`.

## 1. The matrix

`{aarch64, x86_64}` × `{linux-gnu, linux-musl, macos, windows-gnu, windows-msvc, freebsd, netbsd,
openbsd}` = **16 cells**, plus the two baseline-feature cells of `47-00` §4.

**Both checks per cell**: the API probe compile *and* the allowlist-only package consumer. A cell where the
probe compiles but the package consumer fails is a **finding, not a pass** — that combination is exactly
the shipped-`build.zig` defect shape the toplevel is chasing.

**`not targetable` has a definition, and it is deliberately hard to reach.** It applies **only when Zig
0.16 cannot resolve or build a minimal control program for that target** — a hello-world, not rawr.
**A probe or package-consumer failure is `broken`, even when the compiler diagnostic reads like a target
limitation.** Without this rule, a genuine rawr defect gets filed as someone else's problem and the cell
goes quiet.

**So the classification requires the minimal control**: build it first, and record its result alongside
the cell. A cell may only be `not targetable` if that control also failed.

Silent omission is the failure mode `47-00` §5 exists to prevent, and this chunk is where it would
happen.

## 2. Order — highest known risk first

1. **`openbsd` and `freebsd`, both arches.** The only OS values that branch in the shipped `build.zig`,
   and the OpenBSD branch names a file outside `.paths`. **This is a package-integrity risk, not a claim
   that BSD is the largest portability unknown overall** — call-site inspection shows the helpers attach
   only to development modules, never `lib_mod`.
2. **`windows-gnu` and `windows-msvc`.** The largest overall unknown; nothing has ever run there. Two
   ABIs, and neither speaks for the other.
3. Everything else.

Failing early on the risky cells is worth more than a tidy alphabetical sweep.

## 3. Breakage — fix or record, and say which

Per toplevel §7, Tier 1 breakage is **either fixed or recorded as a known limitation with its actual
error**. Both are acceptable outcomes; leaving it ambiguous is not.

**Scope discipline on fixes:**

- A fix must be **correctness-scoped**. Performance is out of scope per toplevel §8 and this chunk makes
  no timing claim.
- **If a fix touches production hot-path code, say so explicitly and flag it for separate performance
  validation.** Do not run a board here and do not assume a compile fix is perf-neutral. The campaign has
  repeatedly found that unrelated edits move rows, so an unflagged hot-path change would quietly enter the
  next board comparison as an unexplained mover.
- A fix that would require OS-conditional code **in a shipped source file** is a design change, not a
  portability fix. **Stop and report it** — toplevel §1's finding is that `src/*.zig` is currently free of
  `builtin.os` and `os.tag`, and that property is worth keeping deliberately rather than losing by
  increment.

## 4. What this chunk cannot conclude

**A compiling target is not a working target.** `compiles` is a status, not a support claim, and no cell
may be described as verified here — nothing has executed. The distinction is the whole point of toplevel
§6's status vocabulary.

**Status attaches to a target triple.** `windows-gnu` compiling says nothing about `windows-msvc`, and
`linux-gnu` says nothing about `linux-musl`.

## Acceptance

- All **16 cells** plus the two baseline-feature cells run, **both checks each**.
- Every cell recorded as `compiles`, `broken` with its error, or `not targetable` with its error —
  **none silently skipped**, demonstrated by the `47-00` §5 reporting path.
- **Every `not targetable` cell backed by a failing minimal-control build**, per §1. A cell whose control
  built but whose rawr checks did not is `broken`.
- Cells run in §2 order, with BSD and Windows results reported first.
- Baseline-feature cells report their **asserted kernel selection**.
- Every breakage **either fixed or recorded**, explicitly labelled which.
- Any fix touching production hot-path code **flagged for separate performance validation**; any fix
  requiring OS-conditional code in a shipped source file **stopped and reported, not applied**.
- **No cell described as `verified`.** No evidence table, no README change.
- Existing suites plus `check-32`, `check-docs`, `check-package` green on the dev host.

## Verification record — implemented, reviewed, ACCEPTED

**Checked in the tree, not taken on report.** `portability_cells` holds exactly **18 entries** — 16 target
triples plus the two baseline-feature cells — and they are declared in §2 risk order: OpenBSD, FreeBSD,
Windows GNU, Windows MSVC, then the rest. **The ordering is structural in `build.zig`, not a claim about
how the run happened.** The controller's expected count is `b.fmt("{d}", .{portability_cells.len})`, so a
passing run proves every declared cell was visited — the `47-00` cell-count fix paying off on its first
real use.

**The two-check design earned its keep immediately.** `aarch64-windows-msvc` failed the probe while its
minimal control **and** its package consumer compiled. That combination is what distinguishes a harness
defect from a library defect:

- probe alone → the cell is recorded `broken` and rawr carries a **false portability defect**;
- package alone → the failure never surfaces at all.

The toplevel anticipated the mirror case (probe compiles, package fails = real defect). This was the
opposite pairing, and the split discriminated both directions. Recording it here because the next reader
will otherwise see only "one cell failed and was fixed".

**The fix is two lines and touches no production source:** `pub const panic = std.debug.no_panic` and
`runProbe() catch unreachable` → `catch @trap()`. `runProbe()` is still called, so analysis of every
enumerated rawr call is preserved; only the error path changed, which is what pulled Windows stack-trace
machinery into a compile-only object.

### Two boundary notes, neither blocking

**"including ReleaseSafe checks" is asserted, not demonstrated.** The probe is compile-only and the
`47-00` control seeds a **type** defect, which semantic analysis catches regardless of panic handling. So
the controls demonstrate that the enumerated calls are still analysed — they demonstrate nothing about
safety-check analysis. The claim is likely true, since safety checks are still emitted and merely trap,
but nothing in this chunk tests it and the probe's purpose is compile-time analysis rather than runtime
checking. **The clause is doing no work; drop it or back it.**

**`tools/check_32_api.zig` is now shared by two guards with only one set of controls.** `check-32` and
`check-portability` both compile it, but `scripts/check-portability-controls.sh` exercises it only through
`check-portability-*-probe`; **no seeded-defect control targets `check-32`.** A future probe edit could
weaken `check-32` while every check still passed — the same invisible-boundary shape spec 40-01 recorded.
Worth a control when `47-02` touches this area, and worth stating in the evidence table either way.

## Estimate

**S/M** — mechanical to run. The size depends entirely on how much breakage falls out, which is the
question the chunk exists to answer.
