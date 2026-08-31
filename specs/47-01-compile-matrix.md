<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 47-01: Run the 16-target compile matrix

Toplevel: [47-portability-matrix.md](47-portability-matrix.md).
Gated on: [47-00](47-00-portability-machinery.md) complete, **with all five controls exercised**.

Runs what `47-00` built and deals with what falls out. **No runtime hosts, no evidence table, no README** —
those are `47-02`.

## 1. The matrix

`{aarch64, x86_64}` × `{linux-gnu, linux-musl, macos, windows-gnu, windows-msvc, freebsd, netbsd,
openbsd}` = **16 cells**, plus the two baseline-feature cells of `47-00` §4.

**Both checks per cell**: the API probe compile *and* the allowlist-only package consumer. A cell where the
probe compiles but the package consumer fails is a **finding, not a pass** — that combination is exactly
the shipped-`build.zig` defect shape the toplevel is chasing.

**Targets Zig 0.16 cannot build at all are recorded as `not targetable`**, with the error. Silent
omission is the failure mode `47-00` §5 exists to prevent, and this chunk is where it would happen.

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
- Cells run in §2 order, with BSD and Windows results reported first.
- Baseline-feature cells report their **asserted kernel selection**.
- Every breakage **either fixed or recorded**, explicitly labelled which.
- Any fix touching production hot-path code **flagged for separate performance validation**; any fix
  requiring OS-conditional code in a shipped source file **stopped and reported, not applied**.
- **No cell described as `verified`.** No evidence table, no README change.
- Existing suites plus `check-32`, `check-docs`, `check-package` green on the dev host.

## Estimate

**S/M** — mechanical to run. The size depends entirely on how much breakage falls out, which is the
question the chunk exists to answer.
