# Spec 08: Coverage-guided differential fuzzing

## Goal

Stand up continuous, coverage-guided fuzzing across rawr's API. The design
principle: **fuzzing without an oracle only finds crashes.** rawr already has an
oracle (CRoaring) and a comparator (`assertAgree`), so the high-value target is
**differential fuzzing** — let the coverage-guided engine generate the inputs,
and assert rawr agrees with CRoaring on every operation. That catches correctness
divergences, not just panics. Crash-only fuzzing is reserved for the parse paths,
where there is no oracle for "handled a hostile byte-blob safely."

This complements, and largely subsumes, the fixed-RNG randomized loop in
`diff_test.zig` (spec 01): same comparator, but inputs are coverage-guided and the
run is continuous rather than a bounded local loop.

## Task 0 — Toolchain spike (do first; de-risks everything)

Zig's fuzzing story is evolving; pin the mechanism on **Zig 0.16** before
building harnesses. Evaluate, in order of preference, and pick the first that
works end-to-end with CRoaring linked:

1. **Zig native fuzzer** — `std.testing.fuzz` + `zig build test --fuzz` (and its
   web UI). Lowest friction since the project is already Zig + `translate-c`.
   Confirm: (a) it's usable/stable enough in 0.16, (b) a fuzz target can link the
   CRoaring C object (via the existing `addTranslatedCImport` helper) and call
   `c.roaring_bitmap_*`, (c) it reports coverage and persists a corpus.
2. **libFuzzer via clang** — a C/C++ fuzz target compiled with
   `-fsanitize=fuzzer,address`, linking both `roaring.c` and rawr (rawr exported
   as a static lib / object). More setup, very mature.
3. **AFL++** — `afl-clang-fast` on a forkserver harness. Most robust for
   long campaigns; heaviest setup.

Deliverable: a one-paragraph decision in this spec recording the chosen mechanism
and why, plus a minimal "hello-world" fuzz target that builds, runs for a few
seconds, links CRoaring, and finds a deliberately-planted disagreement (sanity
check that the oracle wiring works). **Everything below assumes the mechanism
chosen here.**

## Task 1 — Operation-stream decoder (the fuzz input format)

A pure decoder turning the fuzzer's opaque `[]const u8` into a typed sequence of
operations over a small set of working bitmaps. Pure rawr, deterministic, no
allocation beyond the ops list.

- Treat the input bytes as a little program: read an opcode byte, then operands
  (values, ranges, bitmap-register indices) from subsequent bytes; stop at end of
  input. Out-of-range opcodes/operands wrap or are skipped — never error, so all
  inputs are valid programs (maximizes useful coverage).
- Maintain a few "registers" (e.g. 2–4 bitmaps) so binary ops have operands.
- Opcodes cover the **mutating + querying** surface that has a CRoaring oracle:
  `add`, `remove`, `addRange`, `removeRange` (once it exists), `and`/`or`/`xor`/
  `andnot` (+inplace), `runOptimize`, `flip` (once it exists), `rank`/`select`
  (once they exist), `contains`, `cardinality`, `clone`. Gate
  not-yet-implemented ops behind feature availability so this spec can land before
  the Tier-1 parity work.
- Bias operand magnitudes to exercise container-type boundaries (values near
  chunk edges, ranges that cross 4096, etc.) — same spirit as `test_gen`.

## Task 2 — Differential fuzz target (the main harness)

Drive the decoder against **paired** rawr + CRoaring registers: every op is
applied to both, identically. After each mutating op (or at a checkpoint cadence
for speed), assert agreement using the existing comparator:

- Reuse / share `assertAgree` from `diff_test.zig` (byte-identical portable
  serialization + cardinality + membership probes). Factor it into a shared module
  if the fuzz target can't import the exe directly.
- For scalar ops (`rank`/`select`/`cardinality`/`contains`), compare the scalar
  results directly.
- Use the **leak-checking allocator** on the rawr side (as in `diff_test.zig`):
  a fuzzer running millions of op sequences is the ideal place to surface a leak /
  double-free / use-after-free. A leak at teardown is a finding.
- On any divergence or crash, the engine saves the input; ensure the harness
  prints the decoded op sequence for a found crash so it's reproducible as a
  unit test.

This is the spec-01 randomized loop with coverage-guided inputs and a continuous
budget. Keep the fixed-RNG `diff_test.zig` loop too — it's the fast,
deterministic CI gate; the fuzzer is the deep, long-running campaign.

## Task 3 — Parse-path crash-fuzz targets (no oracle)

Separate, simpler targets for the untrusted-byte surface, where the only contract
is "no crash, and `_safe`/validate paths reject cleanly":

- `RoaringBitmap.deserialize` / `deserializeSafe`
- `FrozenBitmap.init` followed by `contains` probes + full `iterator` drain
- `bitmap.validate()` on the result of `deserialize`

Acceptable outcomes: a Zig error, or a valid traversable bitmap. A crash / panic /
hang / leak is a finding. Seed the corpus from serialized `test_gen` bitmaps (all
profiles) so the fuzzer starts from structurally-valid inputs and mutates outward.
(This is the continuous, coverage-guided version of the spec-04/06 malformed smoke
tests.)

## Task 4 — Corpus, CI, and regression capture

- **Seed corpus:** serialize a spread of `test_gen` bitmaps (every profile,
  run-optimized and not) as the starting corpus for both the differential and
  parse-path targets. Check it into the repo (small).
- **CI:** a time-boxed fuzz run (e.g. a few minutes per target) as a CI job,
  distinct from `zig build test`. It must fail the build on any crash/divergence
  and upload the offending input as an artifact.
- **Regression:** every crash the fuzzer finds gets minimized and added as a
  deterministic reproducer test (decoded op sequence for differential finds; raw
  bytes for parse finds), so fixed bugs stay fixed. The corpus also grows with
  these inputs.
- Add `zig build fuzz` (or per-target steps) wiring, mirroring the `difftest`
  step, using `addTranslatedCImport` for the CRoaring-linked targets.

## Acceptance criteria

1. Task 0 decision recorded; a minimal differential fuzz target builds, links
   CRoaring, runs under the chosen engine, and detects a planted disagreement.
2. The differential target (Task 2) runs op sequences against paired rawr +
   CRoaring with `assertAgree` after mutations and a leak-checking allocator;
   shares the comparator with `diff_test.zig` rather than duplicating it.
3. Parse-path targets (Task 3) fuzz `deserialize`/`deserializeSafe`/
   `FrozenBitmap.init`/`validate` with no crash/leak; safe paths reject cleanly.
4. A seed corpus from `test_gen` profiles is checked in; a time-boxed CI fuzz job
   fails on crash/divergence and saves the artifact.
5. A documented path from "fuzzer found X" to "X is a checked-in regression test."
6. `zig build test`, `zig build validate`, `zig build difftest`, and the new
   `zig build fuzz` (short run) all pass.

## Notes / scope

- This overlaps the spec-01 differential loop **by design** — shared comparator,
  different input source. Don't delete the deterministic loop; it stays the fast
  gate.
- Coverage-guided fuzzing of the algebra is only worthwhile *because* of the
  oracle (Task 2). A crash-only algebra fuzzer would be low value — that's why the
  algebra target is differential and only the parse paths are crash-only.
- CI cost is real (continuous fuzzing). The time-boxed job keeps PR latency sane;
  longer campaigns can run out-of-band (nightly / manual) against the same
  targets + corpus.
- Best sequenced **after** at least the Tier-1 parity work (07) so the op-stream
  decoder can cover flip/rank/select/removeRange; but Tasks 0, 1 (partial), and 3
  can proceed now against the current surface and grow as parity lands.
