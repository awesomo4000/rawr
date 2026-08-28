<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 51-02: Ship the branchy hoisted merge

Toplevel: [51-array-union-difference-kernels.md](51-array-union-difference-kernels.md).
Gated on: [51-01](51-01-scalar-merge-candidates.md) complete and accepted.

`51-01` returned **GO for C3** — the branchy three-way body with hoisted values and bulk tails — on both
operations and both hosts. C3 was measurably faster than C2 on Zen 4 (`OR` [0.133, 0.143] against
[0.150, 0.159]; `ANDNOT` [0.114, 0.135] against [0.149, 0.223]), indistinguishable from it on M4, and
smaller code on both. **This chunk ships it.**

## 1. What changes

Two loops, both `pub inline` and both called from exactly one production site:

| loop | production caller | output buffer |
| --- | --- | --- |
| `benchmarkArrayUnionWrite` (`container_ops.zig:540`) | `arrayUnionArray` (`:509`) | freshly allocated by `ArrayContainer.init` |
| `benchmarkArrayDifferenceWrite` (`:1015`) | `arrayDifferenceArray` (`:1002`) | freshly allocated |

Replace each body with the C3 form validated in `51-01`. No signature change, no new public API, no
per-architecture code, no SIMD.

**Rename both.** The `benchmark` prefix was correct when these were extraction hooks; it will be
misleading once they are the shipped merge and the only implementation. Keep the internal export so the
rig still reaches them.

## 2. The aliasing surface — and the third loop nobody measured

`51-01` §2 warned that `@memcpy` becomes undefined behaviour if this loop is ever moved somewhere the
output can alias an input. **That place already exists in the tree**, and it is not one of the two loops
above.

`ArrayContainer.unionInPlace` (`array_container.zig:201`) carries a **third copy of the same branchless
merge**, and it merges within a single buffer: it `@memmove`s `self`'s values to the end of its own array
(`:239`) and then merges forward into the front, relying on the write cursor staying at or behind the read
cursor. Its first drain loop copies `self.values[si]` into `self.values[k]` — **self-overlapping, forward,
`k <= si`**.

Three consequences, all of which must be honoured rather than discovered:

- **`@memcpy` is wrong there.** That drain needs `@memmove`. `@memcpy`'s non-overlap requirement is
  violated by construction, and nothing at the call site would say so.
- **`unionInPlace` is out of scope for this chunk.** `51-01` measured the out-of-place loops only.
  Changing an unmeasured hot path used by the in-place union entry points is precisely the spec-27 trap.
  **Leave it as it is.**
- **Say so in the code.** After this chunk the tree holds two different merge idioms on purpose. Note at
  `unionInPlace` that its branchless form is deliberate and unmeasured, not an oversight, and point at
  this spec. Otherwise the next reader harmonizes them and silently ships an unmeasured change.

**Difference has no aliasing case.** `containerDifferenceInPlace` (`container_ops.zig:409`) routes an
array left operand to the out-of-place `containerDifference`, so there is no in-place array-array
difference merge.

**Guard the precondition rather than documenting it.** Add a debug assertion that `output` overlaps
neither input, so a future caller that violates it fails in Debug and ReleaseSafe instead of corrupting
data in ReleaseFast. A comment cannot fail; this can.

## 3. Codegen must be re-verified after inlining

`51-01`'s disassembly covered **non-inlined batch symbols**, which is a scope limit its own record states.
Production inlines these loops into their call sites, and inlining changes register pressure and what the
optimizer can prove.

**Re-record the disassembly at the production call sites on both hosts** and confirm the data-dependent
value-comparison branches survive there. If they do not, the shipped code is not the thing `51-01`
measured, and the board result is the only remaining evidence.

Update the loop comments. The current *"Branchless merge … on aarch64, LLVM emits csel"* text becomes
false for these two loops. It stays true for `unionInPlace`.

## 4. Keep the rig able to measure this

Once production becomes C3, the rig's `a1_rawr_scalar` arm **is** C3, and the A1-versus-A2 gap the whole
spec was built on disappears from the output.

**Retain the old branchless form as its own arm.** Without it the rig reports a meaningless "no gap" and
the historical baseline is gone — including the ability to detect a future regression back toward it.

## 5. Gates

### 5.1 Canonical board — the corpus-specificity gate

Full canonical board, **both hosts**, per the spec 22 protocol.

This is the real test, and it is adversarial by construction. `51-01` measured streak lengths of p50 8,
p90 41, p99 247 on real data; **synthetic board inputs are where merge decisions are least predictable**,
which is the regime the original branchless decision was made in. `uscensus2000` contributed 21 matched
pairs in `51-01` and constrains almost nothing, so the board is carrying this on its own.

**No row may regress beyond the standing 5% gate.** Report any row that moves with its full range rather
than smoothing it, and tolerate whole-binary layout movement per spec 28 — sub-1.2x M4 ratios sit at the
measurement floor and a shifted untouched row is not evidence of a kernel effect.

### 5.2 Real-data rows — did the thing we were chasing actually move

Re-run the spec 50 harness on all three corpora, both hosts. Layer A recovery does not entail an
end-to-end move; specs 27 and 35 both produced kernel-level wins that failed to reach the row.

**Pre-registered expectations**, from `51-01` recovery applied to `51-00`'s scalar terms. These are
predictions to check against, **not acceptance thresholds** — the point is that a large divergence means
the attribution was wrong somewhere:

| host | operation | scalar term | expected reduction | matched delta today | expected remainder |
| --- | --- | ---: | ---: | ---: | ---: |
| M4 | OR | 0.300 ms | ~0.269 ms | 0.500 ms | ~0.23 ms |
| M4 | ANDNOT | 0.322 ms | ~0.320 ms | 0.265 ms | **at or past parity** |
| Zen 4 | OR | 0.226 ms | ~0.207 ms | 0.517 ms | ~0.31 ms |
| Zen 4 | ANDNOT | 0.263 ms | ~0.277 ms | 0.387 ms | ~0.11 ms |

**Report actual against predicted for all four**, and account for any cell that misses badly. M4 ANDNOT is
the sharpest prediction: the scalar term exceeds the whole matched delta, so if the row does not reach
parity the apportionment has a problem worth understanding before the next chunk.

### 5.3 Correctness

Existing differential suites, plus container-level tests for the cases the corpus cannot reach — empty
either side, single element, disjoint, fully identical, each side exhausting first. `51-01` proved these
paths exist and go untested by corpus data alone, since corpus inputs have `min=2`.

## 6. Reconcile the record

`done/optimization-branchless-merge.md` concluded the opposite for these loops. **Add a pointer there to
this spec** when it lands. Neither record is wrong: that one measured synthetic data, this one measured
real data, and the two regimes genuinely differ. An archive holding both conclusions with no link between
them is the problem.

**Language rule for the claim.** `51-01` measured a **branchy hoisted body**, changing branch structure
and load behaviour together. Say that. **Do not claim branch predictability specifically** — the
measurement does not separate it from load hoisting, and `51-01` says so explicitly.

## Acceptance

- Both out-of-place loops carry the C3 form; renamed off the `benchmark` prefix; internal export retained.
- **`unionInPlace` unchanged**, with a comment recording that its branchless form is deliberate and
  unmeasured and pointing here.
- **Debug assertion that `output` overlaps neither input**, verified to fire against a deliberately
  aliased call.
- Disassembly re-recorded **at the production call sites** on both hosts, stating whether the
  data-dependent branches survived inlining; loop comments corrected.
- **Old branchless form retained as a rig arm**, with a run showing the comparison still measurable.
- Canonical board, both hosts, no row regressing beyond the 5% gate; moved rows reported with ranges.
- Spec 50 harness re-run, three corpora, both hosts, with **actual against predicted for all four cells**
  per §5.2 and an account of any large miss.
- §5.3 correctness tests pass; all four suites plus `check-32`, `check-docs`, `check-package` green.
- Pointer added to `done/optimization-branchless-merge.md`; claim worded per §6.

## Estimate

**M** — the code change is small and already validated. The board run, the real-data re-run, and the
post-inlining disassembly are the work.
