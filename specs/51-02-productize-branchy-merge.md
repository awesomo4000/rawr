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

**Rename both** to `arrayUnionWrite` and `arrayDifferenceWrite`. The `benchmark` prefix was correct when
these were extraction hooks; it is misleading once they are the shipped merge and the only
**out-of-place** implementation — `unionInPlace` keeps its own branchless merge by design (§2). Keep the
internal export so the rig still reaches them.

**Every test added by this chunk calls these production helpers**, never a copy preserved in the
benchmark tree. A test that exercises a duplicate proves nothing about what ships.

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

**The guard compares byte ranges, not pointers.** Pointer equality is the one aliasing case that would
never occur here — the dangerous one is a slice into the middle of an input, which has a different
pointer and overlaps anyway. The condition is range intersection: `output` and each input overlap unless
one ends at or before the other begins.

**Testing it needs a child process.** A failed `assert` panics and cannot be caught by an ordinary Zig
test, so the guard is otherwise unfalsifiable — the exact failure mode this campaign keeps finding.
Require a small process-isolated harness built in **ReleaseSafe**, invoked per case, with the parent
checking the child's exit:

| case | expected |
| --- | --- |
| `output` is the same slice as an input | panic |
| `output` overlaps the head of an input | panic |
| `output` overlaps the tail of an input | panic |
| `output` lies strictly inside an input | panic |
| `output` is adjacent to an input, sharing no byte | **no panic** |
| `output` is a separate allocation | **no panic** |

The last two are the controls, and the adjacent case is the one that matters: it catches an off-by-one
that makes the guard fire on legitimate callers.

## 3. Codegen must be re-verified after inlining

`51-01`'s disassembly covered **non-inlined batch symbols**, which is a scope limit its own record states.
Production inlines these loops into their call sites, and inlining changes register pressure and what the
optimizer can prove.

**Re-record the disassembly on both hosts at the inlined bodies of `arrayUnionArray` and
`arrayDifferenceArray`** — the production call sites, named explicitly, not a helper symbol that may not
survive inlining — and state whether the data-dependent value-comparison branch survives in each.

**The outcome is defined in advance, and it decides the wording, not the adoption:**

| result | adoption | what may be claimed |
| --- | --- | --- |
| **branches survive** | proceed on `51-01` plus the board | **branchy plus hoisted body** |
| **branches removed by inlining** | may still proceed, **on the board alone** | **the C3 source form** — nothing attributing the win to branching |

The second row is not a failure. It means the shipped code is not the thing `51-01` timed, so `51-01`
stops being evidence for it and the board carries the decision by itself. §6's language rule is
conditional on this outcome.

Update the loop comments. The current *"Branchless merge … on aarch64, LLVM emits csel"* text becomes
false for these two loops. It stays true for `unionInPlace`.

## 4. Keep the rig able to measure this

Once production becomes C3, the rig's `a1_rawr_scalar` and `c3_*` arms are the same code, and the
A1-versus-A2 gap the whole spec was built on disappears from the output. The arm set has to be redefined,
not just extended.

**`a1_rawr_scalar` keeps its meaning: rawr as it ships.** That definition is what makes `A1 - A2`
comparable across `51-00`, `51-01`, and everything after. Re-point it at the renamed production helper;
do not repurpose the name for the historical form.

**Add `h1_rawr_branchless_legacy` — a frozen private copy of the pre-`51-02` branchless source.** It must
be its own source text in the benchmark tree, never a call into production, so it cannot silently track
future edits and quietly stop being a baseline. It is the regression detector: `a1` drifting back toward
`h1` becomes visible in one run.

**Retire `c1`, `c2`, and `c3`.** C3 is now `a1`, and C1/C2 were variants of a body that no longer exists.
Keeping them would leave three arm names whose meanings depend on which chunk you are reading.

**Layer A arm set becomes `a1`, `a2`, `a3`, `h1`** — 4 arms × 2 operations × 3 datasets = **24 tuples**.
`51-01`'s protocol mutation control rejects a missing row against an expected-row manifest, so **update
that manifest in the same commit and re-exercise the control**. A stale expected count either fails every
run or, worse, passes while silently expecting arms that no longer exist.

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

Layer A recovery does not entail an end-to-end move; specs 27 and 35 both produced kernel-level wins that
failed to reach the row. Two separate measurements answer two separate questions, and **neither is a
subtraction across runs**.

#### The measured reduction comes from one binary: `h1 - a1`

**Do not quantify the improvement by subtracting a new spec 50 run from the old one.** Different binary,
different layout, and a reference that moves on its own — spec 39 established that a rawr gain of +4.1%
sat next to a CRoaring reference moving −7.8% in the same pair of runs, which is why cross-run arithmetic
was banned.

The retained `h1` arm makes this unnecessary. **`h1 - a1` within a single run is the actual kernel
reduction**, measured on one binary with one layout.

**The predictions below are `wikileaks-noquotes` only.** Both the `51-00` scalar terms and the `51-01`
recovery figures were measured on the target corpus, so comparing `h1 - a1` against them is valid for
that dataset and for no other. **Report `h1 - a1` on `census1881` and `uscensus2000` as controls** — no
predicted value, no miss to account for. Their job is to show the change is not a loss elsewhere, and
`uscensus2000`'s 21 matched pairs constrain little even for that.

| host | operation | scalar term (`51-00`) | `51-01` recovery | predicted `h1 - a1` |
| --- | --- | ---: | ---: | ---: |
| M4 | OR | 0.300 ms | 0.895 | ~0.269 ms |
| M4 | ANDNOT | 0.322 ms | 0.993 | ~0.320 ms |
| Zen 4 | OR | 0.226 ms | 0.917 | ~0.207 ms |
| Zen 4 | ANDNOT | 0.263 ms | 1.052 | ~0.277 ms |

**Predictions to check against, not acceptance thresholds.** A large miss means the attribution was wrong
somewhere, and that is worth knowing before the next chunk.

#### The spec 50 rerun reports the row, and only the row

Re-run the spec 50 harness on all three corpora, both hosts. It answers **where the row now stands** and
whether anything regressed. It does **not** supply the improvement figure.

**Report the new ratios with their ranges, and quote the old ones as context only.** Do not compute a
difference of ratios and present it as a measurement — that is the same cross-run arithmetic in a
different shape, and it is wrong for the same reason. The claim is *the row is now X*, not *the row
improved by Y*.

`51-00`'s matched deltas of 0.500 / 0.265 / 0.517 / 0.387 ms are the standing context. **M4 ANDNOT is the
sharpest case**: its scalar term exceeds the entire matched delta, so it should reach or pass parity.

**If it does not, that is not by itself evidence against the apportionment.** The parity result comes
from a new run while the matched delta is historical, so reference drift and layout movement are live
explanations — the same reason this section bans cross-run subtraction. **Reconcile against the
same-binary `h1 - a1` figure instead.** If that came in near prediction, the kernel change did what
`51-01` said and the row's position is a question about the rest of the pipeline. Only if `h1 - a1` also
misses badly does the attribution itself come into question.

### 5.3 Correctness

Existing differential suites, plus container-level tests for the cases the corpus cannot reach — empty
either side, single element, disjoint, fully identical, each side exhausting first. `51-01` proved these
paths exist and go untested by corpus data alone, since corpus inputs have `min=2`.

## 6. Reconcile the record

`done/optimization-branchless-merge.md` concluded the opposite for these loops. **Add a pointer there to
this spec** when it lands. Neither record is wrong: that one measured synthetic data, this one measured
real data, and the two regimes genuinely differ. An archive holding both conclusions with no link between
them is the problem.

**Language rule for the claim, conditional on §3.** If the data-dependent branches survive inlining,
`51-01` measured a **branchy hoisted body** — branch structure and load behaviour changed together, so
say that and **do not claim branch predictability specifically**. If they do not survive, the claim
narrows to **the C3 source form**, adopted on board evidence, with nothing attributing the win to
branching at all.

## Acceptance

- Both out-of-place loops carry the C3 form, renamed to `arrayUnionWrite` / `arrayDifferenceWrite`;
  internal export retained. **All new tests call these, not benchmark-tree copies.**
- **`unionInPlace` unchanged**, with a comment recording that its branchless form is deliberate and
  unmeasured and pointing here.
- **Byte-range overlap assertion**, with the process-isolated ReleaseSafe harness of §2 covering all four
  panic cases **and both no-panic controls** — the adjacent-slice control specifically.
- Disassembly re-recorded on both hosts **at the inlined `arrayUnionArray` and `arrayDifferenceArray`
  bodies**, with the §3 outcome stated and §6's wording chosen from it; loop comments corrected.
- **Arm set redefined per §4**: `a1` re-pointed at production, `h1_rawr_branchless_legacy` added as frozen
  source, `c1`/`c2`/`c3` retired, 24 tuples, **expected-row manifest updated and its mutation control
  re-exercised in the same commit**.
- **`h1 - a1` reported per operation per host from a single run**, compared against §5.2's predictions
  **on `wikileaks-noquotes` only**, with any large miss accounted for; the other two datasets reported as
  controls with no predicted value.
- Canonical board, both hosts, no row regressing beyond the 5% gate; moved rows reported with ranges.
- Spec 50 harness re-run, three corpora, both hosts, **reporting new ratios with ranges only** — old
  ratios quoted as context, **no difference of ratios computed**.
- §5.3 correctness tests pass; all four suites plus `check-32`, `check-docs`, `check-package` green.
- Pointer added to `done/optimization-branchless-merge.md`; claim worded per §6.

## Estimate

**M** — the code change is small and already validated. The board run, the real-data re-run, and the
post-inlining disassembly are the work.
