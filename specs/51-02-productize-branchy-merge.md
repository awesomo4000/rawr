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
| `arrayUnionWrite` (renamed from `benchmarkArrayUnionWrite`) | `arrayUnionArray` | freshly allocated by `ArrayContainer.init` |
| `arrayDifferenceWrite` (renamed from `benchmarkArrayDifferenceWrite`) | `arrayDifferenceArray` | freshly allocated |

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

## Outcome (08/28/2026)

**GO. The branchy hoisted body with bulk tails is now the production implementation for the two
out-of-place array merges.** `ArrayContainer.unionInPlace` remains branchless and unchanged. Its output
aliases its input, and `51-01` did not measure that path.

The non-aliasing guard uses an explicit Debug/ReleaseSafe panic instead of `std.debug.assert`. Zig lowers
`std.debug.assert` to an unlabelled `unreachable`, while the child-process control needs to distinguish
the intended failure from an unrelated crash. The guard still compiles out in ReleaseFast. The
ReleaseSafe harness observed the named failure in ten operation/case combinations and accepted four
adjacent/separate controls.

### Same-binary kernel reduction

The retained `h1` arm measured the production reduction without cross-run subtraction:

| host | operation | predicted | measured `h1 - a1` | result |
| --- | --- | ---: | ---: | --- |
| M4 | OR | ~0.269 ms | **0.263 ms [0.254, 0.278]** | matches |
| M4 | ANDNOT | ~0.320 ms | **0.300 ms [0.294, 0.317]** | matches |
| Zen 4 | OR | ~0.207 ms | **0.185 ms [0.180, 0.205]** | close, 0.022 ms below |
| Zen 4 | ANDNOT | ~0.277 ms | **0.285 ms [0.283, 0.307]** | matches |

`census1881` also improved on both hosts: OR by 0.125 / 0.106 ms and ANDNOT by 0.149 / 0.148 ms for
M4 / Zen 4. The 21-pair `uscensus2000` cells remained below useful timing resolution. The cross-host
semantic audit passed for all 24 tuples and 120 processes per host. Removing one process row produced
`ProcessCountMismatch`, so the updated expected-row guard remains live.

### Current real-data rows

These are current positions only. The table does not subtract them from historical spec 50 runs.

| host | dataset | operation | rawr ms [min,max] | CRoaring ms [min,max] | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| M4 | uscensus2000 | OR | 0.079 [0.071, 0.083] | 0.139 [0.123, 0.156] | **0.568x** |
| M4 | uscensus2000 | ANDNOT | 0.040 [0.037, 0.043] | 0.091 [0.086, 0.095] | **0.440x** |
| M4 | census1881 | OR | 0.269 [0.258, 0.273] | 0.245 [0.232, 0.254] | **1.098x** |
| M4 | census1881 | ANDNOT | 0.157 [0.153, 0.166] | 0.170 [0.158, 0.179] | **0.924x** |
| M4 | wikileaks-noquotes | OR | 0.444 [0.423, 0.544] | 0.247 [0.246, 0.296] | **1.798x** |
| M4 | wikileaks-noquotes | ANDNOT | 0.170 [0.167, 0.180] | 0.211 [0.211, 0.220] | **0.806x** |
| Zen 4 | uscensus2000 | OR | 0.113 [0.107, 0.178] | 0.135 [0.132, 0.140] | **0.831x** |
| Zen 4 | uscensus2000 | ANDNOT | 0.127 [0.107, 0.160] | 0.064 [0.063, 0.067] | **1.992x** |
| Zen 4 | census1881 | OR | 0.275 [0.264, 0.397] | 0.827 [0.818, 0.839] | **0.333x** |
| Zen 4 | census1881 | ANDNOT | 0.147 [0.142, 0.153] | 0.567 [0.547, 0.577] | **0.259x** |
| Zen 4 | wikileaks-noquotes | OR | 0.612 [0.600, 0.624] | 0.307 [0.303, 0.314] | **1.998x** |
| Zen 4 | wikileaks-noquotes | ANDNOT | 0.224 [0.220, 0.237] | 0.129 [0.109, 0.138] | **1.738x** |

Zen 4 `uscensus2000` ANDNOT remains an open low-resolution row. It has only 21 matched pairs, and rawr's
0.127 ms median spans [0.107, 0.160] against CRoaring's 0.064 [0.063, 0.067] ms. The same-binary
attribution cell is below useful timing resolution, while the current-position table cannot establish a
cross-run change. No kernel conclusion is drawn from its 1.992x ratio.

### Canonical board and code generation

The directly affected canonical rows are flat or faster against their pre-change baselines:

| host | row | pre-change rawr [min,max] | candidate rawr [min,max] |
| --- | --- | ---: | ---: |
| M4 | sparse AND | 0.614 [0.596, 0.620] ms | 0.589 [0.586, 0.592] ms |
| M4 | sparse AND arena control | 0.571 [0.560, 0.603] ms | 0.556 [0.542, 0.562] ms |
| M4 | dense AND | 145.264 [143.066, 145.996] ns/op | 138.916 [136.963, 139.771] ns/op |
| M4 | sparse OR | 1.798 [1.784, 1.811] ms | **1.567 [1.553, 1.571] ms** |
| M4 | dense OR | 234.009 [228.882, 234.375] ns/op | 223.877 [223.145, 224.243] ns/op |
| Zen 4 | sparse AND | 0.914 [0.902, 0.926] ms | 0.906 [0.903, 0.933] ms |
| Zen 4 | dense AND | 230.187 [224.602, 236.866] ns/op | 227.841 [226.145, 237.153] ns/op |
| Zen 4 | sparse OR | 2.864 [2.852, 2.911] ms | **2.765 [2.724, 2.836] ms** |
| Zen 4 | dense OR | 369.975 [364.512, 384.470] ns/op | 369.872 [366.781, 387.476] ns/op |

The paired M4 comparison used an exact `b3ab49f` source export followed immediately by the candidate and
found no rawr row beyond the 5% gate. The earlier sparse-AND arena comparison used a prior-day baseline
and incorrectly reported a 6.2% regression with separated ranges. The paired comparison is 0.571
[0.560, 0.603] against 0.556 [0.542, 0.562] ms, a 2.6% improvement with overlapping ranges.

The strict Zen 4 all-row comparison was not clean: `rankMany` moved from 0.995x to 1.152x and skewed
`andCardinality` from 0.977x to 1.163x. Both functions have instruction-identical production bodies
after address normalization between the clean `b3ab49f` baseline and candidate; their addresses moved
with the changed binary. This is the whole-binary layout effect already recorded by spec 28, not an
array union/difference code path. It is reported here rather than represented as a clean literal 5%
board pass.

The inlined production bodies retain data-dependent value-comparison branches on both hosts. The result
may therefore be described as a **branchy hoisted body with bulk tails**. It does not isolate branch
predictability from the simultaneous load-hoisting change.

### Verification and artifacts

All four test suites, `validate`, `validate64`, `difftest`, `difftest64`, `check-32`, `check-docs`, and
`check-package` pass on both hosts. `check-package` remains at 33 files.

- M4 attribution: `misc/array-attribution-20260828-102552-summary.txt`
- Zen 4 attribution: `misc/array-attribution-20260828-110854-summary.txt`
- M4 real data: `misc/realdata-bench-20260828-104337-summary.txt`
- Zen 4 real data: `misc/realdata-bench-20260828-110925-summary.txt`
- M4 clean-HEAD board: `misc/parity-20260828-143059-summary.txt`
- M4 candidate board: `misc/parity-20260828-145453-summary.txt`
- Zen 4 candidate board: `misc/parity-20260828-111004-summary.txt`
- Zen 4 clean-HEAD board: `misc/parity-20260828-134456-summary.txt`
- Production disassembly: `misc/array-attribution-20260828-m4-production-disassembly.txt` and
  `misc/array-attribution-20260828-zen4-production-disassembly.txt`

## Verification record — implemented, reviewed, ACCEPTED

Checked in the tree rather than from the report: both loops renamed and carrying the C3 body,
`assertDisjointOutput` at the loop head, comments rewritten off the branchless claim, `unionInPlace`
untouched with its deliberate-and-unmeasured note, the follow-up pointer added to
`done/optimization-branchless-merge.md`, 24 tuples with `ProcessCountMismatch` firing live, and both
inlined bodies retaining their data-dependent branches so §3's first row governs the wording.

**The pre-registered prediction that mattered came in.** §5.2 said M4 ANDNOT should reach or pass parity
because its scalar term (0.322 ms) exceeded the entire matched delta (0.265 ms). It landed at **0.806x,
rawr faster**, from 2.005x. That is `51-00`'s apportionment predicting an end-to-end outcome before the
change existed — the strongest evidence the model is right, and worth more than the four same-binary
reductions that came in at 98%, 94%, 89% and 103% of prediction.

**The M4 board scare was a measurement artifact, and it arrived through the one door this spec left
open.** §5.2 banned cross-run subtraction for the improvement figure and for the real-data rows, and said
nothing about the board comparison. The first M4 board result came from a prior-day baseline and reported
sparse-AND arena at +6.2% with separated ranges, which looks exactly like a real regression. The paired
exact-`b3ab49f` run put it at 0.571 → 0.556 ms with overlapping ranges — a small improvement.

**Rule for future chunks: a before/after board claim requires a paired exact-HEAD run in the same
session.** Separated ranges are not evidence of a code effect when the two sides come from different
runs, and this campaign has now been bitten by that in the real-data rows (spec 39), the focused harness
(spec 35) and the board.

**The alias-guard substitution improved on what this spec asked for.** `std.debug.assert` lowers to an
unlabelled `unreachable`, so the child-process control would have accepted *any* crash as a pass — a
segfault would have satisfied the negative test. The explicit named panic is what makes it discriminating.

### Where the campaign stands after this chunk

Removing the scalar term leaves residuals that line up with `51-00`'s other terms in all four cells:

| host | operation | current delta | `51-00` non-scalar terms | remaining lever |
| --- | --- | ---: | ---: | --- |
| M4 | OR | 0.197 ms | 0.200 ms | **normalization** (+0.224) |
| M4 | ANDNOT | −0.041 ms | −0.057 ms | none; rawr ahead |
| Zen 4 | OR | 0.305 ms | 0.291 ms | **normalization** (+0.376) |
| Zen 4 | ANDNOT | 0.095 ms | 0.125 ms | **CRoaring AVX2** (+0.109) |

**This is a consistency observation across runs, not a measurement.** The current deltas and `51-00`'s
terms come from different runs, so no difference is computed and nothing here is a ratio claim. Read only
as: the post-change positions are where the model said they would be.

Two consequences for what comes next. **Normalization is confirmed as the OR lever** on both hosts, which
is the chunk §9 already reserved. And **Zen 4 ANDNOT's residual is essentially the AVX2 term**, so the
only lever left there is per-architecture SIMD — spec 15's territory, and the first point in this campaign
where that argument actually applies.

## Estimate

**M** — the code change is small and already validated. The board run, the real-data re-run, and the
post-inlining disassembly are the work.
