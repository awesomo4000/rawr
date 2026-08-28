<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 51-01: Scalar merge candidates — measured in the Layer A rig

Toplevel: [51-array-union-difference-kernels.md](51-array-union-difference-kernels.md).
Gated on: [51-00](51-00-attribution.md) complete and accepted.

Stage 1 found the scalar merge is the **only term present and material in all four cells**
(0.226–0.322 ms each; 1.111 ms summed) and the sole dominant term for ANDNOT on both hosts. This chunk
measures candidate loops against that term. **It changes no production code** — candidates are additional
Layer A arms. Productizing a winner is `51-02` and does not exist yet.

## 1. What the source comparison shows

Two structural differences separate rawr's loops from CRoaring's scalar `union_uint16` and
`difference_uint16`. Both are portable and neither is SIMD.

**Tail drain.** CRoaring copies the remaining tail with one `memmove`. rawr drains element-at-a-time in a
`while` loop.

**Loop body.** rawr is deliberately branchless — `output[k] = if (a_val <= b_val) a_val else b_val` with
`i += @intFromBool(...)` advances, per `done/optimization-branchless-merge.md`. CRoaring is a branchy
three-way `if`/`else if`/`else` that hoists `val_1`/`val_2` into locals and reloads only the side that
advanced: one load per iteration against rawr's two.

**Both mechanisms are hypotheses, not established facts.** An earlier draft argued them from Stage 1
numbers that do not support them, and the overreach is worth naming so it is not repeated:

- Stage 1's `[min=2, p50=128, p90=683, p99=1320, max=2603]` is a distribution over **inputs**, not over
  **pairs**. Lopsidedness is a per-pair size ratio and was never measured. Tail share does not follow from
  it.
- "864 of 930 results convert to runs" describes the **result**, not the **decision sequence** that
  produced it. Run-convertible output does not establish streaky merge decisions. And ANDNOT performs no
  run conversion at all, so the count says nothing whatever about the operation where the scalar term is
  dominant on both hosts.

The underlying reasoning still stands as a hypothesis: branchless wins when branches are unpredictable,
and the branchless decision was measured on the synthetic board rather than on real data. That is not a
claim it was wrong — it is a claim its measurement did not cover this regime. §1.1 supplies the evidence
that would actually bear on it.

## 1.1 Untimed diagnostics — measure the premises

Computed over the same matched pairs, **per operation and per dataset**, outside any timed region:

- **Tail-length distribution and tail share** — what fraction of each output is produced by the drain
  rather than the merge loop. This is what C1 can address, and it is the number the size quantiles were
  wrongly used to stand in for.
- **Left / right / equal decision counts.**
- **Streak-length distribution** (or decision-transition counts) — consecutive decisions taking the same
  side. This is what bears on branch predictability, and it is measurable rather than inferred from run
  conversion.

These make the C1 and C2 outcomes interpretable instead of merely ranked. A C1 win with negligible tail
share, or a C2 win with no streakiness, would mean the mechanism story is wrong even though the timing
moved — and that is worth knowing before `51-02` justifies a change with it.

**One thing the diagnostics settle immediately.** ANDNOT and OR may have entirely different tail and
streak profiles on the same corpus. If they do, "the scalar merge" is two problems, not one.

**One thing the source does not settle.** rawr's difference loop already contains a conditional store
(`if (a_val < b_val) { output[k] = a_val; k += 1; }`), and the union loop's comment asserts *"on aarch64,
LLVM emits csel"* — a claim made for one architecture and never checked on the other. **What these loops
actually compile to is an open question, not a premise.** §5 makes the disassembly a deliverable.

## 2. Candidates

Three arms, added alongside the existing `a1_rawr_scalar` and `a2_croaring_scalar`. Each applies to both
union and difference.

| arm | change |
| --- | --- |
| **C1 — bulk-copy tails** | Replace the element-wise drain loops with `@memcpy`. Loop body untouched. |
| **C2 — hoisted branchy body** | CRoaring's three-way body with `val_1`/`val_2` in locals, reloading only the advanced side. **Tails stay element-wise.** |
| **C3 — both** | C1 and C2 together. |

**C1 uses `@memcpy`, and its precondition must be stated rather than assumed.** `@memcpy` requires
non-overlapping regions. That holds here because Layer A guarantees output buffers distinct from both
inputs (§6), a property `51-00` verified rather than assumed. State the dependency at the call site — if
`51-02` ever moves this loop somewhere the output can alias an input, `@memcpy` becomes undefined
behaviour and nothing in the loop itself will say so.

**C2 must not accidentally become C3 for ANDNOT.** CRoaring's `difference_uint16` performs a bulk
`memmove` of the remaining left tail **from inside the branchy loop**, at both places where `a2` is
exhausted. Porting the body faithfully would therefore carry C1 along with it and destroy the isolation
the split exists for. **Route those exits into rawr's existing element-wise drain instead.** Only C3 is
permitted to bulk-copy. Union is not exposed to this — `union_uint16` bulk-copies only after the loop —
but implement both operations under the same rule so the arm means one thing.

**C1 and C2 must be measured separately even though C3 is the interesting one.** Spec 30 established this
the hard way: separating fusion from pre-sizing is what stopped a regression from masking a win. Here the
split also decides how risky productizing is. If C1 alone carries the gap, `51-02` is a portable few-line
change with no branch-prediction exposure and no reversal of a prior decision. If C2 is required, `51-02`
is reversing a measured decision and the canonical board becomes the deciding gate, not this corpus.

**C3 is the completeness check.** If C3 lands at A2, these two differences explain the whole scalar term
and the diagnosis is closed. If C3 still trails A2, something else is also producing the gap.

**A residual does not block `51-02`** — an earlier draft said it did, which contradicted the ≥50% gate two
sections later. A candidate that recovers half the term is worth shipping whether or not we understand the
rest, and this campaign has shipped against a documented residual before (`46-01`). **What a residual
blocks is the explanation, not the change.** If C3 leaves a residual, `51-02` may adopt the winning
candidate but may **not** justify it with §1's mechanism story as though it were complete, and the
residual carries forward as an open question with its measured size attached.

**Out of scope:** no SIMD, no per-architecture code, no production change. The Zen 4 union result from
Stage 1 — CRoaring's own vectorized union losing to its own scalar arm on this corpus — is the reason
vectorization is not a candidate here.

## 3. Corpora — one target, two controls

The rig currently hardcodes the dataset (`bench_array_attribution.zig:18`). Make it selectable; the loader
already supports all three.

- **`wikileaks-noquotes` is the target.** It is the row that motivated spec 51 and the only one the GO
  gate reads.
- **`uscensus2000` and `census1881` are regression controls.** A candidate that wins on the target and
  loses on the others is a corpus-specific result, and that is a finding, not a failure — but it must be
  visible **before** anyone productizes rather than discovered on the board afterwards.

**Report the matched array-array pair count per dataset.** A control with few matched pairs constrains
little, and saying so is part of the result.

The §1.1 diagnostics run on all three corpora for the same reason. If the three differ sharply in tail
share or streakiness, that is the direct evidence for or against corpus-specificity, and it is available
without waiting for a board run.

## 4. Correctness — the corpus cannot cover this alone

**Every candidate must produce byte-identical output and an identical return count to `a1_rawr_scalar` on
every pair of every dataset.** The rig's existing digest machinery covers this; reuse it rather than
adding a parallel check.

**That is necessary and not sufficient.** Corpus inputs have `min=2`, so **no pair is ever empty** and the
corpus can never reach an empty-input path. C1 and C3 introduce exactly such a path. Add unit tests for
the candidate loops covering: empty `a`, empty `b`, both empty, single element each, disjoint inputs, fully
identical inputs, `a` exhausted first, `b` exhausted first. Without these, a `memcpy` of length zero or a
misplaced early return ships untested behind a green corpus run.

The difference loop's `<` versus `<=` distinction is the specific thing most likely to be got subtly wrong
while still passing on well-behaved data. The disjoint and fully-identical cases are there for it.

## 5. Disassembly is a deliverable, not a footnote

For `a1_rawr_scalar` and each candidate, on **both hosts**, record the inner-loop disassembly and state
whether the advance compiled branchless (`csel`/`cset` on aarch64, `cmov` on x86_64) or to a branch.

**Pin how the disassembly reaches the code that was actually timed.** The loops are `inline`, so a
standalone symbol for any of them may simply not exist in the binary, and reading a symbol that was not
the one measured proves nothing.

**`@call(.never_inline, ...)` is not sufficient on its own.** The rig dispatches every rawr arm through a
single shared `runRawrMatched`, so a never-inline call there yields **one** symbol covering all arms, not
one per arm. `51-01` must add **explicit per-arm wrappers or comptime-specialized non-inlined batch
functions**, so each arm has its own inspectable symbol, and inspect those exact implementations.

**Distinguish data-dependent branches from loop-control branches.** Every one of these loops has a
back-edge and a bounds test; those are not what C2 is about. The determination concerns the branch on
`a_val` versus `b_val`.

**LLVM may if-convert C2 back to branchless, and that does not make it A1.** An earlier draft said it did;
that was too strong. C2 changes two things at once — the branch structure *and* the hoisted `val_1`/`val_2`
state with its single reload — and if-conversion removes only the first. Read the result as:

- **No data-dependent branch survives** → the branch-predictability hypothesis **was not tested** by this
  run. Say so; do not report C2's timing as evidence for or against it.
- **C2 may still be testing hoisting and load behaviour**, which is a real and separate mechanism.
- **Treat a C2-versus-A1 difference as layout noise only if the disassembly shows equivalent inner-loop
  instructions** — not merely because both are branchless. Spec 28's layout finding applies to
  instruction-identical code, which is a stronger condition than same branch structure.

**Check this before interpreting C2's result, not after.**

This section exists because §1 is a source reading, and the campaign has been wrong about source readings
before — Stage 1's original hypothesis died because a symbol was found without checking reachability.
**If rawr's union loop is not actually branchless on x86_64, that is a finding in its own right** and it
changes what C2 means.

**Accepted limit.** These symbols are non-inlined for inspectability, while production inlines the loop
into its call site. The disassembly therefore describes the timed Layer A implementation and not the
shipped one. That is the same scope limit `51-00` already operates under, and it is a reason `51-02` must
re-check codegen after inlining rather than carrying this record forward as though it settled the
question.

## 6. Protocol

Identical to `51-00`: Layer A conditions (preallocated, non-aliased output buffers, zero timed
allocations), fresh process per cell, warmup then timed iterations, **≥5 process medians with full
ranges**, both hosts, `ReleaseFast`, native CPU.

**All arms in one run per host** means **one build and one controller campaign, with a fresh worker
process per tuple** — not one process executing several arms, which is the allocator-conditioning confound
spec 50 controlled for. Spec 39 established that ratios across runs are invalid because the reference
moves; a candidate compared against an A1 from a different campaign is not a measurement.

Cell count: 2 operations × 5 arms × 3 datasets = **30 tuples per host**.

## 7. Deciding rule — pre-registered

Per **operation**, per **host**. Not summed across operations and not averaged across hosts: spec 35
showed a combined-only gate authorizing a large migration while the binding constraint regressed.

**Recovery is a fraction with an interval, computed exactly as `51-00` §3.1 computed explained share.**
"Closes ≥50% of the gap" is not reproducible without this.

From aggregate medians and full ranges:

```
recovery      = (A1 - Cx) / (A1 - A2)
numerator N   = [A1_min - Cx_max,  A1_max - Cx_min]
denominator D = [A1_min - A2_max,  A1_max - A2_min]
interval S    = [N_min / D_max,  N_max / D_min]
```

**Sign cases are resolved before dividing**, and a definite failure is not the same as an unresolved one.
An earlier draft collapsed both into "inconclusive", which would have let a candidate that measurably does
nothing buy itself a rerun.

Denominator first, since without a gap there is nothing to recover:

| case | verdict |
| --- | --- |
| `D_max <= 0` | **No gap to recover.** A1 is not slower than A2 in this run. Report as such; do not divide, and do not report a ratio. |
| `D_min <= 0 < D_max` | **Inconclusive** — the gap itself is not established. Rerun. |
| `D_min > 0` | Proceed to the numerator. |

Then the numerator:

| case | verdict |
| --- | --- |
| `N_max <= 0` | **NO-GO.** The candidate provides no improvement. This is a resolved answer, not a rerun. |
| `N_min < 0 < N_max` | **Inconclusive** — ranges overlap. Rerun. |
| `N_min >= 0` | Divide and apply the threshold below. |

- **GO** if `S_min >= 0.50` on the target corpus on **both hosts**.
- **NO-GO** if `S_max < 0.50`.
- Otherwise **inconclusive**: rerun once, as `51-00` did for M4 ANDNOT. Still inconclusive is reported as
  such, never rounded toward the nearer verdict.

**Control-corpus regression is a non-overlapping slowdown** — `Cx_min > A1_max` — not "beyond noise",
which is not a criterion anyone can apply. Overlapping ranges on a control are **not** a regression.

### C3 completeness — same interval, threshold at 1.0

**The C3-versus-A2 comparison is reported as its own line with its own interval.** It uses the same
recovery fraction, since `recovery = 1.0` is exactly the statement that C3 reached A2:

- `S_min >= 1.0` → **explains the term.**
- `S_max < 1.0` → **residual remains**, reported with its measured size.
- Otherwise → **inconclusive**, same rerun policy.

A residual scopes what `51-02` may claim; per §2 it does not block `51-02`.

## Acceptance

- C1, C2, C3 implemented as Layer A arms per §2; `a1`/`a2` unchanged.
- **C2 drains element-wise on both operations**, with the ANDNOT mid-loop bulk-copy exits routed to the
  element-wise drain. Only C3 bulk-copies. Verified by reading C2, not inferred from its timing.
- **C1's non-aliasing precondition stated at the call site.**
- **§1.1 diagnostics reported** — tail share, decision counts, streak lengths — per operation and per
  dataset, untimed.
- Dataset selectable; all three corpora run; **matched pair count reported per dataset**.
- Byte-identical output and return count versus `a1` on every pair of every dataset.
- **Unit tests per §4, including the empty-input cases the corpus cannot reach.**
- Disassembly recorded for `a1` and all three candidates on **both hosts** via **per-arm wrappers or
  comptime-specialized non-inlined batch functions** — one symbol per arm, not the shared
  `runRawrMatched` — distinguishing data-dependent from loop-control branches, and **stating whether C2's
  data-dependent branch survived**, including whether the existing aarch64 comment holds on x86_64. If it
  did not survive, say the predictability hypothesis was untested rather than reporting C2's timing
  against it.
- §6 protocol met: one build, one controller campaign per host, fresh worker process per tuple, 30 tuples.
- §7 rule applied exactly — **both sign tables**, with `N_max <= 0` and `D_max <= 0` reported as resolved
  verdicts rather than reruns; inconclusive rerun policy; control regression judged by non-overlap.
- **C3-versus-A2 stated explicitly** with its interval and a verdict at the **1.0** threshold.
- **No production change. No SIMD. No per-architecture code. `51-02` remains unwritten.**
- Existing suites and checks green; no board row moves.

## Outcome (08/27/2026)

**GO for the branchy, hoisted merge body on both operations and both hosts.** C2 and C3 both clear the
50% recovery gate. C3 is the strongest measured arm and is the candidate to carry into `51-02`; this
chunk makes no production change.

| host | operation | C2 recovery | C3 recovery | C3 completeness |
| --- | --- | ---: | ---: | ---: |
| M4 | OR | 0.891 [0.826, 0.952] | 0.895 [0.804, 0.960] | **residual remains** |
| M4 | ANDNOT | 0.993 [0.897, 1.077] | 0.993 [0.926, 1.080] | inconclusive |
| Zen 4 | OR | 0.840 [0.799, 0.954] | 0.917 [0.873, 1.041] | inconclusive |
| Zen 4 | ANDNOT | 0.931 [0.654, 1.114] | 1.052 [0.942, 1.253] | inconclusive |

The Zen 4 rows use the single target-only rerun required by §7. Its aggregate medians and ranges were:

| operation | A1 | A2 | C1 | C2 | C3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OR | 0.335 [0.334, 0.336] | 0.119 [0.118, 0.138] | 0.331 [0.329, 0.332] | 0.154 [0.150, 0.159] | 0.137 [0.133, 0.143] |
| ANDNOT | 0.424 [0.423, 0.431] | 0.132 [0.125, 0.170] | 0.412 [0.410, 0.414] | 0.152 [0.149, 0.223] | 0.117 [0.114, 0.135] |

C1 does not clear the gate. On Zen 4 its rerun recovery intervals were [0.006, 0.037] for OR and
[0.030, 0.080] for ANDNOT, both definite NO-GO results. On M4, the permitted C1 rerun made OR a
definite NO-GO and left ANDNOT overlapping and therefore inconclusive. Since C1 fails the two-host rule,
the unresolved M4 ANDNOT result does not keep it alive.

**The disassembly changes the interpretation of C1.** On both hosts LLVM already recognizes A1's
source-level element-wise drains and emits bulk copies. It does the same for C2. C1 therefore did not
introduce the machine-level tail-copy distinction the source suggested, which explains its failure to
move the target. The explicit C1 form also made the M4 batch symbol larger (980 bytes versus A1's 896).

The branch/hoisting experiment did execute as designed:

- M4 A1 uses `csel`/`cinc` for the merge advances; its existing ANDNOT conditional store still has a
  data-dependent branch.
- Zen 4 A1 uses `cmov`/condition-code materialization for the advances.
- C2 and C3 retain data-dependent value-comparison branches on both hosts, together with the hoisted
  values and one-sided reloads. The branch-predictability hypothesis was therefore tested, not
  if-converted away.
- C3 produces the smallest inspected batch symbol on both hosts: 792 bytes on M4 and 928 bytes on Zen 4,
  versus 872 and 1,024 bytes for C2.

The target diagnostics support, but do not by themselves prove, why that body works here. OR drains
17.50% of its output and ANDNOT drains 15.90%, while merge-decision streaks have p50 8, p90 41, p99 247,
and max 1,263. The timing establishes the branchy/hoisted body as the lever; it does not separate branch
predictability from load hoisting. C3's additional Zen 4 improvement over C2 is an interaction, not an
independent C1 win.

No C3 control row regressed under the non-overlap rule. `census1881` supplied 118 matched pairs per
operation. `uscensus2000` supplied only 21, so its result is a weak control; the strict rule flagged only
Zen 4 C1 ANDNOT there, not C3. All candidate outputs and counts matched A1 across all three corpora, and
the dedicated empty/singleton/disjoint/identical/exhaustion tests passed on both hosts.

The cross-host audit matched all semantic and diagnostic fields across 30 tuples and 150 fresh processes
per host. Protocol mutation checks rejected a missing row, a changed digest, changed diagnostic metadata,
and a nonzero Layer A allocation count. Existing test, documentation, package, loader, and normal build
checks passed; `check-package` remains at 33 allowlisted files. No board row moved.

Artifacts:

- M4: `misc/array-attribution-20260827-230603-summary.txt`, corresponding process TSV, and
  `misc/array-attribution-20260827-m4-disassembly.txt`
- Zen 4: `misc/array-attribution-20260827-234324-summary.txt`, corresponding process TSV,
  `misc/array-attribution-20260827-zen4-target-rerun.tsv`, and
  `misc/array-attribution-20260827-zen4-disassembly.txt`

## Verification record — implemented, reviewed, ACCEPTED

**Recomputed from the published medians and ranges, not taken on report.** All four C3 recovery intervals
and the Zen 4 C1/C2 intervals reproduce to within display rounding. The §7 rules were applied as written:
M4 OR at `S_max = 0.960 < 1.0` is *residual remains*; the other three straddle 1.0 and are correctly
*inconclusive* rather than rounded to the nearer verdict; C1 at `S_max < 0.50` is NO-GO as a **resolved**
answer, which is exactly the case the revised sign table was added to handle.

**C1's death is the most valuable result here, and it is the §5 disassembly requirement paying for
itself.** LLVM already recognizes A1's source-level element-wise drains and emits bulk copies, so C1
introduced no machine-level difference at all. Without the disassembly, C1's 2–4% recovery reads as *tails
do not matter much*; the truth is *the tail difference never existed in the binary*. **Same numbers,
different conclusion.** The general lesson is reusable: a source-level difference is not a difference until
codegen is checked.

**The C2-versus-C3 choice is better founded than "strongest measured arm" — but by a different
comparison.** The recovery intervals for C2 and C3 overlap in every cell, which looks like an unresolved
choice. That is the wrong test: both fractions share A2's range in the denominator, so overlap there says
little. The direct timing ranges separate cleanly on Zen 4 under the same non-overlap rule §7 uses for
controls:

| Zen 4 | C2 | C3 | separated? |
| --- | --- | --- | --- |
| OR | [0.150, 0.159] | [0.133, 0.143] | yes |
| ANDNOT | [0.149, 0.223] | [0.114, 0.135] | yes |

**What is not shown is the same comparison on M4.** The M4 recovery points are 0.895 versus 0.891 and
0.993 versus 0.993 — indistinguishable, but that is inferred from a ratio rather than measured directly.
**`51-02` must state the M4 C2-versus-C3 timing ranges before committing to C3**, because C3 carries the
`@memcpy` non-aliasing precondition into production and C2 does not. Code size favours C3 on both hosts
(792 and 928 bytes against C2's 872 and 1,024, and against A1's 896), so this is probably a formality —
but "probably" is the thing these rules exist to replace.

**Corpus-specificity evidence is thinner than a three-corpus design suggests.** `uscensus2000` supplied
21 matched pairs; only `census1881`'s 118 constrain anything. Reporting that plainly is right, and it
means **the canonical board in `51-02` is the real corpus-specificity gate**, not these controls.

**The diagnostics earned their place.** Tail share of 17.50% (OR) and 15.90% (ANDNOT) capped C1's ceiling
independently of the codegen finding, and streak lengths of p50 8, p90 41, p99 247, max 1,263 support the
predictability hypothesis. The record correctly states that the timing does not separate predictability
from load hoisting — C2 changes both at once, and that limit should survive into `51-02`'s wording rather
than being quietly dropped.

**One record-keeping obligation for `51-02`.** Adopting C2 or C3 reverses `done/optimization-branchless-
merge.md` for these two loops on real data. Add a pointer there when it lands, so the archive does not
hold two contradictory conclusions with no link between them.

## Estimate

**S/M** — three loop variants and a dataset parameter in a rig that already has arm dispatch, digests,
validation, and a cross-host audit. The unit tests and the disassembly pass are most of the work.
