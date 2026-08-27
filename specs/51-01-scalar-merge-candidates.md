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
the one measured proves nothing. **Give each arm a distinct non-inlined batch symbol** — the rig already
dispatches through `@call(.never_inline, ...)` — and inspect those exact implementations.

**Distinguish data-dependent branches from loop-control branches.** Every one of these loops has a
back-edge and a bounds test; those are not what C2 is about. The determination concerns the branch on
`a_val` versus `b_val`.

**LLVM may if-convert C2 back to branchless.** If it does, C2 and A1 differ in source and not in machine
code, the arm measures nothing it was meant to, and any timing difference between them is layout noise of
the kind spec 28 documented. **Check this before interpreting C2's result, not after.**

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

**Sign checks before dividing**, as in `51-00`: require `D_min > 0` and `N_min >= 0`. `D_min <= 0` means
the A1–A2 gap is not established in this run and the cell reports **no gap to recover** rather than a
ratio. `N_min < 0` means the candidate's range overlaps A1's and the cell is **inconclusive** — this
subsumes the non-overlap requirement an earlier draft stated separately.

- **GO** if `S_min >= 0.50` on the target corpus on **both hosts**.
- **NO-GO** if `S_max < 0.50`.
- Otherwise **inconclusive**: rerun once, as `51-00` did for M4 ANDNOT. Still inconclusive is reported as
  such, never rounded toward the nearer verdict.

**Control-corpus regression is a non-overlapping slowdown** — `Cx_min > A1_max` — not "beyond noise",
which is not a criterion anyone can apply. Overlapping ranges on a control are **not** a regression.

**The C3-versus-A2 comparison is reported as its own line**, with its own interval and a verdict of
*explains the term* or *residual remains*. A residual scopes what `51-02` may claim; per §2 it does not
block `51-02`.

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
- Disassembly recorded for `a1` and all three candidates on **both hosts** via **distinct non-inlined
  batch symbols**, distinguishing data-dependent from loop-control branches, and **stating whether C2
  survived if-conversion** — including whether the existing aarch64 comment holds on x86_64.
- §6 protocol met: one build, one controller campaign per host, fresh worker process per tuple, 30 tuples.
- §7 rule applied exactly — interval form, sign checks before dividing, inconclusive rerun, and control
  regression judged by non-overlap.
- **C3-versus-A2 stated explicitly** with its interval and an *explains the term* / *residual remains*
  verdict.
- **No production change. No SIMD. No per-architecture code. `51-02` remains unwritten.**
- Existing suites and checks green; no board row moves.

## Estimate

**S/M** — three loop variants and a dataset parameter in a rig that already has arm dispatch, digests,
validation, and a cross-host audit. The unit tests and the disassembly pass are most of the work.
