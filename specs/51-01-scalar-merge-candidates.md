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
`while` loop. Stage 1 measured input sizes at `[min=2, p50=128, p90=683, p99=1320, max=2603]`, so pairs are
frequently lopsided and the tail can be most of the output.

**Loop body.** rawr is deliberately branchless — `output[k] = if (a_val <= b_val) a_val else b_val` with
`i += @intFromBool(...)` advances, per `done/optimization-branchless-merge.md`. CRoaring is a branchy
three-way `if`/`else if`/`else` that hoists `val_1`/`val_2` into locals and reloads only the side that
advanced: one load per iteration against rawr's two.

**Why the distinction matters for this corpus specifically.** Branchless wins when branches are
unpredictable. 864 of 930 union results convert to runs, so this data is unusually runny — long stretches
where the same side advances repeatedly, which is exactly where a predictor is cheap and a branchless loop
pays for work it did not need. **The branchless decision was measured on the synthetic board, not on runny
real data.** That is not a claim it was wrong; it is a claim its measurement did not cover this regime.

**One thing the source does not settle.** rawr's difference loop already contains a conditional store
(`if (a_val < b_val) { output[k] = a_val; k += 1; }`), and the union loop's comment asserts *"on aarch64,
LLVM emits csel"* — a claim made for one architecture and never checked on the other. **What these loops
actually compile to is an open question, not a premise.** §5 makes the disassembly a deliverable.

## 2. Candidates

Three arms, added alongside the existing `a1_rawr_scalar` and `a2_croaring_scalar`. Each applies to both
union and difference.

| arm | change |
| --- | --- |
| **C1 — memmove tails** | Replace the element-wise drain loops with `@memcpy`. Loop body untouched. |
| **C2 — hoisted branchy body** | CRoaring's three-way body with `val_1`/`val_2` in locals, reloading only the advanced side. Tails untouched. |
| **C3 — both** | C1 and C2 together. |

**C1 and C2 must be measured separately even though C3 is the interesting one.** Spec 30 established this
the hard way: separating fusion from pre-sizing is what stopped a regression from masking a win. Here the
split also decides how risky productizing is. If C1 alone carries the gap, `51-02` is a portable few-line
change with no branch-prediction exposure and no reversal of a prior decision. If C2 is required, `51-02`
is reversing a measured decision and the canonical board becomes the deciding gate, not this corpus.

**C3 is the completeness check, and it is the most important arm.** If C3 lands at A2, these two
differences explain the whole scalar term and the diagnosis is closed. **If C3 still trails A2, something
else is producing the gap and `51-02` must not be written on this chunk's basis.** Report that comparison
explicitly rather than leaving it to be inferred from the table.

**Out of scope:** no SIMD, no per-architecture code, no production change. The Zen 4 union result from
Stage 1 — CRoaring's own vectorized union losing to its own scalar arm on this corpus — is the reason
vectorization is not a candidate here.

## 3. Corpora — one target, two controls

The rig currently hardcodes the dataset (`bench_array_attribution.zig:18`). Make it selectable; the loader
already supports all three.

- **`wikileaks-noquotes` is the target.** It is the row that motivated spec 51 and the only one the GO
  gate reads.
- **`uscensus2000` and `census1881` are regression controls.** A candidate that wins on runny data and
  loses on the others is a corpus-specific result, and that is a finding, not a failure — but it must be
  visible **before** anyone productizes rather than discovered on the board afterwards.

**Report the matched array-array pair count per dataset.** A control with few matched pairs constrains
little, and saying so is part of the result.

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

This exists because §1's premise is a source reading and the campaign has been wrong about source readings
before — Stage 1's original hypothesis died because a symbol was found without checking reachability.
**If rawr's union loop is not actually branchless on x86_64, that is a finding in its own right** and it
changes what C2 means.

## 6. Protocol

Identical to `51-00`: Layer A conditions (preallocated, non-aliased output buffers, zero timed
allocations), fresh process per cell, warmup then timed iterations, **≥5 process medians with full
ranges**, both hosts, `ReleaseFast`, native CPU.

**All arms in one run per host.** Spec 39 established that ratios across runs are invalid because the
reference moves; a candidate compared against an A1 from a different run is not a measurement.

Cell count: 2 operations × 5 arms × 3 datasets = **30 tuples per host**.

## 7. Deciding rule — pre-registered

Per **operation**, per **host**. Not summed across operations and not averaged across hosts: spec 35
showed a combined-only gate authorizing a large migration while the binding constraint regressed.

- **Improvement claim requires non-overlapping full ranges** — `A1_min > Cx_max`. Overlapping ranges are
  **inconclusive**, rerun once as `51-00` did for M4 ANDNOT; still overlapping is reported as **no
  measurable difference**, not as a small win.
- **GO for a candidate on an operation** requires it close **≥50% of the `A1 - A2` gap on the target
  corpus on both hosts**, with no regression beyond noise on either control corpus.
- **The C3-versus-A2 comparison is reported as its own line** with a verdict of *explains the term* or
  *residual remains*, and a residual blocks `51-02`.

## Acceptance

- C1, C2, C3 implemented as Layer A arms per §2; `a1`/`a2` unchanged.
- Dataset selectable; all three corpora run; **matched pair count reported per dataset**.
- Byte-identical output and return count versus `a1` on every pair of every dataset.
- **Unit tests per §4, including the empty-input cases the corpus cannot reach.**
- Disassembly recorded for `a1` and all three candidates on **both hosts**, with the branchless/branchy
  determination stated per §5 — including whether the existing aarch64 comment holds on x86_64.
- §6 protocol met, all arms in one run per host, 30 tuples per host.
- §7 rule applied exactly, including the non-overlap requirement and the inconclusive rerun.
- **C3-versus-A2 stated explicitly** with an *explains the term* / *residual remains* verdict.
- **No production change. No SIMD. No per-architecture code. `51-02` remains unwritten.**
- Existing suites and checks green; no board row moves.

## Estimate

**S/M** — three loop variants and a dataset parameter in a rig that already has arm dispatch, digests,
validation, and a cross-host audit. The unit tests and the disassembly pass are most of the work.
