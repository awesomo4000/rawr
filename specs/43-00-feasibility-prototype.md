<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 43-00: Feasibility prototype — is the ordering gain bigger than its cost?

Toplevel: [43-address-ordered-lazy-construction.md](43-address-ordered-lazy-construction.md).

**No production code changes.** This chunk answers one question before anything is built: does
address-ordering recover more than the machinery costs? A NO here ends spec 43 cheaply.

## 1. What to build

Extend `src/bench_smp_layout.zig` — the existing zero-rawr-code reproducer from spec 37.

**This is a feasibility prototype, not end-to-end.** It does not model unmatched clones, accumulation, or
result assembly. Do not describe it as end-to-end, and do not treat its absolute numbers as predictions
of the canonical row — only the **relative** question is in scope.

### 1.1 Fix the probe's existing misrepresentations first

Both **distort** the result; only the first is known to flatter it:

- **`bench_smp_layout.zig:167`** — the `sort_zero` cell **allocates before the timed region**. Allocation
  is part of the candidate and must be inside. **This flatters the candidate** by hiding real cost.
- **`bench_smp_layout.zig:233`** — it sorts **slices** with **stable `std.mem.sort`**. The candidate uses
  `sortUnstable` (pdq) over a 16-byte struct with an inline key. Stable block sort over fat elements
  most likely **penalizes** the candidate rather than flattering it — either way the measured quantity is
  not the one being proposed. *(An earlier draft claimed both flattered it, which was wrong.)*

### 1.2 Time the structural equivalent of the production representation

**Preserve the probe's zero-rawr-code property.** `bench_smp_layout.zig` imports only `std` and
`builtin` and models the container header with a local `Header` (`:15`) carrying a comptime 16-byte
assertion (`:20`). That independence is why spec 37's result was credible — it reproduced the pathology
with no rawr or CRoaring code at all. Do not import `BitsetContainer` here.

```zig
const Pending = struct {
    payload_addr: usize,   // sort key — no dereference in the comparator
    header: *Header,       // local 16-byte stand-in, existing size assertion retained
};
```

*(An earlier draft wrote `*BitsetContainer`, which would have broken the probe's defining property. The
production type is structurally identical for this purpose: pointer + `i32`, 16 bytes.)*

One scratch allocation of `[]Pending`; `sortUnstable` on `payload_addr`; comparator touches nothing else.

**Full candidate cost inside the timed region:**

- eligible-count pre-pass (model its `O(a.size + b.size)` walk cost — see §1.2.1),
- scratch allocation,
- header **and** payload allocation,
- the sort,
- the zeroing,
- **scratch release**.

**Outside the timed region:** teardown of retained headers and payloads. The canonical construction row
times `lazyOr(...)` alone and calls `result.deinit()` after stopping the clock
(`bench_croaring.zig:507-512`); a prototype that folds result teardown inward measures a different
quantity than the row it is trying to predict.

### 1.2.1 The modelled pre-pass must resist the optimizer

The probe has **no real merge inputs**, so a synthetic pre-pass over constant data is exactly the kind of
loop the compiler folds away — and a pre-pass that costs nothing in the prototype but costs real time in
production would flatter the candidate into a false GO.

Require:

- **Runtime-populated** key and container-type arrays — not comptime-known, not derivable by constant
  propagation.
- **Validate the resulting eligible count** against an independently computed expectation, so the loop
  has an observable result.
- **`std.mem.doNotOptimizeAway`** (or an equivalent data dependency) on that result.

Record the measured pre-pass cost separately, as with the sort.

### 1.3 Cells

Mirror the three arms so the prototype answers the same attribution question:

| Cell | Models |
| --- | --- |
| `interleaved` | today's path: allocate and zero in allocation order |
| `batched_unsorted` | batch-allocate, then zero — **no sort** |
| `batched_sorted` | batch-allocate, sort by `payload_addr`, then zero |

`batched_sorted` vs `batched_unsorted` is the ordering effect. `batched_unsorted` vs `interleaved` is the
batching/scheduling effect — measure it, do not assume it away.

Run on **both hosts**, SMP and libc. libc is the negative control for the whole premise: spec 37 found it
**order-insensitive** (0.011–0.073 ms sorted vs unsorted), so a libc arm that shows a large ordering gain
means the harness is measuring something other than address order.

## 2. Recording

Record per cell: median plus full range, buffer count, and the **sort cost as its own line item**. No
sort-cost estimate is carried from any earlier spec — spec 38's ~86.98 ns/op was for `[]u8` slices and
does not transfer to this element. **This chunk establishes the number.**

## Acceptance

- Probe corrections at `:167` and `:233` made; timed region contains allocation, and the sort is
  `sortUnstable` over `[]Pending`.
- **Zero-rawr-code property preserved** — no rawr import; `Pending.header` is `*Header`, the local
  16-byte stand-in, with the existing comptime assertion retained.
- **Modelled pre-pass resists the optimizer** (§1.2.1): runtime-populated inputs, validated eligible
  count, `doNotOptimizeAway`. Pre-pass cost recorded separately.
- All three cells implemented, run on both hosts, SMP and libc, with medians and full ranges recorded.
- Sort cost recorded as a separate line item.
- **Timing boundary matches the canonical row:** scratch release inside, result teardown outside.
- **Verdict recorded explicitly, against a stated rule. This is a SCREENING threshold, not the gate.**

  *(An earlier draft called ≥50% recovery "clearing the gap with margin" and then justified it by saying
  omitted production work needs extra headroom — which argues for a threshold **above** the full gap, not
  below it. The two halves contradicted each other. Resolved below.)*

  **GO means: enough evidence to justify spending `43-01`, not evidence that Gate 1 will be met.**

  *(Two earlier drafts were wrong here. The first called ≥50% "clearing the gap with margin" while
  justifying it by arguing omitted work needs extra headroom. The second called the prototype gain an
  **upper bound** while still permitting GO at ~0.85 ms against a ~1.7 ms gap — if it really were an
  upper bound, that GO would authorize work that provably cannot close the row. The "upper bound" claim
  is withdrawn; it was false.)*

  **The prototype is neither an upper nor a lower bound — production omits interactions that push both
  ways:**

  - **Dilution (production recovers less).** Unmatched-key clones are allocated interleaved with the
    pending batch (`bitmap.zig:2331`), fragmenting the address space that the sort is trying to linearize.
  - **Amplification (production recovers more).** The prototype models **zeroing only**. In production the
    same buffers are then **accumulated into** — `lazyAccumulateIntoBitset` for both operands — so
    ascending order benefits that traffic too, and it is not modelled here at all.

  **Threshold:** net gain after full candidate cost (pre-pass + scratch + allocation + sort + zeroing +
  scratch release) recovers **≥50% of the ~1.7 ms M4 gap** (~0.85 ms), with **non-overlapping ranges**
  between `batched_sorted` and `batched_unsorted` across repeated runs. Overlap → **rerun**; still
  overlapping → **inconclusive, treated as NO-GO**.

  50% is a **screen against wasted effort**, not a prediction: below it the ordering effect is too weak
  for amplification to plausibly rescue, so `43-01` would likely be spent to fail. At or above it the
  question is genuinely open, and **only Gate 1 can answer it.**
- **libc shows no large ordering effect.** If it does, stop and diagnose the harness rather than
  proceeding — the premise is that ordering matters for SMP specifically.
- **GO/NO-GO stated.** NO-GO ends spec 43 here, with no production code written.

## Estimate

**S** — the probe exists; this extends and corrects it.
