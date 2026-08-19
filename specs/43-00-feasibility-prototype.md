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

Both would flatter the candidate:

- **`bench_smp_layout.zig:167`** — the `sort_zero` cell **allocates before the timed region**. Allocation
  is part of the candidate and must be inside.
- **`bench_smp_layout.zig:233`** — it sorts **slices** with **stable `std.mem.sort`**. The candidate uses
  `sortUnstable` (pdq) over a 16-byte struct with an inline key.

### 1.2 Time the exact production representation

```zig
const Pending = struct {
    payload_addr: usize,          // sort key — no dereference in the comparator
    header: *BitsetContainer,     // association preserved alongside the key
};
```

One scratch allocation of `[]Pending`; `sortUnstable` on `payload_addr`; comparator touches nothing else.

**Full candidate cost inside the timed region:**

- eligible-count pre-pass (model its `O(a.size + b.size)` walk cost),
- scratch allocation,
- header **and** payload allocation,
- the sort,
- the zeroing,
- **scratch release**.

**Outside the timed region:** teardown of retained headers and payloads. The canonical construction row
times `lazyOr(...)` alone and calls `result.deinit()` after stopping the clock
(`bench_croaring.zig:507-512`); a prototype that folds result teardown inward measures a different
quantity than the row it is trying to predict.

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
- All three cells implemented, run on both hosts, SMP and libc, with medians and full ranges recorded.
- Sort cost recorded as a separate line item.
- **Timing boundary matches the canonical row:** scratch release inside, result teardown outside.
- **Verdict recorded explicitly** — the net gain after full candidate cost, and whether it clears the
  ~1.7 ms gap **with margin**.
- **libc shows no large ordering effect.** If it does, stop and diagnose the harness rather than
  proceeding — the premise is that ordering matters for SMP specifically.
- **GO/NO-GO stated.** NO-GO ends spec 43 here, with no production code written.

## Estimate

**S** — the probe exists; this extends and corrects it.
