<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 44-00: Fused arm — implementation, ownership, correctness

Toplevel: [44-fused-address-ordered-construction.md](44-fused-address-ordered-construction.md).

Builds on branch `spec-43-lazy-construction-diagnostic`, commit **`37d0e8b`** (**push it first** — the
reference does not resolve while the branch is local only).

**No default change. No measurement verdict here** — that is `44-01`. This chunk ends when the fused arm
is correct, leak-free, and selectable.

## 1. The fused arm

New `ConstructionMode` variant: batched, **sorted**, **fused**. Per pending entry, **in address order**:

1. `@memset` the 8 KB payload,
2. `lazyAccumulateIntoBitset` for **both** source operands,
3. write the pointer into its pre-computed output slot.

Each payload is touched once while resident — no second pass over the population.

## 2. Metadata — types pinned

```zig
const Pending = struct {
    payload_addr: usize,          // 8 — sort key, no dereference in comparator
    header: *BitsetContainer,     // 8
    src_a: TaggedPtr,             // 8 — packed struct(usize)
    src_b: TaggedPtr,             // 8
    slot: u32,                    // 4 — NOT u16: up to 65,536 output slots
};                                // 36 → 40 bytes with alignment
```

`slot` **must not be `u16`** — the result can hold 65,536 containers, which does not fit.
`sortUnstable` (pdq) on `payload_addr`; the comparator dereferences nothing.

The pre-pass must compute, per eligible pair: both source `TaggedPtr`s and the **destination slot index**.
Assembly writes into slots, so key order is preserved without consuming buffers in key order.

**The parallel-array variant is out of scope entirely** (toplevel §2.1).

## 3. Ownership — new transactional contract

**Do not inherit spec 43's contract.** It was built around `appendOwnedContainer` and a sequential
cursor; this path writes directly into slots, so neither applies.

1. **Allocate scratch before staging any slot**, so scratch OOM leaves the initialized `result`
   untouched and reusable by the baseline fallback.
2. **Initialize `result.containers[0..output_count]` to `.reserved`.** Verified: `Container.deinit` has
   `.reserved => {}` and `getCardinality` returns 0, so a reserved slot is safe to free and safe to
   traverse.
   **Build reserved entries directly** — `Container.toTagged` on `.reserved` is **`unreachable`**, so
   routing through it will panic.
3. **Set `result.size = output_count`**, so a plain `errdefer result.deinit()` walks every slot and frees
   exactly the populated ones.
4. **Pending entries own their bitsets until the pointer is written into its destination slot.**
5. **On handoff, advance the sorted pending cursor**; the result owns that slot thereafter.
6. **Fill unmatched / non-eligible slots directly.**
7. **Before returning success, verify no `.reserved` slot remains.**

Pending cleanup covers untransferred entries; result cleanup covers assigned slots. **No second ownership
bitmap** — cursor plus reserved sentinel carry the whole boundary.

### 3.1 Fallback scope — narrow, and only at the start

**Only the initial scratch allocation failure falls back** to the baseline merge loop, reusing the
untouched initialized `result`.

**Header, payload, metadata, clone, accumulation, and assembly failures propagate** after transactional
cleanup per §3. No mid-flight fallback once buffers are staged.

## 4. Failure injection — required

Inject allocation failure at **scratch, metadata, pending headers, pending payloads, unmatched clones,
and assembly.**

Every injected failure must leave **both inputs untouched** and leak **nothing**. Leak-checking GPA,
never `c_allocator`.

**Additionally assert, on every failure path:** no `.reserved` slot is ever dereferenced, and no slot is
freed twice — the two failure modes this contract is specifically designed to prevent.

## 5. Equivalence coverage

Drive the fused arm **directly** through the internal dispatch:

- forced **and** selective lazy OR;
- eligible-pair counts of **zero, partial, and all** matched pairs;
- array/bitset/run combinations, disjoint keys, empty inputs on either side;
- **repaired output byte-identical** to baseline **and** to CRoaring.

`lazyXor` **byte-identical to baseline**, verified — scope stays `op == .bor`.

## 6. Manifest — 42 → 43 rows

The fused arm adds one row. **Both guards must read exactly 43:**

- `src/bench_parity_worker.zig:778`
- `scripts/run-compare-bench.sh:72`

Four arms, all against the same CRoaring/libc tuple:

| Row | Arm |
| --- | --- |
| `lazy-or-construction` | 1 — baseline, fused, key order (board row) |
| `lazy-or-construction-batched` | 2 — batched, unsorted, unfused |
| `lazy-or-construction-batched-sorted` | 3 — batched, sorted, unfused |
| `lazy-or-construction-batched-sorted-fused` | 4 — batched, sorted, fused |

**Arm 2 is retained deliberately** — without it, `arm2 − arm1` cannot isolate the batching machinery, and
the decomposition the toplevel exists to produce is lost.

## Acceptance

- Fused arm implemented per §1; metadata types exactly as §2, `slot` **not** `u16`.
- Transactional ownership per §3: scratch-before-staging, `.reserved` initialization built directly,
  `result.size = output_count`, cursor handoff, **no reserved slot remaining on success**.
- Fallback scope per §3.1 — **initial scratch failure only**; everything else propagates.
- **Failure-injection suite green at all six points**, plus the no-reserved-dereference and no-double-free
  assertions.
- Equivalence coverage per §5 passing; `lazyXor` byte-identical.
- **Manifest at 43 rows, both guards updated**; all four arms selectable and producing rows.
- No public API added; internal export classified in the manifest; `check-docs` green with an empty
  allow-list.
- Default behaviour unchanged; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **No measurement verdict claimed in this chunk.**

## Estimate

**M** — infrastructure exists; the new work is metadata, slot assembly, the fused loop, and the
transactional contract.
