<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 44-00: Fused arm — implementation, ownership, correctness

Toplevel: [44-fused-address-ordered-construction.md](44-fused-address-ordered-construction.md).

Builds on branch `spec-43-lazy-construction-diagnostic`, commit **`37d0e8b`** (**push it first** — the
reference does not resolve while the branch is local only).

**No default change. No measurement verdict here** — that is `44-01`. This chunk ends when the fused arm
is correct, leak-free, and selectable.

## 1. The two new arms

**Two** new `ConstructionMode` variants — arm 4 (**unfused**) and arm 5 (**fused**) — identical in every
respect except pass structure, because that is what makes `arm5 − arm4` a fusion measurement (toplevel
§3).

- **Arm 4:** two passes over the sorted pending array — zero all payloads, then accumulate into all.
- **Arm 5:** one pass — per pending entry, **in address order**:

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
    slot: u32,                    // 4 — see note below
};                                // 36 → 40 bytes with alignment
```

**On `slot`:** 65,536 slots means indices `0..65,535`, which **do fit `u16`**. Use `u32` anyway — it
removes boundary reasoning at the maximum and alignment padding absorbs it. **40 bytes holds on the
64-bit benchmark hosts only**; it is not pointer-width independent.

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
   `.reserved => {}` and `getCardinality` returns 0 — safe **for deinit and cardinality specifically**,
   **not** "safe to traverse" generally: several container paths treat `.reserved` as `unreachable`.
   **Build reserved entries directly** — `Container.toTagged` on `.reserved` is **`unreachable`** and will
   panic. Do not route a reserved slot through any other container operation before it is filled.
3. **Set `result.size = output_count`**, so a plain `errdefer result.deinit()` walks every slot and frees
   exactly the populated ones.
4. **Pending entries own their bitsets until the pointer is written into its destination slot.**
5. **Handoff ordering, pinned:** write the pointer into the result slot **first**, then advance the
   pending cursor, with **no fallible operation between**. Any gap there is a window where the buffer is
   owned twice or not at all.
6. **Fill unmatched / non-eligible slots directly.**
7. **Before returning success, verify no `.reserved` slot remains.**

Pending cleanup covers untransferred entries; result cleanup covers assigned slots. **No second ownership
bitmap** — cursor plus reserved sentinel carry the whole boundary.

### 3.1 Fallback scope — narrow, and only at the start

**Only the initial scratch allocation failure falls back** to the baseline merge loop, reusing the
untouched initialized `result`.

**Header, payload, metadata, clone, accumulation, and assembly failures propagate** after transactional
cleanup per §3. No mid-flight fallback once buffers are staged.

## 4. Failure injection — the REAL fallible sites

*(An earlier draft listed "scratch, metadata, pending headers, pending payloads, unmatched clones,
assembly" — that names a duplicate and two non-existent sites: the single `[]Pending` allocation **is**
both the scratch and the metadata, and accumulation and direct-slot assignment are **infallible**.)*

Actual fallible allocation sites:

1. **pending scratch** — the one `[]Pending` allocation;
2. **header `create`** per pending bitset;
3. **`words` payload allocation** per pending bitset;
4. **unmatched clone** allocations;
5. **non-eligible union** allocations.

**Use `std.testing.checkAllAllocationFailures`** for exhaustive coverage rather than hand-enumerated
injection points.

**Plus one targeted test proving the fallback boundary:** only failure at site 1 invokes the baseline
fallback; failures at 2–5 **propagate** after transactional cleanup (§3.1).

Every failure must leave **both inputs untouched** and leak **nothing**. Leak-checking GPA, never
`c_allocator`.

**Additionally assert on every failure path:** no `.reserved` slot is dereferenced, and no slot is freed
twice — the two modes this contract exists to prevent.

## 5. Equivalence coverage

Drive **both** new arms directly through the internal dispatch — arm 4 is production code too, not
scaffolding, and an untested arm 4 makes `arm5 − arm4` meaningless:

- forced **and** selective lazy OR;
- eligible-pair counts of **zero, partial, and all** matched pairs;
- array/bitset/run combinations, disjoint keys, empty inputs on either side;
- **repaired output byte-identical** to baseline **and** to CRoaring.

`lazyXor` **byte-identical to baseline**, verified — scope stays `op == .bor`.

## 6. Manifest — 42 → 44 rows

Arms 4 and 5 add two rows. **Both guards must read exactly 44:**

- `src/bench_parity_worker.zig:778`
- `scripts/run-compare-bench.sh:72`

Five arms, all against the same CRoaring/libc tuple:

| Row | Arm |
| --- | --- |
| `lazy-or-construction` | 1 — baseline, fused, key order (board row) |
| `lazy-or-construction-batched` | 2 — batched, unsorted, unfused |
| `lazy-or-construction-batched-sorted` | 3 — batched, sorted, unfused |
| `lazy-or-construction-slotted` | 4 — sorted + metadata + direct-slot, **unfused** |
| `lazy-or-construction-slotted-fused` | 5 — same as arm 4, **fused** |

**Arms 2 and 4 are both required.** Without arm 2, `arm2 − arm1` cannot isolate the batching machinery;
without arm 4, `arm5 − arm3` bundles fusion with metadata, destination-bound traversal, direct-slot
assembly, and reserved handling — which is exactly the attribution failure this design exists to avoid.

## Acceptance

- **Both** arm 4 (unfused) and arm 5 (fused) implemented per §1, identical except pass structure;
  metadata types exactly as §2.
- Transactional ownership per §3: scratch-before-staging, `.reserved` initialization built directly,
  `result.size = output_count`, cursor handoff, **no reserved slot remaining on success**.
- Fallback scope per §3.1 — **initial scratch failure only**; everything else propagates.
- **`checkAllAllocationFailures` green** across the five real fallible sites (§4), plus the targeted
  fallback-boundary test, plus the no-reserved-dereference and no-double-free assertions.
- Equivalence coverage per §5 passing; `lazyXor` byte-identical.
- **Manifest at 44 rows, both guards updated**; all **five** arms selectable and producing rows.
- No public API added; internal export classified in the manifest; `check-docs` green with an empty
  allow-list.
- Default behaviour unchanged; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.
- **No measurement verdict claimed in this chunk.**

## Estimate

**M** — infrastructure exists; the new work is metadata, slot assembly, the fused loop, and the
transactional contract.
