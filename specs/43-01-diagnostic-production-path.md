<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 43-01: Diagnostic production path and Gate 1

Toplevel: [43-address-ordered-lazy-construction.md](43-address-ordered-lazy-construction.md).
Gated on: [43-00](43-00-feasibility-prototype.md) returning GO.

**The default does not change in this chunk.** The batched/sorted path ships behind runtime dispatch and
is exercised only by diagnostic rows. A Gate 1 failure therefore changes no production behaviour.

## 1. Scope

**Lazy OR only — `op == .bor`, both forced and selective.** `lazyMergeTwo` takes
`ManyOp = enum { bor, xor }` (`bitmap.zig:2131`, signature at `:2317`). The forced/selective flag changes
only *how many* pairs take the bitset branch, not the mechanism, so `op` is the clean boundary.

**`lazyXor` is excluded and must be byte-identical to baseline** — shared-helper obligation, verified,
not assumed.

## 2. Private pending allocation — in `bitmap.zig`

A **file-private `initPendingBitset`** in `bitmap.zig`. It cannot live in `bitset_container.zig`: Zig
privacy is file-level, so a non-`pub` declaration there is not callable from `bitmap.zig`, and making it
`pub` would add to the internally-exported container surface. Struct fields carry no visibility modifier,
so the helper can build a `BitsetContainer` directly with no new export.

- `allocator.create(BitsetContainer)` + `allocator.alignedAlloc(u64, .@"64", NUM_WORDS)`, **no
  `@memset`**, `cardinality` set to whatever `init` uses.
- **Assign `words` before any failure point**, so `BitsetContainer.deinit` is safe on a pending container
  — it frees `words` and destroys the header without reading the body. This ordering is the difference
  between clean cleanup and a leak or double-free.
- File-private: enters neither the `check-docs` surface nor the `check-32` probe.

## 3. Eligible-pair pre-pass

The predicate (`bitmap.zig:2344`):

```zig
const use_lazy_bitset = op == .xor or bitset_conversion
    or isBitsetContainer(c_a) or isBitsetContainer(c_b);
```

Forced lazy OR makes every matched pair eligible; **selective does not**. Sizing off matched pairs would
allocate thousands of unused 8 KB buffers — up to 512 MB worst case — turning an ordering optimization
into a memory-consumption defect.

**One pre-pass over the merge walk evaluating the same predicate**, producing the exact eligible count
before any pending allocation. No allocation in the pre-pass. Cost is `O(a.size + b.size)` — the walk
advances through both key arrays; only the *match* count is bounded by `min` — and it is **inside the
timed region**, being part of the candidate.

## 4. Scratch and sort

One allocation of `[]Pending` (`{ payload_addr: usize, header: *BitsetContainer }`), length = eligible
count, from the allocator passed to `lazyOr`. `sortUnstable` (pdq) on `payload_addr`; the comparator
dereferences nothing. Freed before return **on every path**.

**Scratch allocation failure → retry through the existing interleaved path**, propagating only an error
from *that* path. Do not propagate the scratch OOM directly; do not promise success, since an exhausted
allocator may still legitimately fail the retry.

## 5. Ownership

The pending pool **owns** a header and payload until that container has been (1) zeroed, (2) accumulated,
and (3) **successfully appended** through an ownership-taking helper. Only then does the result own it.

So: failure **before** the append frees through the pool, failure **after** frees through the result, and
no container is ever owned by both or neither.

## 6. Runtime three-arm dispatch

A private `ConstructionMode` (`.baseline`, `.batched_unsorted`, `.batched_sorted`) threaded through the
lazy construction path.

**Runtime, not compile-time:** the parity worker is built **once**
(`run-compare-bench.sh` builds `bench-parity-worker -Dcpu=native`) and selects rows at runtime, so a build
option cannot produce three arms from one binary. Compiling three worker variants is **rejected** — it
reintroduces the spec-28 whole-binary layout noise, where added code moves untouched rows with
instruction-identical disassembly, making arm-to-arm deltas uninterpretable at the ~1.2x floor.

Reached by the benchmark through `roaring.zig`'s **internal-export manifest**, with a reason string.
It **does** ship in the package's internal surface (`roaring.zig` is in `.paths`) and carries the same
"may change without notice" status as the other 10 internal exports. What it stays outside is the
**stable public API** — `API.md`, the `check-docs` guarded region, and the `check-32` probe.

Diagnostic rows, all against the same CRoaring/libc tuple:

| Row | Arm |
| --- | --- |
| `lazy-or-construction` | 1 — existing interleaved baseline (board row, unchanged) |
| `lazy-or-construction-batched` | 2 — batched, unsorted |
| `lazy-or-construction-batched-sorted` | 3 — batched, sorted |

## 7. Failure injection — required

Inject allocation failure at **scratch, pending headers, pending payloads, unmatched clones, and result
assembly.**

Every injected failure must leave **both inputs untouched** and leak **nothing** — no assigned container,
no pending container, no scratch. Use a leak-checking GPA, never `c_allocator`.

## Acceptance

- Scope limited to `.bor`; **`lazyXor` behaviour byte-identical to baseline**, verified.
- `initPendingBitset` file-private in `bitmap.zig`, `words` assigned before any failure point, no new
  export.
- Eligible-pair pre-pass exact for both forced and selective; **no unused 8 KB buffers allocated** —
  verified by counting, not asserted.
- Exactly one `[]Pending` scratch allocation, `sortUnstable` on the inline key, freed on every path;
  scratch failure retries the existing path.
- Ownership contract (§5) holds; **failure-injection suite green at all five points** — no leaks, inputs
  untouched.
- Per-bitset allocation count **still two**; scratch reported separately. This is not an
  allocation-count-reduction lever (specs 27 and 35 both regressed M4 SMP that way).
- Three diagnostic rows measured as **equivalent fresh-process cells**.
- **GATE 1:**
  - `lazy-or-construction-batched-sorted` **≤1.10x** vs the shared CRoaring/libc reference;
  - **arm 3 beats arm 2** — the effect is ordering, not batching;
  - **libc does not regress.** A libc regression is a **STOP** (record the result, report the row as not
    closed, open a follow-up opt-in spec that owns the API/docs cost) — **not** a fallback to opt-in
    inside this spec.
- Default behaviour unchanged; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.

## Estimate

**M/L** — the pending path, exact pre-pass, ownership, and failure injection are each substantial.
