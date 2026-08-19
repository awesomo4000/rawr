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
- **Ordering contract:** after the payload allocation **succeeds**, initialize the header immediately —
  before any subsequent fallible operation — so `BitsetContainer.deinit` is valid from that point on (it
  frees `words` and destroys the header without reading the body). If the **payload allocation itself
  fails**, destroy only the header; there is no payload to free. *(An earlier draft said "assign `words`
  before any failure point", which is not achievable — the payload allocation is itself a failure point,
  and `words` cannot be assigned before it returns.)*
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

## 4.1 The pipeline — pinned, because sorting discards key order

Sorting `Pending` by payload address **destroys merge-key order**, yet result containers must be appended
**in key order**. The resolution is that zeroed pending buffers are **interchangeable** — an 8 KB zeroed
bitset is identical to any other — so no mapping back to keys is needed or wanted.

Exact sequence — **scratch must exist before the pending objects it records**:

1. **`result.initCapacity`** — see §4.2. Leave it exactly where baseline has it.
2. **Pre-pass** the merge walk; compute the exact eligible count (§3).
3. **Allocate the scratch** `[]Pending` for that count.
4. **On scratch OOM → fall through into the baseline merge loop, reusing the `result` from step 1.**

   *(An earlier draft said there was "nothing to unwind" at this point. That is wrong: `result` was
   initialized in step 1 and **already owns its key and container arrays**. Returning to a fresh baseline
   call without handling it would leak both.)*

   **Decision: factor the baseline merge loop to operate on an already-initialized `result`**, and on
   scratch OOM continue into it with the existing one. No pending object exists yet, so nothing else
   needs unwinding, and the result is reused rather than freed and rebuilt.

   *(Rejected: `deinit` the result and call baseline from scratch. It works, but discards two allocations
   and re-does them, on a path already under memory pressure — the worst moment to allocate again.)*
5. **Allocate** all eligible headers + payloads via `initPendingBitset` — no zeroing — **recording each
   into the scratch as it is created**.
6. **Sort** the scratch by `payload_addr` (§4).
7. **Zero** the payloads in sorted order — *this is the entire point of the spec*: the 8 KB `@memset`
   traffic now walks memory ascending.
8. **Re-walk the merge in key order**, and for each eligible pair **take the next pending buffer
   sequentially** from the scratch array (a simple cursor), accumulate both operands into it, and append
   it — transferring ownership per §5.

*(An earlier draft ordered allocation before scratch, which is impossible: the pending objects have
nowhere to be recorded until the scratch exists.)*

**Three things this explicitly forbids:**

- **No second mapping** from sorted position back to key. **Step 8** consumes sequentially precisely
  because buffers are interchangeable; adding a key→buffer map re-introduces the scratch cost the design
  avoids.
- **No appending out of key order.** **Step 8** walks keys, not the sorted array; the sorted array is
  only a free-list cursor by that point.
- **No publishing a pending buffer before step 7 completes** for it. Zeroing precedes accumulation, which
  precedes append (§5).

The merge walk therefore runs **twice** — once to count, once to build. Both traversals are inside the
timed region and inside the candidate's cost.

### 4.2 `result.initCapacity` placement — unchanged from baseline, deliberately

`lazyMergeTwo` calls `Self.initCapacity(allocator, max_result_size)` **before** the merge loop
(`bitmap.zig:2325`), and that call allocates. **Allocation order is the mechanism under test**, so where
those allocations sit relative to the pending batch cannot be left to chance.

**Decision: leave `initCapacity` exactly where baseline has it — first, before the pre-pass.** All three
arms then share an identical prologue, and the *only* difference between them is how the pending bitsets
are allocated and ordered. Moving it would change arm 1 as well as arms 2 and 3, confounding the very
comparison the arms exist to make.

If an implementation finds a reason to move it, that is a **separate arm**, not a silent change.

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

### 6.1 Manifest row count — update both guards to 42

The row count is asserted in **two** places, both hardcoded at 40:

- `src/bench_parity_worker.zig:778` — `if (manifest.len != 40) return error.InvalidManifestRowCount;`
- `scripts/run-compare-bench.sh:72` — `if [[ "$row_count" != 40 ]]`

This chunk adds two rows → **update both to 42**. Missing either one fails the run rather than silently
mismeasuring, which is the desired behaviour; the point of naming both here is that a search for "40"
must not stop at the first hit.

`43-02` keeps the count at 42 by **repurposing**, not adding — see that chunk.

### 6.2 Hook shape — root-level internal, never a method

**Do not implement the dispatch as a `pub fn` on `RoaringBitmap`.** `check_docs.zig:72` reflects over
direct `pub fn` declarations of the five stable types, so a public method would be classified as **stable
public API** and demand a `Type.method` entry in the guarded Quick Reference — precisely the surface this
spec declined to add.

Required shape: a **root-level internal function or namespace**, re-exported through `roaring.zig` and
classified in the **internal-export manifest** with a reason string. That path is reflected over for
*classification* but not for method documentation.

### 6.3 Equivalence coverage — the diagnostic modes must actually be executed

**Without this, the batched paths ship untested.** Public `lazyOr` still uses `.baseline` throughout this
chunk, so the unit suite, `difftest`, and `difftest64` **never execute either new mode**. The canonical
sparse corpus does not help — it exercises one shape, and the risk here is the ownership path, not the
arithmetic.

Add tests that drive `.batched_unsorted` and `.batched_sorted` **directly** through the internal dispatch:

- **both** forced and selective lazy OR;
- eligible-pair counts of **zero, partial, and all** matched pairs — zero and all are the pre-pass's
  boundary cases, and partial is the only case where selective and forced diverge;
- **array/bitset/run combinations**, disjoint keys, and empty inputs on either side;
- **byte-identical repaired output** against both `.baseline` and CRoaring.

Equality is on the **repaired** result, since lazy output is not directly comparable.

## 7. Failure injection — required

Inject allocation failure at **scratch, pending headers, pending payloads, unmatched clones, and result
assembly.**

Every injected failure must leave **both inputs untouched** and leak **nothing** — no assigned container,
no pending container, no scratch. Use a leak-checking GPA, never `c_allocator`.

## Acceptance

- Scope limited to `.bor`; **`lazyXor` behaviour byte-identical to baseline**, verified.
- `initPendingBitset` file-private in `bitmap.zig`, header initialized immediately after a successful
  payload allocation and before any later fallible step; payload-allocation failure destroys only the
  header. No new export.
- **Equivalence coverage (§6.3) executes both diagnostic modes** — forced and selective, eligible counts
  of zero/partial/all, array/bitset/run combinations, disjoint keys, empty inputs — with repaired output
  byte-identical to `.baseline` and to CRoaring.
- Eligible-pair pre-pass exact for both forced and selective; **no unused 8 KB buffers allocated** —
  verified by counting, not asserted.
- **Pipeline follows §4.1 exactly:** `initCapacity` (unchanged position, §4.2) → pre-pass → **allocate
  scratch** → *scratch OOM falls through to the baseline loop reusing `result`* → allocate pending into
  scratch → sort → zero in sorted order → re-walk in key order consuming buffers sequentially. **No
  key→buffer mapping**, no out-of-key-order append, no publication before zeroing.
- Exactly one `[]Pending` scratch allocation, `sortUnstable` on the inline key, freed on every path;
  scratch failure retries the existing path.
- Ownership contract (§5) holds; **failure-injection suite green at all five points** — no leaks, inputs
  untouched.
- Per-bitset allocation count **still two**; scratch reported separately. This is not an
  allocation-count-reduction lever (specs 27 and 35 both regressed M4 SMP that way).
- Three diagnostic rows measured as **equivalent fresh-process cells**.
- **Manifest guards updated to 42 in BOTH `bench_parity_worker.zig:778` and
  `run-compare-bench.sh:72`.**
- **Hook is root-level internal** (§6.2), not a `pub fn` on `RoaringBitmap`; classified in the
  internal-export manifest; `check-docs` still green with an **empty allow-list**.
- **GATE 1 — decision rules, not impressions.** Campaign policy applies: ≥5 fresh-process medians with
  full ranges per cell.
  - `lazy-or-construction-batched-sorted` **≤1.10x** vs the shared CRoaring/libc reference, on median;
  - **arm 3 faster than arm 2 with non-overlapping ranges.** If the ranges overlap, **rerun**; if they
    still overlap, the ordering effect is **unresolved → NO-GO**, not a marginal pass. An unresolved
    arm-2/arm-3 difference means the spec's causal claim is unproven, which is a stop even if arm 3's
    median looks better.
  - **libc does not regress — ≤5% on median, ranges considered.** A libc regression is a **STOP** (record the result, report the row as not
    closed, open a follow-up opt-in spec that owns the API/docs cost) — **not** a fallback to opt-in
    inside this spec.
- Default behaviour unchanged; canonical board row unmoved.
- All four suites green — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
  `check-package`.

## Estimate

**M/L** — the pending path, exact pre-pass, ownership, and failure injection are each substantial.
