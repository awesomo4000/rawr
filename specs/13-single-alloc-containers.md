<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 13: Single-allocation container layout (design + umbrella)

Split out of spec 11 (was 11-04) because it is a different animal: an **internal
ABI and ownership redesign**, not a localized kernel optimization. Public API,
results, and wire format stay identical, but container *internal* signatures change
(growth can relocate a container), touching call sites across the 32-bit and
64-bit trees. Highest complexity and regression risk of the perf work — it gets its
own review/testing bar and its own measurements proving the win.

**Design round → prototype → implement.** D1, D3, D4, D5 are **ratified** and D2
is resolved as stored slices. The `13-00` prototype halved per-container allocation
calls but failed the required timing gate, so this initiative is **NO-GO / PARKED**.
Do not write or execute `13-01+` unless new evidence justifies reopening it.

## Chunk plan

`13-00` completed the measurement gate. The `13-01+` work below is parked because
the proposed layout did not produce the required build/clone time win.

- **13-00 — prototype + D2 decision.** Prototype **both** D2 layouts (stored-slice
  and derived-accessor) on `ArrayContainer`; stand up the counting harness + fixed
  workloads (see Testing); record the baseline; **decide D2 with numbers** and the
  go/no-go on the whole spec.
- **13-01 — aliasing + ownership rules** under the *current* layout (define and test
  the D4 rules before the layout changes, so the invariant is pinned independently).
- **13-02 — accessor migration** — **only if D2 selects derived accessors** (the
  rough grep ~207 `.values` / ~160 `.runs` → calls, exact count from 13-00). Skipped entirely if D2
  picks stored slices.
- **13-03 — array single-block layout** — the moving ABI, full caller migration, OOM
  tests.
- **13-04 — run single-block layout** — same, for `RunContainer`.
- **13-05 — shrink / slot audit** across the 32-bit and 64-bit trees + the full
  differential matrix.
- **13-06 — results** — allocation-count + timing numbers, cross-platform builds,
  final docs.

## Goal / evidence

`ArrayContainer.init` = `allocator.create(Self)` **plus** `alignedAlloc` for values
— two allocations, two pointer chases, header and data on different cache lines
(`RunContainer` is the same shape). Co-locating the header with the data in one
aligned block removes an allocation and a pointer chase per container. Unlike the
kernel work (which ports CRoaring's algorithms), this is a layout choice rawr makes
on its own terms: CRoaring, like the current rawr, allocates two blocks per
container — a header and a separate payload — so the single-block layout is a
structural change here, not a port of an existing design.

The payoff is amplified on the 64-bit tree: `Roaring64Bitmap` multiplies container
counts (many small containers under high-32 buckets), so per-container allocation
savings scale there. It must be measured, not assumed — see Acceptance.

## Layout facts (corrected — an earlier draft got several wrong)

- **Run payload is `RunPair` = 4 bytes**, not `u16`/2 bytes. Block sizing and
  `values()`/`runs()` offset math use `@sizeOf(RunPair)` (4), not 2.
- **The run header must retain its cached `cardinality`** — runs cache cardinality
  today; the co-located header keeps that field. Header is not just `{n_runs,
  capacity}`.
- **Bitset data requires 64-byte alignment**, not the 32 an earlier draft proposed.
  If bitsets are in scope, the block alignment/offset must be 64.
- Array header: `{ cardinality: u16, capacity: u16 }`. Run header: `{ n_runs: u16,
  capacity: u16, cardinality: <cached> }`. Pad each header up to the data
  alignment.

## Decisions (D1–D5)

D1, D3, D4, D5 ratified below; D2 is resolved by the `13-00` prototype.

### D1 — Which container types change?

Array and run are the clear wins (two-alloc today). **Bitset** is fixed-size (8 KB
words); single-alloc for it is trivial (header + words in one 64-byte-aligned
block) but adds nothing structurally and forces the 64-byte-alignment question.
**DECIDED: array + run only; bitset a measured follow-up** once array/run land.

### D2 — Stored slices vs derived accessors

The two forms have **different headers and offset math — define both explicitly in
13-00**, because the target layout sketch below (header = cardinality/capacity only)
is the *derived-accessor* layout; the stored-slice form needs the slice in the
header, changing `HEADER_SIZE` and every offset.

- **Keep stored `values`/`runs` slices** — the slice lives *in the header*
  (`Header { cardinality, capacity, values: []Elem }`), pointing into the same block.
  Larger header, larger `HEADER_SIZE`. Preserves every current call site verbatim,
  but does **not** eliminate the pointer chase (the slice ptr still indirects — now
  to the same cache line). **Hazard — refresh the slice after *every* capacity change,
  not just a move:**
  - On a **move** (relocate): the stored pointer dangles into freed memory unless
    reset to the new block. (A `memcpy` of the old header into the new block would
    carry the stale pointer along — reset explicitly.)
  - On an **in-place `allocator.resize`** (pointer stays valid): the pointer is fine
    but the stored slice's **length still reflects the old capacity** — it must be
    updated too.
  So growth **and** shrink reset the stored slice's **pointer *and* length** after
  both the in-place-resize and the move paths. Every capacity-changing site re-derives
  the slice from `dataOffset()` + the new capacity.
- **Derived accessors** (`fn values(self) [*]Elem` = `self + dataOffset()` — the
  padded offset, not raw `@sizeOf(Header)`; never stored) — small header
  (cardinality/capacity only), no stored pointer, no
  reset-after-move hazard. But it's a **repo-wide refactor**: a grep counts ~**207
  `.values` / ~160 `.runs`** — a rough upper bound, since `.values` also matches
  unrelated fields; 13-00 produces the real accessor-migration count. That migration
  is chunk 13-02, and it only exists if this option wins.

**RESOLVED by `13-00`: stored slices.** Derived accessors were 3.12% slower than
stored slices for membership and 1.17% slower for iteration, missing the respective
19.73% and 13.80% (`2*epsilon_w`) win thresholds. The hand-filtered migration count
was 147 real `.values` references plus 132 real `.runs` references, so there is no
performance case for that 279-reference migration. If this initiative is reopened,
the refresh-after-capacity-change rule above remains mandatory.

### D3 — The pointer-update ABI (choose one)

Growth reallocates the *whole block including the header*, so `*ArrayContainer`
itself may move. Two ways to let a mutation update the owning slot:

- **(a) return the possibly-new pointer** — grow-capable methods return `*Self` (or
  a small result struct that also carries values like `added: bool`); the
  **bitmap / container-op helper** reassigns the tagged slot.
- **(b) take a `*TaggedPtr` slot** — the container method updates the caller's slot
  in place.

**DECIDED: (a) pointer-return.** Passing `*TaggedPtr` **into** `array_container.zig`
would make the container layer depend on the tagged-pointer/ownership layer above it
— an inverted dependency and a likely **import cycle**. Keep the container layer
ignorant of tags: grow/shrink-capable methods return the (possibly relocated)
pointer + any result value (a small result struct carrying e.g. `added: bool`), and
the layer that *owns* the slot (bitmap / `container_ops` helpers) writes the tagged
slot.

**Exact return contracts to define before implementation** (13-03/13-04) — each
grow/shrink/convert-capable method returns the possibly-relocated `*Self` (or a
result struct carrying it + values like `added: bool`):

- `ArrayContainer`: `add`, `unionInPlace`, `ensureCapacity`, `shrinkToFit`.
- `RunContainer`: `add`, `addRange`, **`remove`** (a run split can grow the array),
  `ensureCapacity`, `shrinkToFit`.
- **Conversion cases** where an array becomes a bitset (or vice versa) — return a
  **typed local union/result** (the new container of a possibly-different type), **not
  a `TaggedPtr`**. Per D3 the container layer stays tag-ignorant; the owning
  bitmap / `container_ops` layer builds the tag from the returned type.
- **Ownership: state who frees the old pointer** after a relocation or a conversion
  — the container method (which allocated the new block) frees the old block before
  returning, and the caller only rewrites the tagged slot; define this once so no
  path double-frees or leaks.

Shrinking **also moves the block** (it was missing from an earlier draft's
pointer-update audit) — it's in the same ABI and audit as growth.

### D4 — Aliasing rule for moving containers

A self-operation — `unionInPlace(a, a)`, or any op where two arguments alias the same
container — can **relocate and free the block while the other argument still points at
the old one** (use-after-free). **DECIDED: special-case pointer equality *before* any
growth or relocation** so identity ops never reach a grow/relocate path. The handling
**differs by API shape — an *allocating* op must not mutate its input:**

- **Allocating** (`a.bitwiseXor(allocator, &a)`, etc.) — returns a **new** result and
  leaves `a` **unchanged** (never frees `a`'s containers): self-union / self-intersection
  → a fresh clone of `a`; self-XOR / self-difference → a fresh **empty** bitmap.
- **In-place** (`a.bitwiseXorInPlace(&a)`, etc.) — may mutate `a`: self-union /
  self-intersection → **no-op** (leave `a` as-is); self-XOR / self-difference →
  **clear** `a`, deinitializing its containers **exactly once** (no double-free, no
  leak).

**Aliasing scope — what's supported:**
- **`self == other` at the bitmap API boundary** (e.g. `a.bitwiseXorInPlace(&a)`) —
  supported via the pointer-equality short-circuit above.
- **Transient container-op aliases with exactly one owner** (two arguments of a single
  op resolving to the same container pointer) — supported; the short-circuit keys on
  container pointer within that one operation.
- **Two *owning* bitmaps sharing a container** — **unsupported.** Without reference
  counting or copy-on-write, one owner's mutation invalidates the other and deinit
  eventually double-frees. Out of scope — rawr has no COW (the API-gap audit records
  COW as a deliberate omission).

There is **no "self-aliased shrink"** — `shrinkToFit` takes one argument, so aliasing
can't arise; what matters on the shrink path is **updating the owning tagged slot after
the block relocates** (D3), not alias detection.

Tests must verify (a) **alias detection fires before any relocation** for the two
supported cases, and (b) **correct empty-result ownership** for self-XOR/difference
(exactly-once free), under `DebugAllocator`.

## Target layout sketch (array; run analogous with the 4-byte `RunPair` + cached card)

```
offset 0:   Header { cardinality: u16, capacity: u16 }   padded to data alignment
offset A:   data...                                        (A = data alignment; see D5)
```

**D5 — per-type *block* alignment. DECIDED: array 16, run 4, bitset 64 (if in scope).**

**Tag-bit invariant (correctness, not just SIMD):** `TaggedPtr` consumes the **two
low address bits** of the block pointer to encode the container type, so **every
block must be aligned to at least 4** regardless of payload needs. The block
alignment is:

```
A = max(header alignment, payload alignment, 4)   // 4 = TaggedPtr's 2 tag bits
```

- **Array**: the shipped 11-05/11-06 kernels are **128-bit** (`@Vector(8, u16)` =
  16-byte loads), so 16-byte alignment keeps those loads aligned → **A = 16** (≥ 4,
  tag bits fine). (An earlier draft said 32; over-aligned — 32 only matters for a
  256-bit kernel, which deliberately doesn't exist: 256-bit AVX2 array-intersect
  can't compact cleanly (`vpshufb` is lane-restricted), so the wide path is AVX-512,
  out of scope. Extra alignment wastes header padding for no load benefit.)
- **Run**: `@alignOf(RunPair)` is only **2** — **insufficient for the 2-bit tag** — so
  run blocks must be bumped to **A = 4**. Do **not** use natural RunPair alignment; a
  2-byte-aligned block would collide with the tag bits.
- **Bitset**: words → **A = 64** (if in scope, per D1).

**Data offset — align the header size *up* to `A` (the header can be larger than
`A`):** the stored-slice header (D2) likely exceeds 16 bytes, so `dataOffset` is not
simply `A`:

```zig
const data_offset = std.mem.alignForward(usize, @sizeOf(Header), A);
```

**Compile-time assertions (required):** `comptime` assert, per container type, that
(1) `A >= 4` (tag bits available), (2) `data_offset % A == 0`, and (3) `data_offset >=
@sizeOf(Header)` — **not** that the header fits *within* `A` (it needn't). These catch
a too-small alignment or a bad offset at build time rather than as runtime tag/data
corruption.

```zig
fn allocBlock(allocator: Allocator, cap: u16) ![]align(A) u8 {
    // NB: dataOffset(), NOT @sizeOf(Header) — the header is padded up to A, so a
    // 24-byte header at A=32 needs a 32-byte offset; using 24 underallocates by 8.
    const bytes = std.math.add(usize, dataOffset(),
        std.math.mul(usize, cap, @sizeOf(Elem)) catch return error.Overflow) catch return error.Overflow;
    return allocator.alignedAlloc(u8, .fromByteUnits(A), bytes);
    // Elem = u16 for array, RunPair (4 bytes) for run
}
```

**Overflow-safe arithmetic (required):** block size (`dataOffset() + cap*@sizeOf(Elem)`
— **always the padded offset, never the raw `@sizeOf(Header)`**) and any
capacity-growth doubling must use checked/saturating math (as in spec 12's
`ensureTotalCapacity`) — `cap` is bounded (≤ 4096 for arrays before promotion) but
the arithmetic must be total regardless of what a caller passes.

**Canonical block reconstruction (required — one implementation, not per-call-site).**
`deinit`, `clone`, resize, and the OOM/move paths must all compute block size and the
data offset **the same way**, or they'll free/copy mismatched sizes (the exact class
of bug `DebugAllocator` exists to catch). Provide shared, private helpers used
everywhere:
- `dataOffset()` — `alignForward(@sizeOf(Header), A)` (header size rounded up to `A`;
  **not** just `A` — the header may be larger).
- `blockSize(cap)` — checked `dataOffset() + cap*@sizeOf(Elem)` (overflow-safe, below;
  the padded offset, **not** raw `@sizeOf(Header)`).
- a `*Self → []align(A) u8` reconstruction that yields the **exact** aligned
  allocation slice to hand back to `allocator.free`.
- `moveBlock` / `freeBlock` — the single alloc-new+memcpy+free-old and free paths.

Growth: try `allocator.resize` first (in-place keeps the pointer — common with
power-of-two caps); on failure, `moveBlock` (alloc-new + memcpy + free-old), then
update the slot per D3. `clone` / `deinit` go through the same helpers — free/copy one
block, not two. (The **serialized wire size** is a *separate* computation — the
portable-format byte length, unrelated to the allocation-block size, unchanged by this
spec.) `TaggedPtr` scheme unchanged — tag bits still discriminate type; only what the
pointer points at changes.

## Call-site audit (both trees, growth **and** shrink)

Convert every grow/shrink-capable entry point and every site that constructs then
grows a result container. Starting set (verify exhaustiveness by grepping
`ensureCapacity` / `shrinkToFit` callers — treat as a floor, not the full list):

- `ArrayContainer.add`/`ensureCapacity`/`shrinkToFit`; `RunContainer` append/insert/
  `ensureCapacity`/`shrinkToFit`.
- Result-container growth inside `container_ops.zig`.
- **32-bit** bitmap `addToContainer` and every mutation holding a slot.
- **64-bit** (`roaring64.zig`) — the equivalent slot-holding paths; roaring64 landed
  on the two-alloc layout, so its call sites are in scope and amplify the payoff.

## Testing

- **`DebugAllocator` coverage already mostly exists** (corrected — an earlier draft
  wrongly said the differential exes don't use it). In fact `diff_test`,
  `validate_roaring64`, and `diff_test64` **already run under `DebugAllocator`**; only
  `validate_croaring` uses the C allocator. That's the coverage this redesign needs
  — freeing the wrong block size / double-free on a moved block will surface in those
  three suites. **Decide** whether `validate_croaring` also needs switching (likely
  not — three suites already exercise the checking). **Do not** add a vague build
  flag; the checking is already in place.
- **Aliasing tests** (D4): the supported self-aliased cases — allocating vs in-place
  self-XOR/difference/union/intersection at the bitmap boundary — under
  `DebugAllocator` (there is no self-aliased *shrink*; shrink is a slot-update check).
- **Allocation-failure tests**: a failing allocator on the alloc-new-then-move path
  leaves the container and its slot unchanged and usable.
- **Allocation-count proof, not just time — with fixed workloads.** The existing
  allocator-matrix bench reports *time*, not allocation counts. Add a **counting
  allocator** (tallies alloc/free calls + bytes) and run **fixed, specified**
  workloads on both trees — pin **N, container shapes, seeds, and the target/CPU**, and
  state the **go/no-go criterion**, so the result is reproducible and decisive:
  - build-N-containers, clone, and deserialize a large bitmap (32-bit); the same over
    a many-bucket `Roaring64Bitmap`.
  - Record before/after allocation counts + time deltas.
  - **"~½ the allocations" is per array/run container** (two allocs → one). It is
    **not** ½ of total bitmap construction or deserialization — those also allocate
    the top-level index and any bitset containers (bitset is unchanged, per D1). State
    the expected reduction against the *per-container* baseline, not the whole
    workload, so the go/no-go isn't measured against an impossible target.

**Coordination with spec 11:** spec 11's SIMD kernels now compact through a local
stack scratch (not container over-allocation), so **this spec's layout carries no
SIMD headroom slack** — the two efforts are decoupled. If that scratch decision ever
changes, revisit the array block sizing here.

## Acceptance

- D1/D3/D4/D5 ratified; **D2 decided by `13-00` with numbers** before any layout
  chunk (13-01+) is written.
- Chosen ABI implemented for **array and run; bitset excluded** (D1); growth **and
  shrink** both update slots; aliasing rule enforced and tested; overflow-safe
  block/growth math; block alignment ≥ 4 with the tag-bit/offset compile asserts (D5).
- Full call-site audit complete across both trees; the **existing `DebugAllocator`-backed
  suites** (`diff_test`, `validate_roaring64`, `diff_test64`) pass; allocation-failure
  and aliasing tests pass.
- Counting-allocator workloads show the **allocation-count reduction** (numbers, both
  trees) and a net time win on allocation-heavy scenarios — per the "own
  measurements" bar.

## Estimate

**L (multi-week), not M.** The earlier `M` was optimistic given: pointer-moving
APIs, direct `.values`/`.runs` field usage (D2), shrink-also-moves handling,
aliasing rules, and the dual 32/64-bit call-site audit. Scope depends heavily on
D2 (derived accessors → repo-wide refactor). D1 is settled (array + run; bitset
excluded), so it no longer swings the estimate.
