<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 13: Single-allocation container layout (design + umbrella)

Split out of spec 11 (was 11-04) because it is a different animal: an **internal
ABI and ownership redesign**, not a localized kernel optimization. Public API,
results, and wire format stay identical, but container *internal* signatures change
(growth can relocate a container), touching call sites across the 32-bit and
64-bit trees. Highest complexity and regression risk of the perf work — it gets its
own review/testing bar and its own measurements proving the win.

**Design round → prototype → implement.** D1, D3, D4, D5 are now **ratified**
(marked DECIDED below). The one open variable — D2 (stored slices vs derived
accessors) — is resolved by a **`13-00` prototype/measurement chunk** before any
layout implementation is written. So the next step is `13-00`, not the full layout.

## Chunk plan

- **13-00 — prototype + measurement (D2).** Build a throwaway prototype of both
  `values`/`runs` forms (stored slice vs derived accessor) on `ArrayContainer`,
  measure the pointer-chase win in isolation with the counting allocator + the
  workloads below, and **pick slices or accessors** with numbers. Also confirm the
  allocation-count reduction is real before committing to the full audit. Output:
  D2 decided, recorded here; go/no-go on the whole spec.
- **13-01+ — layout implementation** (array, then run), the pointer-return ABI,
  the aliasing rule, growth+shrink slot updates, and the dual-tree call-site audit.
  Written only after 13-00 lands.

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

- **Keep stored `values`/`runs` slices** (a field on the header pointing into the
  same block): preserves every current call site verbatim, but does **not**
  eliminate the pointer chase the redesign is partly justified by (the slice ptr
  still indirects — though now to the same cache line). **Hazard if slices stay:**
  after any move (realloc/relocate), the stored slice pointer must be **reset** to
  the new block — a `memcpy` of the old header into the new block would carry a
  **dangling pointer** into freed memory. Every move site re-derives the slice.
- **Derived accessors** (`fn values(self) [*]u16` computed as `self + HEADER_SIZE`,
  never stored): actually removes the stored pointer *and* the reset-after-move
  hazard, but requires a **repo-wide refactor** of every `.values` / `.runs` field
  access into a call.

**DECIDED: settle this in a `13-00` prototype/measurement chunk** (see chunk list).
Prototype both; **start with stored slices if the derived-accessor result is
marginal** — the allocation-count win is real regardless, the pointer-chase win only
materializes with derived accessors, and it may not justify the repo-wide refactor.
If slices stay, the reset-after-move rule above is mandatory. This choice gates the
size of the whole spec, so it's resolved *first*, with numbers.

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

**Signatures to define once (b) or (a) is chosen** — at minimum:
`add`, `addRange`, `unionInPlace` (and the other in-place set ops that can grow a
result), `ensureCapacity`, and **`shrinkToFit`**. Shrinking **also moves the
block** and was missing from the earlier pointer-update audit — it must be in the
same ABI and audit as growth.

### D4 — Aliasing rule for moving containers

A self-operation — `unionInPlace(a, a)` or any op where two arguments alias the
same container — can **relocate and free the block while the other argument still
points at the old one** (use-after-free). **DECIDED: special-case pointer equality
*before* any growth or relocation**, and define ownership per operation — identity
ops must **avoid growth/relocation entirely** (there is no "self-aliased grow path"
to reach; alias detection happens first):

- **self-union / self-intersection** → the result equals the input; return it
  **unchanged** (no alloc, no free).
- **self-XOR / self-difference** → the result is **empty**; replace with an empty
  container and **deinitialize the old container exactly once** (no double-free, no
  leak).

Tests must verify (a) **alias detection fires before any relocation**, and (b)
**correct empty-result ownership** for self-XOR/difference (exactly-once free),
under `DebugAllocator`.

## Target layout sketch (array; run analogous with the 4-byte `RunPair` + cached card)

```
offset 0:   Header { cardinality: u16, capacity: u16 }   padded to data alignment
offset A:   data...                                        (A = data alignment; see D5)
```

**D5 — per-type data alignment. DECIDED: array 16, run natural alignment, bitset 64
(if in scope).** The shipped 11-05/11-06 array kernels are **128-bit** (`@Vector(8,
u16)` = 16-byte loads), so **16-byte** data alignment is exactly what keeps those
loads aligned. An earlier draft said 32; that is over-aligned — 32 would only matter
for a 256-bit kernel, and there deliberately isn't one (a 256-bit AVX2 array-intersect
doesn't compact cleanly — `vpshufb` is lane-restricted — which is why the wide path is
AVX-512, out of scope in spec 11). Over-aligning wastes header padding per container,
which cuts against the allocation-size goal for no load-speed benefit. Run data has no
SIMD kernel → `@alignOf(RunPair)`. Bitset words → 64 (if in scope, per D1).

```zig
fn allocBlock(allocator: Allocator, cap: u16) ![]align(A) u8 {
    const bytes = std.math.add(usize, HEADER_SIZE,
        std.math.mul(usize, cap, @sizeOf(Elem)) catch return error.Overflow) catch return error.Overflow;
    return allocator.alignedAlloc(u8, .fromByteUnits(A), bytes);
    // Elem = u16 for array, RunPair (4 bytes) for run
}
```

**Overflow-safe arithmetic (required):** block size (`HEADER_SIZE + cap*@sizeOf(Elem)`)
and any capacity-growth doubling must use checked/saturating math (as in spec 12's
`ensureTotalCapacity`) — `cap` is bounded (≤ 4096 for arrays before promotion) but
the arithmetic must be total regardless of what a caller passes.

Growth: try `allocator.resize` first (in-place keeps the pointer — common with
power-of-two caps); on failure, alloc-new + memcpy + free-old, then update the slot
per D3. `clone` / `deinit` free/copy `HEADER_SIZE + cap*@sizeOf(Elem)` as one block,
not two. (The **serialized wire size** is a *separate* computation — it is the
portable-format byte length, unrelated to the allocation-block size, and is
unchanged by this spec.) `TaggedPtr` scheme unchanged — tag bits still discriminate
type; only what the pointer points at changes.

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

- **`DebugAllocator` must actually back the differential executables.** They do
  **not** use it today (unit `test`/`test64` use `std.testing.allocator`, but the
  `validate*`/`difftest*` exes use their own GPA/`ReleaseFast` config). This
  redesign's core risks — freeing the wrong block size, double-free on a moved block
  — are exactly what `DebugAllocator` catches, so wire it in: either build the
  `validate*`/`difftest*` modules with `DebugAllocator` as the harness allocator (a
  build flag/param), or add dedicated `DebugAllocator`-backed harness runs. Specify
  which in the implementation chunk; "green under DebugAllocator" is otherwise not a
  real check.
- **Aliasing tests** (D4): self-aliased grow/shrink under `DebugAllocator`.
- **Allocation-failure tests**: a failing allocator on the alloc-new-then-move path
  leaves the container and its slot unchanged and usable.
- **Allocation-count proof, not just time.** The existing allocator-matrix bench
  reports *time*, not allocation counts — it can't prove the "fewer allocations"
  claim. Add a **counting allocator** wrapper (tallies alloc/free calls + bytes) and
  run concrete workloads on **both trees**: build-N-containers, clone, and
  deserialize a large bitmap for 32-bit; the same over a many-bucket
  `Roaring64Bitmap`. Record before/after **allocation counts** (expect ~½ the
  per-container allocs) alongside the time deltas.

**Coordination with spec 11:** spec 11's SIMD kernels now compact through a local
stack scratch (not container over-allocation), so **this spec's layout carries no
SIMD headroom slack** — the two efforts are decoupled. If that scratch decision ever
changes, revisit the array block sizing here.

## Acceptance

- D1/D3/D4/D5 ratified; **D2 decided by `13-00` with numbers** before any layout
  chunk (13-01+) is written.
- Chosen ABI implemented for array (+ run; bitset per D1); growth **and shrink** both
  update slots; aliasing rule enforced and tested; overflow-safe block/growth math.
- Full call-site audit complete across both trees; differential suites green under a
  **`DebugAllocator`-backed** harness; allocation-failure and aliasing tests pass.
- Counting-allocator workloads show the **allocation-count reduction** (numbers, both
  trees) and a net time win on allocation-heavy scenarios — per the "own
  measurements" bar.

## Estimate

**L (multi-week), not M.** The earlier `M` was optimistic given: pointer-moving
APIs, direct `.values`/`.runs` field usage (D2), shrink-also-moves handling,
aliasing rules, and the dual 32/64-bit call-site audit. Scope depends heavily on
D2 (derived accessors → repo-wide refactor) and D1 (bitset in/out).
