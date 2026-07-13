# Spec 11: Array kernel parity + container layout (umbrella)

Perf work identified by kernel-level benchmarking against CRoaring.
**Behavior-preserving throughout** — public API, results, and wire format
unchanged; replacement kernels must be bit-identical to the current ones
(the reference bench cross-verifies kernel outputs — keep that check).
Internal container-API signatures do change in 11-04 (documented there).
Every task must pass the existing differential fuzz rig against CRoaring
(`zig build validate`, property tests) before merge.

## Background / evidence

Standalone kernel bench (x86-64 AVX2, uniform random u16 draws from the 64K
container domain, median of 9 trials, verified all kernels produce identical
output):

| scenario        | rawr-gallop | branchless merge | CRoaring vec16 (AVX2) | skewed gallop |
|-----------------|------------:|-----------------:|----------------------:|--------------:|
| 4096×4096 (1:1) |     43.2 µs |          25.3 µs |                 7.2 µs |       49.0 µs |
| 1024×1024 (1:1) |     10.5 µs |           6.3 µs |                 1.1 µs |       10.8 µs |
| 256×256 (1:1)   |      2.5 µs |           1.6 µs |                0.26 µs |        2.3 µs |
| 1024×4096 (1:4) |     16.6 µs |          15.9 µs |                 3.2 µs |       20.1 µs |
| 256×4096 (1:16) |      6.6 µs |          13.6 µs |                 2.2 µs |        7.2 µs |
| 64×4096 (1:64)  |      2.6 µs |          13.0 µs |                 2.0 µs |        2.2 µs |
| 16×4096 (1:256) |     0.67 µs |          12.2 µs |                 1.8 µs |       0.33 µs |

Findings, confirmed against the vendored CRoaring source
(`vendor/roaring.c:6930`, `array_container_intersection`):

1. CRoaring **never gallops unconditionally**. Dispatch is: gallop only when
   `card_small * 64 < card_big`; otherwise AVX2 `intersect_vector16` on x86,
   scalar merge elsewhere. rawr made gallop the only path
   (`container_ops.zig:707`), which loses ~1.7× to a plain branchless merge at
   balanced ratios **on every architecture**, and 5–10× to vec16 on x86.
2. CRoaring's kernel algorithms (shuffle-table compaction, threshold=64
   dispatch) are the reference to port — no better-known alternative exists.
   Its memory layout is not: CRoaring, like rawr, is two mallocs per array
   container (`array_container_create_given_capacity`); 11-04 does better.
3. CRoaring's non-x64 balanced path is **plain scalar** (`#else` branch). No
   implementation has a NEON array kernel. Task 11-06 would put rawr ahead of
   CRoaring on all ARM targets.

## Sequencing & strategy (2026-07)

Written after roaring64 (spec 10) reached full CRoaring parity and was validated.
Recommended order, with rationale:

0. **Spec 12 (capacity API) and the roaring64 Phase-3 tail come first / in
   parallel** — both are small, independent of the container ABI, and don't touch
   these kernels. Not blockers either way.

1. **Portable tier first — 11-01, 11-02, 11-03, 11-07.** Afternoon-each, all
   architectures, near-zero risk (the differential fuzz is the correctness net),
   and they fix a *measured, user-facing regression on a hot path* (array∩array is
   gallop-only today — verified at `container_ops.zig:707` — losing ~1.7× at
   balanced ratios everywhere). **11-07 is not optional glue: without a balanced-
   array corpus these wins are invisible** — the current board showed ~1.03×
   "parity" while the kernel was 6× slower. Land 11-07 alongside 11-01 so the gain
   shows. This tier alone takes rawr from gallop-only to competitive-at-balanced
   with **zero SIMD**.

2. **11-04 (single-alloc containers) while the tree is stable.** It changes the
   container ABI (growth can move `*Self`), so its call-site audit spans the 32-bit
   *and* 64-bit trees. roaring64 already landed on the two-alloc layout, but the
   remaining roaring64 tail (10-19/20/21) and spec 12 add **no new grow-capable
   container sites**, so they don't widen the audit. Parity being the last 64-bit
   feature means the audit surface is now **stable and won't grow** — so this is the
   right window to do 11-04, before any future feature work reopens it. Test hard
   under `DebugAllocator` (catches free-size mismatches). The 64-bit tree amplifies
   the payoff (many small containers under high-32 keys).

3. **SIMD tier last — 11-05 (AVX2), 11-06 (NEON) — as a deliberate, heavily-fuzzed
   opt-in, intersect-first.** These buy the remaining 4–6× on x86/ARM but are a
   different risk class: hand-written per-arch kernels with inline `pshufb`/`tbl`,
   the output-headroom and in-place-trampling footguns, harder to maintain than
   portable Zig — a real tradeoff against the project's "idiomatic Zig, clean API"
   goal. Decide them explicitly, not by momentum. 11-05 and 11-06 share the shuffle
   table and compare-any skeleton; build together, structured generic over
   `(shuffle, movemask)` per the 11-06 notes. Scope down to intersect alone first —
   it captures most of the measured gap.

**Bench coordination:** roaring64's `10-21 bench64` and this spec's `11-07` are the
same kind of work ("make the benchmark reveal the real regression"). Do them with
shared thinking — both must include many-bucket / balanced-input cases, or they'll
be as blind as the current board was.

**Framing:** per project stance this is *fixing our own regression + idiomatic
perf*, not competing with CRoaring. The one place rawr genuinely surpasses CRoaring
here is 11-04's layout (CRoaring has the same two-malloc flaw); the kernels are
ports of known-best algorithms, not a race.

---

## 11-01 — Ratio dispatch for array∩array (portable)

**File:** `src/container_ops.zig` — `arrayIntersectArray` (:707),
`arrayIntersectArrayCard` (:889), `arrayIntersectsArray` (:976).

Replace unconditional gallop with CRoaring's three-way dispatch:

```zig
const SKEW_THRESHOLD = 64; // CRoaring's value; "subject to tuning"

if (@as(u32, small.len) * SKEW_THRESHOLD < big.len) {
    // existing gallop path (rename current body to intersectSkewed)
} else {
    // branchless merge — same shape as arrayDifferenceArray (:1073),
    // which already does this correctly:
    while (i < sa.len and j < sb.len) {
        const av = sa[i]; const bv = sb[j];
        out[k] = av;
        k += @intFromBool(av == bv);
        i += @intFromBool(av <= bv);
        j += @intFromBool(bv <= av);
    }
}
```

Notes:
- `arrayDifferenceArray` and `arrayUnionArray` already use branchless merge;
  after this task the intersect family is consistent with them. Optionally add
  the symmetric skew check to difference (CRoaring has
  `difference_skewed`-style dispatch too) — low value, implementor's call,
  measure first.
- The cardinality-only variant (`arrayIntersectArrayCard`) and the boolean
  early-exit variant (`arrayIntersectsArray`) get the same dispatch. For the
  boolean variant, gallop remains correct at *any* skew because it can
  early-exit; keep threshold dispatch anyway for the balanced case.
- When 11-05 lands, the balanced branch becomes a third arm:
  `if (comptime has_avx2) vec16 else merge`.

**Acceptance:** differential fuzz green; kernel microbench shows ≥1.5× at
1024×1024 and no regression ≥1:64 skew.

## 11-02 — Branchless array→bitset conversion loops (portable)

**Files:** `src/container_ops.zig` — `arrayXorArray` overflow path (:1315),
`arrayUnionArray` overflow path (:541), `bitsetUnionArrayInPlace` (:417),
`bitsetXorArrayInPlace` (:489, audit), and any other `for (values) |v| _ =
bc.add(v)` / `contains-then-remove-else-add` loop (grep for `bc.add(v)`).

Root cause: `BitsetContainer.add` carries a bounds shift, a `was_absent`
test, and a `cardinality >= 0` branch **per element**. The XOR path is worse:
`contains` → branch → `remove`/`add`. These loops run during every
array-overflow conversion inside union/xor.

Replacement pattern — raw word ops + single repair, using the lazy-cardinality
machinery from 07-06:

```zig
// union: array elements into bitset words
for (ac.values[0..ac.cardinality]) |v|
    bc.words[v >> 6] |= @as(u64, 1) << @truncate(v);
bc.invalidateCardinality();

// xor: toggling needs no membership test at all
for (values) |v|
    bc.words[v >> 6] ^= @as(u64, 1) << @truncate(v);
bc.invalidateCardinality();
```

Then a single `computeCardinality()` (already vectorized via `countWords`)
where the caller needs the count for the array-demotion decision. Where the
caller is a lazy op, leave `-1` and let repair handle it — do not compute
eagerly.

**Semantic hazard — check before merging:** these loops leave
`bc.cardinality == -1` transiently where it was previously always valid.
That is only safe if every reader on the affected paths goes through the
`cardinality()` accessor (which computes on `-1`) rather than reading the
field directly. Grep for direct `\.cardinality` field reads on
`BitsetContainer` reachable from the converted call sites — in particular the
array-demotion comparisons (`<= MAX_CARDINALITY`) and serialization sizing —
and route any direct read through the accessor or an explicit
`computeCardinality()` first. Public semantics are unchanged; this is purely
an internal invariant shift.

Optional (measure): CRoaring's `bitset_set_list` avoids the read-modify-write
dependency chain by tracking the previous word; skip unless the plain loop
disappoints — on modern OoO cores it usually doesn't.

**Acceptance:** differential fuzz green; microbench a 4096-element array→bitset
union conversion, expect ≥2×.

## 11-03 — `findKey` (portable, minor)

**File:** `src/bitmap.zig:185`.

Current three-way branchy binary search. Replace body with:

```zig
const idx = self.lowerBound(key);           // already branchless-friendly
if (idx < self.size and self.keys[idx] == key) return idx;
return null;
```

Additionally, add a linear-scan fast path for small key arrays — most real
bitmaps have few containers and a predictable forward scan beats binary search:

```zig
if (self.size <= 32) {
    for (self.keys[0..self.size], 0..) |k, idx| {
        if (k == key) return idx;
        if (k > key) return null;
    }
    return null;
}
```

Tune the 32 cutoff on the compare bench. Keep `lowerBound` itself as the
insert-point primitive.

**Acceptance:** no functional change; `contains`-heavy board scenario neutral
or better.

## 11-04 — Single-allocation containers (ABI change; land before roaring64)

**Files:** `src/array_container.zig`, `src/run_container.zig` (same pattern),
`src/bitmap.zig` call-site audit. Bitsets are fixed-size and can be
single-alloc trivially (header + 8 KB words in one block).

Current: `ArrayContainer.init` = `allocator.create(Self)` **plus**
`alignedAlloc` for values — two allocations, two pointer chases, header and
data on different cache lines. CRoaring has the same flaw; co-locating the
header with the data in one aligned block removes an allocation and a pointer
chase per container and is the one structural improvement in this spec that
goes beyond CRoaring rather than matching it.

Target layout — one allocation, header co-located, data offset to SIMD
alignment:

```
offset 0:   Header { cardinality: u16, capacity: u16 }   (array)
            Header { n_runs: u16, capacity: u16 }        (run)
offset 32:  data...                                       (32-byte aligned)
```

```zig
const HEADER_SIZE = 32; // = data alignment; header padded to it

fn allocBlock(allocator: Allocator, cap: u16) ![]align(32) u8 {
    return allocator.alignedAlloc(u8, .fromByteUnits(32),
        HEADER_SIZE + @as(usize, cap) * @sizeOf(u16));
}
// values pointer derived, never stored:
inline fn values(self: *Self) [*]align(32) u16 {
    return @ptrCast(@alignCast(@as([*]u8, @ptrCast(self)) + HEADER_SIZE));
}
```

**The footgun (spell this out in code comments):** growth reallocates the
*whole block including the header*, so `*ArrayContainer` itself may move.
Every mutation that can grow must be able to update the container slot in the
bitmap:

1. Try `allocator.resize` first (in-place growth keeps the pointer — the
   common case with power-of-two capacities and most allocators).
2. On resize failure, alloc-new + memcpy + free-old and **return the possibly
   new pointer**. Change grow-capable container APIs to return `*Self` (or
   have them take a `*TaggedPtr` slot to update). Known grow-capable entry
   points to convert — treat this list as a starting point, not exhaustive;
   verify by grepping for `ensureCapacity` callers:
   - `ArrayContainer.add` (via `ensureCapacity`), `ensureCapacity` itself
   - `RunContainer` append/insert paths (same `ensureCapacity` pattern)
   - anything constructing then growing a result container inside
     `container_ops.zig`
   Audit all call sites, **including the roaring64 tree (spec 10)**: the
   bitmap-level `addToContainer` path already holds the slot index, so the
   plumbing is local there; the 64-bit layer's equivalent slot-holding path
   needs the same treatment.
3. `clone`, `deinit`, serialization sizing: free/copy `HEADER_SIZE + cap*2`
   bytes, not two blocks.

`TaggedPtr` scheme is unchanged — tag bits still discriminate type; only what
the pointer points at changes.

Sequencing: spec 10 (roaring64) has landed on the two-alloc layout, so the
call-site audit includes the 64-bit tree — which also multiplies container
counts (high-48-bit keys over small containers), meaning the per-container
allocation savings are amplified there. Do this before further specs grow the
audit surface.

**Acceptance:** differential fuzz + property tests green under
`DebugAllocator` (catches the free-size mismatches); allocator-matrix bench
rerun — expect visible gains on create/clone/deserialize-heavy scenarios given
the documented 10–40× allocator sensitivity.

## 11-05 — AVX2 `vector16` array kernels (x86)

**New file suggested:** `src/array_simd.zig`. Wire into the balanced branch of
11-01's dispatch for intersect, xor, and (optionally, measure) union.
Reference implementation: CRoaring `array_util.c` (`intersect_vector16`,
`union_vector16`, `xor_vector16`) — the vendored copy in `vendor/roaring.c`
contains all of it (search those names in the amalgamation).

### Gating

```zig
const HAS_AVX2 = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);
```

Comptime gating is correct for rawr: builds are per-target across the
Windows/BSD/ARM matrix already; runtime dispatch (CRoaring-style) is out of
scope. Baseline x86-64 builds without `-Dcpu` including avx2 simply keep the
merge path.

### Shuffle table (shared with 11-06)

256 entries × 16 bytes = 4 KB, comptime-generated:

```zig
fn genEntry(comptime mask: u8) [16]u8 {
    var e: [16]u8 = @splat(0xFF);
    var out: usize = 0;
    inline for (0..8) |i| {
        if (mask & (1 << i) != 0) {
            e[out * 2] = @intCast(i * 2);       // low byte of u16 lane i
            e[out * 2 + 1] = @intCast(i * 2 + 1);
            out += 1;
        }
    }
    return e;
}
pub const shuffle_mask16: [256][16]u8 = blk: {
    var t: [256][16]u8 = undefined;
    for (&t, 0..) |*e, i| e.* = genEntry(@intCast(i));
    break :blk t;
};
```

### Intersect kernel shape

Process both inputs in 8×u16 blocks; scalar tail below 8:

```zig
while (ia < enda and ib < endb) {
    const va: @Vector(8, u16) = A[ia..][0..8].*;
    const vb: @Vector(8, u16) = B[ib..][0..8].*;

    // compare-any: does each lane of va match ANY lane of vb?
    var matches: @Vector(8, bool) = @splat(false);
    inline for (0..8) |i| {
        const bcast: @Vector(8, u16) = @splat(vb[i]);
        matches = matches | (va == bcast);      // 8 vpcmpeqw + vpor
    }
    const mask: u8 = @bitCast(matches);         // vpmovmskb-equivalent

    if (mask != 0) {
        // pshufb-compact matching lanes of va to the front
        const packed_: @Vector(8, u16) = @bitCast(pshufb(
            @as(@Vector(16, u8), @bitCast(va)), shuffle_mask16[mask]));
        // store full 8-lane vector; only popCount(mask) lanes are valid
        out[k..][0..8].* = packed_;             // REQUIRES headroom, see below
        k += @popCount(mask);
    }

    // advance: whichever block's max is smaller; both if equal
    const amax = A[ia + 7]; const bmax = B[ib + 7];
    if (amax <= bmax) ia += 8;
    if (bmax <= amax) ib += 8;
}
// scalar merge for tails
```

`pshufb` on x86: LLVM reliably lowers a runtime-index byte lookup expressed
via `@shuffle`-free code only through the intrinsic path — use inline asm or
verify codegen if using a generic formulation. Simplest reliable form:

```zig
inline fn pshufb(v: @Vector(16, u8), m: [16]u8) @Vector(16, u8) {
    const mv: @Vector(16, u8) = m;
    return asm ("vpshufb %[m], %[v], %[out]"
        : [out] "=x" (-> @Vector(16, u8)),
        : [v] "x" (v), [m] "x" (mv));
}
```

(A non-asm formulation — building the shuffled vector through a generic
per-byte gather and trusting LLVM to pattern-match it — may also lower to
`vpshufb`, but this is backend-sensitive; if going that route, verify with
`zig build-obj -femit-asm` that an actual `vpshufb` is emitted, and keep the
asm version as the fallback.)

### Output headroom — hard requirement

The kernel stores 8 lanes even when fewer match. **Every destination buffer
passed to a vector16 kernel must have `min(cardA, cardB) + 8` u16 capacity**
(CRoaring grows by `sizeof(__m128i)/sizeof(u16)`). Enforce at the container
level: the intersect result container allocation in `arrayIntersectArray`
must over-allocate by 8 when the vec16 path is compiled in. Under 11-04's
single-alloc layout this is 16 bytes of slack — fold it into the capacity
rounding. Add a debug assert in the kernel: `out.len >= min_card + 8`.

In-place variants additionally need a 16-slot `u16` scratch to avoid
trampling the input (an 8-wide store at offset can span 16 slots): buffer
the compacted block in the scratch, then copy `@popCount(mask)` lanes out —
the write cursor trails the read cursor by at least one block, so a bounded
scratch suffices. Or skip in-place vec16 initially and let in-place ops route
through the out-of-place kernel + copy.

### XOR / union kernels

Same block structure with a sorted-merge network instead of compare-any: 8-lane
merge via min/max + rotate steps, dedup against previous max, shuffle-compact
by a "keep" mask (union keeps all unique; xor keeps lanes appearing exactly
once). Port directly from CRoaring `union_vector16` / `xor_vector16` — do not
re-derive. If scoping down, intersect alone captures most of the measured gap;
xor/union vec16 are follow-ups behind the same dispatch.

**Acceptance:** differential fuzz green (this rig is the whole safety story
for a hand-SIMD kernel — run it long, ReleaseSafe and ReleaseFast); kernel
bench ≥4× over merge at 1024×1024; board scenario from 11-07 moves.

## 11-06 — NEON `tbl` array kernels (aarch64)

Same algorithm and same 4 KB table as 11-05; only two primitives differ. No
existing roaring implementation has this (CRoaring's non-x64 balanced path is
scalar) — landing it makes rawr the fastest array∩array on ARM, including
Apple Silicon, ARM BSDs, and Windows-on-ARM.

Gate: `builtin.cpu.arch == .aarch64` (NEON is baseline; no feature check).

**Primitive 1 — dynamic byte shuffle** (`vqtbl1q_u8`). Zig's `@shuffle`
requires a comptime mask, so inline asm:

```zig
inline fn tbl(v: @Vector(16, u8), m: @Vector(16, u8)) @Vector(16, u8) {
    return asm ("tbl %[out].16b, { %[t].16b }, %[m].16b"
        : [out] "=w" (-> @Vector(16, u8)),
        : [t] "w" (v), [m] "w" (m));
}
```

Out-of-range indices (the 0xFF padding in the table) yield 0 on `tbl` — fine,
those lanes are past `@popCount(mask)` and never counted.

**Primitive 2 — movemask.** NEON has no `pmovmskb`. Options, in order of
preference:
1. Write `const mask: u8 = @bitCast(matches);` and inspect codegen — recent
   LLVM lowers i1×8 bitcasts on aarch64 acceptably (shift-and-accumulate or
   `addv`).
2. If codegen is poor: the `shrn` trick — narrow the 8×u16 compare result
   (0xFFFF/0x0000 per lane) with `shrn.8b v, v, #4`, move to GPR, extract
   nibble-spaced bits. Implement behind the same `inline fn movemask8` seam so
   it's swappable.

Everything else — block loop, advance logic, headroom rule, scalar tails,
table — is shared code with 11-05. Structure `array_simd.zig` so the kernel is
generic over `(pshufb_or_tbl, movemask)` and instantiated per-arch at comptime.

**Acceptance:** differential fuzz on an aarch64 host (Apple Silicon dev box
qualifies); kernel bench vs branchless merge on the same host — expect
2–4× at balanced ratios (predicted, not yet measured; the bench harness
below produces the number). Also record gallop-vs-merge crossover on aarch64
since csel codegen may shift it from the x86 value.

## 11-07 — Bench corpus: balanced array∩array scenario

**File:** `src/bench_croaring.zig`.

The existing board reported ~1.03× parity against AVX2 CRoaring while the
array∩array kernel was 6× slower — meaning the corpus rarely exercises
balanced array pairs and would not show 11-01/11-05 landing. Add scenarios:

- two bitmaps, each ~200 containers, all arrays of cardinality 1024–4096,
  ≥80% key overlap; ops: `and`, `andCardinality`, `xor`;
- a skewed variant (16–64 vs 4096) to guard the gallop path against
  regression;
- keep the standalone kernel bench (`bench-aa.zig`, provided separately) in
  tree for per-kernel numbers isolated from allocation and container-walk
  noise.

Expected board movement after 11-01 alone: balanced-array `and` improves
~1.6–1.7×; after 11-05 on x86: 4–6× total; skewed scenario within noise
throughout.

---

## Out of scope

- Runtime CPU dispatch (per-target builds cover the port matrix).
- AVX-512 kernels (`vp2intersect` etc.) — revisit after 11-05, and note the
  known LLVM feature-detection issues that motivated disabling CRoaring's
  AVX512 in the compare bench.
- roaring64 feature work — separate track (spec 10); 11-04 touches its
  container handling but adds no 64-bit features.
- `SKEW_THRESHOLD` tuning beyond confirming 64 is sane on the board.

## Task order and estimates

| task | deps | size | platforms |
|------|------|------|-----------|
| 11-01 dispatch | — | S | all |
| 11-02 conversion loops | — | S | all |
| 11-03 findKey | — | XS | all |
| 11-07 corpus | — | S | all |
| 11-04 single-alloc | — | M (call-site audit incl. roaring64 tree) | all |
| 11-05 AVX2 vec16 | 11-01, 11-07 | M | x86-64 |
| 11-06 NEON tbl | 11-05 (shared skeleton) | S on top of 11-05 | aarch64 |
