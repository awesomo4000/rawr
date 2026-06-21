# Spec: Differential Test Suite for `rawr` (CRoaring Parity)

## Goal

Bring `rawr`'s correctness testing to parity with CRoaring's test suite by
**differentially testing every operation against CRoaring as an oracle**, over
inputs that **deliberately exercise all three container types and all
cross-container type pairs**.

The current suite proves the array-container path and the wire format are
correct. It does **not** exercise bitset/run operands of set operations, container type transitions under operations, the mixed empty/survive container case, or in-place/allocating equivalence for AND/XOR/ANDNOT. This spec closes those gaps.

Out of scope for now: AFL/libFuzzer continuous fuzzing, untrusted/malformed input hardening (`deserialize_safe`). A single optional task at the end seeds malformed-input testing cheaply, but it is not required for parity.

## Definitions

- **Oracle**: CRoaring, via the translated `vendor/croaring_wrapper.h` binding. Extend the wrapper header as needed (see Task 0).
- **Differential check**: perform the same operation in `rawr` and CRoaring on equal inputs, then assert the results agree on **all** of:
  1. byte-identical portable serialization (`serialize` vs `roaring_bitmap_portable_serialize`),
  2. equal cardinality,
  3. equal membership for a sampled set of probe values,
  4. CRoaring can deserialize rawr's bytes and vice versa (round-trip). Byte-identity is the strongest signal and subsumes most of the rest, but keep the cardinality/membership asserts because they localize failures faster than a byte diff.

## Existing surface (do not re-derive — use these exact names)

rawr public API:
`init, deinit, clone, add, addRange, remove, contains, cardinality, isEmpty,
minimum, maximum, bitwiseOr, bitwiseAnd, bitwiseXor, bitwiseDifference,
bitwiseOrInPlace, bitwiseAndInPlace, bitwiseXorInPlace,
bitwiseDifferenceInPlace, andCardinality, intersects, runOptimize, isSubsetOf,
equals, iterator, serialize, deserialize, serializedSizeInBytes, fromSorted,
fromSlice`.

CRoaring oracle (in wrapper, extend if missing):
`roaring_bitmap_{create,free,copy,add,add_range,contains,get_cardinality,
is_empty,and,or,xor,andnot,and_inplace,or_inplace,xor_inplace,andnot_inplace,
run_optimize,portable_serialize,portable_size_in_bytes,
portable_deserialize_safe,equals,is_subset,minimum,maximum}`.

Mapping note: rawr `bitwiseDifference` ⇄ CRoaring `andnot`.
rawr `addRange(start, end)` is **inclusive** on both ends; CRoaring
`roaring_bitmap_add_range(min, max)` is **exclusive** on `max`. Always pass `@as(u64, end) + 1` to CRoaring. (This convention is already handled correctly in `validate_croaring.zig`; preserve it.)

---

## Task 0 — Extend the CRoaring wrapper

`vendor/croaring_wrapper.h` already exposes creation, basic ops, all four set
ops + their in-place forms, `run_optimize`, and portable serialization. The
differential harness additionally needs the following, which are **currently
missing** from the wrapper and must be added:

```c
bool roaring_bitmap_equals(const roaring_bitmap_t*, const roaring_bitmap_t*);
bool roaring_bitmap_is_subset(const roaring_bitmap_t*, const roaring_bitmap_t*);
uint64_t roaring_bitmap_and_cardinality(const roaring_bitmap_t*, const roaring_bitmap_t*);
bool roaring_bitmap_intersect(const roaring_bitmap_t*, const roaring_bitmap_t*);
uint32_t roaring_bitmap_minimum(const roaring_bitmap_t*);
uint32_t roaring_bitmap_maximum(const roaring_bitmap_t*);
```

(`roaring_bitmap_andnot` and `roaring_bitmap_andnot_inplace` are already present
— do not re-add them.) After editing the header, confirm the `translate-c` step
picks them up: they appear as `c.roaring_bitmap_*` in both `validate_croaring.zig`
and the new `diff_test.zig` without any other build change, because the existing
`b.addTranslateC` already points at this header.

Do **not** vendor the whole CRoaring header; keep the wrapper minimal as it is
now. This stays within the spirit of the 0.16 `00-04` interop chunk, which
asked to keep `croaring_wrapper.h` the single imported header and avoid widening
the C surface area beyond what's needed.

---

## Task 1 — The container-type-aware generator (the core of this spec)

Create `src/test_gen.zig`. This is the single most important piece. The current generator (`property_tests.zig: randomBitmap`) draws ~100 uniform-random `u32`s, which almost always yields only **array** containers. The new generator must be able to force **each** container type and **mixtures** within one bitmap.

### Required building blocks

A bitmap is built per-chunk (a chunk = one 16-bit high key = 65536 contiguous values). The generator picks, for each populated chunk, a **container profile**. The container type a profile yields **depends on `run_optimize`** — be explicit about this in tests so they assert the right type:

| Profile        | How to fill the chunk                                              | Type w/o `runOptimize` | Type after `runOptimize` |
|----------------|-------------------------------------------------------------------|------------------------|--------------------------|
| `sparse`       | N random low-16 values, N < 4096                                  | array                  | array (unchanged)        |
| `dense`        | N random low-16 values, N > 4096 (e.g. 4097–60000)               | bitset                 | bitset (unless runs win) |
| `full`         | all 65536 values in the chunk (build via `addRange`, see below)   | bitset                 | run                      |
| `runs`         | K disjoint consecutive ranges (e.g. 3–10 runs of length 50–2000) | array or bitset (by count) | run                  |
| `single`       | one value                                                         | array (size 1)         | array (unchanged)        |
| `boundary`     | low-16 values {0, 1, 65534, 65535}                               | array, edge offsets    | array (unchanged)        |

**Important:** with `run_optimize = false`, `runs` and `full` do **not** produce run containers — `runs` is whatever its raw cardinality dictates (array if < 4096, else bitset), and `full` is a bitset. Run containers only exist after `runOptimize`. Chunk specs must state the expected container type **for the optimize setting they test under** so they don't assert the wrong thing.

The generator API (suggested):

```zig
pub const Profile = enum { sparse, dense, full, runs, single, boundary };

/// Build a bitmap with explicitly chosen container profiles per chunk.
/// `chunks` maps a high-key -> profile. Returns a rawr bitmap AND the sorted
/// unique value list used to build it (so the oracle can be built identically
/// and so membership probes know ground truth).
pub fn build(
    allocator: Allocator,
    rng: std.Random,
    chunks: []const struct { key: u16, profile: Profile },
    run_optimize: bool,
) !struct { bm: RoaringBitmap, values: []u32 };

/// Fully random bitmap: random number of chunks (0..a few hundred), each with a
/// randomly chosen profile. Used by the randomized differential loop.
pub fn randomMixed(allocator: Allocator, rng: std.Random, max_chunks: usize, run_optimize: bool)
    !struct { bm: RoaringBitmap, values: []u32 };
```

Note: `buildOracle` (constructing the CRoaring oracle from a `values` slice) is
**not** part of `test_gen.zig` — it needs the `c` import and therefore lives in
`diff_test.zig`. `test_gen.zig` only ever produces rawr bitmaps plus the
ground-truth `values` slice; the harness builds the oracle from that slice. See
Build wiring.

### Hard requirements on the generator

1. It must be possible to produce a bitmap where **chunk 0 is an array, chunk 1 is a bitset, chunk 2 is a run** in a single bitmap. This is what makes the container-dispatch loop see heterogeneous operands.
2. `randomMixed` must, over a run of a few hundred iterations, produce **every profile** and **every adjacent profile pairing** with high probability. (Don't leave it to chance on a uniform distribution — weight `dense`/`runs`/`full` up so they actually appear; uniform-random would bury them.)
3. The returned `values` slice is the ground-truth oracle input. Build the CRoaring bitmap from the identical slice so the two are guaranteed to start equal, independent of how rawr chose to lay out containers.
4. Always offer a `run_optimize` toggle; run containers only exist after `runOptimize`, so half the matrix must run with it on.
5. **Build `full` (and large `runs`) compactly** via `addRange`, not by adding 65536 individual values — the latter is slow and pointless. Note the cost tradeoff: the ground-truth `values` slice still has to enumerate every value (the oracle is built from it), so a `full` chunk contributes 65536 `u32`s to that slice. Across many `full`/`dense` chunks the slice gets large fast; keep `max_chunks` and the per-profile sizes modest in the randomized loop (see Acceptance criteria) so the harness stays fast. An optional future optimization is to let the oracle also build via `roaring_bitmap_add_range` for `full`/`runs` chunks instead of value-by-value, but value-by-value from the slice is fine to start.

### Generator self-test (ships with this piece)

The generator must come with a small self-test (unit tests, pure rawr — no oracle) proving it can **force each container type**: build a single bitmap with chunk 0 = `sparse`, chunk 1 = `dense`, chunk 2 = `runs` (run-optimized), chunk 3 = `full`, and assert each chunk's container is the expected type (array / bitset / run / run) by inspecting the container tag. This is the proof that the generator does its one job before anything depends on it.

---

## Task 2 — The differential operation harness

Create `src/diff_test.zig` built as a new `zig build difftest` step (mirror the existing `validate` step in `build.zig`). It is an executable (like `validate_croaring.zig`), not a unit test, because it links CRoaring.

### Core comparator

```zig
/// The single assertion used everywhere. Compares a rawr result bitmap against
/// a CRoaring result bitmap on all four axes. Frees nothing (caller owns).
fn assertAgree(name: []const u8, rawr_bm: *RoaringBitmap, oracle: *c.roaring_bitmap_t) !void {
    // 1. cardinality
    // 2. byte-identical portable serialization
    // 3. membership probes: every value in rawr's iterator is in oracle,
    //    and a sample of known-absent values is absent in both
    // 4. cross-deserialize both directions, re-check cardinality + equals
}
```

Reuse the byte-diff reporting already in `validate_croaring.zig` (`validateRoundTrip`) — first-divergence-byte printout is good, keep it.

### The operation matrix

For each operation, run it in rawr and in CRoaring on the **same** pair `(A, B)` and call `assertAgree` on the result. Operations:

| rawr op                     | CRoaring op                       |
|-----------------------------|-----------------------------------|
| `bitwiseOr`                 | `roaring_bitmap_or`               |
| `bitwiseAnd`                | `roaring_bitmap_and`              |
| `bitwiseXor`                | `roaring_bitmap_xor`              |
| `bitwiseDifference`         | `roaring_bitmap_andnot`           |
| `bitwiseOrInPlace`          | `roaring_bitmap_or_inplace`       |
| `bitwiseAndInPlace`         | `roaring_bitmap_and_inplace`      |
| `bitwiseXorInPlace`         | `roaring_bitmap_xor_inplace`      |
| `bitwiseDifferenceInPlace`  | `roaring_bitmap_andnot_inplace`   |

For in-place ops: clone A first (`A.clone()` / `roaring_bitmap_copy`), mutate the clone, then `assertAgree`. **Also** assert the in-place result equals the allocating result (`bitwiseXorInPlace(clone, B)` ⇄ `bitwiseXor(A, B)`). This catches the class of bug where the allocating path is correct but the in-place
path diverges — currently only OR has this cross-check.

Non-producing predicates (compare scalar/bool directly, no `assertAgree`):

| rawr                | CRoaring                         |
|---------------------|----------------------------------|
| `andCardinality`    | `roaring_bitmap_and_cardinality` |
| `intersects`        | `roaring_bitmap_intersect`       |
| `isSubsetOf`        | `roaring_bitmap_is_subset`       |
| `equals`            | `roaring_bitmap_equals`          |
| `cardinality`       | `roaring_bitmap_get_cardinality` |
| `minimum`/`maximum` | `roaring_bitmap_minimum/maximum` |

### Required explicit pairings (the 9-pair matrix)

Beyond randomized testing, write **deterministic** cases that force each operand type pairing, because randomized runs can under-sample specific pairs. For every operation in the matrix above, run it on at least these `(A-profile, B-profile)` pairings, in the **same chunk** (so the containers actually meet) and run with both `run_optimize` off and on:

```
(sparse,  sparse)   -> array  X array
(sparse,  dense)    -> array  X bitset
(dense,   sparse)   -> bitset X array     (asymmetric — test both orders!)
(dense,   dense)    -> bitset X bitset
(sparse,  runs)     -> array  X run
(runs,    sparse)   -> run    X array
(dense,   runs)     -> bitset X run
(runs,    dense)    -> run    X bitset
(runs,    runs)     -> run    X run
(full,    sparse)   -> full-chunk edge
(X,       empty)    -> every op against an empty operand
(empty,   X)        -> and empty on the left
```

Order matters for ANDNOT/difference (non-commutative) — always test both `A op B` and `B op A` for those. AND/OR/XOR are commutative; still test both orders once to catch order-dependent container-allocation bugs.

### Container-transition cases (explicitly)

These target the logic in `optimize.zig` / container promotion-demotion under operations, which is currently untested:

1. **Promotion**: two `sparse` arrays in the same chunk whose union exceeds 4096 → result must become a bitset. `OR` and verify against oracle.
2. **Demotion**: two `dense` bitsets whose **intersection** drops below 4096 → result should become an array. `AND` and verify. (Construct B to share few elements with A.)
3. **Empty-out-one-of-many**: A has containers in chunks {0,1,2}; B equals A's chunk 1 exactly. `A andnot B` must **drop** chunk 1 entirely while chunks 0 and 2 survive. Assert the result has exactly 2 containers and byte-matches oracle. This is the ghost-container invariant — the highest-value missing
   case.
4. **Run boundary**: a `full` chunk (all 65536) `andnot` a single value → verify the run splits correctly. Run-optimize on.

---

## Task 3 — Property tests over the mixed generator, plus oracle-anchored identities

This splits into two parts because of a build constraint: `property_tests.zig`
runs inside `zig build test` (pulled in via `src/roaring.zig`) and that build
**does not link CRoaring**. So the oracle cannot be called from
`property_tests.zig`. Do not try to import `c` there.

**Part A — in `property_tests.zig` (pure rawr, no oracle):**

1. Swap `randomBitmap` for `test_gen.randomMixed` so the existing algebraic
   identities (commutativity, associativity, distributivity, De Morgan, xor
   decomposition, absorption, etc.) run over bitset/run containers instead of
   only arrays. This alone is a large coverage win — the identities currently
   never see a non-array container.
2. Increase iteration counts modestly (e.g. 50 → 200) now that each iteration
   covers more container variety. No AFL-scale volume.

   Note: `test_gen.zig` stays pure rawr, so it imports cleanly into the unit-test
   build. Keep it that way — no `c` import.

**Part B — in `diff_test.zig` (oracle available):**

3. Add oracle-anchored versions of the most bug-revealing identities. The
   blind spot in Part A is that an identity computed two ways in rawr
   (`A∩(B∪C)` vs `(A∩B)∪(A∩C)`) can pass even if **both** sides share the
   same bug. Anchoring at least one side to CRoaring removes that blind spot.
   For each anchored identity, compute the left side in rawr and assert it
   byte-matches the CRoaring result of the equivalent operation via
   `assertAgree`. A handful of the highest-value identities is enough
   (distributivity, xor decomposition); they don't all need anchoring since the
   Task 2 matrix already pins every individual op to the oracle.

The division of labor: Part A proves rawr is *internally consistent* over all
container types; the Task 2 matrix + Part B prove rawr *agrees with CRoaring*.

---

## Task 4 — Round-trip and addRange coverage (extend existing)

`validate_croaring.zig` already does build→serialize→cross-deserialize well. Extend its fixture list to drive the generator profiles rather than hand-rolled arrays:

- Round-trip a `randomMixed` bitmap for each profile combination, both run-optimized and not.
- Keep the existing hand-picked boundary fixtures (chunk boundaries, NO_OFFSET_THRESHOLD container counts 3/4/5) — those are valuable and targeted.
- Add `addRange` differential cases that span container types: a range that produces a run, a range >4096 that produces a bitset, a range crossing several chunk boundaries. Compare bytes against `roaring_bitmap_add_range` (remember the +1 exclusive-end convention).

---

## Task 5 (optional, low effort, high surprise value) — malformed input smoke test

Untrusted input is out of scope as a feature, but a tiny corruption sweep is a cheap way to surface real bugs (panics, OOB reads, infinite loops) in `deserialize`:

1. Serialize a known-good mixed bitmap to bytes.
2. In a loop, copy the bytes and corrupt them: flip a random byte; truncate to a random length; zero the cardinality field; set a container's cardinality to `0xFFFF`.
3. Call `RoaringBitmap.deserialize` on each corrupted buffer. The **only** acceptable outcomes are: returns a valid bitmap, or returns a Zig error. A crash/panic/hang is a finding.

This is ~40 lines, needs no CRoaring, and tends to expose missing bounds checks fast. Treat any non-error crash as a bug to file, not necessarily to fix now.

---

## Build wiring

The 0.16 upgrade already converted CRoaring interop to build-system `translate-c`
(`b.addTranslateC`), imported as `const c = @import("c");`. **There is no
legacy C import builtin anymore** — use the existing `addTranslatedCImport`
helper in `build.zig`. That recipe is:

```zig
const difftest_mod = b.createModule(.{
    .root_source_file = b.path("src/diff_test.zig"),
    .target = target,
    .optimize = .ReleaseFast,
});
difftest_mod.addImport("rawr", bench_lib_mod); // reuse the ReleaseFast lib module
addTranslatedCImport(b, difftest_mod, .{
    .header = "vendor/croaring_wrapper.h",
    .include_dir = "vendor/",
    .c_source = "vendor/roaring.c",
    .target = target,
    .optimize = .ReleaseFast,
});

const difftest_exe = b.addExecutable(.{ .name = "diff_test", .root_module = difftest_mod });
b.installArtifact(difftest_exe);
const difftest_step = b.step("difftest", "Differential tests vs CRoaring");
difftest_step.dependOn(&b.addRunArtifact(difftest_exe).step);
```

Note: the existing `validate`/`bench-compare` modules build at `.ReleaseFast`.
For a *correctness* harness we want a debug-friendly, leak-checking allocator on
the rawr side (see Allocator below), which is independent of the module's
optimize level — the allocator is chosen in `diff_test.zig`, not via the build
graph. Keeping `.ReleaseFast` is fine and matches the other interop targets.

`src/test_gen.zig` is imported by both the `difftest` exe and (via the test
build) `property_tests.zig`. It must compile in both contexts, so **`test_gen.zig`
stays pure rawr** — it never imports `c`. `buildOracle` (which needs `c`) therefore
lives in `diff_test.zig`, not `test_gen.zig`. This is also why oracle-anchoring
for the property tests lives in `diff_test.zig` rather than in the unit-test
build (see Task 3).

### Allocator

The rawr side of the harness MUST use a leak-checking allocator
(`std.heap.GeneralPurposeAllocator(.{})` / `DebugAllocator` with leak detection),
not `c_allocator`. A differential harness running millions of ops is the ideal
place to catch a leak, double-free, or use-after-free in rawr's own container
lifecycle — `c_allocator` would silently hide exactly those. Check
`gpa.deinit() == .ok` at the end and treat a leak as a harness failure. The
CRoaring side allocates via its own malloc and is unaffected.

`diff_test.zig` follows the `validate_croaring.zig` entry-point shape:
`pub fn main() !void` (no `std.process.Init` / `std.Io` needed — the harness has
no timing loop, unlike `bench_croaring.zig`).

---

## Acceptance criteria

The suite is "at parity with CRoaring's test suite" when:

1. Every operation in the Task 2 matrix is differentially checked against CRoaring on **all 9 container-pair combinations** plus the empty-operand and full-chunk edges, run-optimized and not.
2. Both orderings are tested for the non-commutative ops (difference/andnot).
3. In-place results are asserted equal to allocating results for **all four** ops (not just OR).
4. The four container-transition cases in Task 2 pass (promotion, demotion, empty-out-one-of-many, run boundary).
5. Part A: the algebraic property tests in `property_tests.zig` run over the
   mixed generator (`test_gen.randomMixed`), exercising bitset/run containers.
   Part B: a handful of the highest-value identities are oracle-anchored in
   `diff_test.zig` via `assertAgree`.
6. `zig build difftest` runs the full matrix and a randomized loop across mixed profiles with zero failures. The loop's iteration count and profile sizes must be **tunable constants** at the top of `diff_test.zig` so `difftest` stays a fast, run-it-every-time command rather than a soak test. Target ≥1000 random `(A,B)` pairs as the default, but if dense/full-heavy profiles make that impractically slow, cap `max_chunks` / dense sizes so a default run finishes in a few seconds — and leave the constants obvious so a deeper soak is one edit away. A practical `difftest` beats a thorough one nobody runs.
7. `zig build test` and `zig build validate` still pass unchanged.

## Chunk plan

This spec is chunked into the following standalone sub-specs. Each has its own
acceptance criteria and pass/fail. Dependency order is roughly top-to-bottom;
01-04/01-05 can proceed in parallel once 01-03 lands.

| Chunk | Title | Covers | Depends on |
|-------|-------|--------|------------|
| `01-01` | CRoaring wrapper extension | Task 0 | — |
| `01-02` | Container-type-aware generator + self-test | Task 1 | — |
| `01-03` | Prove the rig (wiring, oracle, `assertAgree`, one case) | Task 2 core | 01-01, 01-02 |
| `01-04` | Operation matrix (8 ops + predicates + 9 pairs) | Task 2 matrix | 01-03 |
| `01-05` | Container-transition + edge cases | Task 2 transitions | 01-03 |
| `01-06` | Randomized differential loop + tunability | Task 2 random | 01-03, 01-04 |
| `01-07` | Property tests (A) + oracle-anchored identities (B) | Task 3 | 01-02, 01-03 |
| `01-08` | Round-trip + addRange coverage | Task 4 | 01-02 |
| `01-09` | Malformed input smoke test (optional) | Task 5 | 01-02 |

The critical path is `01-02` (the generator — design-risky, universal dependency)
then `01-03` (prove the rig). `01-03` deliberately bundles the `assertAgree`
comparator with the first passing case: a "passing case" *is* an `assertAgree`
call, so splitting them would only manufacture throwaway inline checks. Get the
one `(sparse, dense)` OR case passing end-to-end through `assertAgree` before
fanning out 01-04/01-05 — that proves the generator, the oracle build, and the
comparator all line up.
