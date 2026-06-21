# Spec 01-02: Container-type-aware generator + self-test

Chunk of [`01-differential-testing.md`](01-differential-testing.md). **The core
of the whole spec and its central dependency** — nearly everything else builds on
this. Isolated deliberately because it is the design-risky piece; its self-test
proves it works before anything depends on it.

## Why this exists

The current generator (`property_tests.zig: randomBitmap`) draws ~100
uniform-random `u32`s, which almost always yields only **array** containers. The
diff suite needs to force **each** container type (array / bitset / run) and
**mixtures within one bitmap** so set ops actually meet heterogeneous operands.

## Carry-over facts

- rawr container model: tagged pointer (`packed struct(u64)`, 2-bit type tag),
  one heap alloc per container. The self-test inspects this tag.
- A "chunk" here = one 16-bit high key = 65536 contiguous values.
- Run containers exist **only after `runOptimize`**.
- Use a leak-checking allocator in any test (`GeneralPurposeAllocator`), not
  `c_allocator`.

## Task — create `src/test_gen.zig`

`test_gen.zig` must stay **pure rawr** — it never imports `c`. It is imported by
both the unit-test build (via `property_tests.zig`) and the difftest exe, so it
must compile in a build that cannot link CRoaring. (`buildOracle` lives in
`diff_test.zig`, not here — see 01-03.)

### Container profiles

For each populated chunk the generator picks a **profile**. The container type a
profile yields **depends on `run_optimize`**:

| Profile     | How to fill the chunk                                          | Type w/o `runOptimize`     | Type after `runOptimize` |
|-------------|---------------------------------------------------------------|----------------------------|--------------------------|
| `sparse`    | N random low-16 values, N < 4096                              | array                      | array                    |
| `dense`     | N random low-16 values, N > 4096 (e.g. 4097–60000)           | bitset                     | bitset (unless runs win) |
| `full`      | all 65536 values (**build via `addRange`**)                  | bitset                     | run                      |
| `runs`      | K disjoint consecutive ranges (3–10 runs, len 50–2000)       | array or bitset (by count) | run                      |
| `single`    | one value                                                     | array (size 1)             | array                    |
| `boundary`  | low-16 values {0, 1, 65534, 65535}                           | array, edge offsets        | array                    |

### API (suggested)

```zig
pub const Profile = enum { sparse, dense, full, runs, single, boundary };

/// Build a bitmap with explicitly chosen container profiles per chunk.
/// Returns a rawr bitmap AND the sorted unique value list used to build it
/// (ground truth: lets the oracle be built identically, and lets probes know
/// the truth).
pub fn build(
    allocator: Allocator,
    rng: std.Random,
    chunks: []const struct { key: u16, profile: Profile },
    run_optimize: bool,
) !struct { bm: RoaringBitmap, values: []u32 };

/// Fully random bitmap: random chunk count (0..max_chunks), each a random profile.
pub fn randomMixed(allocator: Allocator, rng: std.Random, max_chunks: usize, run_optimize: bool)
    !struct { bm: RoaringBitmap, values: []u32 };
```

### Hard requirements

1. Must be able to produce a bitmap where **chunk 0 = array, chunk 1 = bitset,
   chunk 2 = run** in one bitmap (heterogeneous operands for the dispatch loop).
2. `randomMixed`, over a few hundred iterations, must produce **every profile**
   and **every adjacent profile pairing** with high probability. Weight
   `dense`/`runs`/`full` **up** — uniform-random would bury them.
3. The returned `values` slice is the ground-truth oracle input; building the
   oracle from the identical slice guarantees the two start equal regardless of
   rawr's container layout.
4. `run_optimize` toggle always available.
5. Build `full` (and large `runs`) **compactly via `addRange`**, not 65536
   individual adds. Cost note: the `values` slice still enumerates every value
   (the oracle is built from it), so a `full` chunk contributes 65536 `u32`s.
   Across many `full`/`dense` chunks the slice grows fast — keep `max_chunks` and
   per-profile sizes modest in randomized use (01-06 owns the tunability).

## Self-test (ships with this chunk, pure rawr, no oracle)

Unit tests proving the generator does its one job: build a single bitmap with
chunk 0 = `sparse`, chunk 1 = `dense`, chunk 2 = `runs` (run-optimized), chunk 3
= `full` (run-optimized), and **assert each chunk's container type** (array /
bitset / run / run) by inspecting the container tag. Also assert the returned
`values` slice is sorted, unique, and matches `cardinality()`.

## Acceptance criteria

1. `src/test_gen.zig` exists, pure rawr (no `c` import), compiles in the unit-test
   build.
2. `build` can force array+bitset+run in one bitmap (proven by self-test).
3. `randomMixed` exercises all profiles with `dense`/`runs`/`full` weighted up.
4. `full`/large `runs` are built via `addRange`.
5. The self-test passes under `zig build test` with a leak-checking allocator
   (`gpa.deinit() == .ok`).

## Dependencies

None (pure rawr). Blocks: 01-03 through 01-08.
