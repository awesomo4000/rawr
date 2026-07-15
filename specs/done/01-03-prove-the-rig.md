<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 01-03: Prove the rig — wiring, oracle, comparator, one passing case

Chunk of [`01-differential-testing.md`](01-differential-testing.md). This is the
"prove the rig" chunk: it stands up the difftest build, the oracle constructor,
the `assertAgree` comparator, and **one** end-to-end passing case. Once this
lands, the generator, the oracle build, and the comparator are proven to line up,
and the later chunks just fan out.

The comparator and the first passing case are deliberately in the **same** chunk:
a "passing case" means asserting rawr agrees with the oracle on all four axes —
which *is* `assertAgree`. Splitting them would force throwaway inline checks.

## Dependencies

- **01-01** (wrapper has the six oracle decls).
- **01-02** (`test_gen.zig` generator + `values` slice).

## Carry-over facts

- Zig 0.16, build-system `translate-c`, no `@cImport`. Use the existing
  `addTranslatedCImport` helper in `build.zig` — **do not** duplicate translate-c
  boilerplate.
- `diff_test.zig` is an **executable** (like `validate_croaring.zig`), not a unit
  test, because it links CRoaring. Entry point: `pub fn main() !void` (no
  `std.process.Init` / `std.Io` — no timing loop).
- The rawr side MUST use a **leak-checking allocator**
  (`GeneralPurposeAllocator(.{})`), not `c_allocator`. Assert `gpa.deinit() == .ok`
  at the end; a leak is a harness failure. CRoaring uses its own malloc, unaffected.
- rawr `bitwiseDifference` ⇄ CRoaring `andnot`. `addRange(start,end)` inclusive
  both ends; CRoaring `add_range(min,max)` exclusive on max → pass `end + 1`.
- `validate_croaring.zig` already has a working byte-diff round-trip reporter
  (`validateRoundTrip`, first-divergence-byte printout) — reuse it.

## Task 1 — Build wiring

Add a `zig build difftest` step, mirroring the `validate` step, via the helper:

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

`.ReleaseFast` on the module is fine and matches the other interop targets — the
leak-checking allocator is chosen in `diff_test.zig`, independent of optimize level.

## Task 2 — `buildOracle` (lives in `diff_test.zig`, needs `c`)

Construct a CRoaring bitmap from a ground-truth `values` slice produced by
`test_gen`:

```zig
fn buildOracle(values: []const u32) *c.roaring_bitmap_t {
    // roaring_bitmap_create + add each value (or add_range where contiguous).
}
```

This stays out of `test_gen.zig` (which must remain pure rawr).

## Task 3 — `assertAgree` comparator

The single assertion used everywhere. Compares a rawr result against an oracle
result on all four axes; frees nothing (caller owns):

```zig
fn assertAgree(name: []const u8, rawr_bm: *RoaringBitmap, oracle: *c.roaring_bitmap_t) !void {
    // 1. equal cardinality
    // 2. byte-identical portable serialization
    //    (rawr.serialize vs roaring_bitmap_portable_serialize)
    // 3. membership probes: every value in rawr's iterator is in oracle;
    //    a sample of known-absent values is absent in both
    // 4. cross-deserialize both directions, re-check cardinality + equals
}
```

On byte mismatch, reuse `validate_croaring.zig`'s first-divergence-byte printout.
Print `name` so failures localize.

## Task 4 — One passing case end-to-end

Build A = `sparse` chunk, B = `dense` chunk in the **same** high-key chunk via
`test_gen`. Build their oracles via `buildOracle` from the `values` slices.
Compute `rawr.bitwiseOr(A, B)` and `roaring_bitmap_or(oracleA, oracleB)`, then
`assertAgree("or sparse|dense", ...)`. It must pass.

## Acceptance criteria

1. `zig build difftest` builds and runs via the helper (no duplicated translate-c
   boilerplate).
2. `buildOracle` and `assertAgree` exist in `diff_test.zig`; `assertAgree` checks
   all four axes and reuses the byte-diff reporter.
3. The one `(sparse, dense)` OR case passes end-to-end through `assertAgree`.
4. The rawr side uses a leak-checking GPA; `gpa.deinit() == .ok` at exit.
5. `zig build test` and `zig build validate` still pass unchanged.

## Blocks

01-04, 01-05, 01-06, and the Part B half of 01-07 all build on `assertAgree` /
`buildOracle`.
