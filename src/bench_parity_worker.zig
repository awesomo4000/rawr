// SPDX-License-Identifier: MPL-2.0

//! Fresh-process CRoaring parity benchmark worker.
//!
//! The shell controller discovers rows and tuples through `--list`, then starts
//! one worker process for exactly one `(row, implementation, allocator)` tuple.

const std = @import("std");
const c = @import("c");
const bench_time = @import("bench_time.zig");
const dashboard = @import("bench_croaring.zig");

const warmup_runs = 3;
const timed_runs = 21;

const Implementation = dashboard.ParityImplementation;
const AllocatorKind = dashboard.ParityAllocator;
const Operation = dashboard.ParityRow;

const ReportingUnit = enum {
    ms,
    ns_per_op,

    fn name(self: ReportingUnit) []const u8 {
        return switch (self) {
            .ms => "ms",
            .ns_per_op => "ns/op",
        };
    }
};

const AllocationClass = enum {
    allocating,
    non_allocating,
};

const ValidationOracle = enum {
    portable_bytes,
    exact_queries,
    exact_array,
    exact_scalar,
};

const Followup = enum {
    complete,
    calibration_pending,
};

const TupleKey = struct {
    implementation: Implementation,
    allocator: AllocatorKind,
};

const Reference = struct {
    row_id: []const u8,
    variant: TupleKey,
};

const Variant = struct {
    implementation: Implementation,
    allocator: AllocatorKind,
    batch_count: ?usize = null,
};

const ManifestRow = struct {
    id: []const u8,
    display_name: []const u8,
    corpus: []const u8,
    seed: u64,
    rawr_operation: []const u8,
    croaring_operation: []const u8,
    allocation_class: AllocationClass,
    variants: []const Variant,
    reference: ?Reference = null,
    setup_boundary: []const u8,
    teardown_boundary: []const u8,
    validation_oracle: ValidationOracle,
    batch_count: usize = 1,
    reporting_unit: ReportingUnit = .ms,
    followup: Followup = .complete,
    operation: Operation,
};

const allocating_variants = [_]Variant{
    .{ .implementation = .rawr, .allocator = .smp },
    .{ .implementation = .rawr, .allocator = .libc },
    .{ .implementation = .croaring, .allocator = .libc },
};

const rawr_allocating_variants = [_]Variant{
    .{ .implementation = .rawr, .allocator = .smp },
    .{ .implementation = .rawr, .allocator = .libc },
};

const non_allocating_variants = [_]Variant{
    .{ .implementation = .rawr, .allocator = .none },
    .{ .implementation = .croaring, .allocator = .none },
};

const cardinality_variants = [_]Variant{
    .{ .implementation = .rawr, .allocator = .none, .batch_count = 524288 },
    .{ .implementation = .croaring, .allocator = .none, .batch_count = 64 },
};

const arena_variant = [_]Variant{
    .{ .implementation = .rawr, .allocator = .arena },
};

const smp_variant = [_]Variant{
    .{ .implementation = .rawr, .allocator = .smp },
};

const allocating_setup = "input construction outside timing; result construction inside timing";
const allocating_teardown = "result deinit/free inside timing";
const query_setup = "input construction outside timing; query operation inside timing";
const query_teardown = "input deinit/free outside timing";

const manifest = [_]ManifestRow{
    .{
        .id = "add-random",
        .display_name = "add (random 1M)",
        .corpus = "1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.add loop",
        .croaring_operation = "roaring_bitmap_add loop",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "value corpus outside timing; bitmap construction and inserts inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .add_random,
    },
    .{
        .id = "add-sequential",
        .display_name = "add (sequential 1M)",
        .corpus = "u32 values 0 through 999999 in ascending order",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.add loop",
        .croaring_operation = "roaring_bitmap_add loop",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "value corpus outside timing; bitmap construction and inserts inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .add_sequential,
    },
    .{
        .id = "add-many-random",
        .display_name = "addMany (random 1M)",
        .corpus = "1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.addMany",
        .croaring_operation = "roaring_bitmap_add_many",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "value corpus outside timing; bitmap construction and bulk insert inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .add_many_random,
    },
    .{
        .id = "add-many-sequential",
        .display_name = "addMany (sequential 1M)",
        .corpus = "u32 values 0 through 999999 in ascending order",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.addMany",
        .croaring_operation = "roaring_bitmap_add_many",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "value corpus outside timing; bitmap construction and bulk insert inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .add_many_sequential,
    },
    .{
        .id = "add-range",
        .display_name = "addRange (1M)",
        .corpus = "inclusive range 0 through 999999",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.addRange",
        .croaring_operation = "roaring_bitmap_add_range",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "empty bitmap construction and range insert inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8192,
        .reporting_unit = .ns_per_op,
        .operation = .add_range,
    },
    .{
        .id = "contains-hit",
        .display_name = "contains (hit)",
        .corpus = "1000000 inserted deterministic random values queried as hits",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.contains",
        .croaring_operation = "roaring_bitmap_contains",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_queries,
        .operation = .contains_hit,
    },
    .{
        .id = "contains-miss",
        .display_name = "contains (miss)",
        .corpus = "1000000 deterministic random values transformed with high bit for misses",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.contains",
        .croaring_operation = "roaring_bitmap_contains",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_queries,
        .operation = .contains_miss,
    },
    .{
        .id = "sparse-and",
        .display_name = "bitwiseAnd (sparse)",
        .corpus = "500000 deterministic sorted and deduplicated u32 values split into overlapping halves",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.bitwiseAnd",
        .croaring_operation = "roaring_bitmap_and",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .sparse_and,
    },
    .{
        .id = "sparse-and-arena",
        .display_name = "bitwiseAnd (sparse, arena)",
        .corpus = "same sparse corpus and operation as sparse-and with arena result allocation",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.bitwiseAnd",
        .croaring_operation = "roaring_bitmap_and reference from sparse-and",
        .allocation_class = .allocating,
        .variants = &arena_variant,
        .reference = .{ .row_id = "sparse-and", .variant = .{ .implementation = .croaring, .allocator = .libc } },
        .setup_boundary = allocating_setup,
        .teardown_boundary = "arena deinit inside timing",
        .validation_oracle = .portable_bytes,
        .operation = .sparse_and_arena,
    },
    .{
        .id = "dense-and",
        .display_name = "bitwiseAnd (dense)",
        .corpus = "ranges 0..499999 and 250000..749999",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.bitwiseAnd",
        .croaring_operation = "roaring_bitmap_and",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8192,
        .reporting_unit = .ns_per_op,
        .operation = .dense_and,
    },
    .{
        .id = "sparse-or",
        .display_name = "bitwiseOr (sparse)",
        .corpus = "500000 deterministic sorted and deduplicated u32 values split into overlapping halves",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.bitwiseOr",
        .croaring_operation = "roaring_bitmap_or",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .sparse_or,
    },
    .{
        .id = "sparse-or-arena",
        .display_name = "bitwiseOr (sparse, arena)",
        .corpus = "same sparse corpus and operation as sparse-or with arena result allocation",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.bitwiseOr",
        .croaring_operation = "roaring_bitmap_or reference from sparse-or",
        .allocation_class = .allocating,
        .variants = &arena_variant,
        .reference = .{ .row_id = "sparse-or", .variant = .{ .implementation = .croaring, .allocator = .libc } },
        .setup_boundary = allocating_setup,
        .teardown_boundary = "arena deinit inside timing",
        .validation_oracle = .portable_bytes,
        .operation = .sparse_or_arena,
    },
    .{
        .id = "dense-or",
        .display_name = "bitwiseOr (dense)",
        .corpus = "ranges 0..499999 and 250000..749999",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.bitwiseOr",
        .croaring_operation = "roaring_bitmap_or",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8192,
        .reporting_unit = .ns_per_op,
        .operation = .dense_or,
    },
    .{
        .id = "lazy-or-repair",
        .display_name = "lazyOr+repair (sparse)",
        .corpus = "same sparse corpus as sparse-or",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.lazyOr plus repairAfterLazy",
        .croaring_operation = "roaring_bitmap_lazy_or plus roaring_bitmap_repair_after_lazy",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .lazy_or_repair,
    },
    .{
        .id = "lazy-or-repair-baseline",
        .display_name = "lazyOr+repair (pre-adoption baseline)",
        .corpus = "same sparse corpus as lazy-or-repair",
        .seed = 54321,
        .rawr_operation = "pre-adoption lazyOr construction plus repairAfterLazy",
        .croaring_operation = "lazy-or-repair canonical CRoaring reference",
        .allocation_class = .allocating,
        .variants = &rawr_allocating_variants,
        .reference = .{ .row_id = "lazy-or-repair", .variant = .{ .implementation = .croaring, .allocator = .libc } },
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .lazy_or_repair_baseline,
    },
    .{
        .id = "lazy-or-repair-descending",
        .display_name = "lazyOr+repair (sparse, descending frees)",
        .corpus = "same sparse corpus and full-cycle operation as lazy-or-repair with opt-in descending transient-bitset frees",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.lazyOr plus repairAfterLazyWithOptions descending free order",
        .croaring_operation = "roaring_bitmap_lazy_or plus roaring_bitmap_repair_after_lazy reference from lazy-or-repair",
        .allocation_class = .allocating,
        .variants = &smp_variant,
        .reference = .{ .row_id = "lazy-or-repair", .variant = .{ .implementation = .croaring, .allocator = .libc } },
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .lazy_or_repair_descending,
    },
    .{
        .id = "lazy-or-construction",
        .display_name = "lazyOr construction (sparse)",
        .corpus = "same sparse corpus as sparse-or",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.lazyOr construction only",
        .croaring_operation = "roaring_bitmap_lazy_or construction only",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "inputs outside timing; lazy result construction inside timing",
        .teardown_boundary = "result deinit/free outside the internally timed interval",
        .validation_oracle = .portable_bytes,
        .operation = .lazy_or_construction,
    },
    .{
        .id = "lazy-or-construction-baseline",
        .display_name = "lazyOr construction (pre-adoption baseline)",
        .corpus = "same sparse corpus as lazy-or-construction",
        .seed = 54321,
        .rawr_operation = "pre-adoption lazyOr construction only",
        .croaring_operation = "lazy-or-construction canonical CRoaring reference",
        .allocation_class = .allocating,
        .variants = &rawr_allocating_variants,
        .reference = .{ .row_id = "lazy-or-construction", .variant = .{ .implementation = .croaring, .allocator = .libc } },
        .setup_boundary = "inputs outside timing; pre-adoption lazy result construction inside timing",
        .teardown_boundary = "result deinit/free outside the internally timed interval",
        .validation_oracle = .portable_bytes,
        .operation = .lazy_or_construction_baseline,
    },
    .{
        .id = "lazy-or-repair-only",
        .display_name = "lazyOr repair (sparse)",
        .corpus = "same sparse corpus as sparse-or",
        .seed = 54321,
        .rawr_operation = "RoaringBitmap.repairAfterLazy",
        .croaring_operation = "roaring_bitmap_repair_after_lazy",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "inputs and lazy result construction outside timing; repair inside timing",
        .teardown_boundary = "result deinit/free outside the internally timed interval",
        .validation_oracle = .portable_bytes,
        .operation = .lazy_or_repair_only,
    },
    .{
        .id = "or-many",
        .display_name = "orMany (32 mixed)",
        .corpus = "32 deterministic mixed array, bitset, and run-heavy bitmaps",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.orMany",
        .croaring_operation = "roaring_bitmap_or_many",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 128,
        .reporting_unit = .ns_per_op,
        .operation = .or_many,
    },
    .{
        .id = "or-many-heap",
        .display_name = "orManyHeap (32 mixed)",
        .corpus = "32 deterministic mixed array, bitset, and run-heavy bitmaps",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.orManyHeap",
        .croaring_operation = "roaring_bitmap_or_many_heap",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 128,
        .reporting_unit = .ns_per_op,
        .operation = .or_many_heap,
    },
    .{
        .id = "xor-many",
        .display_name = "xorMany (32 mixed)",
        .corpus = "32 deterministic mixed array, bitset, and run-heavy bitmaps",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.xorMany",
        .croaring_operation = "roaring_bitmap_xor_many",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 128,
        .reporting_unit = .ns_per_op,
        .operation = .xor_many,
    },
    .{
        .id = "array-balanced-and",
        .display_name = "bitwiseAnd (array balanced)",
        .corpus = "200 overlapping array-container pairs of 2048 values",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.bitwiseAnd",
        .croaring_operation = "roaring_bitmap_and",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 16,
        .reporting_unit = .ns_per_op,
        .operation = .array_balanced_and,
    },
    .{
        .id = "array-balanced-and-cardinality",
        .display_name = "andCardinality (array balanced)",
        .corpus = "200 overlapping array-container pairs of 2048 values",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.andCardinality",
        .croaring_operation = "roaring_bitmap_and_cardinality",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_scalar,
        .batch_count = 32,
        .reporting_unit = .ns_per_op,
        .operation = .array_balanced_and_cardinality,
    },
    .{
        .id = "array-balanced-xor",
        .display_name = "bitwiseXor (array balanced)",
        .corpus = "200 overlapping array-container pairs of 2048 values",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.bitwiseXor",
        .croaring_operation = "roaring_bitmap_xor",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8,
        .reporting_unit = .ns_per_op,
        .operation = .array_balanced_xor,
    },
    .{
        .id = "array-skewed-and",
        .display_name = "bitwiseAnd (array skewed)",
        .corpus = "200 sparse 32-value arrays intersected with offset 4096-value arrays",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.bitwiseAnd",
        .croaring_operation = "roaring_bitmap_and",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 128,
        .reporting_unit = .ns_per_op,
        .operation = .array_skewed_and,
    },
    .{
        .id = "array-skewed-and-cardinality",
        .display_name = "andCardinality (array skewed)",
        .corpus = "200 sparse 32-value arrays intersected with offset 4096-value arrays",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.andCardinality",
        .croaring_operation = "roaring_bitmap_and_cardinality",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_scalar,
        .batch_count = 128,
        .reporting_unit = .ns_per_op,
        .operation = .array_skewed_and_cardinality,
    },
    .{
        .id = "iterate",
        .display_name = "iterate (1M values)",
        .corpus = "bitmap built from 1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.Iterator",
        .croaring_operation = "roaring_uint32_iterator_t pull loop in C",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_array,
        .operation = .iterate,
    },
    .{
        .id = "to-array",
        .display_name = "toArray (1M values)",
        .corpus = "bitmap built from 1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.toArray",
        .croaring_operation = "roaring_bitmap_to_uint32_array",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = "bitmap and output buffer construction outside timing; conversion inside timing",
        .teardown_boundary = "bitmap and output buffer deinit/free outside timing",
        .validation_oracle = .exact_array,
        .operation = .to_array,
    },
    .{
        .id = "to-array-alloc",
        .display_name = "toArrayAlloc (1M values)",
        .corpus = "bitmap built from 1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.toArrayAlloc",
        .croaring_operation = "allocate plus roaring_bitmap_to_uint32_array",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "bitmap construction outside timing; output allocation and conversion inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .exact_array,
        .operation = .to_array_alloc,
    },
    .{
        .id = "serialize",
        .display_name = "serialize",
        .corpus = "bitmap built from 1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.serialize",
        .croaring_operation = "roaring_bitmap_portable_serialize",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "bitmap construction outside timing; output allocation and serialization inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .serialize,
    },
    .{
        .id = "deserialize",
        .display_name = "deserialize",
        .corpus = "portable bytes from the 1000000-value deterministic random bitmap",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.deserialize",
        .croaring_operation = "roaring_bitmap_portable_deserialize_safe",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "serialized input construction outside timing; bitmap construction inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .operation = .deserialize,
    },
    .{
        .id = "deserialize-arena",
        .display_name = "deserialize (arena)",
        .corpus = "same portable bytes and operation as deserialize with arena result allocation",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.deserialize",
        .croaring_operation = "roaring_bitmap_portable_deserialize_safe reference from deserialize",
        .allocation_class = .allocating,
        .variants = &arena_variant,
        .reference = .{ .row_id = "deserialize", .variant = .{ .implementation = .croaring, .allocator = .libc } },
        .setup_boundary = "serialized input construction outside timing; bitmap construction inside timing",
        .teardown_boundary = "arena deinit inside timing",
        .validation_oracle = .portable_bytes,
        .operation = .deserialize_arena,
    },
    .{
        .id = "cardinality",
        .display_name = "cardinality",
        .corpus = "bitmap built from 1000000 deterministic random u32 values",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.cardinality",
        .croaring_operation = "roaring_bitmap_get_cardinality",
        .allocation_class = .non_allocating,
        .variants = &cardinality_variants,
        .setup_boundary = "bitmap construction outside timing; batched cardinality calls inside timing",
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_scalar,
        .batch_count = 524288,
        .reporting_unit = .ns_per_op,
        .operation = .cardinality,
    },
    .{
        .id = "rank",
        .display_name = "rank (dense)",
        .corpus = "1000000 deterministic rank probes over dense range 0..499999",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.rank",
        .croaring_operation = "roaring_bitmap_rank",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_queries,
        .operation = .rank,
    },
    .{
        .id = "select",
        .display_name = "select (dense)",
        .corpus = "1000000 deterministic select probes over dense range 0..499999",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.select through one noinline benchmark boundary",
        .croaring_operation = "roaring_bitmap_select loop in C through one benchmark boundary",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_queries,
        .operation = .select,
    },
    .{
        .id = "rank-many",
        .display_name = "rankMany (dense)",
        .corpus = "200000 ascending probes over dense range 0..499999",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.rankMany",
        .croaring_operation = "roaring_bitmap_rank_many",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = "bitmap, probes, and output buffer outside timing; rankMany inside timing",
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_array,
        .batch_count = 8,
        .reporting_unit = .ns_per_op,
        .operation = .rank_many,
    },
    .{
        .id = "range-cardinality-small",
        .display_name = "rangeCardinality small (bitset)",
        .corpus = "1000000 deterministic ranges up to 1023 values over a bitset container",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.rangeCardinality",
        .croaring_operation = "roaring_bitmap_range_cardinality_closed",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_queries,
        .operation = .range_cardinality_small,
    },
    .{
        .id = "range-cardinality-large",
        .display_name = "rangeCardinality large (bitset)",
        .corpus = "1000000 deterministic ranges of 30000 to 49999 values over a bitset container",
        .seed = 12345,
        .rawr_operation = "RoaringBitmap.rangeCardinality",
        .croaring_operation = "roaring_bitmap_range_cardinality_closed",
        .allocation_class = .non_allocating,
        .variants = &non_allocating_variants,
        .setup_boundary = query_setup,
        .teardown_boundary = query_teardown,
        .validation_oracle = .exact_queries,
        .operation = .range_cardinality_large,
    },
    .{
        .id = "flip",
        .display_name = "flip wide range (dense)",
        .corpus = "flip closed range 100000..650000 over dense range 0..499999",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.flip",
        .croaring_operation = "roaring_bitmap_flip_closed",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = allocating_setup,
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8192,
        .reporting_unit = .ns_per_op,
        .operation = .flip,
    },
    .{
        .id = "remove-range",
        .display_name = "removeRangeCopy wide (dense)",
        .corpus = "copy dense range 0..499999 with closed range 100000..650000 removed",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.removeRangeCopy",
        .croaring_operation = "roaring_bitmap_copy plus roaring_bitmap_remove_range_closed",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "input construction outside timing; modified-copy construction inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8192,
        .reporting_unit = .ns_per_op,
        .operation = .remove_range,
    },
    .{
        .id = "clone",
        .display_name = "clone (dense)",
        .corpus = "deep-copy dense range 0..499999; CRoaring COW disabled",
        .seed = 0,
        .rawr_operation = "RoaringBitmap.clone",
        .croaring_operation = "roaring_bitmap_copy with copy-on-write disabled",
        .allocation_class = .allocating,
        .variants = &allocating_variants,
        .setup_boundary = "input construction outside timing; deep copy inside timing",
        .teardown_boundary = allocating_teardown,
        .validation_oracle = .portable_bytes,
        .batch_count = 8192,
        .reporting_unit = .ns_per_op,
        .operation = .clone,
    },
};

const RequestedTuple = struct {
    row: *const ManifestRow,
    variant: Variant,
};

pub fn main(init: std.process.Init) !void {
    try validateManifest();

    // Argument parsing must not precondition either allocator under test.
    var args = try init.minimal.args.iterateAllocator(std.heap.page_allocator);
    defer args.deinit();
    _ = args.skip();

    var list = false;
    var header = false;
    var row_id: ?[]const u8 = null;
    var implementation: ?Implementation = null;
    var allocator: ?AllocatorKind = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--list")) {
            list = true;
        } else if (std.mem.eql(u8, arg, "--header")) {
            header = true;
        } else if (std.mem.startsWith(u8, arg, "--row=")) {
            row_id = arg[6..];
        } else if (std.mem.startsWith(u8, arg, "--implementation=")) {
            implementation = std.meta.stringToEnum(Implementation, arg[17..]) orelse return error.UnknownImplementation;
        } else if (std.mem.startsWith(u8, arg, "--allocator=")) {
            allocator = std.meta.stringToEnum(AllocatorKind, arg[12..]) orelse return error.UnknownAllocator;
        } else {
            return error.UnknownArgument;
        }
    }

    if (list) {
        if (header or row_id != null or implementation != null or allocator != null) return error.ConflictingArguments;
        printManifest();
        return;
    }
    if (header) {
        if (row_id != null or implementation != null or allocator != null) return error.ConflictingArguments;
        printHeader();
        return;
    }

    const requested = try resolveTuple(
        row_id orelse return error.MissingRow,
        implementation orelse return error.MissingImplementation,
        allocator orelse return error.MissingAllocator,
    );
    const median_ns = try runTuple(requested);
    const batch_count = effectiveBatchCount(requested);

    // RESULT is emitted only after the untimed validation in runTuple succeeds.
    bench_time.print("RESULT\t{s}\t{s}\t{s}\t{s}\t{d}\t{d}\n", .{
        requested.row.id,
        @tagName(requested.variant.implementation),
        @tagName(requested.variant.allocator),
        requested.row.reporting_unit.name(),
        batch_count,
        median_ns,
    });
}

fn validateManifest() !void {
    if (manifest.len != 42) return error.InvalidManifestRowCount;
    for (&manifest, 0..) |*row, i| {
        if (row.id.len == 0 or row.variants.len == 0 or row.batch_count == 0) return error.InvalidManifestRow;
        if (dashboard.parityRequiresAllocator(row.operation) != (row.allocation_class == .allocating)) {
            return error.InvalidAllocationClass;
        }
        if (hasProtocolDelimiter(row.id) or
            hasProtocolDelimiter(row.display_name) or
            hasProtocolDelimiter(row.corpus) or
            hasProtocolDelimiter(row.rawr_operation) or
            hasProtocolDelimiter(row.croaring_operation) or
            hasProtocolDelimiter(row.setup_boundary) or
            hasProtocolDelimiter(row.teardown_boundary))
        {
            return error.InvalidManifestText;
        }
        for (manifest[i + 1 ..]) |other| {
            if (std.mem.eql(u8, row.id, other.id)) return error.DuplicateManifestRow;
        }
        for (row.variants, 0..) |variant, variant_index| {
            if (variant.batch_count == 0) return error.InvalidBatchCount;
            if (variant.batch_count != null and row.reporting_unit != .ns_per_op) {
                return error.InvalidBatchOverride;
            }
            if ((row.allocation_class == .allocating) == (variant.allocator == .none)) {
                return error.InvalidAllocatorVariant;
            }
            for (row.variants[variant_index + 1 ..]) |other| {
                if (variant.implementation == other.implementation and variant.allocator == other.allocator) {
                    return error.DuplicateManifestVariant;
                }
            }
        }
        if (row.reference) |reference| {
            if (!manifestHasTuple(reference.row_id, reference.variant)) return error.InvalidManifestReference;
        } else if (hasRawrVariant(row) and findCRoaringVariant(row) == null) {
            return error.MissingCRoaringReference;
        }
    }
}

fn hasProtocolDelimiter(value: []const u8) bool {
    return std.mem.indexOfAny(u8, value, "\t\r\n") != null;
}

fn hasRawrVariant(row: *const ManifestRow) bool {
    for (row.variants) |variant| if (variant.implementation == .rawr) return true;
    return false;
}

fn manifestHasTuple(row_id: []const u8, key: TupleKey) bool {
    for (&manifest) |*row| {
        if (!std.mem.eql(u8, row.id, row_id)) continue;
        for (row.variants) |variant| {
            if (variant.implementation == key.implementation and variant.allocator == key.allocator) return true;
        }
        return false;
    }
    return false;
}

fn printHeader() void {
    bench_time.printBenchEnvironment();
    bench_time.print("# requested-cpu: native\n", .{});
    bench_time.print("# protocol: {d}w/{d}t median\n", .{ warmup_runs, timed_runs });
    bench_time.print("# croaring-avx512: {s}\n", .{
        if (c.CROARING_COMPILER_SUPPORTS_AVX512 != 0) "on" else "off",
    });
}

fn printManifest() void {
    for (&manifest) |*row| {
        const reference = row.reference;
        bench_time.print("ROW\t{s}\t{s}\t{s}\t{d}\t{s}\t{s}\t{s}\t{s}\t{d}\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n", .{
            row.id,
            row.display_name,
            row.corpus,
            row.seed,
            row.rawr_operation,
            row.croaring_operation,
            @tagName(row.allocation_class),
            row.reporting_unit.name(),
            row.batch_count,
            row.setup_boundary,
            row.teardown_boundary,
            @tagName(row.validation_oracle),
            if (reference) |value| value.row_id else "-",
            if (reference) |value| @tagName(value.variant.implementation) else "-",
            if (reference) |value| @tagName(value.variant.allocator) else "-",
            @tagName(row.followup),
        });

        for (row.variants) |variant| {
            if (variant.implementation == .rawr) {
                const comparison = row.reference orelse Reference{
                    .row_id = row.id,
                    .variant = findCRoaringVariant(row) orelse unreachable,
                };
                bench_time.print("TUPLE\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\t{d}\n", .{
                    row.id,
                    @tagName(variant.implementation),
                    @tagName(variant.allocator),
                    comparison.row_id,
                    @tagName(comparison.variant.implementation),
                    @tagName(comparison.variant.allocator),
                    variant.batch_count orelse row.batch_count,
                });
            } else {
                bench_time.print("TUPLE\t{s}\t{s}\t{s}\t-\t-\t-\t{d}\n", .{
                    row.id,
                    @tagName(variant.implementation),
                    @tagName(variant.allocator),
                    variant.batch_count orelse row.batch_count,
                });
            }
        }
    }
}

fn findCRoaringVariant(row: *const ManifestRow) ?TupleKey {
    for (row.variants) |variant| {
        if (variant.implementation == .croaring) {
            return .{ .implementation = variant.implementation, .allocator = variant.allocator };
        }
    }
    return null;
}

fn resolveTuple(row_id: []const u8, implementation: Implementation, allocator: AllocatorKind) !RequestedTuple {
    for (&manifest) |*row| {
        if (!std.mem.eql(u8, row.id, row_id)) continue;
        for (row.variants) |variant| {
            if (variant.implementation == implementation and variant.allocator == allocator) {
                return .{ .row = row, .variant = variant };
            }
        }
        return error.UnsupportedTuple;
    }
    return error.UnknownRow;
}

fn runTuple(requested: RequestedTuple) !u64 {
    dashboard.parityPrepare(requested.row.operation, requested.variant.implementation);
    defer dashboard.parityCleanup();

    const median_ns = measure(requested);
    try dashboard.parityValidate(requested.row.operation, requested.variant.allocator);
    return median_ns;
}

fn effectiveBatchCount(requested: RequestedTuple) usize {
    return requested.variant.batch_count orelse requested.row.batch_count;
}

fn measure(requested: RequestedTuple) u64 {
    for (0..warmup_runs) |_| _ = runBatch(requested);

    var times: [timed_runs]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = runBatch(requested);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return times[timed_runs / 2];
}

fn runBatch(requested: RequestedTuple) u64 {
    const batch_count = effectiveBatchCount(requested);
    if (dashboard.parityTiming(requested.row.operation) == .internal) {
        var elapsed: u64 = 0;
        for (0..batch_count) |_| {
            elapsed +%= dashboard.parityRun(
                requested.row.operation,
                requested.variant.implementation,
                requested.variant.allocator,
            );
        }
        std.mem.doNotOptimizeAway(elapsed);
        return elapsed;
    }

    const start = bench_time.monotonicNanos();
    for (0..batch_count) |_| {
        _ = dashboard.parityRun(
            requested.row.operation,
            requested.variant.implementation,
            requested.variant.allocator,
        );
    }
    return bench_time.monotonicNanos() - start;
}
