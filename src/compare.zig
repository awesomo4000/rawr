const std = @import("std");
const RoaringBitmap = @import("bitmap.zig").RoaringBitmap;
const Container = @import("container.zig").Container;
const TaggedPtr = @import("container.zig").TaggedPtr;
const ArrayContainer = @import("array_container.zig").ArrayContainer;
const BitsetContainer = @import("bitset_container.zig").BitsetContainer;
const RunContainer = @import("run_container.zig").RunContainer;

/// Check if a bitmap is a subset of another. O(n) where n is total container size.
pub fn isSubsetOf(self: *const RoaringBitmap, other: *const RoaringBitmap) bool {
    // Fast-path: if self has more keys, it can't be a subset
    if (self.size > other.size) return false;

    var j: usize = 0;
    for (0..self.size) |i| {
        // Find matching key in other
        while (j < other.size and other.keys[j] < self.keys[i]) : (j += 1) {}

        if (j >= other.size or other.keys[j] != self.keys[i]) {
            return false; // Key not found in other
        }

        // Check container subset with O(n) algorithms per container type
        if (!containerIsSubset(self.containers[i], other.containers[j])) {
            return false;
        }
        j += 1;
    }
    return true;
}

/// Check if two bitmaps are equal. Single pass O(n).
pub fn equals(self: *const RoaringBitmap, other: *const RoaringBitmap) bool {
    if (self.size != other.size) return false;

    for (0..self.size) |i| {
        if (self.keys[i] != other.keys[i]) return false;
        if (!containerEquals(self.containers[i], other.containers[i])) {
            return false;
        }
    }
    return true;
}

/// O(n) container subset check.
fn containerIsSubset(a_tp: TaggedPtr, b_tp: TaggedPtr) bool {
    const a = Container.fromTagged(a_tp);
    const b = Container.fromTagged(b_tp);

    // Quick cardinality check
    if (a.getCardinality() > b.getCardinality()) return false;

    return switch (a) {
        .array => |ac| switch (b) {
            // array ⊆ array: merge-walk O(n+m)
            .array => |bc| arraySubsetArray(ac, bc),
            // array ⊆ bitset: O(n) lookups, each O(1)
            .bitset => |bc| arraySubsetBitset(ac, bc),
            // array ⊆ run: O(n) lookups
            .run => |rc| arraySubsetRun(ac, rc),
            .reserved => false,
        },
        .bitset => |ac| switch (b) {
            // bitset ⊆ array: bitset can have <4096 cardinality, check each bit
            .array => |bc| bitsetSubsetArray(ac, bc),
            // bitset ⊆ bitset: (a & ~b) == 0, O(1024) words
            .bitset => |bc| bitsetSubsetBitset(ac, bc),
            // bitset ⊆ run: check each set bit
            .run => |rc| bitsetSubsetRun(ac, rc),
            .reserved => false,
        },
        .run => |ac| switch (b) {
            // run ⊆ array: check each run value
            .array => |bc| runSubsetArray(ac, bc),
            // run ⊆ bitset: check each run value
            .bitset => |bc| runSubsetBitset(ac, bc),
            // run ⊆ run: check each run
            .run => |rc| runSubsetRun(ac, rc),
            .reserved => false,
        },
        .reserved => false,
    };
}

fn arraySubsetArray(a: *ArrayContainer, b: *ArrayContainer) bool {
    // Merge-walk: O(n+m)
    var i: usize = 0;
    var j: usize = 0;
    while (i < a.cardinality) {
        if (j >= b.cardinality) return false;
        if (a.values[i] < b.values[j]) return false; // a has element not in b
        if (a.values[i] == b.values[j]) i += 1;
        j += 1;
    }
    return true;
}

fn arraySubsetBitset(a: *ArrayContainer, b: *BitsetContainer) bool {
    for (a.values[0..a.cardinality]) |v| {
        if (!b.contains(v)) return false;
    }
    return true;
}

fn arraySubsetRun(a: *ArrayContainer, b: *RunContainer) bool {
    for (a.values[0..a.cardinality]) |v| {
        if (!b.contains(v)) return false;
    }
    return true;
}

fn bitsetSubsetBitset(a: *BitsetContainer, b: *BitsetContainer) bool {
    // (a & ~b) == 0 means all bits in a are also in b
    for (a.words, b.words) |aw, bw| {
        if ((aw & ~bw) != 0) return false;
    }
    return true;
}

fn bitsetSubsetArray(a: *BitsetContainer, b: *ArrayContainer) bool {
    // Check each set bit in bitset is present in array
    for (a.words, 0..) |word, word_idx| {
        var w = word;
        while (w != 0) {
            const bit = @ctz(w);
            const v: u16 = @intCast(word_idx * 64 + bit);
            if (!b.contains(v)) return false;
            w &= w - 1;
        }
    }
    return true;
}

fn bitsetSubsetRun(a: *BitsetContainer, b: *RunContainer) bool {
    for (a.words, 0..) |word, word_idx| {
        var w = word;
        while (w != 0) {
            const bit = @ctz(w);
            const v: u16 = @intCast(word_idx * 64 + bit);
            if (!b.contains(v)) return false;
            w &= w - 1;
        }
    }
    return true;
}

fn runSubsetArray(a: *RunContainer, b: *ArrayContainer) bool {
    for (a.runs[0..a.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (!b.contains(@intCast(v))) return false;
        }
    }
    return true;
}

fn runSubsetBitset(a: *RunContainer, b: *BitsetContainer) bool {
    for (a.runs[0..a.n_runs]) |run| {
        var v: u32 = run.start;
        while (v <= run.end()) : (v += 1) {
            if (!b.contains(@intCast(v))) return false;
        }
    }
    return true;
}

fn runSubsetRun(a: *RunContainer, b: *RunContainer) bool {
    // Merge-walk: O(n_runs_a + n_runs_b) instead of O(cardinality × log(n_runs))
    var j: usize = 0;
    for (a.runs[0..a.n_runs]) |run_a| {
        const start_a = run_a.start;
        const end_a = run_a.end();

        // Skip B runs that end before A's run starts
        while (j < b.n_runs and b.runs[j].end() < start_a) : (j += 1) {}

        // Check that B's runs fully cover [start_a, end_a]
        var pos: u32 = start_a;
        while (pos <= end_a) {
            if (j >= b.n_runs) return false; // No more runs in B
            if (b.runs[j].start > pos) return false; // Gap in B's coverage
            // B.runs[j] covers up to its end
            pos = b.runs[j].end() + 1;
            if (pos <= end_a) j += 1; // Need next run in B
        }
    }
    return true;
}

/// O(n) container equality check.
fn containerEquals(a_tp: TaggedPtr, b_tp: TaggedPtr) bool {
    const a = Container.fromTagged(a_tp);
    const b = Container.fromTagged(b_tp);

    // Different types can still be equal if same cardinality and same values
    if (a.getCardinality() != b.getCardinality()) return false;

    return switch (a) {
        .array => |ac| switch (b) {
            .array => |bc| std.mem.eql(u16, ac.values[0..ac.cardinality], bc.values[0..bc.cardinality]),
            .bitset => |bc| arrayEqualsBitset(ac, bc),
            .run => |rc| arrayEqualsRun(ac, rc),
            .reserved => false,
        },
        .bitset => |ac| switch (b) {
            .array => |bc| arrayEqualsBitset(bc, ac),
            .bitset => |bc| std.mem.eql(u64, ac.words, bc.words),
            .run => |rc| bitsetEqualsRun(ac, rc),
            .reserved => false,
        },
        .run => |ac| switch (b) {
            .array => |bc| arrayEqualsRun(bc, ac),
            .bitset => |bc| bitsetEqualsRun(bc, ac),
            // Same values can have different run encodings, so compare element-by-element
            .run => |rc| runEqualsRun(ac, rc),
            .reserved => false,
        },
        .reserved => false,
    };
}

fn arrayEqualsBitset(a: *ArrayContainer, b: *BitsetContainer) bool {
    // Cardinality already checked equal
    for (a.values[0..a.cardinality]) |v| {
        if (!b.contains(v)) return false;
    }
    return true;
}

fn arrayEqualsRun(a: *ArrayContainer, b: *RunContainer) bool {
    for (a.values[0..a.cardinality]) |v| {
        if (!b.contains(v)) return false;
    }
    return true;
}

fn bitsetEqualsRun(a: *BitsetContainer, b: *RunContainer) bool {
    for (a.words, 0..) |word, word_idx| {
        var w = word;
        while (w != 0) {
            const bit = @ctz(w);
            const v: u16 = @intCast(word_idx * 64 + bit);
            if (!b.contains(v)) return false;
            w &= w - 1;
        }
    }
    return true;
}

fn runEqualsRun(a: *RunContainer, b: *RunContainer) bool {
    // Cardinality already checked equal, so |A|=|B| ∧ A⊆B → A=B
    return runSubsetRun(a, b);
}
