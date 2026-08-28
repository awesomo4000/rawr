<!-- SPDX-License-Identifier: MPL-2.0 -->

# Optimization: Branchless Merge Walk for Array Containers

> **Follow-up (08/28/2026):** [Spec 51-02](../51-02-productize-branchy-merge.md) replaces the
> out-of-place union and difference loops with the real-data-tested C3 source form. The in-place union
> remains branchless because its output aliases its input and that path was not part of the later
> measurement. The two results cover different workloads and call paths.

**Applies to:** `src/container_ops.zig`
**Functions:** `arrayUnionArray`, `arrayIntersectArray`, `arrayDifferenceArray`
**Depends on:** Nothing. Standalone change.
**Do after:** Refactor merge + comptime cleanups (no conflicts either way, but cleaner diff)

## Background

The current merge walks in array-on-array operations use branchy comparisons:

```zig
if (sa[i] < sb[j]) {
    // take from a
    i += 1;
} else if (sa[i] > sb[j]) {
    // take from b
    j += 1;
} else {
    // equal
    i += 1;
    j += 1;
}
```

On Apple Silicon (M4), branch mispredictions cost ~14 cycles. When two arrays have
interleaved values (common in real workloads), the branch predictor can't learn the
pattern and mispredicts ~33% of iterations. Lemire (2021) showed that a branchless
variant using conditional moves is 10%+ faster on ARM with LLVM — and Zig compiles
through LLVM.

## Change 1: `arrayIntersectArray` (intersection)

**File:** `src/container_ops.zig`, currently around line 284.

**Current code** (simplified):
```zig
while (i < sa.len and j < sb.len) {
    if (sa[i] < sb[j]) {
        i += 1;
    } else if (sa[i] > sb[j]) {
        j += 1;
    } else {
        result.values[k] = sa[i];
        i += 1;
        j += 1;
        k += 1;
    }
}
```

**Replace with:**
```zig
while (i < sa.len and j < sb.len) {
    const a_val = sa[i];
    const b_val = sb[j];

    // Branchless: always advance whichever pointer is behind (or both if equal).
    // On aarch64, LLVM emits csel/cset for these — no branch, no mispredict.
    i += @intFromBool(a_val <= b_val);
    j += @intFromBool(b_val <= a_val);

    // Only write on match. This IS a branch, but it's well-predicted
    // because intersections are typically sparse (most iterations don't match).
    if (a_val == b_val) {
        result.values[k] = a_val;
        k += 1;
    }
}
```

The equality check stays as a branch because:
- In most datasets, matches are rare (sparse intersection), so the branch is
  heavily biased toward not-taken — easy for the predictor.
- Making the write branchless would require always writing + conditionally
  incrementing k, which touches more memory unnecessarily.

The two pointer advances are the ones that benefit from being branchless — they
alternate unpredictably when arrays are interleaved.

## Change 2: `arrayUnionArray` (union)

**File:** `src/container_ops.zig`, currently around line 115 (the merge loop inside
the `max_card <= MAX_CARDINALITY` branch).

**Current code** (simplified):
```zig
while (i < sa.len and j < sb.len) {
    if (sa[i] < sb[j]) {
        result.values[k] = sa[i];
        i += 1;
        k += 1;
    } else if (sa[i] > sb[j]) {
        result.values[k] = sb[j];
        j += 1;
        k += 1;
    } else {
        result.values[k] = sa[i];
        i += 1;
        j += 1;
        k += 1;
    }
}
```

**Replace with:**
```zig
while (i < sa.len and j < sb.len) {
    const a_val = sa[i];
    const b_val = sb[j];

    // Always write the smaller value (or either if equal).
    result.values[k] = if (a_val <= b_val) a_val else b_val;
    k += 1;

    // Advance whichever pointer(s) contributed.
    i += @intFromBool(a_val <= b_val);
    j += @intFromBool(b_val <= a_val);
}
```

This produces exactly one output per iteration. When `a_val == b_val`, both
pointers advance (dedup). When unequal, only the smaller side advances.

**Key:** No second branch for the dedup case. The `if` for the output value
compiles to a `csel` (conditional select) on aarch64, not a branch.

## Change 3: `arrayDifferenceArray` (difference: A \ B)

**File:** `src/container_ops.zig` (find `arrayDifferenceArray` or equivalent).

Same pattern. Keep element from A only when A < B or B is exhausted.

```zig
while (i < sa.len and j < sb.len) {
    const a_val = sa[i];
    const b_val = sb[j];

    // Write a_val only when it's strictly less than b_val (not in B).
    if (a_val < b_val) {
        result.values[k] = a_val;
        k += 1;
    }

    // Advance pointers branchlessly.
    i += @intFromBool(a_val <= b_val);
    j += @intFromBool(b_val <= a_val);
}
// Drain remaining from A.
while (i < sa.len) : (i += 1) {
    result.values[k] = sa[i];
    k += 1;
}
```

## Also applies to: `ArrayContainer.unionInPlace`

**File:** `src/array_container.zig`, the forward-merge loop in `unionInPlace`.

Same branchless pattern as Change 2. The move-to-end + forward-merge algorithm
doesn't change — only the inner comparison loop gets the branchless treatment.

## Verification

1. `zig build test` — all existing tests cover these code paths extensively.
2. Benchmark: run `zig build bench` before and after on M4. Compare:
   - `bitwiseAnd (sparse 500K x 500K)` — this exercises arrayIntersectArray heavily
   - `bitwiseOr (sparse 500K x 500K)` — exercises arrayUnionArray
   - `bitwiseDifference (sparse)` — exercises arrayDifferenceArray
3. Expected improvement: 5-15% on sparse operations (many array containers).
   Dense operations (bitset containers) won't change.

## Why this works on M4 specifically

Apple Silicon has a wide out-of-order core that's very good at executing
conditional selects (`csel`, `csinc`) in parallel with other work. LLVM
reliably lowers `@intFromBool(comparison)` to `cset` on aarch64. The
resulting instruction sequence has zero branches in the hot path — every
iteration executes the same instructions regardless of data values.

Lemire confirmed this: "With LLVM, there is a sizeable benefit (over 10%)
on both the Apple (ARM) processor and the Zen 2 processor."

Zig uses LLVM. This is free.

## Assembly Analysis (2026-02-13)

To verify LLVM generates the expected branchless code, we isolated the hot loops:

```zig
// merge_test.zig - isolated inner loops for assembly comparison
export fn intersect_branchy(sa: [*]const u16, sa_len: usize, sb: [*]const u16, sb_len: usize, out: [*]u16) usize {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    while (i < sa_len and j < sb_len) {
        if (sa[i] < sb[j]) {
            i += 1;
        } else if (sa[i] > sb[j]) {
            j += 1;
        } else {
            out[k] = sa[i];
            i += 1; j += 1; k += 1;
        }
    }
    return k;
}

export fn intersect_branchless(sa: [*]const u16, sa_len: usize, sb: [*]const u16, sb_len: usize, out: [*]u16) usize {
    var i: usize = 0;
    var j: usize = 0;
    var k: usize = 0;
    while (i < sa_len and j < sb_len) {
        const a_val = sa[i];
        const b_val = sb[j];
        i += @intFromBool(a_val <= b_val);
        j += @intFromBool(b_val <= a_val);
        if (a_val == b_val) {
            out[k] = a_val;
            k += 1;
        }
    }
    return k;
}
```

### ARM64 (Apple Silicon M4)

```bash
zig build-lib merge_test.zig -OReleaseFast -femit-asm=merge_test.s
```

| Function | Branch instructions | Conditional ops |
|----------|--------------------|--------------------|
| intersect_branchy | 6 | 0 |
| intersect_branchless | 2 | `cinc x10, x10, ls` + `cinc x9, x9, ls` |
| union_branchy | 6 | 0 |
| union_branchless | 1 | `csel w13, w11, w12, lo` + 2x `cinc` |

The branchless version correctly generates `cinc` (conditional increment) and
`csel` (conditional select) instructions, reducing branches by ~70%.

### x86_64 (Intel/AMD)

```bash
zig build-lib merge_test.zig -OReleaseFast -target x86_64-linux -femit-asm=merge_x86_64.s
```

| Function | Jump instructions | Conditional ops |
|----------|------------------|-----------------|
| intersect_branchy | 9 | 0 |
| intersect_branchless | 4 | `setbe` + `sbb r10, -1` |
| union_branchy | 9 | 0 |
| union_branchless | 3 | `cmovb r14d, ebx` |

The branchless version generates `cmov` (conditional move) and `sbb` (subtract
with borrow for conditional increment), reducing jumps by ~60%.

### Benchmark Results

Despite correct codegen, M4 benchmarks showed **no improvement** (within noise):

- bitwiseAnd (sparse): ~1.8ms both versions
- bitwiseOr (sparse): ~7.3ms both versions

**Hypothesis:** M4's branch predictor is exceptionally good, and the sparse
operations are dominated by allocation overhead, not branch mispredictions.
The branchless code may help more on:
- x86_64 where misprediction penalties are steeper
- Workloads with truly random/unpredictable interleaving
- When combined with arena allocation to remove the allocation bottleneck

We're keeping this change because:
1. The codegen is correct and theoretically sound
2. It may help on other architectures
3. It doesn't hurt performance on M4
4. Once allocation overhead is addressed, this may become the bottleneck

## Commit

```
perf: branchless merge walks for array container operations
```
