<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 11-03: `findKey` lookup tuning (portable, minor)

Chunk of [array-kernel performance](11-array-kernel-perf.md). Replace the branchy
three-way binary search in the container-index lookup with `lowerBound` + equality,
plus a linear-scan fast path for the common small-container case.
Behavior-preserving.

**Dependency order:** after [11-00](11-00-kernel-extraction-bench.md) for the bench
harness. Independent of 11-01/11-02.

## Change — move the lookup into `array_kernels.zig` (no new bitmap method)

Put the key-slice lookup in the shared kernel module as a **pure free function**,
not a new `RoaringBitmap` method (adding a `pub` bitmap method would violate the
umbrella's "no public API change" rule). Both `bitmap.zig` and `bench_aa.zig` call
it:

```zig
// array_kernels.zig
pub fn findKey(keys: []const u16, key: u16, comptime cutoff: usize) ?usize {
    if (keys.len <= cutoff) {            // linear scan for small key arrays
        for (keys, 0..) |k, idx| {
            if (k == key) return idx;
            if (k > key) return null;
        }
        return null;
    }
    const idx = lowerBound(keys, key);   // insert-point primitive (also lives here)
    if (idx < keys.len and keys[idx] == key) return idx;
    return null;
}
```

`cutoff` is **`comptime`** so the branch specializes per call site. **Move**
`lowerBound` into `array_kernels.zig` (don't mirror it — one implementation) and
have **both** bitmap lookup functions (`findKey` and any other `lowerBound` caller
in `bitmap.zig`) delegate to the shared version. `bitmap.zig:185`'s `findKey`
becomes a one-line delegate:
`return array_kernels.findKey(self.keys[0..self.size], key, LINEAR_CUTOFF);`
No public bitmap surface changes; `array_kernels.zig` is already in `.paths` (from
11-00).

## Benchmark seam

`bench_aa.zig` benches `array_kernels.findKey` **directly** on constructed key
slices — the lookup in isolation, no container membership check. Prefer this over
benchmarking public `contains` (lookup + a small membership check adds enough noise
to blur the ±3% threshold).

## Benchmark cases (define "neutral or better" concretely)

**Record the current binary-search baseline first** (before replacing it), so
"within ±3%" is measured against a real number, not assumed.

Add to `bench-aa`, all under the fixed build config:
- container counts `{4, 16, 32, 64, 256, 1024}` (spans the cutoff);
- **hit** (present) and **miss** (absent) for each;
- **hit position**: first / middle / last container; **miss position**:
  before-first (key < all), between keys (interior gap), after-last (key > all) —
  linear scan is position-sensitive, binary search isn't.

## Acceptance

- No functional change; differential suites green (`zig build test test64 validate
  validate64 difftest difftest64`).
- Across all cases above, `findKey` is **within ±3% or faster** than the current
  binary search.
- `LINEAR_CUTOFF` chosen as the value that's neutral-or-better across the small-count
  cases; record the chosen value and its bench. **If no positive cutoff satisfies the
  ±3% rule**, fall back to `LINEAR_CUTOFF = 0` (no linear scan) or land **only the
  `lowerBound` consolidation** (moving it into `array_kernels.zig`) — the linear-scan
  fast path is optional, the shared-primitive consolidation is the guaranteed win.
