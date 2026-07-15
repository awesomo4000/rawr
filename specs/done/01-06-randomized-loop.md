<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 01-06: Randomized differential loop + tunability

Chunk of [`01-differential-testing.md`](01-differential-testing.md). The
randomized breadth on top of the deterministic matrix — and the chunk that owns
keeping `difftest` a **practical, run-it-every-time** command rather than a soak
test.

## Dependencies

- **01-03** (`assertAgree`, `buildOracle`).
- **01-02** (`randomMixed`).
- Sequences after **01-04** (matrix) so the loop reuses the same per-op
  comparison code rather than duplicating it.

## Task

Add a randomized loop to `diff_test.zig`: for each iteration, generate A and B via
`test_gen.randomMixed`, build their oracles from the `values` slices, and run the
full operation set (the producing ops + predicates from 01-04) through
`assertAgree` / direct comparison. Vary `run_optimize` across iterations.

### Tunability (the point of this chunk)

The loop's iteration count and profile sizes must be **named constants at the top
of `diff_test.zig`**, e.g.:

```zig
const RANDOM_ITERS: usize = 1000;     // default
const MAX_CHUNKS: usize = 8;          // cap per-bitmap size
const DENSE_MAX: usize = 20000;       // cap dense fill
```

- Default target: **≥1000 random `(A,B)` pairs** across mixed profiles.
- If dense/full-heavy profiles make 1000 impractically slow, **cap `MAX_CHUNKS`
  / dense sizes** so a default run finishes in a few seconds. A practical
  `difftest` beats a thorough one nobody runs.
- Leave the constants obvious so a deeper soak is one edit away.
- Use a **fixed seed** (or a seed printed at startup) so a failure is
  reproducible; print the seed and iteration index on failure.

## Acceptance criteria

1. `zig build difftest` runs ≥1000 random `(A,B)` pairs (default) across mixed
   profiles with zero failures, in a few seconds.
2. Iteration count and profile-size caps are named constants at the top of the
   file.
3. On failure, the seed + iteration index are printed for reproduction.
4. Leak-checking allocator clean across the whole loop (`gpa.deinit() == .ok`).
