# Spec 04: FrozenBitmap bounds-validation (malformed-input safety)

## Goal

Make `FrozenBitmap.init` reject malformed or truncated serialized bytes instead
of constructing a view that performs **out-of-bounds reads** on first access.
After `init` returns success, every accessor (`contains`, `iterator`,
`cardinality`) must be guaranteed in-bounds for the borrowed buffer.

This closes the one concrete finding from the first security review (baseline,
this session). Scope is deliberately narrow: **the frozen zero-copy path only.**
A full `deserialize_safe` for the allocating path, and semantic validation
(key ordering, cardinality-vs-population consistency), are explicitly **out of
scope** here — see "Out of scope" below.

## Background / threat model

`FrozenBitmap` (`src/frozen.zig`) is a zero-copy read-only view over borrowed
bytes — its doc comment advertises "*zero-copy reads from mmap'd LMDB values*".
Whenever those bytes can originate from anything the application did not itself
produce and fully trust (a network peer, a user upload, a shared/mmap'd store a
hostile process can write), they are attacker-influenced input.

Zig built in `ReleaseFast` (the mode the interop/bench targets use) has **no
bounds checking**, so an out-of-range slice access is a real OOB memory read, not
a checked panic.

### The defect (what init fails to check)

`FrozenBitmap.init` (`frozen.zig:20-68`) validates only the **header** region:
cookie, run-bitset fit (`:34`), `data.len >= 8` for the no-run cookie (`:38`),
and that the keys + offset headers fit (`pos > data.len`, `:57`). It does **not**
validate:

1. **The container-data region.** Container bytes live at/after `data_offset`,
   whose extent is never checked against `data.len`. A buffer with a valid header
   but **truncated** container data passes `init`, then OOB-reads on first
   `contains`/`iterator`.
2. **Offset-header values.** `getContainerDataOffset` (`:121-125`) reads a `u32`
   absolute offset straight from the buffer and returns it as an index into
   `self.data` with no range check — fully attacker-controlled.
3. **`n_runs`** read from the data prefix (`:143`, `:168`, `:335`) is used to size
   reads with no check that `2 + n_runs*4` fits.

Unchecked read sites that depend on the above: `containerContains` bitset
(`:176`) and array (`binarySearchArray`, `:184-201`); run (`:168-170`,
`searchRuns :203-224`); and the `Iterator` for all three types (`:267-365`,
`initContainer :330-365`).

### Residual assumption (document, don't fix here)

Zero-copy means the borrowed buffer must remain **immutable for the lifetime of
the view**. Validation at `init` is only sound if the bytes don't change
afterward; a buffer an attacker can mutate concurrently (e.g. a live mmap) is a
TOCTOU surface inherent to any zero-copy reader and is out of scope. State this
invariant in the `FrozenBitmap` doc comment.

## Task 1 — One-pass bounds validation in `FrozenBitmap.init`

Harden `init` itself (not a separate `initSafe`). After the existing header
checks, add a single walk over all `size` containers that verifies each
container's declared data region lies within `data`. On any failure return
`error.InvalidFormat`.

For each `idx` in `0..size`:

1. `card = getCardinality(idx)` — reads the descriptive header, which is already
   inside the validated header region, so this read is safe. `card ∈ [1, 65536]`.
2. `is_run = isRunContainer(idx)` — `run_bitset` is length `(size+7)/8` and
   `idx < size`, so `idx/8` is in bounds; safe.
3. Determine the container's start offset:
   - **Offset header present** (`offsets_offset != 0`): read the `u32` from the
     offsets region (safe — inside validated header). The **value** is untrusted:
     require `data_offset >= self.data_offset` (must point at/after the data
     region, not back into the header) **and** `data_offset <= data.len`.
   - **No offset header** (small run format, `size < NO_OFFSET_THRESHOLD`):
     compute sequentially, accumulating from `self.data_offset`. Each step needs
     the previous container's size, which for runs requires reading `n_runs` —
     bounds-check each `n_runs` read (step 4) before trusting it.
4. Compute and bounds-check the container's extent:
   - **run**: require `data_offset + 2 <= data.len` before reading `n_runs`;
     then require `data_offset + 2 + @as(usize, n_runs)*4 <= data.len`.
   - **bitset** (`card > ArrayContainer.MAX_CARDINALITY`): require
     `data_offset + BitsetContainer.SIZE_BYTES <= data.len` (8192).
   - **array**: require `data_offset + @as(usize, card)*2 <= data.len`.

All arithmetic in `usize` (the struct already widens `size*4` to `usize` at
`:46/:54`, so the Finding-2 `u32` overflow does not exist here — keep it that
way). If a check fails, `return error.InvalidFormat`.

Because the borrowed buffer is immutable for the view's lifetime, the `card` and
`n_runs` values read during validation equal those read during access, so a
successful `init` guarantees every later accessor is in-bounds.

**Zero-allocation requirement:** `FrozenBitmap.init` is allocation-free today and
must stay that way. The validation is a **bounds walk only** — no heap
allocation, no scratch buffers. It reads fields already in the buffer and
compares offsets; nothing more.

Implementation note (non-binding): a `validateContainerBounds()` helper called
from `init` after the offsets are constructed, plus a checked
`containerDataOffset(idx, sequential_offset)`-style helper (or a local loop),
keeps this clean. This walk recomputes the same per-container offsets that
`getContainerDataOffset` derives. The no-offset-header fallback in
`getContainerDataOffset` (`:128-134`) is already O(n²) (re-walks from the start
each call); the validation walk is O(n) and may share a helper with it, but
refactoring that fallback for speed is **not** required by this spec — correctness
(rejecting OOB) is the only goal.

## Task 2 — `size` cap (fold in)

In the no-run path, `size` is read as a full `u32` from the buffer (`:39`). The
maximum legal container count is 65536 (one per high-16 key). Reject `size >
65536` with `error.InvalidFormat` **immediately after reading the no-run size,
before any header-size arithmetic** (the `pos += size * 4` computations at
`:46`/`:54`) — don't compute offsets from an unbounded `size` first. (The
run-cookie path derives `size` from 16 bits + 1, max 65536, so it is already
bounded.) Cheap, and it caps the validation walk and all header-size arithmetic.

## Task 3 — Deterministic malformed reproducers (write these FIRST, red → green)

Drive this fix test-first. Before touching `init`, write a deterministic
reproducer for each unsafe shape below. Each constructs a buffer that **today's
`init` wrongly accepts** (and whose first `contains`/`iterator` access would read
OOB — in Debug that may already panic). After Tasks 1–2 land, each test's final
asserted behavior is simply:

```zig
try std.testing.expectError(error.InvalidFormat, FrozenBitmap.init(bad_bytes));
```

The point of writing them first is to *prove the bug exists* (init currently
accepts the buffer) and to lock in regression coverage on the exact bad fields —
randomized corruption can't be relied on to hit them. Required cases, one
deterministic test each:

1. **Valid header, truncated array container data** — header declares an array of
   cardinality N, buffer cut so fewer than `N*2` value bytes remain.
2. **Valid header, truncated bitset container data** — header declares a bitset,
   buffer cut so fewer than 8192 word bytes remain.
3. **Valid header, truncated run container data** — header declares a run
   container; `n_runs = N` in the data prefix but fewer than `2 + N*4` bytes
   follow.
4. **Offset table entry before `data_offset`** — an offset-header entry points
   back into the header region (`< data_offset`).
5. **Offset table entry past `data.len`** — an offset-header entry points beyond
   the buffer end.
6. **No-run cookie with `size > 65536`**.

Build the start fixtures by serializing a real bitmap of the right container
type, then hand-corrupt the specific field/length. **The bitset fixture must be a
real bitset container** — do **not** use `addRange` for it, because contiguous
ranges now store as run containers. Force a bitset with scattered individual
adds, `fromSorted` of >4096 non-contiguous values, or the `test_gen` `dense`
profile. Likewise build an actual run container via `addRange` + `runOptimize`
for cases 3.

## Task 4 — Extend the randomized malformed smoke test to the frozen path

Keep this as a backstop *after* the deterministic reproducers pass. The existing
smoke test (`bitmap_tests.zig`, "deserialize malformed input smoke") corrupts a
serialized bitmap and calls `RoaringBitmap.deserialize`; it does **not** exercise
`FrozenBitmap`. Extend it (same file, no CRoaring needed):

1. Serialize a known-good **mixed** bitmap that truly contains **array + bitset +
   run** containers so all three validation branches are hit. Per Task 3, build
   the bitset branch with scattered adds / `fromSorted` / `test_gen` `dense` (not
   `addRange`), and the run branch via `addRange` + `runOptimize`.
2. For each corrupted buffer (reuse the existing corruption modes: bit-flip,
   truncate to random length, zero/corrupt the size field, set a container
   cardinality high), call `FrozenBitmap.init`. The **only** acceptable outcomes:
   - `init` returns a Zig error, **or**
   - `init` succeeds **and** a follow-up `contains` over a sweep of probe values
     plus a full `iterator` drain complete without a crash/panic.
3. A panic/crash is a failure. Run under `zig build test` (Debug, bounds checks
   on) so any surviving OOB surfaces as a panic and fails the test.

## Acceptance criteria

1. Each of the six deterministic reproducers in Task 3 asserts
   `expectError(error.InvalidFormat, FrozenBitmap.init(bad_bytes))` and passes.
   (Each must have been demonstrated to *fail* against the pre-fix `init` — i.e.
   pre-fix `init` wrongly accepted the buffer — so the tests prove the bug.)
2. After a successful `init`, `contains` and `iterator` never read outside
   `data` for any input (guaranteed by the Task 1 walk).
3. `FrozenBitmap.init` remains **allocation-free** (Task 1).
4. The randomized frozen smoke test (Task 4) passes under `zig build test` — for
   every corrupted buffer, `init` either errors or the view is safely traversable.
5. All existing tests still pass (`zig build test`, `zig build validate`,
   `zig build difftest`); valid round-trips through `FrozenBitmap` are unchanged.
6. The immutability/TOCTOU invariant is documented on `FrozenBitmap`.

## Out of scope (explicitly)

- `deserialize_safe` / hardening of the allocating `serialize.zig:deserialize`
  path (it is already bounded by `Reader.fixed`; the only soft spot is the
  `size*2` `u32` overflow at `serialize.zig:227`, gated behind allocation
  failure — track separately if desired).
- Semantic validation: key strict-ascending order, and cardinality-vs-actual-
  population consistency. These produce wrong answers, not OOB, and belong with a
  future `deserialize_safe`.
- Defending against concurrent mutation of the borrowed buffer (TOCTOU) — inherent
  to zero-copy; documented, not fixed.

## Sequencing for the implementer

Test-first, single pass:

1. **Task 3 reproducers** — write the six deterministic malformed buffers and
   confirm pre-fix `init` wrongly accepts them (the bug, demonstrated). This is
   the red step.
2. **Task 2** (size cap, trivial) then **Task 1** (the validation walk — the
   substance) until all six reproducers go green.
3. **Task 4** (randomized smoke backstop) and the doc note (AC 6).

No chunking unless Task 1 grows beyond expectation while wiring the
no-offset-header fallback.
