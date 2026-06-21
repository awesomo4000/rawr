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

Implementation note: this walk recomputes the same per-container offsets that
`getContainerDataOffset` derives. The no-offset-header fallback in
`getContainerDataOffset` (`:128-134`) is already O(n²) (re-walks from the start
each call); the validation walk is O(n) and may share a helper with it, but
refactoring that fallback for speed is **not** required by this spec — correctness
(rejecting OOB) is the only goal.

## Task 2 — `size` cap (fold in)

In the no-run path, `size` is read as a full `u32` from the buffer (`:39`). The
maximum legal container count is 65536 (one per high-16 key). After reading
`size`, reject `size > 65536` with `error.InvalidFormat`. (The run-cookie path
derives `size` from 16 bits + 1, max 65536, so it is already bounded.) Cheap,
and it caps the validation walk and all header-size arithmetic.

## Task 3 — Extend the malformed-input smoke test to the frozen path

The existing smoke test (`bitmap_tests.zig`, "deserialize malformed input
smoke") corrupts a serialized bitmap and calls `RoaringBitmap.deserialize`. It
does **not** exercise `FrozenBitmap`, which is exactly where the missing checks
bite. Extend it (same file, no CRoaring needed):

1. Serialize a known-good **mixed** bitmap (array + bitset + run containers, so
   all three validation branches are hit). Reuse `test_gen` profiles if handy.
2. For each corrupted buffer (reuse the existing corruption modes: bit-flip,
   truncate to random length, zero/corrupt the size field, set a container
   cardinality high), call `FrozenBitmap.init`. The **only** acceptable outcomes:
   - `init` returns a Zig error, **or**
   - `init` succeeds **and** a follow-up `contains` over a sweep of probe values
     plus a full `iterator` drain complete without a crash/panic.
3. A panic/crash is a failure. Run under `zig build test` (Debug, bounds checks
   on) so any surviving OOB surfaces as a panic and fails the test — that's what
   makes this an effective check.

## Acceptance criteria

1. `FrozenBitmap.init` returns `error.InvalidFormat` for: a truncated buffer
   whose header is valid but container data is short; an offset-header entry
   pointing past `data.len` or before the data region; a run container whose
   `n_runs` would read past `data.len`; `size > 65536`.
2. After a successful `init`, `contains` and `iterator` never read outside
   `data` for any input (guaranteed by the Task 1 walk).
3. The frozen malformed-smoke test (Task 3) passes under `zig build test` — for
   every corrupted buffer, `init` either errors or the view is safely traversable.
4. All existing tests still pass (`zig build test`, `zig build validate`,
   `zig build difftest`); valid round-trips through `FrozenBitmap` are unchanged.
5. The immutability/TOCTOU invariant is documented on `FrozenBitmap`.

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

Small enough to do in one pass: Task 2 (trivial), Task 1 (the validation walk —
the substance), Task 3 (the test that proves it). No chunking unless Task 1 grows
beyond expectation while wiring the no-offset-header fallback.
