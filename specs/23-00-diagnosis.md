<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 23-00: Iterate diagnosis — four-path decomposition

First chunk of [iterate parity](23-iterate-parity.md). **Diagnosis only, benchmark-only.**
Deliverable: the reported iterate gap (1.52x M4 / 1.88x Zen 4) decomposed so we know how much
is a **pull-vs-push benchmark model mismatch** vs a **real like-for-like pull-iterator gap** —
the input that decides whether `23-01` does a perf fix or just corrects the row.

## The four measured paths (all measured correctly)

- **rawr pull** — `iterator().next()` loop; iterator state built **inside** the timed region.
- **rawr push (diagnostic)** — a benchmark-only direct per-container traversal in Zig that
  **accumulates inline** (comptime sink, **no runtime callback**) — the shape a real `forEach`
  would take. Report explicitly that this inlines where CRoaring's push uses a runtime function
  pointer; it is not identical work.
- **CRoaring pull** — a benchmark-only **C wrapper** that stack-initializes a
  `roaring_uint32_iterator_t`, runs the **complete** pull loop **in C**, returns a **local**
  checksum. **Not** `roaring_uint32_iterator_advance` per value from Zig (that measures a million
  FFI calls).
- **CRoaring push** — a benchmark-only **C wrapper** running `roaring_iterate` with a **C**
  callback accumulating into a **local** (not a Zig callback / global).

## Measurement rules

- **Timed sink is minimal and identical** on every path: **count + wrapping sum only.** No
  order-sensitive rolling hash in the timed loop (a dependent hash can dominate an array walk and
  hide the iterator overhead).
- Local/context-owned accumulation (never a global); scan state constructed inside the timed
  scan.
- Canonical spec-22 protocol: **3 warmup / 21 timed / median**, **≥5 fresh processes**, full
  min/max range, on **M4 and Zen 4**. Iteration does not allocate → the single **rawr
  non-allocating** tuple vs CRoaring.
- **One path per fresh worker process per run.** Each of the four paths is its own isolated
  measurement — do **not** time all four sequentially in a single process, which would
  reintroduce the process-sharing/allocator-history bias spec 22 eliminated. (Each path is a
  distinct tuple in the per-`(row, impl, allocator)` isolation model.)
- Normalize by the **actual deduplicated cardinality**, not the 1,000,000 attempted inserts →
  report **ns/value**.

## Corpus characterization

Record exact **array / bitset / run** container counts mechanically. The corpus (~1M random
values across 65,536 keys, ~15/container) is expected **array-dominated**, so the dominant inner
loop is the array walk — do **not** spend effort on bitset `ctz` attribution unless the counts
implicate bitsets.

## Correctness (untimed)

**Every one of the four paths** has an **untimed validation mode that writes its traversed
values into a caller-provided buffer**; compare each buffer to the **sorted-value oracle** (equal
count, full sequence equality). An aggregate oracle alone does not prove the new C wrappers
traverse correctly. The timed paths' count + wrapping sum must agree; rolling hash + full
sequence comparison are untimed.

## Acceptance

- The 1.5–1.9x is decomposed on **both hosts** into:
  - **like-for-like iterator comparison** — rawr-pull vs CRoaring-pull (the clean kernel number);
  - **push API comparison** — rawr-push vs CRoaring-push (traversal + callback model, not
    like-for-like);
  - **within-implementation pull-vs-push API-model delta** — pull − push per side.
- Exact container-mix counts reported; ns/value normalized by dedup cardinality.
- All four paths self-validated (per-path buffer == sorted oracle); differential green.
- **Benchmark-only:** no library API added, no vendored-source change; build green under
  `ReleaseSafe` and `ReleaseFast`; results committed to `docs/parity-measurement.md`.
- Validation: `zig build test`; the four-path diagnostic executable run via its five-process
  runner on both hosts; `zig build -Doptimize=ReleaseSafe` / `ReleaseFast`.

## Result to record (feeds `23-01`)

Whether the reported gap is mostly the **pull-vs-push model** (like-for-like pull is near
parity) or a **real like-for-like pull-iterator kernel gap** — this decides `23-01`'s lever. The
`iterate` row is corrected to pull-vs-pull **regardless** (in `23-01`).
