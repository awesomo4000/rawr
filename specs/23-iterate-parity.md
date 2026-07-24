<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 23: Iterate parity — diagnosis-first

Address the **largest persistent reported gap** on the accurate parity board: **iterate**,
1.52x (M4) / 1.88x (Zen 4) rawr/CRoaring, present on **both** architectures. Whether it is a
*real* rawr gap is exactly what this spec determines — so it is **diagnosis first, no
preselected cause** (the 20a / 21 discipline: verify the comparison is fair and attribute the
cost before touching code). A fix is a conditional second phase.

Scope note: `iterate` is the idiomatic pull-iteration path (`while (it.next()) |v|`). It does
**not** underlie `toArray` or serialization — those use their own dedicated per-container
loops, not `Iterator.next()` — so closing this gap helps the pull-iteration path, not those
rows.

## The first thing to check — are we comparing like for like?

The two benched paths use **different iteration models**:

- **rawr** — `bm.iterator()` then a pull `while (it.next()) |v|` loop (`src/bitmap.zig:2218`,
  `:2325`): one `next()` call per value, saving/restoring iterator state across every value.
- **CRoaring** — `roaring_iterate(bm, callback, …)` (`src/bench_croaring.zig:1164`): a **push**
  model where CRoaring drives a tight per-container inner loop, no per-value state save/restore.

CRoaring's **push** `roaring_iterate` is typically faster than its own **pull** iterator, so an
unknown share of the 1.5–1.9x may be a **benchmark model mismatch** (the sparse-AND/OR shape),
not a rawr kernel deficiency. This must be resolved before attributing anything to rawr.

## Phase 1 — Diagnosis (benchmark-only; canonical harness + focused split)

Measure **four** paths so the model tax and the like-for-like kernel gap are both quantifiable.
All measured correctly:

- **rawr pull** — `iterator().next()` loop, local checksum, iterator state built **inside** the
  timed region.
- **rawr push (diagnostic)** — a **benchmark-only direct per-container traversal** in Zig that
  **accumulates inline** (a comptime sink, **no runtime callback**), context-owned checksum —
  the shape a real `forEach` would take. Note this is not identical work to CRoaring push: rawr
  can **inline** its sink where `roaring_iterate` uses a **runtime function pointer** — a real
  (language) difference to **report explicitly**, not hide. Without this path there is no rawr
  push number and the model tax cannot be computed; Phase 2 decides whether it deserves a public
  API.
- **CRoaring pull** — a benchmark-only **C wrapper** that stack-initializes a
  `roaring_uint32_iterator_t`, runs the **complete** pull loop **in C**, and returns a **local**
  checksum. Do **not** call `roaring_uint32_iterator_advance` per value from Zig — that measures
  a million Zig→C FFI calls, not iteration.
- **CRoaring push** — a benchmark-only **C wrapper** that runs `roaring_iterate` with a **C**
  callback accumulating into a **local**, returning the checksum (not a Zig callback updating a
  global — that is asymmetric FFI/global work).

Symmetry requirements on every path: **local/context-owned accumulation** (never a global);
iterator/scan state constructed **inside** the timed scan; and during timing accumulate a
**count**, a **sum/checksum**, **and an order-sensitive rolling hash** (a scalar checksum alone
does not catch a wrong sequence or order). For correctness, a **full untimed sequence-equality**
check across paths afterward. Normalize by the **actual deduplicated cardinality**, not the
1,000,000 attempted inserts → report **ns/value**.

From these:
- **model tax** = pull − push on each side (rawr, CRoaring);
- **like-for-like kernel gap** = rawr-pull vs CRoaring-pull, and rawr-push vs CRoaring-push.

### Corpus characterization (expect array-dominated)

The corpus inserts ~1M random values across 65,536 high keys — averaging ~15 values/container,
so it should be **overwhelmingly array containers**. Record exact **array / bitset / run
counts** mechanically, but the dominant inner loop is the **array walk** — do **not** spend
effort on bitset `ctz` attribution unless the counts actually implicate bitsets.

Phase 1 stands alone: "how much is model vs like-for-like kernel, on array containers, on both
hosts" is the deliverable even if no fix follows.

## Phase 2 — Fix (conditional; lever follows the attribution)

- **If the model tax dominates:** add a **bulk push API** (a `forEach`-style call mirroring the
  diagnostic per-container traversal). Additive — the existing pull `iterator()` semantics are
  unchanged. It does **not** auto-accelerate `toArray`/serialization (dedicated loops).
- **If rawr's pull iterator is genuinely slower like-for-like:** tighten `next()` — reduce
  per-value state work, per-container fast path — without changing iteration semantics.
- Other container types / dispatch only if the counts and attribution implicate them.

### Canonical-row outcome (pin it)

- The `iterate` row is scoped as **idiomatic pull iteration**, so it becomes **rawr pull vs
  CRoaring pull** (like-for-like) **unconditionally** — the current pull-vs-push comparison is
  corrected regardless of what the attribution shows.
- If a **public push API** is added, push iteration gets a **separate manifest row**. That
  changes the current **38-row** manifest/count checks (`--list` → 39) and the canonical
  `docs/parity-measurement.md` — call it out explicitly and update both.

## Constraints / measurement

- **Correctness:** every path yields the same value sequence — verified by **full untimed
  sequence equality** (not just an equal scalar checksum), and a differential check (rawr
  iteration == CRoaring order == sorted values) stays green.
- Canonical spec-22 protocol: **3 warmup / 21 timed / median**, **≥5 fresh processes**, full
  min/max range, on **M4 and Zen 4** (the gap is on both). Iteration **does not allocate**, so
  the measured tuple is the **single rawr non-allocating tuple** vs CRoaring — no SMP/libc split.
- Phase 1 is **benchmark-only** (the C wrapper and the rawr diagnostic traversal add no library
  API); a Phase-2 `forEach` would be an additive production API.

## Acceptance

- **Phase 1 GO:** the reported 1.5–1.9x is decomposed into **model tax vs like-for-like kernel
  gap** (all four paths), with the container mix recorded, on both hosts.
- **Phase 2 GO (if attempted):** the like-for-like iterate ratio (the single **rawr
  non-allocating** tuple vs CRoaring) is **≤ 1.10x on both M4 and Zen 4**, with **no canonical
  row regressing by more than 5%**, differential green. If the finding is "like-for-like is
  already near parity and the board was comparing pull-vs-push," that is a valid terminal
  outcome — **correct the row** and record it rather than optimizing a kernel that isn't slow.

## NO-GO

- Phase 1 shows the gap is essentially the benchmark model mismatch and like-for-like iteration
  is already at/near parity → fix the row's comparison (and optionally add the bulk API for
  ergonomics), do not chase a kernel that isn't slow.

## Estimate

S for Phase 1 (four measured paths incl. the C pull wrapper + rawr diagnostic traversal, on the
existing harness). Phase 2 is S–M: a bulk `forEach` is small and additive; a `next()` tightening
is a focused kernel change — chosen by the diagnosis.
