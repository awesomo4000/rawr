<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 49 (parked): Multithreaded tiny-bitmap contention

**Parked, not drafted.** This is the single follow-up retained from spec 48, recorded so the question is
not lost and is not re-derived from scratch.

## The question

Spec 48 measured tiny bitmaps (cardinality ≤12) at **77.5% of objects but only ~5% of lifecycle time**,
and the owner closed it with **no design change**.

One number survived unresolved: **the tiny tail is 17.075% of allocation calls** — roughly 3.7x its time
share. Spec 48's harness is **single-threaded**, so it **neither confirms nor rejects** that this matters
under allocator contention.

The archetype-F deployment shape from the workload survey is exactly where it might: millions of tiny
bitmaps, concurrently, across threads.

## What would have to be true to act

**No production redesign until a real concurrent workload demonstrates material impact.** Owner decision,
2026-08-23.

## If it is ever picked up

- **Reuse the spec 48 harness** — fixtures, pools, hashes, accounting, and the plain-list references are
  all built and validated (`48-00`, committed `abdf517`). Do not rebuild them.
- The candidate this would inform is **inline small-set storage**, not lazy top-level allocation: spec 48
  established the cost is **per-container and rises across the tiny range**, so lazy allocation of the
  top-level arrays cannot address it.
- Carry spec 48's finding that **the plain-list comparison is host- and allocator-conditioned** — a
  contention benchmark needs its own stable reference, not that one.

## Why it is parked rather than specced

Nothing measurable is blocked on it, and a contention benchmark without a real target workload would
measure a synthetic thread pattern of our own invention. **The trigger is a workload, not a schedule.**
