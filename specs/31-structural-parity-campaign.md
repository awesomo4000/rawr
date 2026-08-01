<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 31: Structural parity campaign (umbrella)

The map for closing the **remaining M4 SMP gaps above 1.10x** after spec 30:

| row | M4 gap | targeted by |
|---|---:|---|
| clone (dense) | 1.786x | E1, (E5 fallback) |
| lazy OR construction | 1.708x | E1 (array clones), E3 (forced bitsets) |
| bitwiseAnd dense | 1.570x (spec 29 open) | E1, (E5 fallback) |
| select (dense) | 1.486x | E4 |
| orMany (mixed) | 1.248x | E2 |
| lazy OR + repair | 1.155x | E1 + E3 (via construction) |

**This is not chunked directly.** It is the campaign plan: the shared thesis, the five experiments
(E1–E5), their ordering and information dependencies, and the discipline every experiment inherits.
Each experiment is promoted to **its own numbered toplevel spec** when activated (draft → review →
chunk), in the recommended order but **subject to what earlier experiments reveal**.

## Thesis (the spec-30 lesson, generalized)

Spec 30 closed removeRange by removing **provably wasted construction** (8 containers built, 6 of
them immediately freed) — **not** by reducing allocation counts in general. That distinction is the
campaign's spine:

- **Demand reduction wins only where work is provably temporary or redundant** (built-then-freed,
  or metadata guaranteed to die). Clone reproduces every container, dense-AND keeps every produced
  container, `select` allocates nothing — the removeRange trick **does not transfer** to them.
- Those rows need **different levers**: **structural compaction** (E1 — smaller headers, same
  count, same payload class), **avoiding doomed metadata** (E3 — headerless transient bitsets),
  **compute/bandwidth** (E2 n-way OR, E4 select kernel), and, only as a fallback, **allocation
  ordering** (E5).

## Do not re-tread (closed NO-GOs)

Every experiment must show it is **not** one of these in disguise:

- **Spec 13** — header+payload co-located in one allocation → the combined block crossed into the
  next SMP size class. *(E1 explicitly keeps payload separate to avoid exactly this.)*
- **Spec 17** — transient bump-arena over temp containers: arena bulk-free costs more than
  individual SMP frees; lifetime fails the memory gate.
- **Spec 18** — allocator swap / segregated libc-like heap: hurts container-heavy ops broadly.
- **Spec 27** — exact-capacity pre-sizing clone: fewer allocations but **regressed M4 SMP**;
  combined run storage crosses worse classes.
- **Spec 29** — dense-AND scratch-bypass (alloc/free on empties) and pre-sized OR: both regressed
  M4.

A new experiment is legitimate only if it introduces a **new proven mechanism**, not another tuning
pass over one of the above.

## The five experiments

### E1 — Compact separate container headers (highest leverage; run first)

**Targets:** clone, dense-AND, select (locality), lazy-OR construction (array clones) — the widest
blast radius on the board.

**Mechanism (grounded):** `ArrayContainer` and `RunContainer` store a **slice** (`ptr + len`, 16 B)
*and* a separate `capacity` and `cardinality` — so their headers are **24 B → land in the 32-byte
SMP class**. Replace the slice with an **aligned many-pointer** (`[*]align(N) T`, 8 B), reconstruct
bounds at use sites from `capacity`/`cardinality`; the header drops to **16 B → the 16-byte class**.
**Payload stays separate, in its existing power-of-two class** (this is the spec-13 firewall).
Allocation **count unchanged**; only the header slot halves. CRoaring's headers are similarly
compact.

**Not applicable to Bitset:** `BitsetContainer` is already `ptr(8) + i32(4) = 16 B`. E1 is
Array + Run only.

**Risk / surface:** this changes the **core container representation** — every op touches it, and
losing the slice means reconstructing bounds (including `ReleaseSafe` bounds checks) at use sites.
Large correctness surface; benchmark-only prototype first.

**First step:** add a separate-payload/**compact-header** variant to the existing single-allocation
prototype. Measure reserved build, growth, clone, deinit, membership, iteration, dense run-AND, and
select on both hosts. **Assert 16-byte headers and unchanged payload classes** before any production
migration.

### E2 — Fused N-way bitset accumulation for `orMany` (independent; run early)

**Target:** orMany 1.248x — **not** a top-level allocation gap (attribution: ~14.18 of 14.71 µs is
mixed-container accumulation).

**Mechanism:** the corpus maps multiple bitset inputs to one output key; today rawr streams the
destination **once per input bitset**. A **word-major N-way OR** loads each input word, reduces in
registers, and stores the destination **once** — cutting destination memory traffic K-fold. Cells:
(1) baseline zero-then-input-major; (2) clone first bitset instead of zero+OR; (3) word-major N-way
OR per key; (4) first-bitset seeding + word-major.

**Discipline:** first **split accumulation time by array/bitset/run source** — the word-major kernel
only helps the **bitset** share; establish that share before building it. **OR-specific; do not
touch `xorMany`** (already well ahead).

**Independence:** a compute/bandwidth lever, orthogonal to the header work — can run in parallel
with E1.

### E3 — Headerless transient lazy bitsets (after E1)

**Target:** lazy-OR construction / repair — the **forced-bitset** share (forced lazy OR must produce
bitsets; `bitset_conversion=true` is pinned, so routing through eager array union is **out of
bounds**, and the spec-17 arena is closed).

**Mechanism (the closest analogue to removeRangeCopy — avoid metadata guaranteed to die):** an
**unrepaired** lazy bitset has implicitly-unknown cardinality and may not need a separately
allocated 16-byte `BitsetContainer` header until repair. Allocate **only the aligned 8 KB words**;
represent internally as a **transient lazy-bitset tag** (the `reserved = 0b11` tag slot is free);
repair computes cardinality directly; **if it demotes, free the words with no header ever
allocated**; if it survives, allocate the normal 16-byte header and adopt the words.

**Stop-gate:** prototype benchmark-only; **count exactly how many header allocations disappear** and
measure construction, repair, and full lifecycle (including repeated lazy ops and
deinit-before-repair). **If removing the small headers cannot materially move the M4 result, stop
before changing the container union** — the payoff is the doomed-header **alloc call**, not bytes,
so it must clear the bar on call-count alone.

**Depends on E1:** if E1 already makes headers cheap, E3's marginal value shrinks — evaluate E3
against E1's measured header cost.

### E4 — `select`: container-skip kernel matrix (independent)

**Target:** select 1.486x — **no allocation** (select allocates nothing); the **top-level
cardinality walk** dominates.

**Matrix:** current scalar walk; 2-container and 4-container unrolled walks; homogeneous-run
specialization; **precomputed prefix-cardinality lookup as a ceiling experiment only**; plus rawr
vs CRoaring disassembly and branch counts on the canonical corpus.

**Decision rule:** if unrolling or homogeneous dispatch closes it → ship, no storage change. If
**only** prefix cardinalities close it, choose **explicitly** between (a) an **optional caller-owned
`RankSelectIndex`** (helps indexed users, does **not** close the base row) or (b) **maintained
bitmap metadata** (must pay mutation + memory gates across the whole board). **Do not add a
permanent index until the ceiling experiment proves it recovers the full gap.**

### E5 — Clone / dense-AND allocation ordering (fallback only)

**Only if E1 does not close clone / dense-AND.** The remaining unexplored allocator lever is
**ordering**, not count or size: interleaved vs all-headers-then-payloads-grouped-by-class;
interleaved vs grouped teardown; and for dense-AND a **two-pass run-result plan** that fills all
permanent outputs directly **without scratch construction** (distinct from spec 29's rejected
bypass: compute cardinalities first, then allocate exact and fill once — no scratch, no empty
allocs). Matches the observed allocator-history sensitivity while preserving counts and
representation.

**Lower confidence** (exact sizing, combined blocks, scratch bypass already failed) —
**benchmark-only unless focused M4 exceeds noise and Zen 4 stays neutral.**

## Recommended order + information dependencies

**E1 → E2 → E3 → E4 → E5.**

- **E1 first** is both highest-leverage *and* information-ordering: its result **gates E5** (if
  compact headers close clone/dense-AND, E5 is moot) and **informs E3** (a cheap 16-byte header
  weakens the case for removing it). One structural experiment that could move three of the largest
  rows, with a layout that specifically dodges spec 13's size-class failure.
- **E2 and E4 are independent** compute levers (orMany, select) — orthogonal to headers, safe to
  interleave / parallelize; sequenced here after E1 only to keep one active structural change at a
  time.
- **E3 after E1**, evaluated against E1's measured header cost.
- **E5 last**, and only if E1 leaves clone/dense-AND open.

## Shared experimental discipline (every experiment inherits)

1. Assert the **canonical corpus** and exact container/type inventory before timing.
2. **Benchmark-only A/B cells before any production change.**
3. Report **allocations, frees, requested bytes, effective SMP-class bytes, and teardown** — kept
   distinct (container instances ≠ allocator calls, per spec 30).
4. Validate **byte / set identity outside timing** (+ CRoaring differential).
5. **Five fresh-process medians + full ranges on M4 and Zen 4** (canonical protocol).
6. Adopt **one architecture-neutral shape** only (per the spec-30 Zen 4 policy: within-noise passes;
   a real regression needs an explicit owner exception).
7. Fresh **full-board before/after gate**; investigate any untouched movement > 5% (spec-28 layout
   exception: stable focused timing *and* instruction-identical disassembly for **untouched** rows).
8. **Retain partial wins only when they introduce a new proven mechanism**, not another tuning pass
   over a prior NO-GO. Parity stays a hard requirement — a row **closes** at ≤ 1.10x; a partial is
   adopted by owner judgement and the row stays open.

## Numbering plan

- **31** — this umbrella (not chunked).
- Each experiment → **its own toplevel spec number** on activation (E1 first), with its own
  diagnostic-first chunks (`NN-00` prototype+measure, later chunks conditional on the numbers).
  Numbers are assigned **when activated**, not reserved now — order and inclusion may change with
  findings.

## Immediate next step

Promote **E1 (compact headers)** to its own toplevel spec, draft it in full (prototype variant,
measurement matrix across both hosts, the assert-16-byte-headers gate, the production-migration
decision), take review, then chunk. Hold E2–E5 as briefs here until E1's numbers land.
