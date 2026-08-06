<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 31: Structural parity campaign (umbrella)

> **Wave 1 outcome (2026-08-06).** **E1 (spec 32) — Run header GO (major), Array header NO-GO.** The
> compact `RunContainer` header (32→16-byte SMP class) closed a whole cluster on M4: clone
> 1.788x→0.672x, dense-AND 1.587x→0.845x, dense-OR 1.138x→0.702x, select 1.387x→0.864x, removeRange
> 0.803x→0.290x — all rawr-faster. Also closed **spec 29** (dense-AND) as a side effect. Array header
> made lazy-OR worse → rejected. **E2 (spec 33) — orMany GO** (seeded word-major reduction, shipped).
> **E4 (spec 34) — kernel NO-GO**; select closed via E1's Run-header locality, not unrolling.
> **E5 is now MOOT** — clone/dense-AND closed by E1, so no allocation-ordering fallback needed.
> **Remaining open: lazy-OR construction (~1.7x)** — Array header (the expected E1 lever there) was
> NO-GO, so **E3 (headerless transient lazy bitsets) is the live remaining experiment** (Wave 3,
> against a post-E1 rebaseline). Specs 32/33/34 + chunks in `specs/done/`; shipped `d7d357b`.

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

**Array and Run are decided independently.** They target different rows (Array headers →
lazy-OR-construction array clones; Run headers → clone / dense-AND / select) and may produce
different results. The child spec requires **independent prototype cells and independent GO/NO-GO
decisions** — do **not** couple the two representation migrations, and do **not** infer Run
performance from the prior Array prototype: E1 needs **real compact-header Run replicas** measured on
dense clone, dense-AND, and select.

**Pointer contract (child spec must pin):**
- **`cardinality`/`n_runs` bound the readable** values/runs; **`capacity` controls growth and
  deallocation** (the freed length).
- Use sites reconstruct a **temporary slice** (`ptr[0..cardinality]` / `ptr[0..capacity]`) so
  **`ReleaseSafe` bounds checking is restored** at access.
- **Tagged-pointer alignment stays valid** (the 2-bit tag still fits the pointer's low bits).
- Header **`@sizeOf` / `@alignOf`** (24 B → 16 B) are **compile-time asserted**. The **SMP slot
  class** (32-byte → 16-byte) is **allocator behavior, not a compile-time type property** — it must
  be **calculated and reported by the benchmark's class accounting on each host**, not asserted from
  the type.
- Production migration gets **exhaustive allocation-failure testing** (each op's changed path).

**Risk / surface:** this changes the **core container representation** — every op touches it, and
losing the slice means reconstructing bounds at use sites. Large correctness surface; benchmark-only
prototype first.

**First step:** add a separate-payload/**compact-header** variant to the existing single-allocation
prototype (Array and Run as **separate** cells). Measure reserved build, growth, clone, deinit,
membership, iteration, dense run-AND, and select on both hosts. **Assert 16-byte headers and
unchanged payload classes** before any production migration.

### E2 — Fused N-way bitset accumulation for `orMany` (independent; run early)

**Target:** orMany 1.248x — **not** a top-level allocation gap (attribution: ~14.18 of 14.71 µs is
mixed-container accumulation).

**Mechanism:** the corpus maps multiple bitset inputs to one output key; today rawr streams the
destination **once per input bitset**. A **word-major N-way OR** loads each input word, reduces in
registers, and stores the destination **once** — cutting destination memory traffic K-fold. Cells:
(1) baseline zero-then-input-major; (2) clone first bitset instead of zero+OR; (3) word-major N-way
OR per key; (4) first-bitset seeding + word-major.

**Pin how the per-key bitset pointers are gathered** for word-major traversal — a **fresh allocation
per output key** to hold the pointer list could **erase the gain**. The child spec must state the
collection mechanism (e.g. a reused scratch pointer buffer) and **include collection overhead inside
the full mixed-corpus cell's timing** — it may **not** be silently excluded.

**Discipline:** first **split accumulation time by array/bitset/run source** — the word-major kernel
only helps the **bitset** share; establish that share before building it. **Assert bitset
multiplicity per output key** — word-major accumulation only helps keys that receive **multiple**
bitset inputs; a corpus where each key gets one bitset gains nothing. Add a **bitset-only ceiling
cell** (the maximum the kernel could recover) **before** the full mixed-corpus cell, and verify
**unknown-cardinality handling** and **input immutability** (inputs are read, never mutated).
**OR-specific; do not touch `xorMany`** (already well ahead).

**Independence:** a compute/bandwidth lever, orthogonal to the header work — can run in parallel
with E1.

### E3 — Headerless transient lazy bitsets (after E1)

**Target:** lazy-OR construction / repair — the **forced-bitset** share (forced lazy OR must produce
bitsets; `bitset_conversion=true` is pinned, so routing through eager array union is **out of
bounds**, and the spec-17 arena is closed).

**Mechanism (the closest analogue to removeRangeCopy — avoid metadata guaranteed to die):** an
**unrepaired** lazy bitset has implicitly-unknown cardinality and may not need a separately
allocated 16-byte `BitsetContainer` header until repair. Allocate **only the aligned 8 KB words**;
represent internally as a **transient lazy-bitset tag**; repair computes cardinality directly; **if
it demotes, free the words with no header ever allocated**; if it survives, allocate the normal
16-byte header and adopt the words.

**The `0b11` tag slot is NOT operationally free — and the child spec RENAMES it.** `Container` *does*
have a `.reserved` member, but it is **`reserved: void`** and `fromTagged` **discards the pointer**;
existing generic paths **return false/zero, skip deallocation, or treat it as unreachable**. Keeping
that name would let those dangerous arms keep compiling, so spec 35 **renames the tag to
`.lazy_bitset = 0b11` with a real payload** (`lazy_bitset: *align(64) [1024]u64`) — the rename makes
every unhandled switch arm a **compile error**, turning the required dispatch/lifecycle inventory
into a compiler-checked invariant. That inventory covers: repeated lazy operations, **clone**,
**repair failure**, **deinit-before-repair**, **serialization**, **validate**, **eager-op dispatch**,
and **generic queries** (contains, cardinality, rank/select, iteration) — each with pinned, tested
behavior, no default fall-through. See [35-headerless-transient-lazy-bitsets.md](35-headerless-transient-lazy-bitsets.md).

**Eliminated vs deferred headers (the load-bearing distinction).** A **surviving** lazy bitset still
needs its normal header **at repair** — its allocation is merely **moved from construction to
repair**, not removed. Only bitsets that **demote** permanently eliminate a header allocation. The
diagnostic must report all five:

1. headers **permanently eliminated** (via demotion),
2. headers **deferred** to repair (survivors),
3. **construction-only** allocation reduction,
4. **full construction-plus-repair** allocation reduction,
5. **repair regression** from allocating surviving headers there.

Without this split, E3 could greatly improve the **construction-only** row while doing nothing for —
or **regressing** — the **combined** row. The gate is the **combined** row.

**Numeric stop-gate:** prototype benchmark-only; **count exactly how many header allocations
disappear** (permanently, per split above) and measure construction, repair, and full lifecycle
(repeated lazy ops, deinit-before-repair). Pin a **numeric bar** before touching the container
union — either a **required focused-time improvement on the combined construction-plus-repair
lifecycle**, or a **demonstrated projected path to the ≤ 1.10x row gate** (permanently-eliminated
header calls × measured per-call cost ≥ the residual). "Materially move" is not acceptable as the
gate; the payoff is the **permanently-doomed** header **alloc call**, not bytes and not deferred
survivors.

**Depends on E1 as a REBASELINE, not a header-cost change.** E1 excludes Bitset, so it does **not**
make E3's bitset headers cheaper. But E1's **Array** compact-header lands in the same lazy-OR
construction path (the 2-way-merge array clones), so **lazy-OR construction / repair must be
re-measured after E1** before E3's numbers mean anything. E3's own lever (skipping doomed bitset
headers) is independent of E1's header size.

### E4 — `select`: container-skip kernel matrix (independent)

**Target:** select 1.486x — **no allocation** (select allocates nothing); the **top-level
cardinality walk** dominates.

**Matrix:** current scalar walk; 2-container and 4-container unrolled walks; homogeneous-run
specialization; **precomputed prefix-cardinality lookup as a ceiling experiment only**; plus rawr
vs CRoaring disassembly and branch counts on the canonical corpus.

**Homogeneous-run specialization risks repeating a rejected experiment** — a prior integrated run
loop already **regressed** (`docs/parity-measurement.md`). The child spec must **explain how
homogeneous-run dispatch differs** from that, or **retain it explicitly as a control**, not a
presumed candidate. **Prefix cardinalities remain the strongest ceiling experiment.**

**Tooling:** **disassembly and focused timing are mandatory** on both hosts; **branch-counter
collection is best-effort where host tooling permits** — Apple M4 branch counters may not be
reachable through the same tooling as Zen 4, so a missing M4 branch count does not block the
experiment.

**Decision rule:** if unrolling or homogeneous dispatch closes it → ship, no storage change. If
**only** prefix cardinalities close it, choose **explicitly** between (a) an **optional caller-owned
`RankSelectIndex`** (helps indexed users, does **not** close the base row) or (b) **maintained
bitmap metadata** (must pay mutation + memory gates across the whole board). **Do not add a
permanent index until the ceiling experiment proves it recovers the full gap.**

### E5 — Clone / dense-AND allocation ordering + direct construction (fallback only; two experiments)

**Only if E1 does not close clone / dense-AND.** Conceptually **two separate experiments**, do not
conflate:

- **E5a — allocation ordering, counts/classes unchanged:** interleaved vs
  all-headers-then-payloads-grouped-by-class; interleaved vs grouped teardown. Preserves counts and
  representation; probes the observed allocator-history sensitivity only.
- **E5b — two-pass direct run-result constructor (dense-AND):** eliminate scratch allocation by
  filling all permanent outputs directly. Distinct from spec 29's rejected bypass, but **"compute
  cardinalities first" is insufficient** — the **first pass must determine the exact non-empty
  output keys, each result's container type, and each run count, WITHOUT allocating**; only then does
  the second pass allocate exact and fill once (no scratch, no empty allocs).

**Lower confidence** (exact sizing, combined blocks, scratch bypass already failed) —
**benchmark-only unless focused M4 exceeds noise and Zen 4 stays neutral.**

## Waves (diagnostics parallelize; production integration is serial)

**Diagnostics parallelize; only production integration serializes** (one representation/behavior
change adopted at a time so a board-gate movement is attributable to a single change).

- **Wave 1 — diagnostics, in parallel:** **E1, E2, E4.** Independent at the diagnostic stage — E1 is
  structural (headers), E2 and E4 are compute levers (orMany, select) orthogonal to headers.
  Prototype and measure all three concurrently.
- **Wave 2 — adoption, one at a time:** integrate at most **one production change at a time**, each
  behind its own full-board gate. **If E1 is a GO, adopt its Array and Run changes independently and
  BEFORE any E2/E4 production change**, then **rebase and re-measure E2/E4** — both access the
  affected container representations, so their pre-E1 numbers are stale once E1 lands.
- **Wave 3 — E3**, after the **post-E1 lazy-OR rebaseline** (E1's Array-clone change moves the
  lazy-OR construction baseline E3 measures against; E1 does **not** change bitset-header cost).
- **Wave 4 — E5**, only if E1 leaves clone/dense-AND open — and E5a (ordering) vs E5b (direct
  construction) are separate experiments.

**Why E1 still leads structurally** (not sequentially): it is highest-leverage (could move clone,
dense-AND, select, lazy-array-clones), its result **gates E5** (if it closes clone/dense-AND, E5 is
moot) and **shifts the lazy-OR baseline E3 measures against** — with a layout that dodges spec 13's
size-class failure. Leading structurally does **not** mean E2/E4 wait: their diagnostics run in
Wave 1 alongside E1.

**Parallel-work hygiene:** every diagnostic agent/branch **records the same baseline commit and
benchmark artifact**; a production candidate is **re-run after rebasing onto the latest accepted
campaign state** before its board gate (no candidate is judged against a stale baseline). Give
parallel agents **disjoint diagnostic files/scripts** (E1 / E2 / E4 each own their own bench module);
**shared integration points — `build.zig`, parity infrastructure, measurement documentation — are
owned by the implementer**, not edited concurrently by diagnostic branches.

## Shared experimental discipline (every experiment inherits)

1. Assert the **canonical corpus** and exact container/type inventory before timing.
2. **Benchmark-only A/B cells before any production change.**
3. Report **allocations, frees, requested bytes, effective SMP-class bytes, and teardown** — kept
   distinct (container instances ≠ allocator calls, per spec 30).
4. Validate **operation-appropriate identity outside timing** — byte-identity where a serialized
   form is defined, set-identity + CRoaring differential otherwise; pick per experiment, don't
   assume byte-identity everywhere.
5. **Five fresh-process medians + full ranges on M4 and Zen 4** (canonical protocol).
6. Adopt **one architecture-neutral shape** only (per the spec-30 Zen 4 policy: within-noise passes;
   a real regression needs an explicit owner exception).
7. **Full-board before/after gates PRODUCTION ADOPTION, not every diagnostic prototype** — the
   benchmark-only A/B cells (step 2) do not each trigger a full board run; the fresh full-board gate
   runs when a shape is proposed for the shipping path. Investigate any untouched movement > 5%
   (spec-28 layout exception: stable focused timing *and* instruction-identical disassembly for
   **untouched** rows).
8. **Any ownership change** (new representation, adopted words, transient tags) requires
   **OOM / allocation-failure injection** — valid-or-cleanly-errored, source untouched, no leak.
9. **Retain partial wins only when they introduce a new proven mechanism**, not another tuning pass
   over a prior NO-GO. Parity stays a hard requirement — a row **closes** at ≤ 1.10x; a partial is
   adopted by owner judgement and the row stays open.

## Numbering plan

- **31** — this umbrella (not chunked).
- Each experiment → **its own toplevel spec number** on activation, with its own diagnostic-first
  chunks (`NN-00` prototype+measure, later chunks conditional on the numbers). Numbers are assigned
  **when activated**, not reserved now — order and inclusion may change with findings.

## Immediate next step

Promote the **Wave 1** experiments — **E1 (compact headers), E2 (n-way OR fusion), E4 (select
kernel matrix)** — to their own toplevel specs and **chunk them concurrently**; their diagnostics
run in parallel. Each: prototype variant, measurement matrix across both hosts, the experiment's
assert/ceiling gate, the production-migration decision, take review, then chunk. **Production
integration stays serial** (Wave 2 — one change at a time). E3 (Wave 3) and E5 (Wave 4) stay briefs
here until their preconditions (post-E1 rebaseline; E1 leaving rows open) are met.
