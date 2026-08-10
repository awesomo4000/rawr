<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 39: Descending free order for mass bitset release

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). Predecessor:
[38-00](38-00-address-sort-measurement.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md),
[smp-free-order.md](../docs/smp-free-order.md).

## Target workload — CORRECTED

`38-00` measured an **all-bitset teardown corpus**, which does **not** correspond to any production step:
in canonical sparse lazy-OR the 16,364 transient bitsets are **demoted and freed inside
`repairAfterLazy`**, and the *resulting* bitmap holds **65,496 array containers**. So `deinit()` of a
lazy-OR result is an **array** teardown, not a bitset teardown, and **a default-off `deinit` option
cannot close the lazy-OR construction gap.**

**The production home of "free ~16,364 8 KB bitsets" is `repairAfterLazy`'s demote path.** That is this
spec's primary target.

### Which canonical row can actually move — and which cannot

| row | can this improve it? |
|---|---|
| `lazy-or-construction` | **NO — that row never calls repair.** |
| `lazy-or-repair-only` | **Probably NO** — the expected benefit lands on the *next* construction, which is **outside that row's timer.** |
| **steady-state `lazyOr+repair`** | **YES — this is the only row that can move**, because one iteration's descending frees condition the next iteration's allocations. |

### BLOCKING: an opt-in cannot close a default-path row

**`repairAfterLazyWithOptions` is opt-in and `repairAfterLazy()` is unchanged — so this cannot close the
canonical default-path row.** Claiming the canonical row from an opt-in measurement would be false
reporting. (This is the same contradiction spec 38 resolved and this spec reintroduced; it is resolved
here explicitly.)

**Two admissible positions — `39-00` measures for both, the owner picks:**

- **(A) Optional SMP throughput feature.** The canonical default row is **NOT claimed**. Results are
  reported against a **separate opt-in benchmark variant row** — there is precedent on the board for
  exactly this shape (`bitwiseAnd (sparse, arena)` / `bitwiseOr (sparse, arena)` sit beside their default
  rows). Honest, low-risk, **moves no board row**.
- **(B) Becomes the default.** Requires `39-00` to show the mechanism is **non-harmful on every
  allocator** — note `38-00` found **libc regressed on M4** for teardown; whether that holds for the
  repair-demote path is **unknown and must be measured**. Only if libc is unharmed is (B) available.

**Reporting rule, non-negotiable:** while the API is opt-in, results are reported **as a variant row**,
never as the canonical row. The canonical row may be claimed **only** if (B) is adopted and the default
path actually changes.

**Diagnostic sequence (pinned):**

**One timed cycle is exactly — and teardown is INSIDE it:**

```
timed span = [ lazyOr construction  →  repairAfterLazy(candidate)  →  result teardown ]
```

**Corrected from an earlier draft, which excluded teardown.** The canonical row includes it:
`bench_parity_worker.zig` declares `allocating_teardown = "result deinit/free inside timing"`, and
`benchRawrLazyOrSparseRepairWithAllocator` runs `defer result.deinit()` before the timed function
returns. **The hard gate must therefore be `construction + repair + result teardown`.**

**This is not merely about matching the row — excluding teardown would hide the most likely way the
mechanism fails.** Result teardown is a *larger* free burst (~65,496 arrays) running immediately after
repair's reordered frees, so it can **overwrite the very allocator conditioning those frees produced**
before the next construction ever sees it. A measurement that stops before teardown would report a
benefit the full cycle does not have.

- **Teardown order itself stays UNCHANGED (default) across all arms** — it is not a second intervention;
  only repair's frees vary. But it is **timed**, not excluded.
- **Phase timings may still be reported separately** — construction, repair, and teardown — as
  attribution beneath the full-cycle gate.

**Report separately:** (i) **repair cost**, (ii) **following construction cost**, (iii) **full cycle**.
**The hard gate is FULL-CYCLE improvement.** A repair-only improvement that does not appear in the full
cycle is not a win; a repair-only regression offset by a larger construction gain **is** one — which is
why the cycle is the gate.

**Noise is a DIAGNOSTIC, not the gate.** The injected stage-3 allocator noise exists to prove the effect
survives a shared allocator. **The adoption number must come from a separate no-injected-noise run of a
CANONICAL-EQUIVALENT OPT-IN VARIANT** — identical to the canonical row in corpus, boundaries and protocol,
differing *only* in that it opts into the candidate. (A genuinely *unchanged* canonical row cannot
exercise an opt-in mechanism at all, so "rerun the unchanged row" — an earlier draft's wording — would
measure nothing.) Report both; never substitute the noise-injected figure for the variant one.

**Note what `38-00` did and did not test:** it sorted repair's **cardinality read traversal** (regressed)
and separately measured **descending frees on a synthetic bitset corpus** (won, survived stage-3 noise).
It did **not** test **descending demote frees inside repair** — the combination this spec targets.

Two workloads must be benchmarked, and a **representative container mix** used for anything claiming to
be teardown:

| workload | contents | role |
|---|---|---|
| **repair demote frees** | 16,364 transient 8 KB bitsets, released in a **deferred descending pass** (see below) | **primary** — production path |
| **result teardown** | the real lazy-OR result: **~65,496 arrays** (+ any surviving bitsets/runs) | representative teardown; the array-dominated reality |
| all-bitset teardown | 16,364 bitsets, no interleaving | `38-00`'s corpus — retained as a **control only**, not a production claim |

**No workload-shape claims beyond what is measured.** (An earlier draft asserted a "datalog-backend
shape" fit; removed — unsupported.)

## Deferred-free design — PINNED (descending frees cannot stay interleaved)

`repairAfterLazy` today does, per demoted container: **`bitsetToArray` allocation → immediately free the
old bitset.** A *global* descending free order is therefore impossible without restructuring. Two options
existed; **option (a) is selected:**

- **(a) SELECTED — key-order conversion, deferred descending free.** Keep the conversion pass in key
  order (preserving result allocation order and key order exactly as today), **collecting the old bitset
  pointers**; then run a **separate descending free pass**.
- (b) Rejected — reordering the conversion pass itself, which would change result allocation order and
  perturb far more than the frees.

**Consequences that must be measured, not assumed:**

- **Frees are NO LONGER INTERLEAVED with allocations.** That is a deliberate change of shape, not an
  incidental one, and it may alter the result independently of ordering. **Three named arms are therefore
  required**, so that *deferral* and *direction* are separated:

  | arm | description |
  |---|---|
  | **I — interleaved** | today's behaviour: convert and free each container in key order |
  | **D-key — deferred, key order** | conversions deferred, then freed in **key order** (i.e. deferral only, no reordering) — **isolates the cost/benefit of deferral itself** |
  | **D-desc — deferred, reverse/address-descending** | conversions deferred, then freed in **descending payload-address order** — the candidate |

  `D-desc − D-key` is the **direction** effect; `D-key − I` is the **deferral** effect. Reporting only
  `D-desc − I` would conflate them.
- **Temporary live-memory peak RISES.** Old bitsets are held while new arrays are allocated: up to
  ~16,364 × 8 KB ≈ **134 MB retained simultaneously** that today is released incrementally. **Measure and
  report peak RSS**; this is a real cost the win must justify.
- **OWNERSHIP HAZARD — deferred-free failure handling (pinned).** Once conversions begin, the scratch
  list becomes the **sole owner** of old bitsets that are **no longer reachable through
  `self.containers`** (their slots now hold the new arrays). Therefore:
  - **On any later conversion error, every bitset collected so far must be freed exactly once**, via
    `errdefer` over the scratch list.
  - **The partial-repair invariant is a NEW GUARANTEE INTRODUCED BY `39-01` — it does not exist today.**
    Verified in source: `repairAfterLazy` commits `self.size = @intCast(write_idx)` **only after the
    complete loop**, so a mid-loop allocation failure today leaves compacted/overwritten entries with the
    **old** `size`. Spec 35 *designed* the in-place partial-commit invariant but **35-01 never shipped**
    (35 was NO-GO at the diagnostic). So `39-01` must **introduce** it, and its scope explicitly includes
    **tail compaction and final state commit**: on failure, compact the untouched tail behind the repaired
    prefix, commit `size`, leave cardinality unknown, no dangling entries. Do not describe this as
    preserving existing behaviour.
  - **Scratch allocation failure is the easy case: fall back BEFORE any mutation** to today's interleaved
    free path — no partial state, no error propagated.
  - Failure injection must hit **first / middle / last** collected-bitset positions and the scratch
    allocation itself; verified leak-free and double-free-free with a shadow bitmap.
- **SCRATCH SIZING — PINNED: allocate `self.size` pointers (upper bound), no prepass.** Exact sizing
  would need the demotion count *before* mutation, which means a cardinality/demotion prepass — and that
  restructures repair, adding a second variable to an experiment about free order. Two options were
  considered:
  - **(SELECTED) upper-bound `self.size` pointers**, filled as demotions are found, count tracked. Simple,
    no restructure. **Allocated from `self.allocator`**, per the project allocation contract — the same
    allocator the mechanism is compensating for (noted, accepted), and its failure falls back **before
    mutation**. **Cost: ~65,496 × 8 ≈ 524 KB** rather than the ~131 KB an exact fit would need —
    roughly **4×**, and **that larger memory cost must be reported** alongside peak RSS.
  - (deferred) a cardinality/demotion prepass **inside timing**, with cached cardinality reads during
    conversion — exact scratch *and* it avoids recomputing cardinality, but it is a repair restructure and
    belongs in its own spec if the 524 KB proves material.
- **Rung 0 is NOT free on this path.** Deferring requires storing the collected old-bitset pointers —
  the `self.size`-sized scratch above, plus the fill. "Reverse iteration" then means reverse
  iteration *of that deferred array*. So rung 0's cost is **scratch allocation + fill, not zero**; it is
  still far cheaper than any sort, but the cost table below must be read with that correction. (Rung 0 is
  genuinely free only for an in-place traversal we already perform — which is the read side, not this
  one.)

## Direction discipline — frees and reads want OPPOSITE orders

- **Frees want DESCENDING.** `SmpAllocator.free` pushes onto a **LIFO** freelist, so descending frees
  hand back **ascending** addresses to the next allocation burst. This is the `38-00` win.
- **Reads want ASCENDING.** A cardinality/zeroing traversal has no LIFO involved; it wants monotone
  ascending order.

**Consequence for the repair retest:** repair's read traversal must be retested **ascending**, which is
the *opposite* direction from the teardown win. And **if rung 0 (reverse iteration) wins for frees, its
read-side equivalent is simply the existing forward order — i.e. it supplies no new repair candidate at
all.** Do not describe the retest as "the winning rung applied to repair" without inverting direction.

## Reorder-mechanism ladder — cheapest sufficient wins

`38-00`'s verdicts were measured with `std.mem.sort`, which Zig 0.16 implements as a **stable block
sort** (not pdq). Stability is unnecessary — payload addresses are unique — and block sort pays for
stability *and* in-place rotation, so **both `38-00` verdicts carried an unquantified, probably large
reorder tax.** Quantify rung 4 so that tax is known.

**⚠ The costs below are PRIOR ESTIMATES from the 1-vCPU Cascade Lake Linux VM** in
`smp-free-order.md` §4.1 — **not** from M4 or the current Zen 4 host, and from a machine that could not
exercise slot rotation. They establish the **relative ordering** of mechanisms only. **Rung selection
must use fresh two-host measurements.**

| rung | mechanism | prior estimate @ n≈16,364 |
|---|---|---:|
| **0** | reverse iteration of the **deferred pointer array** | **scratch alloc + fill only** (not zero — see Deferred-free design) |
| **1** | 1-pass span-adaptive bucket partition | ~0.184 ms |
| **2** | LSD radix on normalized payload addresses | ~0.240 ms |
| **3** | `std.mem.sortUnstable` (pdq) | ~1.423 ms (reference) |
| **4** | `std.mem.sort` (block, stable) — the `38-00` baseline | **unmeasured — measure it** |

**Stop at the cheapest rung that captures the effect.** Report **measured** reorder cost as its own
column at every rung.

### Rung 0 — must be qualified, not assumed

The container array is **fundamentally key-ordered**, not allocation-ordered. In `lazyMergeTwo`
construction and append coincide, so reverse iteration *may* approximate descending payload order there —
but **mutation, container replacement, array↔bitset conversion, and long-lived bitmaps all break that
correlation.** Therefore rung 0 requires **measured address-order quality (travel and/or page-locality)
on the actual target lifecycles**, not only on a fresh synthetic corpus. If the correlation does not hold
on the repair-demote path, rung 0 is out for that path regardless of its cost.

### Rung 1 — bucket partition, specified

- Key: **`(address − min_address) >> shift`**, `min_address` from a first pass.
- **Shift — span-adaptive, portable, and UNDERFLOW-SAFE.** Do **not** write `max(0, a − b)`: Zig evaluates
  the unsigned subtraction first, so it underflows when `significant_bits < bucket_bits`. Pin:

  ```zig
  const significant_bits = @bitSizeOf(usize) - @clz(span);   // never hardcoded 64
  const bucket_bits = std.math.log2_int(usize, nbuckets);
  const shift: std.math.Log2Int(usize) = if (significant_bits > bucket_bits)
      @intCast(significant_bits - bucket_bits)
  else
      0;
  ```

  Production must compile on 32-bit targets even though the gates run on M4 and Zen 4. A **fixed** shift is
  a recorded trap: 256 buckets over a 160 MB span left travel **unchanged at 6856x**.
- **`span == 0` (zero or one item) must be pinned:** with `n <= 1` there is nothing to reorder — return
  immediately; `@clz(0)` is `@bitSizeOf(usize)` so the shift formula degenerates safely, but the
  early-return is required so the count/scatter passes are never entered with a degenerate span.
- **Clamp** the bucket index to `nbuckets − 1` (guards the `hi` element and any rounding).
- Steps: **count → prefix-sum → scatter into scratch → copy back**.
- **Within-bucket order is unspecified/arbitrary** (a bucket spans `span/nbuckets`); state that this is
  accepted, and that descending output is produced by **iterating buckets high→low**.
- `nbuckets` default **4096** (2 × 4096 × 4 B counters ≈ 32 KiB — trades L1 residency against
  resolution); sizing recorded.

### Rung 2 — radix, specified

- **Three LSD passes do not sort a full `usize`** — that must be pinned, not implied.
- Key: **normalized `address − min_address`**, so only the **span's** significant bits need sorting.
- **Bits per pass** and **pass count adaptive to the observed span**: passes =
  `ceil(significant_bits / bits_per_pass)` where
  **`significant_bits = @bitSizeOf(usize) − @clz(span)`** — **portable, not hardcoded 64.** State
  `bits_per_pass` (8 or 11) and the resulting count-table size.
- **`span == 0` / `n <= 1`: early-return, no passes.** Same requirement as rung 1.
- **Descending output** produced explicitly (reverse the final scatter, or reverse-iterate the result) —
  do not leave direction implicit.
- **Scratch and count-table sizes** stated; **allocation failure falls back to unreordered**.

## API — CONDITIONAL on A/B (deferred to `39-01`)

**Corrections to the previous draft:** `deinit()` always uses the bitmap's **stored** allocator, so the
"per-call allocator can differ" rationale does **not** apply to teardown. And with allocator detection
rejected, **a size gate cannot exclude libc-on-M4** — that has to be a contract, not a mechanism.

**The API, libc policy, and benchmark gate are CONDITIONAL on the A/B scope decision — they are NOT
pinned here.** An earlier draft of this umbrella stated A's contract unconditionally; under (B)
`repairAfterLazy()` changes, the canonical row itself is gated rather than a variant, and libc is
necessarily on the default path.

**Single source of truth: the conditional A/B table in
[`39-01`](39-01-free-order-production.md#api-libc-policy-and-benchmark-gate--all-conditional-on-ab).**
Do not restate it here — that duplication is exactly what drifted.

**Common to both positions (safe to state unconditionally):**

- **EXCLUDED surfaces:** `deinit`, `clearRetainingCapacity`, `Roaring64Bitmap`, `OwnedBitmap`.
  `OwnedBitmap` on principle — **its arena owns teardown**, so container-level free order is irrelevant.
- **Allocator detection stays rejected** — opaque `Allocator` vtable; comparing against
  `smp_allocator.vtable` breaks for every wrapped allocator. (Under (A) this is why the opt-in is
  caller-declared; under (B) it is why the mechanism must be safe for *all* allocators.)
- The effect is **SMP-specific, and Zen 4 gains MORE in absolute terms** (−3.577 vs −2.086 ms) — **neither
  position may be framed as an M4 fix.**

`OwnedBitmap` is excluded on principle (**its arena owns teardown**, so container-level free order is
irrelevant); the other three are excluded to keep the first shipped surface minimal — they may be
revisited in a later spec on their own evidence, not on this one's.

## Size gate — unit, and the timing contradiction it creates

**Unit (for MEASUREMENT / crossover derivation): the number of BITSETS ACTUALLY DEMOTED** — not total
containers, not all reorderable payloads. `38-00`'s crossovers (**64 M4 / 1,024 Zen 4**) were measured in
8 KB bitset payload count, and that is the population this mechanism reorders; a total-container gate
would be wrong by ~4× on a lazy-OR result (65,496 containers vs 16,364 demoted bitsets).

**The array-result teardown remains DIAGNOSTIC EVIDENCE ONLY** — it characterises the array-dominated
reality of a lazy-OR result, but must **neither select nor block** the bitset-demotion mechanism.

### CONTRADICTION: that count is not known in time to gate on it

The demoted-bitset count is only known **after** walking containers and computing cardinalities — i.e.
**after conversions have already been deferred.** By then it is **too late to fall back to today's
interleaved path**. So a *runtime* gate on that unit is unimplementable as previously written. It is fine
as a **measurement unit**; it is not usable as a runtime switch without one of the following.

**Resolution: the gate choice FOLLOWS FROM the scope decision — it is not an independent choice.**

| scope position | gate mechanism | why |
|---|---|---|
| **(A) optional opt-in variant** *(recommended)* | **Option 3 — NO runtime gate; activation entirely caller-controlled via the opt-in API** | Consistent with the caller-declared design already pinned: the caller declares allocator suitability, so they can equally declare workload size. No prepass, no forced deferral, simplest `39-01`. |
| **(B) default adoption** | **Option 1 — a demotion PREPASS is REQUIRED**, so the gate controls the whole deferred-free mechanism *before* mutation | A default-on mechanism must be able to **decline cheaply** for small inputs; it cannot force deferral (and its raised memory peak) on every caller. Accepts the repair restructure and its cost on the default path. |

**Option 2 — unconditional deferral, gating only the reordering — is REJECTED** in both positions: it
imposes the deferral cost and the raised memory peak on **every** call regardless of benefit, and makes
**`D-key` the silent floor**, so below-gate callers get a behaviour change they neither asked for nor
benefit from.

**`39-00` is unaffected** — it measures all three arms and derives the crossover regardless. **This is a
`39-01` decision only**, and it is settled by choosing (A) or (B).

## Gate re-derivation

The 64/1,024 crossovers came from **block sort**. The **selected rung needs its own size sweep**, on both
hosts, with **stage-3 noise re-verified per rung** (survival at rung 4 does not transfer to a cheaper,
lower-quality reorder) and **measured address travel / order quality** reported alongside timing.

## Scope

- **Steady state only.** M4 first-cycle was inconclusive in `38-00`.
- **Not in scope:** compaction/relayout, the private pool (`smp-free-order.md` Idea B), any upstream
  `SmpAllocator` change, and repair's read traversal beyond the single inverted retest.

## Measurement

Canonical protocol; **rawr/SMP and rawr/libc**; **M4 and Zen 4/WSL2**; five fresh-process medians + full
ranges; one process per tuple; reorder cost **inside** the timed region and reported separately; range
separation before any claim; stage-3 noise control per rung.

## Correctness

- **Free order is correctness-invariant:** every container freed **exactly once** at every rung —
  shadow-bitmap verified; no leak, no double-free; failure injection green.
- Rungs 1–2 must be **permutations** (same multiset in and out).
- Reorder-scratch failure **degrades to unreordered**, never propagates an error.
- Repair's demote/compaction results must be **byte-identical** to today regardless of free order.
- `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast`.

## Acceptance

- **Primary:** the **three-arm matrix I / D-key / D-desc** measured inside `repairAfterLazy`, both
  allocators, both hosts — reporting `D-key − I` (deferral) and `D-desc − D-key` (direction)
  **separately**, plus **peak RSS** for the raised temporary live footprint.
- **Hard gate is FULL-CYCLE improvement** on steady-state `lazyOr+repair`, where the timed span is
  **construction + repair + result teardown** (matching the canonical row's
  `"result deinit/free inside timing"`), with **construction / repair / teardown / full-cycle reported
  separately** as attribution beneath the gate. Teardown *order* stays default across arms but is timed. `lazy-or-construction` and `lazy-or-repair-only` documented as rows this mechanism
  **cannot** move.
- **Noise-injected results reported as DIAGNOSTIC only**; the adoption number comes from a
  **canonical-equivalent opt-in variant** with no injected noise (identical corpus/boundaries/protocol,
  differing only in opting in).
- **Scope position recorded — (A) optional variant or (B) default** — and while the API is opt-in,
  results are reported **as a variant row, never as the canonical row**; **under (B) the canonical row
  itself is the result** and there is no variant row. (B) requires libc shown unharmed on the
  repair-demote path.
- **Deferred-free failure handling verified:** scratch failure falls back before mutation; mid-conversion
  errors free every collected bitset exactly once; **`39-01` INTRODUCES the partial-repair invariant**
  (tail compaction + final `size`/cardinality commit) — new behaviour, not preservation — with
  first/middle/last positional injection green.
- **Scratch sized at `self.size` (upper bound), no prepass, allocated from `self.allocator`**, with the
  ~4× memory cost (~524 KB vs ~131 KB exact) reported alongside peak RSS.
- Representative **result teardown (~65,496 arrays)** measured; the all-bitset corpus reported as a
  control only.
- Rungs 0–2 measured (3 reference, 4 quantified); **rung 0 qualified by measured order quality on the
  real lifecycles**; cheapest sufficient rung selected.
- Repair read-traversal retest run **once, ASCENDING**, with the explicit note that rung 0 offers no
  read-side candidate; verdict recorded either way.
- Size-gate **measurement unit** is **demoted bitsets**; crossover **re-derived for the selected rung** on
  both hosts. **The runtime gate mechanism is a `39-01` decision that follows from the scope position** —
  (A) ⇒ no runtime gate, caller-controlled; (B) ⇒ demotion prepass required. `39-00` derives the crossover
  either way and does not depend on the choice.
- **API, libc policy and benchmark gate shipped per the chosen A/B row** of `39-01`'s conditional table —
  one position's row, not a blend. Common to both: `deinit`, `clearRetainingCapacity`, `Roaring64Bitmap`,
  `OwnedBitmap` **excluded**; detection stays rejected.
- Algorithms **portable**: `@bitSizeOf(usize) − @clz(span)` (never hardcoded 64), `span == 0` / `n <= 1`
  early-return pinned, compiles on 32-bit targets.
- Reorder costs reported as **fresh two-host measurements**, with the `smp-free-order.md` figures cited
  only as prior estimates.
- Correctness surface green.

## Chunk plan

- **[`39-00`](39-00-free-order-measurement.md)** — measurement: three arms (I / D-key / D-desc), full-cycle
  timing with teardown inside the span, the ladder with rung-0 order-quality qualification, libc on the
  repair-demote path, inverted (ascending) read retest, crossover candidates. **No production change.
  Ready to implement.**
- **[`39-01`](39-01-free-order-production.md)** — production: deferred descending free at the selected
  rung, the surface **determined by the A/B decision** (see its conditional table), and the **new**
  partial-repair invariant. **BLOCKED** on (i) a full-cycle win from `39-00` and (ii) the scope decision
  (A optional-variant / B default), which determines the API, the gate mechanism, the libc position, and
  which row is reported.

## Estimate

M for `39-00` — rung 0 carries **no sorting cost** (though it still needs the deferred-pointer scratch,
see Deferred-free design) and rung 1 is small, but the repair-demote instrumentation, the three-arm
I / D-key / D-desc matrix, the array-dominated teardown corpus, the cycle harness, and per-rung stage-3
re-verification are real work. M for `39-01`.
