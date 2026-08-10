<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 39-00: Free-order measurement — three arms, full cycle

Toplevel: [39-descending-free-order.md](39-descending-free-order.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md),
[smp-free-order.md](../docs/smp-free-order.md).

**Diagnosis only — no production library change.** Produce the numbers `39-01` needs, and decide whether
descending demote frees are worth shipping at all. **Independent of the `39-01` scope decision**
(A optional-variant vs B default) — this chunk measures the same thing either way.

## The three arms

Separates **deferral** from **direction**; reporting only `D-desc − I` would conflate them.

| arm | behaviour |
|---|---|
| **I** — interleaved | today: per container, `bitsetToArray` allocate → immediately free the old bitset, in key order |
| **D-key** — deferred, key order | conversions in key order **collecting** old bitset pointers, then free them in **key order** |
| **D-desc** — deferred, descending | same collection, then free in **descending payload-address order** — the candidate |

**Report `D-key − I` (deferral effect) and `D-desc − D-key` (direction effect) separately.**

## Timed span — teardown is INSIDE it

```
timed span = [ lazyOr construction  →  repairAfterLazy(arm)  →  result teardown ]
```

Matches the canonical row: `bench_parity_worker.zig` declares
`allocating_teardown = "result deinit/free inside timing"`, and
`benchRawrLazyOrSparseRepairWithAllocator` runs `defer result.deinit()` inside the timed function.

- **Hard gate: FULL-CYCLE improvement.** Measured across **consecutive iterations**, so iteration *k*'s
  frees condition iteration *k+1*'s allocations.
- **Teardown order stays DEFAULT across all arms** — only repair's frees vary — but it **is timed**.
  Excluding it would hide the most likely failure mode: teardown is a *larger* free burst
  (~65,496 arrays) that can **overwrite** the conditioning repair's frees just produced.
- **Report construction / repair / teardown / full-cycle separately** as attribution beneath the gate.

## Rows this cannot move — state in the writeup

| row | |
|---|---|
| `lazy-or-construction` | **cannot** — never calls repair |
| `lazy-or-repair-only` | **probably cannot** — the benefit lands on the *next* construction, outside that row's timer |
| steady-state `lazyOr+repair` | **the only row that can move** |

## Workloads

| workload | contents | role |
|---|---|---|
| **repair demote frees** | 16,364 transient 8 KB bitsets, deferred-release | **primary** |
| **result teardown** | the real lazy-OR result: **~65,496 arrays** | representative teardown, inside the timed span |
| all-bitset teardown | 16,364 bitsets, no interleaving | `38-00`'s corpus — **control only** |

## Reorder ladder — cheapest sufficient wins

**Costs below are PRIOR ESTIMATES** from the 1-vCPU Cascade Lake VM (`smp-free-order.md` §4.1) — relative
ordering only. **Selection must use fresh two-host measurements.**

| rung | mechanism | prior estimate |
|---|---|---:|
| **0** | reverse iteration of the deferred pointer array | **no sorting cost** (still pays scratch + fill) |
| **1** | 1-pass span-adaptive bucket partition | ~0.184 ms |
| **2** | LSD radix on normalized addresses | ~0.240 ms |
| **3** | `sortUnstable` (pdq) | ~1.423 ms (reference) |
| **4** | `std.mem.sort` (block, stable) — `38-00`'s baseline | **unmeasured — quantify it** |

- **Rung 0 must be QUALIFIED**, not assumed: the container array is **key-ordered**, and mutation,
  replacement, conversion and long-lived bitmaps break any allocation-order correlation. Report
  **measured order quality (travel and/or page-locality)** on the real repair-demote lifecycle.
- **Rung 1:** key `(addr − min) >> shift`. **The shift MUST NOT be written as `max(0, a − b)`** — Zig
  evaluates the unsigned subtraction *first*, so it underflows (panic in safe modes, wrap in fast) whenever
  `significant_bits < bucket_bits`. Pin it as:

  ```zig
  const significant_bits = @bitSizeOf(usize) - @clz(span);   // portable, never hardcoded 64
  const bucket_bits = std.math.log2_int(usize, nbuckets);
  const shift: std.math.Log2Int(usize) = if (significant_bits > bucket_bits)
      @intCast(significant_bits - bucket_bits)
  else
      0;
  ```

  Then: clamp index to `nbuckets − 1`; count → prefix → scatter → copy back; within-bucket order arbitrary;
  descending by iterating buckets **high→low**; `nbuckets` default 4096.
- **Rung 2:** normalized `addr − min`; `significant_bits = @bitSizeOf(usize) − @clz(span)`; passes =
  `ceil(significant_bits / bits_per_pass)`; state `bits_per_pass` and count-table size; **descending output
  constructed explicitly**.
- **Both:** `span == 0` / `n <= 1` ⇒ **early return**, no passes.
- **Report measured reorder cost as its own column at every rung.**

## Scratch

- **`self.size` pointers (upper bound), no prepass**, filled as demotions are found, count tracked.
- **Allocated from `self.allocator`** per the project allocation contract — the same allocator being
  compensated for (accepted, noted).
- Cost **~524 KB** vs ~131 KB for an exact fit (**~4×**) — **report it, alongside peak RSS.**
- **Allocation failure falls back before any mutation** to arm `I`.

## Memory

**Deferral raises the temporary live peak** — up to ~16,364 × 8 KB ≈ **134 MB** held simultaneously that
today is released incrementally. **Report peak RSS per arm.** This is a real cost the win must justify.

## Protocol

- Canonical: **3 warmup / 21 timed**, **five fresh-process medians + full ranges**, one process per
  `(row, implementation, allocator)` tuple, implementation-specific init only.
- Arms: **rawr/SMP and rawr/libc**; hosts: **M4 and Zen 4/WSL2**.
- **Range separation** (spec-37 discipline) before any win is claimed.
- **Noise is DIAGNOSTIC.** Injected stage-3 allocator noise proves the effect survives a shared allocator
  and must be **re-verified per rung** (survival at rung 4 does not transfer to a cheaper, lower-quality
  reorder). The **adoption number** comes from a **no-injected-noise run of a canonical-equivalent opt-in
  variant** — identical corpus/boundaries/protocol, differing only in opting in. **Never substitute the
  noise figure for it.**
- **libc must be measured on this path.** `38-00` found libc regressed on M4 for *teardown*; whether that
  holds for repair-demote is unknown and **decides whether scope position (B) is even available.**

## Inverted repair read-traversal retest

`38-00`'s repair NO-GO used rung 4 (block sort). Retest **once**, and note the direction inversion:

- **Frees want descending; reads want ascending.** The read retest must be **ascending**.
- **If rung 0 wins for frees, it offers NO read-side candidate** — its read equivalent is simply today's
  forward order.
- One retest only. If repair's read traversal still regresses with a cheap reorder, that NO-GO is final.

## Crossover derivation

- **Measurement unit: bitsets actually demoted** (not total containers — that would be ~4× wrong on a
  lazy-OR result).
- Sweep sizes; require **separated ranges favouring the candidate**, with **monotonicity** — a separated
  point followed by a regression at a larger size is a **crossover, not a threshold**.
- Report per allocator and per host; **do not select a shipping value here** — that follows the `39-01`
  scope decision.
- `38-00`'s 64/1,024 came from block sort and **do not carry over**.

## Correctness

- **Free order is correctness-invariant:** every container freed **exactly once** in every arm and rung —
  shadow-bitmap verified; no leak, no double-free.
- Rungs 1–2 must be **permutations** (same multiset in and out).
- Repair results **byte-identical** to today across all arms — same container kinds, cardinalities, values,
  key order.
- **Deferred-free ownership on the SUCCESSFUL path:** once conversions begin, the scratch list is the
  **sole owner** of bitsets no longer reachable through `self.containers`. Verify on success that every
  collected bitset is freed exactly once and the scratch is released.
- **Scratch-allocation failure** is verified here, because it fails **before any mutation** and simply
  falls back to arm `I` — a well-defined state with no partial repair.

### Mid-conversion failure injection is DEFERRED to `39-01` — deliberately

**`39-00` must NOT require positional mid-conversion failure injection.** Verifying leak/double-free
freedom after a mid-conversion error presupposes the bitmap is safely deinit-able in that state — which
requires **tail compaction and a committed `size`**, i.e. the **partial-repair invariant that does not
exist today and that `39-01` introduces**. Without it the failed bitmap's state is not well-defined, so
the verification would be checking an undefined property.

Two ways out; **the first is chosen:**

- **(SELECTED) move positional conversion-failure injection entirely to `39-01`**, alongside the invariant
  it depends on. `39-00` retains only the successful-path ownership checks and the pre-mutation
  scratch-failure case above.
- (rejected) implement benchmark-local partial-commit cleanup in `39-00` — duplicates `39-01`'s work in a
  throwaway diagnostic, for no measurement benefit.
- `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast`.

## Acceptance

- **Three arms (I / D-key / D-desc)** measured, both allocators, both hosts, five fresh processes;
  **`D-key − I`** and **`D-desc − D-key`** reported separately.
- **Full-cycle gate** (construction + repair + teardown) reported, with the four phase timings beneath it.
- Rungs 0–2 measured, 3 as reference, **4 quantified**; **rung 0 order-quality qualified on the real
  lifecycle**; cheapest sufficient rung identified.
- **Peak RSS and scratch cost reported per arm.**
- Stage-3 noise **re-verified per rung**; adoption number from the **canonical-equivalent opt-in variant**
  with no injected noise.
- **libc measured on the repair-demote path**, with an explicit statement of whether scope (B) remains
  available.
- Inverted (ascending) read retest run once; verdict recorded.
- Crossover **candidates and monotonicity** reported per allocator/host; **no shipping value selected**.
- Rows that cannot move documented.
- Correctness surface green — **successful-path ownership + pre-mutation scratch failure only**;
  positional mid-conversion injection explicitly **deferred to `39-01`** with the invariant it needs.
- **No production library change.**
- `docs/parity-measurement.md` updated; `docs/allocator-address-order-pathology.md` cross-referenced.

## Estimate

M — rung 0 has no sorting cost and rung 1 is small, but the three-arm matrix, the cycle harness with
teardown inside timing, deferred-free plumbing in a diagnostic build, per-rung noise re-verification, and
the two-host sweep are real work.
