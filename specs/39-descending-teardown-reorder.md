<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 39: Descending-payload teardown reorder

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md). Predecessor:
[38-00](38-00-address-sort-measurement.md). Background:
[allocator-address-order-pathology.md](../docs/allocator-address-order-pathology.md),
[smp-free-order.md](../docs/smp-free-order.md).

Ship the one effect `38-00` established, using the **cheapest reorder that achieves it**, behind a
**caller-declared** opt-in.

## What `38-00` established

- **Descending payload-order teardown improves SMP steady-state reuse, and it survived the mandatory
  stage-3 allocator-noise control:** M4 **7.436 → 5.350 ms (−2.086)**, Zen 4 **22.742 → 19.165 ms
  (−3.577)**. Not a sole-allocator-client artifact.
- Confirms the **LIFO direction-inversion** prediction: descending frees → ascending next-cycle
  allocations.
- **M4 first-cycle is INCONCLUSIVE** (ranges overlap) — the win is established for **repeated
  build/teardown cycles only**.
- **libc is MIXED** — regresses on M4, improves on Zen 4.
- Crossover **64 containers (M4) / 1,024 (Zen 4)**.
- **Repair sorting regressed** on both hosts — but see the retest below; that verdict was measured with
  an expensive reorder.

## Three corrections this spec is built on

1. **This is an SMP effect, not an M4 effect — and Zen 4 gains MORE in absolute terms** (−3.577 vs
   −2.086 ms). **Do not platform-gate to M4**; that would forgo the larger win. What must be excluded is
   **libc-on-M4**, which regresses.
2. **rawr cannot detect the allocator.** `std.mem.Allocator` is an opaque vtable. Comparing
   `allocator.vtable == std.heap.smp_allocator.vtable` is technically possible but **rejected**: it
   breaks for every wrapped allocator (arena-over-SMP, counting wrappers, `OwnedBitmap`'s own arena) and
   couples rawr to std internals. **Therefore the opt-in must be caller-declared.**
3. **The `38-00` baseline used `std.mem.sort`, which Zig 0.16 implements as a STABLE BLOCK SORT** — not
   pdq. Stability is unnecessary here (payload addresses are unique), and block sort pays for both
   stability and in-place rotation, so its cost is plausibly **worse** than pdq's measured 86.98 ns/op.
   **Both `38-00` verdicts were measured net of an unquantified and probably large reorder tax.**

## Reorder-mechanism ladder — cheapest sufficient wins

`38-00`'s numbers are net of the block-sort cost, so the *effect* is larger than the reported win. Walk
this ladder and **stop at the cheapest rung that captures the effect**; do not build radix if a cheaper
rung suffices.

| rung | mechanism | cost @ n≈16,364 | order quality |
|---|---|---:|---|
| **0** | **reverse iteration of the containers array — NO reorder at all** | **0.000 ms** | approximate descending *if* append order tracks allocation order |
| **1** | 1-pass **span-adaptive bucket partition** (4096 buckets), iterated high→low | **0.184 ms** | approximate descending (travel 6857x → 3.65x) |
| **2** | 3-pass **LSD radix** on payload address bits | **0.240 ms** | true descending |
| **3** | `std.mem.sortUnstable` (pdq) | 1.423 ms | true descending — **reference only** |
| **4** | `std.mem.sort` (block, stable) | **unmeasured** | the `38-00` baseline — **measure it** so the tax is known |

**Rung 0 is free and must be tried first.** In `lazyMergeTwo` containers are **appended in allocation
order**, and bump allocation ascends within a slab — so iterating the array backwards may approximate
descending payload order for **zero cost**. Slot rotation across ~`cpu_count` bump streams (§2.5 of
`smp-free-order.md`) will partially break this, which is exactly what the measurement decides.

**Rung 1 requires the span-adaptive shift** — `shift = max(0, 64 - clz(span) - log2(nbuckets))`. A fixed
shift is a documented trap: 256 buckets over a 160 MB span left travel **unchanged at 6856x**.

**Report reorder cost as its own column at every rung** (the `38-00` lesson: a combined number hides
it), and quantify rung 4 so we know what the previous verdicts actually paid.

## Conditional repair retest (scoped, not assumed)

Repair's NO-GO was measured with rung 4. If the winning rung is ≥10× cheaper, **retest repair once**
with it: M4's +1.896 ms regression is plausibly mostly reorder tax. **One retest only** — if repair
still regresses with a cheap reorder, the NO-GO is final and repair leaves scope permanently.

## API — caller-declared, per the correction above

- **No allocator auto-detection.** See correction 2.
- **Not a creation-time flag** — the per-call allocator can differ from whatever the bitmap was created
  with (the `38-01` finding).
- Candidates, decided at review: **per-operation option** (an options struct on the teardown path) or an
  **explicit variant** (e.g. a distinct descending-teardown entry point). Both fit; pick one.
- **Default: off.** With libc mixed and first-cycle inconclusive, an unconditional default is not
  defensible.
- **Document what the caller is asserting**: "my allocator benefits from descending free order, and I
  tear down repeatedly" — not "I am on M4."

## Scope

- **Steady state only.** First-cycle is inconclusive on M4; the spec targets **repeated build/teardown
  cycles** (the datalog-backend shape). Say so in the docs; do not imply a one-shot win.
- **Size gate required.** Crossovers differ 16× (64 M4 / 1,024 Zen 4). The conservative single value is
  1,024, but that forgoes the M4 win between 64 and 1,024 — **decide deliberately, not by rule**, and
  record the choice.
- **Not in scope:** repair (except the single retest), compaction/relayout, a private pool
  (`smp-free-order.md` Idea B), and any upstream `SmpAllocator` change.

## Measurement

- Canonical protocol; **rawr/SMP and rawr/libc**, **M4 and Zen 4/WSL2**; five fresh-process medians +
  full ranges; one process per tuple.
- **Every rung must survive the stage-3 noise control** — the effect did at rung 4; that does not
  transfer automatically to a cheaper, lower-quality reorder. Re-verify per rung.
- **Reorder cost inside the timed region**, and reported separately.
- **libc-on-M4 must be shown excluded** by the chosen gate — it regresses, and shipping must not expose
  it.
- Range separation (spec-37 discipline) before any win is claimed.

## Correctness

- **Free order is correctness-invariant**: every container freed **exactly once** regardless of rung,
  verified with a shadow bitmap; no leak, no double-free; failure-injection green.
- Rung 1's partition must be a **permutation** — same multiset in and out.
- Reorder-scratch allocation failure must **degrade to unreordered teardown**, never propagate an error.
- `zig build test`, `zig build difftest`, `ReleaseSafe` and `ReleaseFast`.

## Acceptance

- Rungs 0–2 measured (3 as reference, 4 quantified), reorder cost per rung reported, **cheapest
  sufficient rung selected**.
- Selected rung **survives stage-3 noise** on both hosts; range separation applied.
- Repair retest run **once** if the winning rung is ≥10× cheaper than rung 4; verdict recorded either
  way.
- Caller-declared opt-in shipped, **default off**, size gate chosen deliberately and recorded,
  **libc-on-M4 demonstrably excluded**.
- Correctness surface green; steady-state-only scope documented.

## Chunk plan

- **`39-00`** — ladder measurement (rungs 0–4), stage-3 re-verification per rung, conditional repair
  retest. No production change.
- **`39-01`** — ship the selected rung behind the caller-declared opt-in, with the size gate and
  correctness surface.

## Estimate

S–M for `39-00` — rung 0 is free, rung 1 is ~40 lines, and the harness from `38-00` already exists.
M for `39-01`.
