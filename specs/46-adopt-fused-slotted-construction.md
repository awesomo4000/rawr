<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 46: Adopt fused slotted lazy-OR construction, and close the campaign

**Purpose.** Promote spec 44's fused slotted path (arm 5) to production, strip the diagnostic machinery,
re-measure, and close the parity campaign with an **accepted residual** rather than a parity claim.

## 1. Owner decision — recorded, with its rationale

**Owner decision, 2026-08-20: accept a residual on `lazy-or-construction` as an explicit exception to the
`≤1.10x` gate.** Rationale as accepted:

- Forced lazy-OR construction is a **narrow, explicitly invoked** operation, not a broad path.
- The candidate improves rawr **~27.3%** on this row (≈1.70x → **1.235x** measured).
- **Further vehicles have failed cleanly** — across specs 43, 44 and 45, **every tested vehicle either
  regressed or left a residual above the former gate**; spec 45's per-operation chunk allocation came out
  *worse than baseline* on both hosts.
- **Zen 4 also improves** (20.749 → 18.570 ms).
- The remaining cost is **not a bitmap-algorithm deficiency**. Two distinct facts, kept distinct
  (see `46-01` §7.1):
  - the **baseline gap** is primarily **allocator / address-order** related;
  - **arm 5's residual also includes the machinery required to recover that ordering** — it is not purely
    allocator cost.

  And two distinct **measurements**, not interchangeable: **47.5% (M4)** is standalone-probe headroom
  (scattered vs sorted zeroing in `bench_smp_layout.zig`, no rawr code); **−2.211 ms (M4)** is production
  ordering recovery (spec 44 arm 3 vs arm 2, inside the real merge).

  **Every tested vehicle for obtaining that ordering either regressed or left a residual above the former
  gate.**

**Owner also accepts the measured M4 libc regression (+21.2%)**, consistent with the standing policy that
libc regressions are reportable but not blocking, and that SMP is the performance-relevant allocator.

### 1.1 Language — binding

- **Record as: "accepted residual: 1.235x on M4"** (or whatever §4 actually measures).
- **Never** "parity", "row closed", "at parity", or "at parity when enabled" — that last phrase belongs
  to opt-in outcomes like spec 39-01, and this is a **default** change with a residual.
- The board row remains **open with an accepted residual**. Those are different from closed, and the
  distinction must survive into the umbrella record.

## 2. What is promoted

Spec 44's **arm 5** — fused slotted construction — becomes the **default** for `op == .bor` (forced and
selective lazy OR).

### 2.0 FIRST: materialize the source, before any other work

**The arm 5 implementation is not on `spec-43-lazy-construction-diagnostic`.** That branch tip
(`4b6bec4`) carries only baseline / batched / sorted. Verified.

**Arm 5 survives only in stash commit `a1cb8c726686897b2e82dfed879dac540b52c8cd`** — reachable today,
but a stash entry is held by `refs/stash` alone and is **reclaimable by `gc` if dropped**. The entire
1.235x result is currently one `git stash drop` from gone.

**Step one of this spec, before reading anything else: give that commit a durable LOCAL ref** — branch or
tag — so `gc` can no longer reclaim it. **The immutable hash is already
`a1cb8c726686897b2e82dfed879dac540b52c8cd`; creating a ref does not produce a new one**, so there is
nothing further to pin.

**Pushing is an explicit owner action, not part of this spec** — matching `46-00` §0. Creating the local
ref makes the work safe; publishing it is a separate decision.

Then **selectively port** arm 5 onto `main`. **Do not apply the stash commit wholesale** — it sits atop
the diagnostic branch and carries the multi-arm machinery §3 removes. Do not productize from a stash.

Everything the diagnostic version required is now **production code and the bar rises, not falls**:

- eligible-pair pre-pass (exact count, not matched-pair count);
- single `[]Pending` scratch, `sortUnstable` on `payload_addr`;
- `initPendingBitset` file-private in `bitmap.zig`;
- `.reserved` slot initialization built directly (never via `Container.toTagged`, which is `unreachable`
  for `.reserved`), `result.size = output_count`, no-reserved-slot check on success;
- **ownership contract as arm 5 actually works** — *(an earlier draft described spec 43's
  `appendOwnedContainer` handoff, which this path does not use: it assigns **directly into reserved
  slots**)*:
  **assign the tagged pointer into its slot, then advance `transferred_count`**, with no fallible
  operation between. **Pending cleanup owns only the untransferred suffix; `result.deinit()` owns the
  populated slots.**
- **`cached_cardinality = -1` before success**;
- fused zero + accumulate per buffer in address order;
- scratch-OOM falls through to the baseline loop reusing the initialized `result`; every other site
  propagates.

## 3. What is removed

**Delete, do not leave dormant:**

- the `ConstructionMode` dispatch and arms 2, 3, and 4 code paths;
- the **mode-selection** export (replaced by the single baseline export above);
- source-travel / container-type diagnostic instrumentation;
- the diagnostic rows added on the branch.

**Keep the pre-adoption baseline path and BOTH of its rows:**

| Retained row | Reference for |
| --- | --- |
| `lazy-or-construction-baseline` | the construction gate |
| `lazy-or-repair-baseline` | the **combined** gate |

*(An earlier draft retained only the construction baseline while still gating the combined row — leaving
that gate with **no in-binary reference**, which forces exactly the cross-run comparison spec 43-02
forbids.)*

**Manifest:** `main` is at **40**; +2 retained baseline rows → **42**. Both guards:
`src/bench_parity_worker.zig:778`, `scripts/run-compare-bench.sh:72`.

**Internal access is narrowed, not removed.** *(An earlier draft said to delete the internal export
outright, which contradicts retaining baseline rows — the worker must still be able to call the old
implementation.)* Replace the multi-mode dispatch with **one narrowly named internal baseline export**,
retaining a corresponding reason string in `check_docs.zig`'s internal manifest. Remove the
`ConstructionMode` enum and arms 2–4; keep exactly the one entry point the baseline rows need.

## 4. Re-measure — the number is NOT inherited

**1.235x was measured in a five-arm binary on an uncommitted branch. It does not carry over.** Spec 28
established that adding or removing code moves untouched rows — including CRoaring's — with
instruction-identical disassembly. Stripping four arms is exactly that kind of change.

**The accepted residual is whatever the post-cleanup canonical run measures**, recorded with its date.
1.235x is the *expectation*, not the guarantee.

- Canonical harness only, fresh process per cell, 3 warmup / 21 timed, **≥5 process medians with full
  ranges**, both hosts, all three tuples, **whole board**.
- Report `lazy-or-construction`, `lazyOr+repair`, **and both retained baseline rows**.

### 4.1 Acceptance thresholds

- **M4 SMP construction ≤ 1.30x** and **materially better than the pre-adoption baseline** (non-overlapping
  ranges). If it lands repeatedly between ~1.235x and 1.30x, **investigate, report with the explanation,
  and obtain explicit owner re-acceptance before retaining the candidate as the final default and closing
  the campaign** — see `46-01` §3 for the full branch. Ordinary local or cross-host commits for testing
  are unaffected.
- **Combined `lazyOr+repair` does not regress** beyond 5% on median.
- **Zen 4: `candidate / baseline ≤ 1.05`** on both rows.
- **Untouched rows: `>5%` triggers a RERUN and targeted inspection, not an immediate failure.** This
  comparison is across different binaries, and spec 28 established that adding or removing code moves
  untouched rows with instruction-identical disassembly. **Only a repeated, attributable regression fails
  adoption**; unexplained one-off movement within layout noise does not.
- **libc: reported, not gated.** Record the final number; ~+21.2% is accepted.

### 4.2 Rollback

If §4.1 fails, **restore the previous default** and record the result. Adoption is not automatic just
because the diagnostic looked good.

### 4.3 Negative control, in-binary

**Both** candidate rows against **their own** retained baseline, in the **same binary**:

- `lazy-or-construction` vs `lazy-or-construction-baseline` — must beat it with **non-overlapping
  ranges**;
- `lazy-or-repair` vs `lazy-or-repair-baseline` — **must not regress beyond 5% on median**, matching
  §4.1 and `46-01` §3.

A default that cannot be distinguished from the path it replaced has not earned the residual exception.
Comparing either against a number from a different run is exactly the error spec 43-02 forbids.

## 5. Correctness — production bar

All of `44-00`'s coverage re-run against the production path, not the diagnostic one:

- repaired output **byte-exact vs the previous default**, and **set-equal + cross-deserializable vs
  CRoaring** — see `46-00` §5. *(Byte-identity to both at once is not always satisfiable: one set has
  multiple valid `RoaringFormatSpec` encodings.)* Coverage spans forced and selective lazy OR, eligible
  counts of zero/partial/all, array/bitset/run combinations, disjoint keys, empty inputs;
- **`cardinality()` checked before `repairAfterLazy`**, not only after;
- **`lazyXor` byte-identical** to its current behaviour;
- **failure injection** via `checkAllAllocationFailures` at every real fallible site — `initCapacity`,
  pending scratch, header `create`, `words` payload, unmatched clone, non-eligible union — with inputs
  untouched, nothing leaked, leak-checking GPA, and assertions that no `.reserved` slot is dereferenced
  and no slot is freed twice;
- **only** initial scratch failure falls back to baseline.

All four suites — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
`check-package`, under `ReleaseSafe` and `ReleaseFast`.

## 6. Documentation — DECIDED: shipped docs unchanged

**Shipped `README.md` and `API.md` are not modified.** Spec 41 deliberately removed every performance
claim from them one spec ago; re-introducing allocator performance guidance would reverse that decision
immediately after making it.

**Record in `docs/parity-measurement.md`** (repo-only, not in `.paths`): the accepted residual with its
date, the final libc figure, and the campaign summary.

*(Considered and declined: a neutral "lazy-OR construction is allocator-sensitive" note in `API.md`'s
allocator guide. It is defensible — a +21.2% libc regression on a default path is a caveat rather than a
claim — but it reopens a settled decision for a narrow, explicitly-invoked operation. Revisit only if
users report it.)*

## 7. Campaign closure

On success, update the umbrella (spec 31):

- `lazy-or-construction`: **open with accepted residual**, with the measured number and date;
- every other material row: at or under gate;
- the closed families: allocator replacement (18), transient arenas (17), header elimination (35),
  first-touch/residency (36), read-traversal sorting (38), payload-address sorting (43), slotted+fused
  machinery beyond this adoption (44), **per-operation** chunk allocation (45);
- **the standing finding**, with its two measurements kept distinct (`46-01` §7.1): **47.5% (M4)** is
  standalone **zeroing** headroom in `bench_smp_layout.zig` (scattered vs sorted, no rawr code), while
  **−2.211 ms (M4)** is **production ordering recovery** (spec 44 arm 3 vs arm 2, inside the real merge).
  **Every tested vehicle** for obtaining that ordering either regressed or left a residual above the
  former gate. Any future proposal must state how it avoids that.

**Scope the closures precisely — they are narrower than "ordering is closed":**

- Spec 45 closes **per-operation chunk allocation**. It does **not** close a persistent pool amortized
  across calls, an allocator change, or an upstream Zig `SmpAllocator` fix. Those remain unexplored, and
  spec 45's data does not speak to them.
- Spec 44's machinery closure is about **that** vehicle, not about all future construction strategies.

## 8. Out of scope

- Further attempts at **this campaign's per-operation ordering vehicles** — payload-address sorting,
  batched/slotted machinery, per-operation chunk allocation. Those are closed (§7).
  **Allocator-level work and persistent pools are NOT closed** — they are untested here and require a
  new spec, not a re-litigation.
- Opt-in variants: this is a default change; there is no enable/disable knob.
- Changes to `lazyXor` or any non-`bor` path.

## 9. Chunking — two chunks

- **[46-00](46-00-productize-arm5.md)** — materialize arm 5 from the stash, productize it, retain both
  baseline paths and rows, remove the other diagnostics, complete correctness and failure testing.
- **[46-01](46-01-measure-and-close.md)** — two-host full-board measurement, rollback decision,
  documentation, campaign record.
