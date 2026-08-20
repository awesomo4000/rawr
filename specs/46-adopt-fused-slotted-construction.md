<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 46: Adopt fused slotted lazy-OR construction, and close the campaign

**Purpose.** Promote spec 44's fused slotted path (arm 5) to production, strip the diagnostic machinery,
re-measure, and close the parity campaign with an **accepted residual** rather than a parity claim.

## 1. Owner decision — recorded, with its rationale

**Owner decision, 2026-08-20: accept a residual on `lazy-or-construction` as an explicit exception to the
`≤1.10x` gate.** Rationale as accepted:

- Forced lazy-OR construction is a **narrow, explicitly invoked** operation, not a broad path.
- The candidate improves rawr **~27.3%** on this row (≈1.70x → **1.235x** measured).
- **Further vehicles have failed cleanly** — specs 43, 44, and 45 exhausted the
  "allocate differently to obtain ordering" family; spec 45 came out *worse than baseline* on both hosts.
- **Zen 4 also improves** (20.749 → 18.570 ms).
- The remaining cost is **allocator-related**, not a bitmap-algorithm deficiency: ordering is worth ~47%
  of construction time (re-confirmed three times), and every mechanism to obtain it costs more than it
  returns.

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
selective lazy OR), from branch `spec-43-lazy-construction-diagnostic`.

Everything the diagnostic version required is now **production code and the bar rises, not falls**:

- eligible-pair pre-pass (exact count, not matched-pair count);
- single `[]Pending` scratch, `sortUnstable` on `payload_addr`;
- `initPendingBitset` file-private in `bitmap.zig`;
- `.reserved` slot initialization built directly, `result.size = output_count`, cursor handoff **before**
  `appendOwnedContainer`, no-reserved-slot check on success;
- **`cached_cardinality = -1` before success**;
- fused zero + accumulate per buffer in address order;
- scratch-OOM falls through to the baseline loop reusing the initialized `result`; every other site
  propagates.

## 3. What is removed

**Delete, do not leave dormant:**

- the `ConstructionMode` dispatch and arms 2, 3, and 4 code paths;
- the internal export that exposed mode selection, and its manifest entry in `check_docs.zig`;
- source-travel / container-type diagnostic instrumentation;
- the diagnostic rows added on the branch.

**Keep exactly one thing from the diagnostics: the pre-adoption baseline path, as a single named
diagnostic row** — `lazy-or-construction-baseline`. Without it there is no in-binary reference for the
negative control in §4.3, and spec 43-02 established that cross-run comparisons do not hold.

**Manifest:** `main` is at **40**. Adoption adds the one retained baseline row → **41**. Both guards:
`src/bench_parity_worker.zig:778`, `scripts/run-compare-bench.sh:72`.

## 4. Re-measure — the number is NOT inherited

**1.235x was measured in a five-arm binary on an uncommitted branch. It does not carry over.** Spec 28
established that adding or removing code moves untouched rows — including CRoaring's — with
instruction-identical disassembly. Stripping four arms is exactly that kind of change.

**The accepted residual is whatever the post-cleanup canonical run measures**, recorded with its date.
1.235x is the *expectation*, not the guarantee.

- Canonical harness only, fresh process per cell, 3 warmup / 21 timed, **≥5 process medians with full
  ranges**, both hosts, all three tuples, **whole board**.
- Report `lazy-or-construction`, `lazyOr+repair` combined, and the retained baseline row.

### 4.1 Acceptance thresholds

- **M4 SMP construction ≤ 1.30x** and **materially better than the pre-adoption baseline** (non-overlapping
  ranges). If it lands worse than ~1.235x by more than measurement noise, **stop and investigate before
  committing** — that would mean the cleanup cost something the diagnostic build was hiding.
- **Combined `lazyOr+repair` does not regress** beyond 5% on median.
- **Zen 4: `candidate / baseline ≤ 1.05`** on both rows.
- **No other board row moves** beyond the 5% layout tolerance.
- **libc: reported, not gated.** Record the final number; ~+21.2% is accepted.

### 4.2 Rollback

If §4.1 fails, **restore the previous default** and record the result. Adoption is not automatic just
because the diagnostic looked good.

### 4.3 Negative control, in-binary

Against the retained `lazy-or-construction-baseline` row, in the same binary: the adopted default must
beat it with **non-overlapping ranges**. A default that cannot be distinguished from the path it replaced
has not earned the residual exception.

## 5. Correctness — production bar

All of `44-00`'s coverage re-run against the production path, not the diagnostic one:

- repaired output **byte-identical** to the previous default **and** CRoaring — forced and selective lazy
  OR, eligible counts of zero/partial/all, array/bitset/run combinations, disjoint keys, empty inputs;
- **`cardinality()` checked before `repairAfterLazy`**, not only after;
- **`lazyXor` byte-identical** to its current behaviour;
- **failure injection** via `checkAllAllocationFailures` at every real fallible site — `initCapacity`,
  pending scratch, header `create`, `words` payload, unmatched clone, non-eligible union — with inputs
  untouched, nothing leaked, leak-checking GPA, and assertions that no `.reserved` slot is dereferenced
  and no slot is freed twice;
- **only** initial scratch failure falls back to baseline.

All four suites — `test`, `difftest`, `test64`, `difftest64` — plus `check-32`, `check-docs`,
`check-package`, under `ReleaseSafe` and `ReleaseFast`.

## 6. Documentation — a decision, not an assumption

**Recommendation: shipped docs unchanged.** Spec 41 deliberately removed every performance claim from
`README.md` and `API.md`, and re-introducing allocator performance guidance would reverse a decision made
one spec ago.

**Record instead in `docs/parity-measurement.md`** (repo-only, not in `.paths`): the accepted residual, its
date, the libc figure, and the campaign summary.

**Open question for the owner, flagged not decided:** a **+21.2% libc regression on a default path** is
arguably a *caveat* rather than a performance claim, and callers passing `c_allocator` are affected
without any signal. If you want it surfaced to users, the honest form is a neutral note in `API.md`'s
allocator guide — *"lazy-OR construction is allocator-sensitive; measure with your allocator"* — with **no
numbers**. I have not written it; say the word either way.

## 7. Campaign closure

On success, update the umbrella (spec 31):

- `lazy-or-construction`: **open with accepted residual**, with the measured number and date;
- every other material row: at or under gate;
- the closed families: allocator replacement (18), transient arenas (17), header elimination (35),
  first-touch/residency (36), read-traversal sorting (38), payload-address sorting (43), slotted+fused
  machinery beyond this adoption (44), chunked/slab/arena allocation (45);
- **the standing finding**: ordering is worth ~47% of construction time on M4, and every mechanism to
  obtain it costs more than it returns. Any future proposal must state how it avoids that.

## 8. Out of scope

- Any further attempt at the ordering lever — the family is closed (§7).
- Opt-in variants: this is a default change; there is no enable/disable knob.
- Changes to `lazyXor` or any non-`bor` path.

## 9. Chunking

Not chunked — pending review. Plausibly single-chunk: promote, delete, re-measure, record.
