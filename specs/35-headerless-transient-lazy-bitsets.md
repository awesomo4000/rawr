<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 35: Headerless transient lazy bitsets (E3)

Campaign: [31-structural-parity-campaign.md](31-structural-parity-campaign.md) (Wave 3 — the
finale). Close the **last material M4 gap**: **lazyOr construction 1.663x** (post-`d7d357b`
canonical board). **lazyOr + repair 1.178x is downstream of the same phase** — repair-alone is
**1.049x** (fine), so closing construction is expected to close the combined row too; both are
gated here. Once these close, **every material M4 row is within the defined ≤ 1.10x gate** — note
that several small rows (addMany sequential 1.073x, serialize 1.045x, add sequential 1.036x,
toArrayAlloc 1.033x) remain **above 1.0x** while inside the gate; "within the gate" is the claim,
not "at or under parity."

**Parity is a hard requirement** — rows close at ≤ 1.10x; a partial is adopted by owner judgement
(spec-30 policy) and the row stays open.

## Post-Wave-1 baseline (pinned)

This spec measures against the fresh post-`d7d357b` board (Wave 1 shipped: compact Run header,
word-major orMany). M4: lazyOr construction **5.746 ms vs 3.456 ms = 1.663x**; lazyOr+repair
**14.612 vs 12.403 = 1.178x**; lazyOr repair-alone **8.315 vs 7.928 = 1.049x**. Zen 4 reference
numbers are captured fresh in `35-00` (the no-regress gate needs them). The **compact Array header
was NO-GO here** (spec 32 made lazy-OR *worse*) — that lever is closed; this spec attacks the
**transient bitset accumulators**, which spec 32 never touched.

## Verified structural facts (function names authoritative; mirror line numbers may lag `d7d357b`)

In `lazyMergeTwo` (`bitmap.zig`, ~`:2210` pre-Wave-1):

- For every **matched key**, `use_lazy_bitset = (op == .xor) or bitset_conversion or either side is
  a bitset`. The canonical row passes `bitset_conversion = true`, so **every matched pair takes the
  bitset path** regardless of cardinality.
- That path does **`BitsetContainer.init(allocator)`** = **2 allocator calls** (16 B header `create`
  + 8 KB aligned `words`) **plus an 8 KB `@memset` zero-fill**, then accumulates both sides into it.
- **Unmatched keys** are `cloneContainer`ed (the thousands-of-array-clones share; spec 17/18 closed
  the arena/allocator levers for those, and spec 32's Array-header NO-GO closed the header lever).
- `repairAfterLazy` then, per bitset: `computeCardinality()` (full 1024-word scan); **card ≤ 4096 →
  `bitsetToArray` + `bc.deinit` — the header AND the 8 KB words both die**; card 0 → both die;
  only card > 4096 survives.
- `BitsetContainer.words` is **already a separate allocation** from the header — so a survivor path
  that "allocate header late, adopt words" requires **no copy**; adoption is pointer assembly.

**Corpus shape (assert in `35-00`, don't trust this estimate):** the sparse corpus (500 k random
u32 across the full space; `a = [0..half]`, `b = [half/2..]`) gives roughly **~16 k matched keys**,
each holding tiny arrays (~7–15 values) — so essentially **every transient bitset demotes** at
repair. On this corpus the eliminated-vs-deferred split (umbrella) collapses to the good case:
**headers (and words) are born-to-die**, E3's exact target. `35-00` must pin the actual counts:
matched keys, transient bitsets created, demoted vs surviving, per-key cardinalities.

## Attribution first — extend the EXISTING harness, do not build a second one

`35-00` **extends `src/bench_lazy_or_attribution.zig` and `tools/croaring_lazy_attribution.h`** —
the repository already has this attribution system (the earlier arena experiment measured **130,994
allocations** with roughly **32,726 transient-container allocation calls**). Reconcile against those
recorded numbers; **do not build a parallel attribution.**

Attribute the 1.663x across:

1. **Transient-bitset lifecycle** — header create, 8 KB words alloc, 8 KB zero-fill, accumulate,
   repair scan, demote copy, header free, words free — per matched key, summed.
2. **Unmatched clone traffic** — the known residual share (closed levers; measured for share, not
   as a target).
3. **Top-level assembly** — `initCapacity(min(a.size+b.size, 65536))` etc.

**Why this is not the failed arena (spec 17) again — state explicitly in the chunk:** spec 17
redirected transient bitsets into a bump arena, which lost because **bulk-freeing one slab cost more
than individual SMP frees** and the arena's lifetime failed the memory gate. E3 changes **neither**:
the **8 KB words remain individually SMP-allocated and individually freed, with unchanged
lifetime**. Only the **16 B header allocation and its matching free** disappear. Different
mechanism, different failure mode.

**CRoaring materialization count — an assertion, not an open question.** The vendored authoritative
source settles the semantics: in `roaring_bitmap_lazy_or`, when `bitsetconversion` is true and
neither matched container is already a bitset, CRoaring calls **`container_to_bitset`** and then
`container_lazy_ior` (`vendor/roaring.c`, the `bitsetconversion &&` branch). **CRoaring does not
threshold tiny pairs** — it materializes a bitset exactly as rawr does. So `35-00` keeps the
per-key materialization count **as an assertion that both sides materialize the same way** (guarding
against a future divergence or a mis-set flag), and there is **no semantics-mismatch fork and no
owner contract decision** — the comparison is already like-for-like.

## Activation scope (pinned — every path that takes the transient branch)

The transient tag is used **exactly where `use_lazy_bitset` is true** in `lazyMergeTwo`:

```
use_lazy_bitset = (op == .xor) or bitset_conversion
                  or isBitsetContainer(c_a) or isBitsetContainer(c_b)
```

So the affected public surface is **wider than lazyOr(…, true)** and must all be covered:

| entry point | takes transient path when |
|---|---|
| `lazyOr(…, bitset_conversion = true)` | **always** (every matched key) — the target row |
| `lazyOr(…, bitset_conversion = false)` | **when either matched side is already a bitset** |
| `lazyOrInPlace(…, bitset_conversion)` | same as `lazyOr` (delegates to it) |
| `lazyXor(…)` | **always** — it calls `lazyMergeTwo(.xor, …, true)` and `op == .xor` forces it |
| `lazyXorInPlace(…)` | same as `lazyXor` |

**Correction to an earlier draft:** `bitset_conversion = false` is **not** untouched, and the XOR
path involved is **`lazyXor` / `lazyXorInPlace`** — **not** `xorMany`, which is a different code
path and genuinely untouched. `35-01` must carry **`lazyXor` / `lazyXorInPlace` correctness
coverage** (including the XOR-specific `lazyToggle` / `lazyXorWith` accumulation and the
`card == 0` drop rule), and the false-flag bitset-input case, not just the sparse OR row.

## The lever (L1) — never allocate a header that is going to die

For the transient accumulator on the lazy path:

- Allocate **only the aligned 8 KB words** (one allocator call instead of two; still zeroed —
  correctness requires a zeroed accumulator for OR).
- Track the transient state via the **`reserved = 0b11` tag**.

**Correction — the `reserved` member exists but is `void`.** `Container` **does** have a
`.reserved` member; it is **`reserved: void`**, and `fromTagged` maps `.reserved => .{ .reserved =
{} }`, **discarding the pointer**. So this is not "a union with no member" — it is a member that
**throws away the payload**. Production therefore needs a **real words-pointer payload** (e.g.
`reserved: *align(64) [1024]u64`, or equivalent accessors that recover the words pointer from the
tagged pointer) — that representation change is part of `35-01`, not a free tag reuse.

**Behavior contract (explicit, not "defined behavior"):** a transient container **behaves as an
unknown-cardinality bitset** for:

- **read-only queries** — `contains`, `getCardinality` (computes from words), `rank`, `select`,
  `minimum`/`maximum`, iteration, `toArray`, equality/subset;
- **clone** — clones to a normal owned bitset (or a transient with its own words), never aliases;
- **repeated lazy operations** — a transient accumulator can be accumulated into again (the
  `lazyOrInPlace`-then-`lazyOr` case) without materializing a header;
- **repair** — the demote/survive path below;
- **deinit** — frees the words; **no header to free**.

**Mechanical inventory required (not a hand list):** `35-01` must `rg` **every** `.reserved` switch
site and give each a defined arm — the mirror shows **~98 sites across 17 files**, of which the
production ones are `bitmap.zig`, `container.zig`, `container_ops.zig`, `compare.zig`, `optimize.zig`,
`serialize.zig`, `roaring64.zig`. Explicitly include `validate`, conversion/optimization
(`runOptimize`/`optimize`), range and set operations, `toArray`, equality/subset, minimum/maximum,
and clearing. **No arm may remain a default fall-through or a silent `unreachable` on a path a
transient container can actually reach.**

- **Repair:** compute cardinality directly from the words; **demote → free the words, no header was
  ever allocated** (the eliminated case); **survive → allocate the 16 B header and adopt the words**
  (deferred case — no copy, words are already a separate allocation).

## Eliminated vs deferred (the load-bearing accounting — umbrella, verbatim obligation)

The diagnostic reports all five, and **the gate is the COMBINED construction+repair row**:

1. headers **permanently eliminated** (demotion),
2. headers **deferred** to repair (survivors),
3. **construction-only** allocation reduction,
4. **full construction+repair** allocation reduction,
5. **repair regression** from allocating surviving headers there.

On the canonical sparse corpus, `35-00`'s pinned counts are expected to show ~all-eliminated /
~zero-deferred — but the accounting must hold for bitset-heavy corpora too, so the deferred path's
repair cost is **gated, not merely measured**:

### Dense survivor control — numeric gate (not just "measured")

A survivor-heavy corpus (matched pairs whose union stays > 4096, so headers are **deferred** to
repair rather than eliminated) is run as a control, and **all three of its rows — construction,
repair-only, and combined — must stay within noise (≤ 5%) on BOTH hosts.** Rationale: without this
gate the optimization could **move header cost from construction into repair**, passing the sparse
construction gate while **regressing survivor-heavy workloads**. A dense-control regression beyond
noise fails the chunk (owner-exception route per spec-30 policy remains, explicitly recorded).

## Numeric stop-gate (before touching the container union)

Benchmark-only prototype first. Pin the bar: **permanently-eliminated header calls × measured
per-call SMP cost, plus the removed header frees, must project the combined construction+repair row
to ≤ 1.10x** — or show a required focused-time improvement that does. If the header-call arithmetic
cannot get there (e.g. the 8 KB zero-fill + repair scan dominate, not the 16 B create), **stop
before changing the container union** and report what *does* dominate — that attribution then
drives whatever follows (possibly the owner decision from the like-for-like check above).

## Measurement discipline

- Canonical protocol: **3 warmup / 21 timed, five process medians + full range**, fresh-process,
  **M4 and Zen 4**, one CRoaring reference per host; E3-owned diagnostic module (shared
  `build.zig`/runner/docs edits are implementer-owned).
- **Accounting per cell:** allocations, frees, requested bytes, effective SMP-class bytes,
  teardown — container instances ≠ allocator calls.
- **Construction and repair measured separately AND combined** (the gate is combined; the split is
  the attribution).
- **Both-flag correctness before performance** (spec-32 discipline): the transient-tag build passes
  `zig build test`, `zig build difftest`, `ReleaseSafe`, `ReleaseFast` before numbers are accepted.
- **Zen 4 policy (spec 30):** within-noise passes (repeated focused timing + range overlap); a real
  regression fails by default, adoptable only via explicit owner exception.
- **Board gate + spec-28 layout exception** on production adoption; one architecture-neutral shape.

## Correctness (production chunk)

- **Output invariants:** post-repair result has the same container kinds / cardinalities / values
  as baseline lazyOr+repair, identical portable bytes where serialize is valid; CRoaring
  set-parity differential.
- **The full transient-tag lifecycle inventory** (above) tested explicitly — including
  deinit-before-repair (words freed, no header leak).

### Repair failure — concrete transactional strategy required ("build-then-commit" is not enough)

`repairAfterLazy` **mutates in place**: it compacts with a `write_idx` and **frees containers
(demoted bitsets, empty containers) before committing `self.size`**. An allocation failure partway
through (today: `bitsetToArray`; with E3 additionally: the **deferred survivor header allocation**)
can therefore leave **stale entries** — freed containers still referenced beyond `write_idx`, or a
half-converted array. E3 **adds a failure point**, so `35-01` must pin **one** of:

- **(a) Two-phase replacement** — build the repaired key/container arrays **beside** the originals,
  committing only on full success (originals freed after commit); or
- **(b) Explicit rollback bookkeeping** — a recorded undo log sufficient to restore a
  consistently-deinit-able bitmap on failure at any point.

Either way the **post-failure invariant** is explicit: the bitmap is **valid and deinit-able with no
leak and no double-free**, and its logical contents are either fully repaired or the documented
partial state — never dangling.

**Failure injection must hit, at minimum:** the **first**, a **middle**, and the **last** demotion
position; the **first / middle / last survivor-header** allocation position; the words allocation
during construction; and demote-array allocation — each verified valid-or-cleanly-errored, inputs
untouched, no leak.
- **Coverage follows the activation table** — `lazyOr(true)`, `lazyOr(false)` **with a bitset input**,
  `lazyOrInPlace`, **`lazyXor`, `lazyXorInPlace`** all exercised. **`xorMany` and the eager set ops
  are genuinely untouched** (different code path).

## Acceptance

- **Phase 1 GO (35-00):** corpus counts pinned (matched keys, transient bitsets, demote/survive
  split); the three-way attribution reported **through the extended
  `bench_lazy_or_attribution` / `croaring_lazy_attribution.h`, reconciled against the recorded
  130,994 allocations / ~32,726 transient-container calls**; **CRoaring materialization-count
  assertion** (both sides materialize identically) green; headerless prototype measured
  benchmark-only with the five-figure eliminated/deferred accounting, both hosts; stop-gate
  arithmetic explicit. No production change.
- **Phase 2 GO — hard (35-01):** **lazyOr construction AND lazyOr+repair reach ≤ 1.10x on M4 SMP**,
  **dense survivor control within noise (≤ 5%) on construction, repair-only, and combined, both
  hosts**, Zen 4 within noise, the **mechanical `.reserved` inventory complete** (every site an
  explicit arm), the **pinned repair transactional strategy** with first/middle/last demotion and
  survivor-position failure injection green, activation-table coverage (incl. `lazyXor` /
  `lazyXorInPlace`) green, invariants + board gate held. Partial adoption per spec-30 policy (owner
  judgement, row stays open).
- `zig build test`; `zig build difftest`; `ReleaseSafe`/`ReleaseFast`; canonical
  `run-compare-bench.sh` both hosts on adoption; `docs/parity-measurement.md` updated.

## Proposed chunk plan (confirm at review)

- **`35-00`** — extend the **existing** attribution harness (lifecycle / clone-share / assembly,
  reconciled to the recorded numbers) + **CRoaring materialization-count assertion** + pinned corpus
  counts + benchmark-only headerless prototype with eliminated/deferred accounting + **dense survivor
  control** + stop-gate arithmetic, both hosts. No production change.
- **`35-01`** — production transient-tag migration (conditional on `35-00` GO): the **words-pointer
  payload** representation, the **mechanical `.reserved` site inventory**, the pinned **repair
  transactional strategy** (two-phase or rollback), activation-table coverage incl. `lazyXor` /
  `lazyXorInPlace`, invariants, positional failure injection, board gate, ship on both-host numbers.

## Estimate

M for `35-00` (attribution + instrumented CRoaring count + prototype, two hosts). M–L for `35-01`
(container-union change with the full lifecycle inventory) — if it runs.
