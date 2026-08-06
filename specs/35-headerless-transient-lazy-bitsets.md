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
  correctness requires a zeroed accumulator for **OR and XOR**).
- Track the transient state in the `0b11` tag slot — **renamed, not reused as `.reserved`**.

**RENAME the tag — do not keep `.reserved`.** `Container` **does** have a `.reserved` member today;
it is **`reserved: void`**, and `fromTagged` maps `.reserved => .{ .reserved = {} }`, **discarding
the pointer**. Keeping that name would let **dangerous existing arms keep compiling** — every
`.reserved => {}`, `=> false`, `=> null` silently becomes a wrong answer for a real transient
container. So `35-01` **renames the tag and gives it a payload**:

```zig
// ContainerType
lazy_bitset = 0b11,          // was: reserved
// Container union
lazy_bitset: *align(64) [1024]u64,   // was: reserved: void
```

**The rename is the enforcement mechanism:** Zig then emits a **compile error at every unhandled
switch arm**, converting the mechanical inventory below from a discipline into a
compiler-checked invariant. A silent-fall-through arm becomes impossible rather than merely
discouraged.

**Behavior contract (explicit, not "defined behavior"):** a transient container **behaves as an
unknown-cardinality bitset** for:

- **read-only queries** — `contains`, `getCardinality` (computes from words), `rank`, `select`,
  `minimum`/`maximum`, iteration, `toArray`, equality/subset;
- **clone — PINNED: clone to a NORMAL owned unknown-cardinality bitset** (allocate a real header,
  copy the words, cardinality `-1`). Not "either/or": this is closest to current clone semantics and
  **stops the transient representation from propagating** into cloned bitmaps. Never aliases.
- **repeated lazy operations** — a transient accumulator can be accumulated into again (the
  `lazyOrInPlace`-then-`lazyOr` case) without materializing a header;
- **Serialization — PINNED: ERROR on an unrepaired bitmap, at EVERY writing entry point**
  (documented `error.UnrepairedLazyResult`): the portable format has no transient type,
  cardinalities are unknown, and silently repairing inside a const-ish serialize would surprise.
  Callers repair first. **Name them all so none is overlooked: `serialize`, `serializeToWriter`, and
  `OwnedBitmap.serialize`** (plus any other writer that walks containers — the frozen/`roaring64`
  writers are checked in the inventory). All already return error unions, so this is additive.
- **`serializedSizeInBytes` — PINNED: keep the signature, compute the true size.** Its public
  signature is **`fn serializedSizeInBytes(self: *const Self) usize`** — **no error channel** — so it
  **cannot** return `error.UnrepairedLazyResult`. Chosen (least disruptive, no API break, no panic):
  **keep `usize` and compute the correct size for a transient container by scanning its words**
  (derive the cardinality, then the size the container *would* serialize as). **Rejected
  alternatives, recorded:** an unconditional panic on unrepaired input (hostile for a pure size
  query), and a breaking change to `!usize` (API break for an internal optimization).
- **`validate` — PINNED: return `UnrepairedLazyResult` on ANY `.lazy_bitset`.** There is **no
  bitmap-level repaired-state flag** — the tag itself is the only state indicator — so `validate`
  **cannot** distinguish "valid because unrepaired" from "invalid because it survived repair".
  Therefore: `validate` **adds `UnrepairedLazyResult` to `ValidateError` and returns it whenever it
  encounters a transient container**; **normal validation must then pass after `repairAfterLazy`**
  (that pairing is what machine-checks "no transient survives repair"). Pointer/alignment checks for
  the transient itself live in an **internal transient-state test/helper**, not in public `validate`
  — **no state field is added to every bitmap.**
- **repair** — the demote/survive path below;
- **deinit** — frees the words; **no header to free**.

**Eager set ops:** "untouched" means they **never produce** a transient container. Their **dispatch
must still consume one correctly** if the contract admits an unrepaired bitmap as an input — so each
eager-path arm is either a real implementation (treating it as an unknown-cardinality bitset) or an
explicit documented rejection. Which one is chosen per op is pinned in `35-01`; silence is not an
option.

**Mechanical inventory (compiler-enforced by the rename):** `35-01` must `rg` **every** tag switch
site and give each an explicit arm — the mirror shows **~98 sites across 17 files**, of which the
production ones are `bitmap.zig`, `container.zig`, `container_ops.zig`, `compare.zig`, `optimize.zig`,
`serialize.zig`, `roaring64.zig`. Explicitly include `validate`, conversion/optimization
(`runOptimize`/`optimize`), range and set operations, `toArray`, equality/subset, minimum/maximum,
and clearing. **No arm may remain a default fall-through, and no `unreachable` may sit on a path a
transient container can actually reach** — the rename makes each one a compile error until addressed.

- **Repair:** compute cardinality directly from the words; **demote → free the words, no header was
  ever allocated** (the eliminated case); **survive → allocate the 16 B header and adopt the words**
  (deferred case — no copy, words are already a separate allocation).

## Eliminated vs deferred (the load-bearing accounting — umbrella, verbatim obligation)

The diagnostic reports all five, and **BOTH the construction row AND the combined construction+repair
row are hard gates** (construction ≤ 3.802 ms, combined ≤ 13.643 ms on M4 — construction is the
binding constraint):

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
repair rather than eliminated) is run as a control on **all three rows — construction, repair-only,
and combined — on BOTH hosts**, under a **one-sided no-regression gate**:

```
candidate / baseline ≤ 1.05      (per row, per host)
```

**One-sided on purpose:** dense **construction is expected to IMPROVE** (its header allocation moves
out of construction into repair), so a two-sided "within noise" band would wrongly flag the very
effect we want. **Improvements are always allowed**; only regression beyond 1.05 fails. Apply the
same explicit one-sided rule to repair-only and combined, **plus process-range analysis** (repeated
fresh-process medians with overlapping ranges) rather than a single-number comparison.

Rationale: without this gate the optimization could **move header cost from construction into
repair**, passing the sparse construction gate while **regressing survivor-heavy workloads**. A
dense-control regression beyond 1.05 fails the chunk (owner-exception route per spec-30 policy
remains, explicitly recorded).

## Numeric stop-gate (before touching the container union)

**The removable half — do not double-count.** The recorded **~32,726** transient-container
allocation calls are approximately **one header call + one words call for each of ~16,363 matched
keys**. **E3 removes ONLY the ~16,363 header calls (and their matching frees); the ~16,363 words
allocations REMAIN** — the accumulator still needs its 8 KB. The available benefit is therefore
**about half** the transient-call figure, and the stop-gate projection **must be computed on the
~16,363 header calls, never on the ~32,726 total.**

Benchmark-only prototype first. Pin the bar: **~16,363 eliminated header calls × measured per-call
SMP cost, plus the matching frees, must project **BOTH** hard rows — **construction to ≤ 3.802 ms
AND combined construction+repair to ≤ 13.643 ms** (M4; either failing stops the work) —
or show a required focused-time improvement that does. If that arithmetic cannot get there (e.g. the
8 KB zero-fill + repair scan dominate, not the 16 B create), **stop before changing the container
union** and report what *does* dominate; that attribution drives whatever follows.

## Measurement discipline

- Canonical protocol: **3 warmup / 21 timed, five process medians + full range**, fresh-process,
  **M4 and Zen 4**, one CRoaring reference per host; E3-owned diagnostic module (shared
  `build.zig`/runner/docs edits are implementer-owned).
- **Accounting per cell:** allocations, frees, requested bytes, effective SMP-class bytes,
  teardown — container instances ≠ allocator calls.
- **Construction and repair measured separately AND combined** — **hard gates are construction and
  combined; repair-only is attribution.**
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
half-converted array. E3 **adds a failure point**, so a strategy must be pinned. Three are
permitted; **`35-01` SELECTS (c)**:

- **(a) Two-phase replacement** — build the repaired key/container arrays **beside** the originals,
  committing only on full success (originals freed after commit). **Rejected:** a second top-level
  array allocation per repair can cancel the saving E3 is chasing.
- **(b) Explicit rollback bookkeeping** — a recorded undo log sufficient to restore a
  consistently-deinit-able bitmap on failure at any point. **Rejected:** same cost objection.
- **(c) SELECTED — per-container build-before-free with in-place partial commit.** Allocate each
  replacement (demoted array, or the survivor header adopting the words) **before** retiring the old
  container. On failure: the repaired **prefix** `[0, write_idx)` stands, the **untouched tail is
  compacted behind it** (those entries keep their existing valid containers), `self.size` is updated,
  **`cached_cardinality` stays `-2`** (transients remain in the tail, so the reject-preflight sentinel
  must not be cleared), and the error is returned. **No parallel arrays, no undo log** — so the
  repair gain survives. A failed repair **may be retried.**

The **post-failure invariant** is explicit: the bitmap is **valid and deinit-able with no leak and no
double-free**, every entry in `[0, self.size)` is a live owned container, and its logical contents are
either fully repaired or the documented partial state — never dangling.

**Failure injection must hit, at minimum:** the **first**, a **middle**, and the **last** demotion
position; the **first / middle / last survivor-header** allocation position; the words allocation
during construction; and demote-array allocation — each verified valid-or-cleanly-errored, inputs
untouched, no leak.
- **Coverage follows the activation table** — `lazyOr(true)`, `lazyOr(false)` **with a bitset input**,
  `lazyOrInPlace`, **`lazyXor`, `lazyXorInPlace`** all exercised.
- **Eager set operations never PRODUCE a transient container, but their dispatch IS updated** per the
  pinned consume/reject policy above (each arm a real unknown-cardinality-bitset implementation or an
  explicit documented rejection) — they are **not** "untouched".
- **The many-ops likewise: they produce no transients, but they can RECEIVE one.** `orMany`,
  `orManyHeap`, `xorMany` (and `orManyOwned` / `xorManyOwned`) take **`[]const *const Self`** — a
  caller can pass an **unrepaired lazy bitmap** in that slice. So their **input dispatch follows the
  same eager consume/reject policy**; "cannot receive one" is false and must not be assumed.

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
  hosts (one-sided ≤ 1.05 per row)**, Zen 4 within noise, the **tag renamed to `.lazy_bitset` with
  the compiler-enforced inventory complete** (every site an explicit arm; no fall-through, no
  reachable `unreachable`), the **pinned repair transactional strategy** with first/middle/last
  demotion and
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
- **`35-01`** — production transient-tag migration (conditional on `35-00` GO): the **tag rename
  `.reserved` → `.lazy_bitset` with a words-pointer payload** (compile errors drive the site
  inventory), the pinned **clone / serialize / validate / eager-dispatch behaviors**, the pinned **repair
  transactional strategy** (**(c)** per-container build-before-free with in-place partial commit),
  activation-table coverage incl. `lazyXor` /
  `lazyXorInPlace`, invariants, positional failure injection, board gate, ship on both-host numbers.

## Estimate

M for `35-00` (attribution + instrumented CRoaring count + prototype, two hosts). M–L for `35-01`
(container-union change with the full lifecycle inventory) — if it runs.
