<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 35-01: `.lazy_bitset` production migration

Toplevel: [35-headerless-transient-lazy-bitsets.md](35-headerless-transient-lazy-bitsets.md) (E3).
Ship the headerless transient accumulator in production. **Gated on `35-00`'s dual stop-gate
projecting BOTH hard rows: lazyOr construction ≤ 3.802 ms AND lazyOr+repair (combined) ≤ 13.643 ms
on M4** (construction is the binding constraint). This is the risky chunk: it changes the core
container discrimination.

## 1. Tag rename + payload (the enforcement mechanism)

```zig
// ContainerType
lazy_bitset = 0b11,                  // was: reserved
// Container union
lazy_bitset: *align(64) [1024]u64,   // was: reserved: void  (fromTagged discarded the pointer)
```

**Rename, do not reuse `.reserved`** — the rename makes every unhandled switch arm a **compile
error**, so dangerous existing arms (`.reserved => {}` / `=> false` / `=> null`) cannot silently
survive. This converts the dispatch inventory below into a **compiler-checked invariant**.

## 2. Lever

On the lazy path, allocate **only the aligned 8 KB words** (zeroed — required for OR **and** XOR);
no header. **Repair:** compute cardinality from the words; **demote → free words, no header ever
allocated**; **survive → allocate the 16 B header and adopt the words** (no copy — words are already
a separate allocation).

## 3. Activation scope (all must be covered)

Transient containers arise exactly where `use_lazy_bitset` is true
(`op == .xor or bitset_conversion or either side is a bitset`):

| entry point | takes transient path when |
|---|---|
| `lazyOr(…, true)` | always — the target row |
| `lazyOr(…, false)` | when either matched side is already a bitset |
| `lazyOrInPlace(…)` | as `lazyOr` |
| `lazyXor(…)` | always (`lazyMergeTwo(.xor, …, true)`) |
| `lazyXorInPlace(…)` | as `lazyXor` |

XOR-specific behavior (`lazyToggle` / `lazyXorWith`, the `card == 0` drop rule) is covered
explicitly, not just the sparse OR row.

## 4. Pinned behavior contract

A transient behaves as an **unknown-cardinality bitset** for read-only queries (`contains`,
`getCardinality` from words, `rank`, `select`, `minimum`/`maximum`, iteration, `toArray`,
equality/subset), and:

- **clone** → a **normal owned unknown-cardinality bitset** (allocate header, copy words,
  cardinality `-1`); never aliases, never propagates the transient representation.
- **repeated lazy operations** → a transient can be accumulated into again without materializing a
  header.
- **serialization → `error.UnrepairedLazyResult` at EVERY writing entry point**: `serialize`,
  `serializeToWriter`, `OwnedBitmap.serialize`, plus any other container-walking writer
  (frozen / `roaring64`). All already return error unions.
- **`serializedSizeInBytes` → keep the `usize` signature**; compute the true (post-repair-equivalent)
  size by scanning the transient's words. **No panic, no `!usize` API break.**
- **`validate` → add `UnrepairedLazyResult` to `ValidateError` and return it on ANY `.lazy_bitset`**;
  normal validation must **pass after `repairAfterLazy`**. That pairing machine-checks "no transient
  survives repair." Transient pointer/alignment checks live in an **internal helper** — **no
  repaired-state field is added to any bitmap.**
- **deinit** → frees the words; no header to free.

### Consume-vs-reject policy — DECIDED (not "pinned per op" later)

**Governing rule:** a transient is **bit-identical to a `BitsetContainer`'s words with cardinality
`-1`**. So **read paths CONSUME** by reusing the existing bitset arm (near-zero new code), and
everything that **mutates, combines, or converts REJECTS** with `error.UnrepairedLazyResult` (small,
uniform, and keeps the transient from leaking into kernels). Callers repair first.

**Hard constraint driving the table:** an operation **can only REJECT if it already returns an error
union.** Every no-error signature (`void` / `bool` / `u64` / `f64` / `?u64`) **must CONSUME** —
rejecting would require an API break or a panic.

| category | decision | rationale |
|---|---|---|
| Single-bitmap **read queries** — `contains`, `getCardinality`, `rank`, `select`, `minimum`/`maximum`, iteration, `toArray` | **CONSUME** | reuse the bitset arm on the words; cardinality computed, not cached |
| **Equality / subset** (`compare.zig`) | **CONSUME** | pure reads; bitset arm applies to both sides |
| **No-error reads (complete inventory)** — `differenceCardinality` (`u64`), `jaccardIndex` (`f64`), `containsRange` (`bool`), `rankMany` (`void`), `getIndex` (`?u64`), `isEmpty` (`bool`) | **CONSUME** | **no error channel** — cannot reject without an API break or panic; all pure reads |
| **`clearRetainingCapacity`** (`void`) | **CONSUME / cleanup** | cannot reject (`void`). **Frees transient words** like any container, clears the bitmap, and sets **`cached_cardinality = 0`** (empty, known) — clearing an unrepaired bitmap is well-defined |
| **Eager set ops** by value — `bitwiseAnd`/`Or`/`Xor`/`Difference` | **REJECT** | would require transient arms in every `container_ops` kernel — large surface, no benefit |
| **In-place mutations** — `add`, `remove`, `addRange`, `removeRange`, `bitwiseOrInPlace`, etc. (all return error unions) | **REJECT** | mutating an unrepaired accumulator has no coherent cardinality semantics; O(1) preflight below rejects **before** any mutation |
| **Pairwise cardinality-only** — `andCardinality`, `orCardinality`, `xorCardinality`, `intersects`, `intersectsRange` | **CONSUME** | **they CANNOT reject**: signatures are plain `u64` / `bool` with **no error channel**, so rejecting would force an API break or a panic. They are **pure reads**, so they consume the transient words as an unknown-cardinality bitset — same class as equality/subset |
| **Many-ops** — `orMany`, `orManyHeap`, `xorMany`, `orManyOwned`, `xorManyOwned` | **REJECT** (input dispatch) | take `[]const *const Self` and **can receive an unrepaired bitmap**; consuming would need transient arms in every many-kernel |
| **Optimization** — `runOptimize`, `optimize` | **REJECT** | converting representation before repair is meaningless |
| **Range ops** — `rangeCardinality` (read) | **CONSUME** | read-only; bitset arm applies |
| **Range ops** — `flip`, `removeRangeCopy` and other range **mutations/constructions** | **REJECT** | mutation/conversion class |
| **Serialization** — see above | **REJECT** | no transient type in the portable format |
| **`validate`** — see above | **REJECT** (`UnrepairedLazyResult`) | pairs with post-repair pass to machine-check the invariant |

Eager and many-ops still **never PRODUCE** a transient — only their **input dispatch** changes, per
this table.

### How rejecting APIs detect an unrepaired bitmap — O(1) preflight via a `cached_cardinality` sentinel

A rejecting **in-place** method must **reject before mutating any container**, and a **full tag scan
on every normal eager operation would be an O(n) regression** on the overwhelmingly common repaired
case. Pinned mechanism — **reuse the existing `cached_cardinality: i64` field** (no new bitmap field,
no extra memory):

```
cached_cardinality >= 0  → known cardinality (repaired)
                   -1    → unknown, no transients        (existing meaning, unchanged)
                   -2    → NEW: contains transient lazy containers
```

State transitions (pin all of them):

- **lazy construction** sets **`-2` only when it actually emits a transient** — a lazy merge that
  emits none keeps `-1` (so `bitset_conversion = false` with no bitset inputs is unaffected);
- **read queries (CONSUME)** may compute their result but **preserve `-2`** — they must not
  "upgrade" the state by caching a cardinality over it;
- **clone** materializes transients into normal bitsets → the clone gets **`-1`** (never `-2`);
- **failed repair** (partial commit) **retains `-2`** — transients remain in the tail, and retry is
  valid;
- **successful repair** stores the **real cardinality** (`>= 0`);
- **rejecting APIs** do an **O(1) preflight**: `if (cached_cardinality == -2) return
  error.UnrepairedLazyResult;` **before any mutation**.

This gives **no partial mutation**, **no per-operation container scan on the normal path**, and adds
**no field**. Test the sentinel's transitions explicitly, including: lazy-with-no-transient stays
`-1`, a consume-class read leaves `-2` intact, clone yields `-1`, failed repair still `-2` and a
retry succeeds, successful repair yields `>= 0`.

## 5. Dispatch inventory (compiler-driven)

Work the **compile errors** from the rename to completion — the mirror shows **~98 tag-switch sites
across 17 files**, production being `bitmap.zig`, `container.zig`, `container_ops.zig`, `compare.zig`,
`optimize.zig`, `serialize.zig`, `roaring64.zig`. Explicitly include `validate`, conversion /
optimization (`runOptimize` / `optimize`), range and set operations, `toArray`, equality/subset,
minimum/maximum, and clearing. **No default fall-through; no `unreachable` on any path a transient
can actually reach.**

## 6. Repair transactional strategy — SELECTED: per-container build-before-free + in-place partial commit

`repairAfterLazy` mutates in place and **frees containers before committing `self.size`**; E3 adds a
failure point (the **deferred survivor header** allocation). **Selected strategy (c)** — cheaper than
the two alternatives and it **does not erase the repair gain**:

- **Per container, build before free:** allocate the replacement first (demoted array, or the
  survivor's 16 B header adopting the words); only once it exists, free/retire the old container and
  write it at `write_idx`.
- **On failure, commit the partial in place:** the successfully repaired **prefix** `[0, write_idx)`
  is already correct. **Compact the untouched tail behind that prefix** (the not-yet-visited entries
  keep their existing, still-valid containers), **update `self.size`** to prefix + tail,
  **set `cached_cardinality = -2`** (**not `-1`** — transients remain in the tail, so the sentinel
  must stay `-2` or rejecting APIs would happily operate on them), and **return the error**.
- No parallel top-level arrays, no undo log — **both rejected** because allocating a second
  key/container array (a) or maintaining an undo log (b) adds per-repair cost that could **cancel the
  very saving E3 is chasing**.

**Post-failure invariant (explicit):** the bitmap is **valid and deinit-able, no leak, no
double-free**, every entry in `[0, self.size)` is a live owned container, and cardinality is
**unknown**; contents are either fully repaired or the documented partial state — never dangling. A
failed repair **may be retried** (the remaining transients are still in the tail).

## 7. Failure injection (positional)

At minimum: **first / middle / last demotion** position; **first / middle / last survivor-header**
allocation position; words allocation during construction; demote-array allocation. Each →
valid-or-cleanly-errored, **inputs untouched**, no leak.

## 8. Gates

- **Output invariants:** post-repair results match baseline lazyOr+repair in container kinds,
  cardinalities, and values; identical portable bytes where serialize is valid; CRoaring set-parity
  differential.
- **Dense survivor control:** construction, repair-only, combined — **one-sided
  `candidate / baseline ≤ 1.05`**, both hosts, with process-range analysis.
- **Zen 4 policy (spec 30):** within noise passes (repeated focused timing + range overlap); a real
  regression fails by default, adoptable only via explicit owner exception.
- **Board gate + spec-28 layout exception**; **one architecture-neutral shape**.

## Acceptance

- **lazyOr construction AND lazyOr+repair reach ≤ 1.10x on M4 SMP**; dense survivor control within
  the one-sided gate on both hosts; Zen 4 within noise.
- Tag renamed with the **compiler-enforced inventory complete**; the pinned clone / serialize /
  `serializedSizeInBytes` / `validate` behaviors and the **decided consume-vs-reject table**
  implemented and tested; activation-table coverage (incl. `lazyXor` / `lazyXorInPlace`) green;
  **build-before-free + in-place partial commit** repair with positional failure injection green
  (including the retry-after-failed-repair case); board gate held.
- Partial adoption per spec-30 policy (owner judgement; row stays open above 1.10x).
- `zig build test`; `zig build difftest`; `ReleaseSafe` / `ReleaseFast`; canonical
  `run-compare-bench.sh` both hosts; `docs/parity-measurement.md` updated.
