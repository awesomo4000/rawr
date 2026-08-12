<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 41: Documentation parity — `README.md`, `API.md`, and a coverage guard

**Goal.** Bring the two user-facing documents in line with the API that actually ships, and add a
mechanized guard so they cannot drift again silently. No production code behaviour changes.

**Why now.** The library is feature-complete and 32-bit-clean, and the repo is now public.
`API.md`, `README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md` are the only non-source entries in
`build.zig.zon`'s `.paths` allowlist — verified by building a consumer against an allowlist-only package
tree. `specs/` and `docs/` are repo-only. So those two files carry the entire documented contract.

## 1. Measured drift — a floor, not a total

**`RoaringBitmap` — 4 undocumented:** `bitwiseOrInPlaceConsume` (spec 19; ownership contract),
`removeRangeCopy` (spec 30), `repairAfterLazyWithOptions` (spec 39-01), `clone`.

**`Roaring64Bitmap` — 10 undocumented, and no section of its own:** `addBulk`, `containsBulk`,
`removeBulk`, `clone`, `flipInPlace`, `fromRange`, `fromSortedSlice`, `fromRoaring32`, `toRoaring32`,
`statistics`. `API.md` mentions the type six times in passing and gives it **no section**, while
`OwnedBitmap`, `FrozenBitmap`, and `Frozen64Bitmap` each have one.

**`FrozenBitmap` — 5 undocumented**, new in spec 42: `rank`, `select`, `getIndex`, `minimum`, `maximum`.

**These 19 are a lower bound, and the count is the wrong thing to trust.** It was produced by bare-name
matching, which §2.3 shows is vacuous. The **guard's first run under type-qualified rules is the
authoritative inventory** — expect it to be larger, because a name documented for one type currently
satisfies the search for every type. `41-01` documents what the guard reports, not what this section
lists.

## 2. Deliverables

### 2.1 `API.md`

- **A `Roaring64Bitmap` section**, peer to the existing type sections: construction, the `*Bulk` family,
  `fromRange`/`fromSortedSlice`, the 32↔64 conversion pair, `statistics`, and a pointer to the
  portable-format note at line ~305.
- Document the `RoaringBitmap` and `FrozenBitmap` gaps in their existing topical sections, not an
  appendix.
- **State the stability boundary** (§2.3 pins it mechanically).
- **Replace the single Quick Reference (line ~403) with a per-type Quick Reference** carrying
  **type-qualified** entries — `` `FrozenBitmap.rank` ``, not `rank`. This is what the guard reads, so
  it is a functional requirement, not formatting.

### 2.2 `README.md`

- **No performance claims. At all.** No numbers, ratios, comparisons against the reference C library, or
  "fast"/"faster"/"fastest" framing. A README benchmark claim is read as universal while every number we
  hold is scoped to one harness, three allocators, and two architectures — and it goes stale silently on
  every later commit. Measurements live in `docs/` and the specs, where their scope travels with them.
- **Existing claims that must be removed or neutralized** (verified present):
  - **line 103** — "**Faster** for deserialize and set operations"
  - **line 142** — "`std.heap.c_allocator` was roughly **1.3-1.8x slower** than alternatives"
  - **lines 148–153** — the allocator table's **`Fastest` / `Fast` / `Good`** Speed column, and the
    "Recommended allocators, **fastest to most flexible**" lead-in
  - Keep the allocator guidance itself — it is genuinely useful — but re-cast the column as
    *characteristics/trade-offs* rather than a speed ranking, and keep the pointer to
    `docs/parity-measurement.md` for anyone who wants numbers.
  - *(An earlier draft claimed the README calls the library "high-performance". It does not — that phrase
    is in `src/roaring.zig`'s doc comment. Out of scope here.)*
- **Refresh the bitmap-types table (line ~92)** — it lists only `RoaringBitmap`, `OwnedBitmap`,
  `FrozenBitmap`. **`Roaring64Bitmap` and `Frozen64Bitmap` are missing**, so the table omits the entire
  64-bit half of the library.
- **Refresh Project structure (line ~205)** — stale well beyond `tools/`. Missing at least
  `roaring64.zig`, `frozen64.zig`, the `roaring64_*` test/support files, `range_ops.zig`,
  `array_kernels.zig`, `array_simd.zig`, and under `tools/` the new `check_32_api.zig` and
  `cross_width_fixture.zig`.
- Confirm the **32-bit section** (spec 40) matches the final target list and commands.

### 2.3 `check-docs` — the guard

**Shape:** a repository-only executable (not shipped in `.paths`) that `@embedFile("../API.md")`,
reflects over an explicit type list, and reports **every** missing entry — never stopping at the first.

**Public method = a direct `pub fn` declaration on exactly these five types:** `RoaringBitmap`,
`Roaring64Bitmap`, `OwnedBitmap`, `FrozenBitmap`, `Frozen64Bitmap`.

A Zig 0.16 probe confirmed `@typeInfo(T).@"struct".decls` exposes public declarations and omits private
ones, and that filtering for `.fn` works.

**Matching must be type-qualified.** A bare-name search is **vacuous**: all five of `FrozenBitmap`'s new
`rank`/`select`/`getIndex`/`minimum`/`maximum` are undocumented, yet every one of those names already
appears in `API.md` for another type — so a bare-name guard **passes while the defect is fully present**.
Require the token `` `Type.method` ``. This is the same failure the umbrella's standing question exists to
catch, and it is worth stating plainly that the first draft of this spec proposed exactly that vacuous
check.

**Three manifests, all explicit:**

1. **Stable-type manifest** — the five types above.
2. **Internal-export manifest** — the **10** root-level internal exports in `roaring.zig`
   (`ArrayContainer`, `BitsetContainer`, `RunContainer`, `Container`, `TaggedPtr`, `container_ops`,
   `optimize`, `test_gen`, `roaring64_test_gen`, `roaring64_test_support`), each with a reason string.
   *(An earlier draft said eleven.)*
3. **Root-declaration classification check** — every `pub` declaration in `roaring.zig` must appear in
   manifest 1 or manifest 2. A newly exported type that lands in neither **fails the guard**. Without
   this, adding a public type silently escapes documentation requirements — the 40-01 failure mode
   exactly.

**Accepted limits:** token presence cannot distinguish documentation from a passing mention. That is
fine — it catches *omission*, the failure actually observed. No prose analysis. An **allow-list with
required reason strings** covers deliberate omissions; empty is the goal.

## 3. Behavioural contracts to write down (`API.md` only)

Describe the contract, not the benchmark — no ratios in `API.md` either. *"Frees in descending order,
which some allocators reward"* is usable; *"1.033x on M4"* rots.

- **`bitwiseOrInPlaceConsume`** — **ownership**, the important one. It consumes its right operand;
  document precisely what the caller may and may not do with that operand afterwards. A correctness
  contract, not a performance note.
- **`repairAfterLazyWithOptions`** — opt-in; default `repairAfterLazy` unchanged; changes free order only;
  any benefit is **allocator-dependent**; measure on your own allocator and workload. Claim no speedup.
- **`removeRangeCopy`** — constructs only surviving containers rather than cloning and discarding.
- **`FrozenBitmap.minimum`/`maximum`** (spec 42) — array and run are direct reads; **bitset scans up to
  1,024 words**. Do not write "O(1)". `rank`/`select`/`getIndex` are **O(containers + one container
  probe)** — the frozen descriptor has no prefix sums.

## 4. Out of scope

- Any change to production behaviour, signatures, or the parity board.
- Generated API reference tooling. `API.md` stays hand-written prose; the guard checks coverage, it does
  not author text.
- CI. Same position as spec 40: `check-docs` is a local build step a future CI job can invoke.
- The "high-performance" phrasing in `src/roaring.zig`.

## 5. Acceptance

- `API.md`: `Roaring64Bitmap` section present; every function the guard reports documented; stability
  boundary stated; **per-type Quick Reference with type-qualified entries**.
- `README.md`: the three §2.2 claim sites removed or neutralized; bitmap-types table includes
  `Roaring64Bitmap` and `Frozen64Bitmap`; Project structure refreshed; 32-bit section verified.
  **No parity-status paragraph** — *(an earlier draft required both "no performance claims" and a
  "measured, dated parity status naming lazy-OR". That was a direct contradiction; the no-claims rule
  wins and the parity-status item is removed.)*
- `zig build check-docs` passes, with an empty allow-list or a reasoned entry per exemption, and reports
  **all** misses rather than the first.
- **Negative control 1:** remove one **type-qualified** entry from `API.md`; guard fails and names
  `Type.method`.
- **Negative control 2:** add a `pub fn` to one of the five types without documenting it; guard fails and
  names it. This is the case that actually recurs.
- **Negative control 3:** add a `pub` type to `roaring.zig` listed in neither manifest; guard fails.
- Both controls must test **type-qualified** behaviour — a bare-name control would pass under the
  vacuous scheme and prove nothing.
- Consumer-facing set unchanged; a consumer build against an allowlist-only tree still succeeds.
- No production code changed; all four 64-bit suites remain green.

## 6. Chunking sketch

Pending review of this revision.

- **41-00** — `check-docs` guard, three manifests, stability boundary in `API.md`, per-type Quick
  Reference scaffolding. Lands **first** so its first run produces the authoritative inventory for 41-01.
  All three negative controls here.
- **41-01** — `API.md` content: `Roaring64Bitmap` section, everything the guard reports, §3 contracts.
  Ends with the allow-list empty.
- **41-02** — `README.md`: claim removal, both stale inventories, 32-bit verification.

## 7. Estimate

**M** — mostly writing. The guard is **S**; the `Roaring64Bitmap` section is the bulk; 41-02 needs care
rather than volume.
