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
    *characteristics/trade-offs* rather than a speed ranking.
  - **Remove the relative link to `docs/parity-measurement.md` (line 144).** `docs/` is **not** in
    `.paths`, so for a consumer who received only the package the link is **broken** — it resolves to
    nothing. Removal is cleanest and consistent with the no-claims direction; an absolute GitHub URL or
    adding the file to `.paths` would also work but both re-introduce measurements into the shipped set.
    *(An earlier draft required retaining this link while also asserting consumers receive only
    README/API/license — those cannot both hold.)*
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

**Matching must also be region-scoped, for the same reason one step later.** Searching all of `API.md`
re-opens the hole once 41-01 adds prose: deleting a Quick Reference entry would still pass because
`` `Type.method` `` appears in the prose elsewhere. The Quick Reference would then be unguarded exactly
when it starts to matter.

- **Delimit the Quick Reference** with explicit markers (e.g. HTML comments
  `<!-- check-docs:begin -->` / `<!-- check-docs:end -->`) and **search only between them**.
- Prose mentions outside the region are then irrelevant to the guard — which is correct, since the
  region is what the guard claims to enforce.
- **Negative control 1 must delete the token from inside the guarded region**, not merely from the
  document; a control that removes a prose mention tests nothing.

*(Rejected alternative: make the guard API-wide and treat Quick Reference completeness as manual
acceptance. That trades a mechanical check for a human one in the exact place the mechanical check was
introduced.)*

**Manifests — note the two scopes are different sets, which is what an earlier draft got wrong:**

1. **Stable root-export manifest** — the five types above **plus `ValidateError`**
   (`roaring.zig:17`, `pub const ValidateError = RoaringBitmap.ValidateError`). It is a stable public
   export but **not** a struct with methods. In the previous draft it was classified nowhere and would
   have tripped check 4 immediately.
2. **Method-reflection scope — DERIVED from manifest 1, never written by hand.** Filter manifest 1 to
   struct types at comptime; `ValidateError` drops out because it is not a struct. Two hand-maintained
   lists would drift, and the drift is silent in the dangerous direction: a future public struct added to
   the stable manifest but forgotten in the reflection scope would escape method documentation
   **permanently**, with the guard green. Deriving makes that unrepresentable rather than merely
   detectable.

   *(A mechanical cross-check — verify every stable struct export also appears in the reflection scope —
   is the acceptable fallback if derivation proves awkward in Zig 0.16. Prefer derivation: it removes the
   failure mode instead of testing for it.)*
3. **Internal-export manifest** — the **10** root-level internal exports in `roaring.zig`
   (`ArrayContainer`, `BitsetContainer`, `RunContainer`, `Container`, `TaggedPtr`, `container_ops`,
   `optimize`, `test_gen`, `roaring64_test_gen`, `roaring64_test_support`), each with a reason string.
   *(An earlier draft said eleven.)*
4. **Root-declaration classification check** — every `pub` declaration in `roaring.zig` must appear in
   manifest 1 or manifest 3. A newly exported type that lands in neither **fails the guard**. Without
   this, adding a public type silently escapes documentation requirements — the 40-01 failure mode
   exactly.

**The build step must RUN the checker.** Wire it with **`b.addRunArtifact`** — a step that merely
compiles the executable would pass without ever reading `API.md`, which is a guard that cannot fail for
the reason it exists. (`check-32` is compile-only *by design* because compilation **is** its test;
`check-docs` is the opposite case.)

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
- **Negative control 1:** remove one **type-qualified** entry **from inside the delimited Quick
  Reference region**; guard fails and names `Type.method`. Removing a prose mention outside the region
  must **not** be what the control tests.
- **Negative control 2:** add a `pub fn` to one of the five types without documenting it; guard fails and
  names it. This is the case that actually recurs.
- **Negative control 3:** add a `pub` declaration to `roaring.zig` listed in neither the stable
  root-export manifest nor the internal-export manifest; guard fails. *(This one is a classification
  check — it is **not** type-qualified, unlike controls 1 and 2.)*
- **Controls 1 and 2 must test type-qualified behaviour** — a bare-name control would pass under the
  vacuous scheme and prove nothing.
- **Consumer-facing set changes only by adding `img/rawr.png`** (per `41-02` §1.1 — the packaged README
  embeds the logo and currently renders it broken). No other `.paths` change. **The allowlist-only
  consumer check stays green**, verified by a pinned reproducible procedure — not a prior one-off.
  Add it as a repo helper (script or build step) so acceptance can be re-run:
  1. parse `.paths` from `build.zig.zon`;
  2. copy exactly those files into a scratch tree, preserving structure;
  3. build a throwaway consumer with a `.path` dependency on that tree, importing `rawr` and calling a
     handful of public methods;
  4. require a successful build **and** run.

  This is what proved the current allowlist self-contained (the shipped set builds even though
  `build.zig` references `tools/`, `vendor/`, and `src/bench_*.zig`, because a consumer never reaches
  those steps). Note step 3 needs a package fingerprint; take the value Zig reports on first failure.
- No production code changed; **all four suites green — `test`, `difftest`, `test64`, `difftest64`** —
  plus `ReleaseSafe` and `ReleaseFast`.

## 6. Chunking — cleared, three chunks

- **[41-00](41-00-check-docs-guard.md)** — guard, manifests, guarded Quick Reference, consumer helper.
- **[41-01](41-01-api-md-content.md)** — `API.md` prose, `Roaring64Bitmap` section, contracts, Allocator
  Guide neutralization.
- **[41-02](41-02-readme-refresh.md)** — `README.md` claims and stale inventories. Independent; may land
  in any order.

Each chunk stands alone with its own pass/fail; the rationale that shaped them (green-on-landing,
guard-first inventory, helper ownership) now lives in the chunks themselves.

## 6.1 Outcome — CLOSED

All three chunks implemented and accepted; each carries its own verification record. Landed upstream as
`0b8e67d` (41-00) and `705fd82` (41-01 + 41-02), independently re-checked afterwards with no issues.

End state: `check-docs` guards **168** direct public methods, `check-package` proves the shipped set
(**33** files, logo included) builds and runs from the allowlist alone, no performance claim remains in
either shipped document, and both 64-bit types are documented for the first time.

**Highest-value finding was not mechanical.** The manual prose audit caught that
`Roaring64Bitmap.fromRange` is **half-open** while `addRange` beside it is inclusive both ends — so
`API.md`'s blanket "Ranges Are Inclusive" heading had been wrong since that constructor shipped. No token
check could ever have found it: `check-docs` verifies a name is present, never that the surrounding
sentence is true.

**Residual limitation, intentional and recorded** (`41-00` §3.1): the guard covers direct `pub fn` only,
not nested public types/constants (`BulkContext`, `Statistics`, `RepairAfterLazyOptions`, the
`ValidateError` pair). Changes there still need the manual audit. This is stated in the guard's own output
and header so a passing check is never read as "the complete public API is documented" — the same
overstatement risk that spec 40-01 established when the `check-32` probe surface turned out to *be* the
guard boundary.

## 7. Estimate

**M** — mostly writing. The guard is **S**; the `Roaring64Bitmap` section is the bulk; 41-02 needs care
rather than volume.
