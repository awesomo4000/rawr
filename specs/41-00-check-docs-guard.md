<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 41-00: `check-docs` guard, manifests, and the guarded Quick Reference

Toplevel: [41-documentation-parity.md](41-documentation-parity.md).

**Lands green.** No **topical API prose** is written here (the stability boundary in §2 is prose, and
belongs to this chunk) — but the guarded region must be **complete** on this commit,
so `check-docs` passes the moment it exists. No production code changes.

## 1. The guard

A **repository-only** executable (not in `.paths`) that `@embedFile`s `API.md`, reflects over the public
types, and reports **every** missing entry — never stopping at the first.

**Wire it with `b.addRunArtifact`.** A step that only compiles the executable would pass without ever
reading `API.md` — a guard that cannot fail for the reason it exists. (`check-32` is compile-only *by
design*, because compilation **is** its test. `check-docs` is the opposite case.)

### 1.1 Manifests

1. **Stable root-export manifest** — `RoaringBitmap`, `Roaring64Bitmap`, `OwnedBitmap`, `FrozenBitmap`,
   `Frozen64Bitmap`, **plus `ValidateError`** (`roaring.zig:17`). `ValidateError` is a stable public
   export but not a struct with methods.
2. **Method-reflection scope — DERIVED from manifest 1, never hand-written.** Filter to struct types at
   comptime; `ValidateError` drops out. Two hand-maintained lists drift silently in the dangerous
   direction: a future public struct added to manifest 1 but forgotten in the reflection scope would
   escape method documentation **permanently, with the guard green**. Derivation makes that
   unrepresentable rather than merely detectable. *(Fallback if derivation proves awkward on Zig 0.16:
   mechanically verify every stable struct export appears in the reflection scope. Prefer derivation.)*
3. **Internal-export manifest** — the **10** internal exports in `roaring.zig` (`ArrayContainer`,
   `BitsetContainer`, `RunContainer`, `Container`, `TaggedPtr`, `container_ops`, `optimize`, `test_gen`,
   `roaring64_test_gen`, `roaring64_test_support`), each with a **reason string**.
4. **Root-declaration classification check** — every `pub` declaration in `roaring.zig` must appear in
   manifest 1 or manifest 3. One landing in neither **fails the guard**.

**Public method** = a direct `pub fn` declaration on a type in the reflection scope.
`@typeInfo(T).@"struct".decls` exposes public declarations and omits private ones; filter for `.fn`.
(Confirmed by probe on Zig 0.16.)

### 1.2 Matching — type-qualified AND region-scoped

Both constraints exist because each alone is vacuous:

- **Type-qualified.** Require the token `` `Type.method` ``. A bare-name search passes with the defect
  fully present: `FrozenBitmap`'s five new positional methods are undocumented, yet every one of those
  names already appears in `API.md` for another type.
- **Region-scoped.** Search **only** between explicit delimiters around the Quick Reference (e.g.
  `<!-- check-docs:begin -->` / `<!-- check-docs:end -->`). Searching the whole document re-opens the
  hole as soon as `41-01` adds prose — deleting a Quick Reference entry would still pass on a prose
  mention elsewhere, leaving the region unguarded exactly when it starts to matter.

**Accepted limit:** token presence cannot distinguish documentation from a passing mention. That is fine
— it catches *omission*, the failure actually observed. No prose analysis.

**Allow-list** with required reason strings for deliberate omissions. Empty is the goal.

## 2. `API.md` changes in this chunk

- **Complete, populated, type-qualified per-type Quick Reference inside the delimiters** — every
  `Type.method` for all five reflected types. **Complete, not scaffolded:** a scaffold means
  `check-docs` fails on the commit that introduces it. *(Rejected alternative: temporary allow-list
  entries cleared by 41-01. Populating the table is barely more work, keeps every chunk green, and
  guards the full inventory immediately.)*
- **State the stability boundary** — the public API versus the 10 internal exports that "may change
  without notice". The guard needs this definition; a reader of `roaring.zig` currently gets no guidance.

**No topical API prose.** Topical write-ups, the `Roaring64Bitmap` section, and the behavioural contracts
are 41-01. The stability boundary above is the one piece of prose this chunk owns — the guard needs that
definition to exist.

## 3. Record the **method inventory** — and only that

Record the guard's first-run output as the **complete reflected method inventory for Quick Reference
coverage**. Since `API.md` currently contains **no** type-qualified tokens, that first run reports
essentially **every** method across the five types — on the order of 150 entries. It is the work list for
**this chunk's table**, and nothing else.

**It is explicitly NOT a prose work list.** An earlier draft called it "the authoritative inventory" and
had `41-01` write prose for every entry — which would have meant ~150 write-ups instead of the ~19 real
prose gaps, from a guard that **cannot verify prose by design**. `41-01` maintains its own focused
prose-gap inventory; see that chunk.

## 3.1 Accepted limit — reflection covers `pub fn` only

`@typeInfo(...).decls` filtered to `.fn` does **not** cover public **nested types and constants** that
appear in signatures — verified present: `Roaring64Bitmap.BulkContext` (`roaring64.zig:54`),
`Roaring64Bitmap.Statistics` (`roaring64.zig:65`), `Roaring64Bitmap.ValidateError` (`roaring64.zig:49`),
`RoaringBitmap.RepairAfterLazyOptions` (`bitmap.zig:1710`), `RoaringBitmap.ValidateError`
(`bitmap.zig:44`).

This is an **accepted limit, stated so it is not mistaken for coverage**: `check-docs` guards *methods*,
not the *complete public API*. Say so in the guard's own output or header, so a future reader does not
infer a guarantee the tool never made. Extending reflection to nested declarations is a possible later
change, not part of this chunk.

## 4. Package-consumer helper

Add a repo helper (script or build step) so the allowlist claim is reproducible rather than a one-off:

1. parse `.paths` from `build.zig.zon`;
2. copy exactly those files into a scratch tree, preserving structure;
3. build a throwaway consumer with a `.path` dependency on that tree, importing `rawr` and calling
   several public methods;
4. require a successful build **and** run.

This is what established that the shipped set is self-contained — it builds even though `build.zig`
references `tools/`, `vendor/`, and `src/bench_*.zig`, because a consumer never reaches those steps.
Step 3 needs a package fingerprint; take the value Zig reports on first failure.

## Acceptance

- `zig build check-docs` **runs** (via `addRunArtifact`) and **passes** on this commit, allow-list empty
  or every entry reasoned.
- Reports **all** misses, not the first.
- Reflection scope **derived** from the stable root manifest; `ValidateError` and all 10 internal
  exports classified.
- Quick Reference complete inside its delimiters; stability boundary documented.
- **Method inventory** (first-run output) recorded, and labelled as Quick-Reference coverage — **not** a
  prose work list.
- §3.1 limit stated in the guard's output/header: it guards methods, not the complete public API.
- **Negative control 1:** delete one `` `Type.method` `` **from inside the delimited region**; guard
  fails and names it. Deleting a prose mention outside the region must **not** be what this control
  tests.
- **Negative control 2:** add a `pub fn` to a reflected type without a Quick Reference entry; guard fails
  and names it. This is the case that actually recurs.
- **Negative control 3:** add a `pub` declaration to `roaring.zig` in neither manifest 1 nor manifest 3;
  guard fails. *(Classification check — **not** type-qualified, unlike 1 and 2.)*
- Package-consumer helper present; allowlist-only consumer builds **and** runs.
- No production code changed; **all four suites green — `test`, `difftest`, `test64`, `difftest64`** —
  plus `ReleaseSafe` and `ReleaseFast`. `check-32` still passes.

## Estimate

**S/M** — the checker is small; populating the complete Quick Reference is the bulk.
