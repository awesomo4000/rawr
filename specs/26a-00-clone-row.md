<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26a-00: Canonical `clone (dense)` row + manifest 39

First chunk of [clone attribution](26a-clone-attribution.md). **Mechanical harness work, no
production change.** Puts `clone` on the canonical board and updates the count checks.

## Deliverables

- **`clone (dense)` canonical row**: rawr `clone` (**SMP + libc** — allocating op) vs
  `roaring_bitmap_copy`, on the same wide-dense corpus as `remove-range` so the pair reads
  together. Manifest entry per the schema (corpus/seed, matched ops, allocating class,
  variants, boundaries, oracle, unit). Note in the manifest that CRoaring `copy` runs with COW
  disabled (our build), so both sides deep-copy.
- **Validation (outside timing):** the clone's portable bytes are **identical to its source**
  (rawr), plus **CRoaring set parity** via the established oracle path.
- **Manifest 38 → 39:** update all **active** executable assertions, scripts, and current
  documentation (`--list`, `validateManifest`, runner checks, current tables' prose).
  Completed specs and historical results stay historical.
- **Local test gates:** `zig build test`; `zig build difftest`; a canonical-runner smoke
  showing the new row measured and validated; `ReleaseSafe` / `ReleaseFast` green.

## Acceptance / checklist

- [ ] `clone (dense)` row live: SMP + libc vs CRoaring, validated outside timing
- [ ] Byte-identity-vs-source + CRoaring parity green
- [ ] `--list` = 39; `validateManifest` and all active count checks updated; historical refs
      untouched
- [ ] No production/library change; test / difftest / ReleaseSafe / ReleaseFast green
- [ ] No existing canonical row perturbed (> 5%) by the addition
