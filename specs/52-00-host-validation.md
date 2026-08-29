<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 52-00: Stage 0 — is the x86_64 host representative

Toplevel: [52-x86-64-parity.md](52-x86-64-parity.md).

**No production change. No kernel. No fix.** This chunk measures and reconciles, and it decides whether
the rest of spec 52 happens and what it is about.

Two deliverables. **Part B needs no new hardware and can start immediately** while the native host is
being provisioned.

## Part A — the same board in two environments

### A.1 Both environments measured fresh

**Run the canonical board on native Linux x86_64 and on WSL2, on the same physical machine, from the same
commit, in the same working session.**

**Do not reuse the 08/28 WSL2 boards.** The hypothesis under test is that this environment's behaviour has
drifted; a stale baseline assumes the answer. Every number in Part A comes from this session.

### A.2 Pinned configuration

Identical on both sides, and recorded in the artifact:

- **source commit**, **Zig version**, `ReleaseFast`, `-Dcpu=native`;
- spec 22 process protocol: fresh process per cell, warmup then timed, **≥5 process medians with full
  ranges**;
- **every existing manifest variant**, with SMP/libc compared **where both exist**. The board does not
  pair them universally: `lazy-or-repair-descending` is SMP-only, the default/non-allocating and arena
  rows have no pair, and the pre-adoption rows reference **another row's** CRoaring tuple so their libc
  ratio is not like-for-like. Report what exists; do not synthesise a missing pair.
- **CRoaring dispatch**: report `croaring_hardware_support()` once per environment and map each row to its
  source-gated expected path. **Label this as a source mapping, not branch observation** — it exists to
  confirm the reference did not change branch between two environments running the same binary.

### A.3 Report absolute times, not only ratios

**Both sides' medians and full ranges in both environments.** Each ratio is valid because it is computed
within one run, but a ratio moving between environments does not say which side moved. Spec 39 recorded
rawr gaining 4.1% while the CRoaring reference lost 7.8% in the same pair of runs. **If CRoaring's
absolute times differ between the two environments, that is a finding in itself** and it changes how every
row is read.

**Do not compute a difference of ratios.** Apply the verdict rule independently in each environment and
report the pair.

### A.4 Verdict rule, per row, per environment, per allocator variant

| condition | verdict |
| --- | --- |
| `rawr_min / croaring_max > 1.10` | **gap survives** |
| `rawr_max / croaring_min <= 1.10` | **gap closes** |
| otherwise | **inconclusive** — rerun once, then report as inconclusive |

Applied to the **15 unique rows**: the 13 of toplevel §3 plus the `bitwiseAnd (sparse)` and
`bitwiseAnd (array skewed)` controls.

**Classify each row** by the pair of verdicts: *survives both*, *closes on native*, *survives native only*,
or *inconclusive*. A row that closes on native is **environment-conditioned** — see §D.

### A.5 One row that needs its own line

`bitwiseAnd (array balanced)` reads **2.688x under SMP and 9.138x under libc**. Report it in both
environments under both allocators and state whether the libc behaviour reproduces. It is the one failing
row where the allocator argument runs backwards, and nothing in Part A should smooth that over.

## Part B — reconcile the `serialize` SMP discrepancy

Spec 28 measured **Zen 4 SMP `serialize` at 0.81x** (`1.035 → 0.824 ms`, commit `2ba714a`). It now reads
**2.771x**. That is a 3.4x swing on a row whose serialization code has not changed.

**Three candidate causes, and the design must separate them:**

| candidate | why it is live |
| --- | --- |
| **environment drift** | the host is WSL2 and Part A exists because it may have moved |
| **binary-level change from unrelated edits** | `bitmap.zig` has been modified many times since `2ba714a`; spec 28's own finding was that whole-binary layout moves untouched rows |
| **harness or row-definition change** | the row must mean the same thing at both commits, or the comparison is void |

### B.1 The decisive experiment

**Build and run both commits in the current WSL2 environment, in one session**: `2ba714a` and current
`HEAD`, `serialize` row only, SMP and libc, full protocol.

| result | conclusion |
| --- | --- |
| `2ba714a` reproduces **~0.81x** today | the environment is exonerated; **the binary changed** |
| both commits read **~2.77x** today | **the environment moved** since spec 28 |
| neither matches either figure | the harness or row definition changed — go to B.2 before concluding |

This runs entirely on existing hardware.

### B.2 Verify the row means the same thing

Before comparing anything, **diff the `serialize` row's manifest entry and worker code between `2ba714a`
and `HEAD`**. If the timing boundary, buffer handling, or input construction changed, the two figures are
not comparable and no conclusion about drift or regression may be drawn from them. **State this check's
result either way** — a silent pass here would make B.1's table meaningless.

### B.3 If the binary changed

Do not bisect speculatively. Report the finding, and note that spec 28's serialize work is the record that
would need revisiting. **Identifying the responsible change is not this chunk's job.**

## D. What this chunk cannot conclude

**Native boot does not isolate one variable.** It preserves the CPU and changes the kernel, libc and
userland, the scheduler, page management, and possibly power and frequency configuration.

A row that closes on native Linux is **environment-conditioned**, and that is the entire claim. It is
**not** evidence that WSL2 specifically is the cause, and the words "host artifact" may not appear in the
report. Attributing it further would need experiments this chunk does not run.

## Acceptance

- **Part A**: canonical board on native Linux x86_64 and WSL2, same machine, same commit, **both measured
  in this session**, no reuse of the 08/28 boards.
- §A.2 configuration pinned and recorded, including Zig version and commit; manifest variants reported as
  they exist with no synthesised pairs; CRoaring dispatch reported and **labelled a source mapping**.
- **Absolute medians and ranges for both sides in both environments**, with any movement in CRoaring's own
  times called out explicitly. **No difference of ratios computed anywhere.**
- §A.4 verdict applied to all **15 unique rows** per environment per existing allocator variant, each row
  classified by its verdict pair.
- `bitwiseAnd (array balanced)` reported separately per §A.5, including whether the libc behaviour
  reproduces.
- **Part B**: `2ba714a` and `HEAD` both built and run in the current environment, `serialize` row, SMP and
  libc, with §B.1's table resolved to one of its three outcomes.
- **§B.2 row-definition diff performed and its result stated**, whether or not it found a change.
- Report contains **no claim that WSL2 specifically caused anything**, and does not use "host artifact".
- No production change; all four suites plus `check-32`, `check-docs`, `check-package` green.

## Estimate

**M** — mostly running existing harnesses under a pinned protocol. Part B's two-commit build and the
row-definition diff are the fiddly parts; provisioning the native host is outside the estimate.
