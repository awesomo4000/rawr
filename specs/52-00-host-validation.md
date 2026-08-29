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
  confirm the reference did not change branch between the two environments.
- **Binary identity, stated rather than assumed.** Both environments are Linux x86_64, so **prefer
  building the worker once, copying it across, and recording its SHA-256**. If differing glibc or
  toolchain versions prevent that, build separately, **record both SHA-256 values, and state that the
  binaries differ** — in which case the comparison is "same source and build configuration", not "same
  binary", and the report must say so. Two separate builds are not proven identical.

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

**Classify each row by the complete verdict pair.** An earlier draft omitted the *closes both* case,
which is the one that would retire a row from the campaign entirely:

| WSL2 | native | classification |
| --- | --- | --- |
| survives | survives | **real gap** — carries into Stage 1 |
| survives | closes | **environment-conditioned** — see §D |
| closes | survives | **environment-conditioned in the other direction** — report, do not explain away |
| closes | closes | **not a gap in this session** — reconcile against the 08/28 board rather than assuming it went away |
| any | inconclusive | **inconclusive** — no classification, rerun once per §A.4 |

A row that closes on native is **environment-conditioned** and nothing stronger — see §D.

### A.5 One row that needs its own line

`bitwiseAnd (array balanced)` reads **2.688x under SMP and 9.138x under libc**. Report it in both
environments under both allocators and state whether the libc behaviour reproduces. It is the one failing
row where the allocator argument runs backwards, and nothing in Part A should smooth that over.

## Part B — reconcile the `serialize` movement

### B.0 The anchor is rawr's own time, not a ratio

An earlier draft anchored this on "spec 28 recorded 0.81x, it now reads 2.771x". **That anchor is not
sound.** Spec 28's record does state `0.81x`, but the Zen 4 production artifact it rests on,
`misc/parity-20260728-204017-summary.txt`, **is not retained** — it is absent from `misc/`. A separate
`0.802x` figure in that spec comes from the **factorial diagnostic harness**, not the production board.
**The historical production parity ratio is therefore unknown**, and this chunk may not treat it as given.

**What is durable is rawr's own absolute production SMP time:**

| | rawr SMP `serialize` |
| --- | ---: |
| spec 28 outcome, commit `2ba714a` | **0.824 ms** |
| clean-`b3ab49f` board, 08/28 | **2.004 ms** |

**2.43x slower on the same operation**, and this statement does not depend on the missing artifact or on
what CRoaring was doing in either run. **Part B investigates that movement.** Any parity ratio recovered
along the way is reported as a secondary observation.

### B.1 Two independent contrasts, both of which may move

Three causes are live — environment drift, a binary-level change from unrelated edits (`bitmap.zig` has
been modified many times since `2ba714a`, and spec 28's own finding was that whole-binary layout moves
untouched rows), and a harness or toolchain change.

**They are not mutually exclusive**, and an earlier draft's three-outcome table wrongly forced a single
answer. The experiment gives two contrasts:

| contrast | what it isolates |
| --- | --- |
| historical `2ba714a` (0.824 ms) **vs** current-session `2ba714a` | **session- and environment-conditioned** movement, valid only if B.2 passes |
| current-session `2ba714a` **vs** current-session `HEAD` | **commit- and binary-conditioned** movement |

**Report absolute median and full range for every cell of both contrasts**, SMP and libc, and **allow both
findings to be true at once**. The environment may have moved *and* later binary layout may contribute.
No approximate matching against remembered figures, and no forced single cause.

Both contrasts run on existing hardware.

### B.2 Comparability audit — a prerequisite, not a follow-up

**Run this before B.1 is interpreted.** Comparing a row across commits assumes the row means the same
thing; if it does not, B.1's numbers are uninterpretable rather than merely noisy. Audit between
`2ba714a` and `HEAD`:

- the `serialize` **manifest entry**, and the **setup / timing / teardown boundaries**;
- the **corpus initialisation** called and the **benchmark body**;
- the **controller protocol** and **`bench_time.zig`**;
- the **Zig version and build flags** — a toolchain change is neither environment drift nor our code, and
  would be missed by both contrasts;
- the **vendored CRoaring revision and its C flags**.

**State the result of every item, whether or not it found a change.** Movement in CRoaring or in the
harness would otherwise be mislabelled as a binary-layout effect — which is exactly the error spec 52 §2.1
already had to correct once.

### B.3 If the binary is implicated

Do not bisect speculatively. Report the finding and note that spec 28's serialize record is what would
need revisiting — including that its production artifact is missing. **Identifying the responsible change
is not this chunk's job.**

### B.4 Outcome: complete 08/29/2026

Part B ran in one WSL2 session with Zig 0.16.0, `ReleaseFast`, `-Dcpu=native`, AVX-512 disabled, and five
fresh worker processes per tuple. The retained artifact is
`misc/serialize-history-20260829-103843-summary.txt`. The artifact remains gitignored; the reproducible
controller is `scripts/run-serialize-history.sh`.

The comparability audit passed:

| item | result |
| --- | --- |
| manifest and boundaries | the runtime `serialize` manifest rows are byte-identical, including seed, corpus, setup, timed work, teardown, and validation oracle |
| corpus and benchmark body | both commits call the same deterministic one-million-value corpus setup and the same rawr and CRoaring serialize bodies |
| serialization implementation | `src/serialize.zig` is unchanged after `2ba714a` |
| controller and clock | `bench_time.zig`, the 3-warmup/21-timed process protocol, and five-process aggregation are unchanged; the controller's only relevant diff is the expected total manifest count, 39 to 42 |
| toolchain and flags | both current-session workers use Zig 0.16.0, `ReleaseFast`, `-Dcpu=native`, and the same CRoaring C flags (`-std=c11 -O3 -DNDEBUG`, AVX-512 disabled) |
| vendored reference | `vendor/roaring.c` is unchanged; wrapper additions do not change the portable serialize declaration or implementation |
| binary identity | the worker SHA-256 values differ, as expected from unrelated additions to the benchmark binary; the row-level audit above is therefore required rather than assuming binary identity |

The current-session measurements are:

| commit | implementation | allocator | median ms [full range] |
| --- | --- | --- | ---: |
| `2ba714a` | rawr | SMP | **2.032 [2.011, 2.324]** |
| `2ba714a` | rawr | libc | 0.575 [0.566, 0.652] |
| `2ba714a` | CRoaring | libc | 0.838 [0.768, 3.105] |
| `7d295e0` | rawr | SMP | **2.094 [2.063, 2.261]** |
| `7d295e0` | rawr | libc | 0.602 [0.587, 0.725] |
| `7d295e0` | CRoaring | libc | 0.852 [0.832, 0.884] |

The historical `0.824 ms` result is below the complete current-session `2ba714a` range. The old commit is
2.466x slower in this session, so the historical movement is **session- and environment-conditioned**.
The current-session `2ba714a` and `7d295e0` ranges overlap for rawr/SMP, rawr/libc, and CRoaring/libc.
Later source and whole-binary changes therefore show **no resolved movement in this run**. Both results
can be true: the session/environment contrast moved, while the same-session commit/binary contrast did
not. The missing historical production artifact was not recovered, so its parity ratio remains unknown.

Part A remains open until native Linux is available on the same physical Zen 4 machine. The canonical
runner now records source state, worker SHA-256, OS and libc, Zig version, and CRoaring runtime support via
a separate executable so the dispatch report cannot change the parity worker's code layout.

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
- **Binary identity resolved per §A.2**: one worker copied across with its SHA-256 recorded, or both
  SHA-256 values recorded and the report stating the binaries differ.
- **Absolute medians and ranges for both sides in both environments**, with any movement in CRoaring's own
  times called out explicitly. **No difference of ratios computed anywhere.**
- §A.4 verdict applied to all **15 unique rows** per environment per existing allocator variant, each row
  placed in the **complete** classification matrix including *closes both*.
- `bitwiseAnd (array balanced)` reported separately per §A.5, including whether the libc behaviour
  reproduces.
- **Part B anchored on rawr absolute time per §B.0**, not on the unbacked historical ratio. If the missing
  `parity-20260728-204017` artifact is recovered, say so; otherwise state that the historical parity ratio
  remains unknown.
- **§B.2 comparability audit run before B.1 is interpreted**, with **every listed item's result stated**
  including the ones that found no change.
- **Both §B.1 contrasts reported** with absolute median and range per cell, SMP and libc, and **both
  findings permitted to be true** — no single forced cause, no approximate matching to remembered figures.
- Report contains **no claim that WSL2 specifically caused anything**, and does not use "host artifact".
- No production change; all four suites plus `check-32`, `check-docs`, `check-package` green.

## Estimate

**M** — mostly running existing harnesses under a pinned protocol. Part B's two-commit build and the
row-definition diff are the fiddly parts; provisioning the native host is outside the estimate.
