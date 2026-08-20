<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 45-00: Chunked-allocation prototype

Toplevel: [45-chunked-payload-arena.md](45-chunked-payload-arena.md).

**No production code.** This chunk answers one question before anything is built: does chunk bump
allocation deliver a useful share of the ordering benefit **without** sorting?

**Two different outcomes, and only one of them ends spec 45:**

- **Candidate NO-GO** — the mechanism was validly measured and does not deliver. **Ends spec 45.**
- **INVALID / STOP** — the ordering control failed, so the probe is a defective instrument. **Blocks
  progression pending a diagnosed, valid rerun.** It says nothing about chunking and must never be
  recorded as a candidate result.

## 1. Where

Extend `src/bench_smp_layout.zig` — **zero rawr code**, per spec 37/43 practice. Model the container
header **locally**; do **not** import `BitsetContainer`. That independence is why spec 37's result was
credible.

**Add the four cells below; do NOT alter the existing ones.** `zero_sorted_*` and `sort_zero_*` keep
their cell names, protocol schema, timed boundaries, and semantics.

*(An earlier draft called the existing cells "defective" and asked for them to be fixed. They are not
defective — their boundaries **deliberately isolate zeroing and sort cost**, which is what spec 37 needed.
They are merely **unsuitable as spec-45 candidate models**, because this spec must time complete
candidate cost including allocation. Changing them would break comparability with spec 37/43 results
still in use.)*

The new cells therefore carry their own boundaries: each times its own allocation, and payload-address
sorting uses `sortUnstable`.

## 2. Four cells

| Cell | What it does |
| --- | --- |
| `scattered_interleaved` | today's structure: allocate and zero **per buffer**, interleaved |
| `batched_unsorted` | allocate all, then zero in **allocation order** |
| `batched_sorted` | allocate all, **sort payload addresses** (`sortUnstable`), zero in **sorted order** |
| `chunked_<size>` | chunk bump-allocate and zero **per buffer**, interleaved — **the candidate** |

The candidate is **interleaved like the baseline**, not batched. Chunking obtains ordering *without*
deferring the zeroing — that is the whole idea, and it is why the candidate cell must not be built on the
batched cells.

**Chunk sizes swept: 256 KiB, 1 MiB, 4 MiB** minimum. At 1 MiB a chunk holds **128 payloads**, so 16,364
payloads occupy **128 chunks / 127 boundary jumps**.

## 3. Timing boundary

Each cell times its **complete** cost: allocation, any metadata, any sort, and zeroing.

- **Inside** timing: release of **temporary** batched/sort metadata — it exists only to build the result.
- **Outside** timing: teardown of **retained** headers, payloads, and the candidate's chunk list — these
  are the result, and the canonical construction row calls `result.deinit()` after the clock stops
  (`bench_croaring.zig:507-512`).

Payload alignment is **64 bytes** in every cell; chunk bases 64-byte aligned with a stride that preserves
it. Include the chunk-list append and its capacity reservation.

## 4. Protocol

The existing fresh-process controller: **≥5 process medians, each 3 warmup / 21 timed**, full ranges
recorded.

**Both hosts, SMP and libc.** libc is **report-only** (toplevel §7.1) and must not influence any decision
in this chunk.

## 5. Gate — two ordered steps

**Step 1 — the ordering control must hold, on M4 SMP:**

```
available = batched_unsorted - batched_sorted
```

- `batched_sorted` **beats** `batched_unsorted`, **non-overlapping ranges**, hence **`available > 0`**.
- If not: **INVALID / STOP — not a candidate NO-GO.** The probe failed to reproduce the ordering
  mechanism specs 37 and 44 established, so it is a defective instrument and **no candidate number from
  that run may be interpreted**. Diagnose, then re-run. Recording it as a candidate NO-GO would blame the
  idea for an instrument failure.

**Step 2 — only if `available > 0`, evaluate the candidate on M4 SMP:**

```
recovered = scattered_interleaved - chunked_<size>
```

- **GO requires `recovered >= 0.50 * available`**, with **non-overlapping ranges** between
  `chunked_<size>` and `scattered_interleaved`.
- **Zen 4: `chunked / scattered_interleaved <= 1.05`** on median.
- Overlapping ranges → **rerun**; still overlapping → **inconclusive → NO-GO**.

The two quantities are **deliberately different comparisons**: `available` is measured in the batched
world because that is the only place a payload-address sort is possible; `recovered` is measured against
the real baseline structure because that is what the candidate must actually beat.

50% is a **screen against wasted effort**, not a prediction — below it the mechanism cannot plausibly
cover the residual once production overheads are added. **Only the toplevel §7 gates can decide the row.**

## 6. Chunk-size selection

Using **SMP medians on both hosts**, choose the **smallest** size within **5%** of the best on both.
**libc must not influence selection.** If no single size satisfies both hosts, **report that and stop** —
host-specific tuning needs explicit sign-off, never a silent default.

## Acceptance

- **Four new cells added; existing cells untouched** — `zero_sorted_*` and `sort_zero_*` keep their
  **cell names, protocol schema, timed boundaries, and semantics** unchanged. *(Not "byte-identical
  output": timings and addresses are nondeterministic, so that could never pass.)* **Zero rawr imports**;
  header modelled locally.
- New cells time their own allocation, and payload-address sorting uses `sortUnstable` — the boundaries
  the *existing* cells deliberately do not have.
- Four cells implemented per §2, chunk sizes swept per §2.
- Timing boundary per §3, including temporary-versus-retained teardown placement and chunk-list cost.
- Protocol per §4 on both hosts; libc recorded but not decisive.
- **Step 1 reported first.** `available` and its ranges stated explicitly before any candidate number is
  interpreted.
- **Step 2 evaluated only if `available > 0`.**
- Chunk-size selection per §6, or an explicit report that no size satisfies both hosts.
- **Verdict stated explicitly as one of GO / candidate NO-GO / INVALID-STOP**, with reasoning:
  - **GO** → proceed to `45-01`;
  - **candidate NO-GO** → **ends spec 45**, no production code written;
  - **INVALID-STOP** → **does not end spec 45**; diagnose the probe and rerun. No candidate number from
    that run may be reported.

## Outcome — candidate NO-GO. Spec 45 ends here. No production code written.

**Step 1 — ordering control PASSED on both hosts**, so this is a valid candidate result, not an
INVALID/STOP. The two-step gate did its job: the probe reproduced the mechanism before any candidate
number was interpreted.

| Host | `scattered_interleaved` | sorted control | best `chunked_<size>` | chunked / scattered |
|---|---:|---:|---:|---:|
| M4 SMP | 6.884 ms | 3.615 ms | 8.922 ms | **1.30x** |
| Zen 4 SMP | 19.080 ms | 16.156 ms | 79.476 ms | **4.17x** |

**Step 2 — `recovered` is NEGATIVE on both hosts:** M4 **−2.038 ms**, Zen 4 **−60.396 ms**. The candidate
is not merely short of the 50% screen; it is **worse than the baseline it had to beat**. No range
analysis is needed at this magnitude.

**Ordering headroom re-confirmed** (M4 6.884 → 3.615 = **−3.269 ms, 47.5%**). The mechanism is real, for
the third independent time. **Chunk allocation simply costs more than the entire ordering benefit.**

### Zen 4 is 3.2x worse relatively — hypothesis, NOT established

Zen 4's relative penalty (4.17x) is **3.2x** M4's (1.30x). A plausible explanation is that freshly
requested large chunks on Linux become new `mmap` regions whose pages fault in on first touch, whereas
the existing path reuses resident pages.

**This is a hypothesis and the probe does not isolate it.** Recording it as mechanism would repeat an
error this campaign has made repeatedly. **No lever may be built on it without a measurement that
establishes it.**

**It does NOT contradict spec 36.** Spec 36 refuted first-touch/page-faults for the **existing** path,
where pages were measured resident and reused (40 faults across ~134 MB, 100% reuse). Freshly allocated
large chunks are a **different regime**. Do not read this as reopening spec 36.

## Estimate

**S** — the probe exists; this adds cells alongside it.
