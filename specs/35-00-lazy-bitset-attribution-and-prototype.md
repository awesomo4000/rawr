<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 35-00: Lazy-OR attribution + headerless prototype

Toplevel: [35-headerless-transient-lazy-bitsets.md](35-headerless-transient-lazy-bitsets.md) (E3).
Attribute the **lazyOr construction 1.663x** gap, prototype the headerless transient accumulator
benchmark-only, and decide the stop-gate. **No production change; no container-union change.**

## Extend the EXISTING attribution harness

Extend **`src/bench_lazy_or_attribution.zig`** and **`tools/croaring_lazy_attribution.h`** — do
**not** build a second attribution system. **Reconcile against the recorded numbers** from the
earlier arena experiment: **130,994 allocations**, roughly **32,726 transient-container allocation
calls** (`docs/parity-measurement.md`).

## Pinned baseline + corpus counts (assert before timing)

Post-`d7d357b` canonical M4: lazyOr construction **5.746 ms vs 3.456 ms = 1.663x**; lazyOr+repair
**14.612 vs 12.403 = 1.178x**; repair-alone **8.315 vs 7.928 = 1.049x**. Capture fresh **Zen 4**
references here (the no-regress gate needs them).

Corpus = canonical sparse (`DefaultPrng.init(54321)`, 500 k `int(u32)`, sorted+deduped to
`sparse_len`; `a = sparse_values[0..half]`, `b = sparse_values[half/2..]`, `half = sparse_len/2`).
**Pin and assert the actual counts** (do not trust the toplevel's estimate): matched keys, transient
bitsets created, **demoted vs surviving**, and per-key cardinalities.

## Attribution (three-way)

1. **Transient-bitset lifecycle** — header create, 8 KB words alloc, 8 KB zero-fill, accumulate,
   repair scan, demote copy, header free, words free — per matched key, summed.
2. **Unmatched clone traffic** — measured for share only (closed levers: specs 17/18, and spec 32's
   Array-header NO-GO).
3. **Top-level assembly** — `initCapacity(min(a.size + b.size, 65536))` etc.

## CRoaring materialization assertion (not an open question)

`vendor/roaring.c` `roaring_bitmap_lazy_or`: with `bitsetconversion` true and neither matched
container a bitset, it calls **`container_to_bitset`** then `container_lazy_ior` — **no thresholding
of tiny pairs**. Assert **both sides materialize per matched key the same way** (guards against a
mis-set flag or future divergence). **No semantics fork, no owner contract decision.**

## Headerless prototype (benchmark-only)

Prototype allocating **only the aligned 8 KB words** (zeroed — required for OR **and** XOR), with
the transient state tracked locally in the diagnostic. **No `ContainerType` / `Container` change in
this chunk** — the rename and payload land in `35-01`.

### Eliminated vs deferred accounting (report all five; gate is the COMBINED row)

1. headers **permanently eliminated** (demotion),
2. headers **deferred** to repair (survivors),
3. **construction-only** allocation reduction,
4. **full construction + repair** allocation reduction,
5. **repair regression** from allocating surviving headers there.

### Dense survivor control — one-sided gate

Survivor-heavy corpus (matched unions staying > 4096, so headers **defer** rather than eliminate).
All three rows — **construction, repair-only, combined** — on **both hosts**, under
**`candidate / baseline ≤ 1.05`** (one-sided: dense construction is **expected to improve** as its
header moves to repair; improvements always allowed), **plus process-range analysis**.

## Stop-gate arithmetic (do not double-count)

The recorded **~32,726** transient calls are ~**one header + one words** call for each of
**~16,363** matched keys. **E3 removes ONLY the ~16,363 header calls (and matching frees); the
~16,363 words allocations REMAIN.** Project the gate on **~16,363**, **never ~32,726**.

Bar: **~16,363 eliminated header calls × measured per-call SMP cost + matching frees must project
the combined construction+repair row to ≤ 1.10x** (or an equivalent measured focused-time
improvement). If it cannot — e.g. the 8 KB zero-fill and repair scan dominate, not the 16 B
create — **stop before `35-01`** and report what *does* dominate.

## Measurement discipline

- Canonical protocol: **3 warmup / 21 timed, five fresh-process medians + full range**, **M4 and
  Zen 4**, one CRoaring reference per host. E3-owned diagnostic module; shared `build.zig` / runner /
  docs edits are implementer-owned.
- **Accounting per cell:** allocations, frees, requested bytes, effective SMP-class bytes, teardown —
  container instances ≠ allocator calls.
- **Construction and repair measured separately AND combined** (gate = combined; split = attribution).
- Explain in the writeup **why this is not the spec-17 arena**: words stay **individually
  SMP-allocated with unchanged lifetime**; only the 16 B header alloc/free disappears.

## Acceptance

- Existing harness extended and **reconciled** to the recorded 130,994 / ~32,726 figures; corpus
  counts pinned (matched keys, transient bitsets, demote/survive split); three-way attribution
  reported; CRoaring materialization assertion green; fresh Zen 4 baselines captured.
- Headerless prototype measured with the five-figure eliminated/deferred accounting, both hosts;
  **dense survivor control** run under the one-sided gate; **stop-gate arithmetic explicit and
  computed on ~16,363**.
- **No production change; no container-union change.** Decision recorded: proceed to `35-01` only if
  the stop-gate projects the **combined** row to ≤ 1.10x.
- `zig build test`; `zig build difftest` green; diagnostic section of `docs/parity-measurement.md`
  updated.
