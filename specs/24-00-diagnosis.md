<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 24-00: Select diagnosis — call-boundary matrix

First chunk of [select parity](24-select-parity.md). **Diagnosis only, benchmark-only.**
Deliverable: the **matrix-selected public-API** select ratio (fair call boundary, FFI removed)
on both hosts, and where rawr's select cost lives — the input that decides `24-01`.

## Call-boundary matrix (four paths, one per fresh process)

`benchCRoaringSelectDense` currently calls `roaring_bitmap_select` per query (~1M Zig→C FFI
calls); an in-C loop removes the crossings but not the non-inlined-public-call asymmetry. Measure
four paths to separate language-boundary, function-call, and implementation costs:

- **rawr inline loop** — `select` **forced-inline** (`@call(.always_inline, …)`), **confirmed by
  disassembly** to be incorporated into the loop;
- **rawr via a `noinline` select wrapper** — same kernel, forced non-inlined call;
- **CRoaring via the current Zig loop** — per-query Zig→C FFI;
- **CRoaring via an in-C loop** — one FFI call, loop in C.

**Canonical comparison** = likely **rawr-noinline vs CRoaring-in-C** (both a non-inlined public
call, no Zig→C tax).

**Directional sanity checks** (measured ranges must support these; if any fails materially,
investigate codegen/harness shape, don't accept the ratio):
- rawr-inline no slower than rawr-noinline;
- CR-in-C no slower than the repeated Zig→C path (or ranges overlap);
- the board ratios (**1.675x M4, 1.20x Zen 4**) are a **lower-bound per host** for
  rawr-noinline / CR-in-C — the fair number must not come out *better* for rawr than the
  FFI-inflated board on either host.
Disassembly confirms both canonical paths keep **one non-inlined public call per query**.

## Where rawr's select cost lives

Split `select(rank)` into the **container-skip** (cumulative-cardinality walk across the top-level
array — O(containers)/call, or is there an index?) vs the **intra-container select** (array index
/ bitset rank-select / run walk). Compare structure against `roaring_bitmap_select`. Characterize
the dense corpus (container types, `select_queries` rank distribution). ns/query with a named
residual.

## Measurement / correctness

- Only the accumulator/checksum is local to the timed loop; bitmap + `select_queries` built
  outside timing. Canonical protocol: 3w/21t median, ≥5 fresh processes, one path per process,
  both hosts. Select does not allocate → single rawr non-allocating tuple vs CRoaring.
- Validate identical results untimed (rawr `select` == CRoaring `select` for every rank, incl.
  boundary/empty).

## Acceptance

- The **matrix-selected public-API select ratio** on both hosts, with the four-path matrix +
  directional sanity checks passing (or a flagged codegen/harness issue), rawr's cost split
  container-skip vs intra-container, container/rank mix recorded.
- **Benchmark-only:** no library API added; `zig build test`, `ReleaseSafe`, `ReleaseFast` green;
  results in `docs/parity-measurement.md`.

## Result to record (feeds `24-01`)

The true public-API ratio (per host) and the dominant rawr select cost — decides whether `24-01`
optimizes (>1.10x) or just corrects the row (≤1.10x).
