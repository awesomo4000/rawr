<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 26a-01: Attribution diagnostic + cross-host verdict

Second chunk of [clone attribution](26a-clone-attribution.md). The experimental half: build the
non-manifest diagnostic, measure both hosts, inventory representations, and document the
verdict on the M4 `clone + removeRange` 1.840x.

## Gate

- `26a-00` complete (the canonical `clone` row exists — its medians feed the teardown
  estimates).

## Deliverables

- **Diagnostic executable, outside the canonical manifest:** `zig build bench-range-attrib` →
  `./zig-out/bin/bench_range_attrib`, five-process runner
  `scripts/run-bench-range-attrib.sh`. Three matched-boundary internal-timing measurements per
  implementation, on the wide-dense corpus, one condition per fresh process:
  - **clone body only** (deinit outside timing);
  - **removeRange body only** (fresh untimed clone before each timed invocation, untimed
    deinit after);
  - **clone + removeRange body only** (deinit outside timing).
- **Untimed representation/allocation inventory** (before any cause is named): container
  types/counts, allocation counts/bytes, copied payload bytes — both implementations where
  measurable. (The corpus is `addRange`-built → run containers in rawr; no component hypothesis
  is assumed.)
- **Attribution estimates** from independently measured medians, with underlying ranges, no
  exact-additivity claim:
  ```text
  full teardown        = canonical clone     − clone body
  reduced teardown     = canonical composite − clone+remove body
  interaction residual = clone+remove body − clone body − remove body
  ```
- **Diagnostic allocator variants (pinned):** the attribution equations are fed by **rawr-SMP
  vs CRoaring** — the pairing behind the 1.840x. **rawr-libc** is additionally measured as the
  **allocator A/B control** when diagnosing allocator traffic (if SMP and libc clone-body times
  diverge, allocator traffic is implicated; if they match, look at payload copying /
  per-container overhead). "Per implementation" always means these three.
- **Cross-host measurement:** M4 and Zen 4, canonical protocol (3w/21t median, ≥5 fresh
  processes, median + full range).
- **Full-board regression gate (moved here from `26a-00`):** run the complete **39-row**
  canonical tables on **both** M4 and Zen 4; **no existing row worsens by more than 5%** vs the
  post-spec-26 baseline (rerun on range overlap).
- **Verdict, documented in `docs/parity-measurement.md`:** the M4 1.840x attributed across
  **clone work / mutation work / teardown / non-additive residual**, with the dominant clone
  component named from the inventory (allocator traffic / payload copying / per-container
  overhead) **only if clone dominates** — and the follow-up recommendation (clone-optimization
  spec / removeRange revisit / documented residual) stated per the parent's conditional.

## Acceptance / checklist

- [ ] `bench_range_attrib` + runner exist; diagnostics are **not** manifest rows (board stays 39)
- [ ] Three matched-boundary measurements × (rawr-SMP, rawr-libc, CRoaring) × both hosts, one
      condition per fresh process; equations fed by rawr-SMP vs CRoaring, libc as the A/B control
- [ ] Full 39-row canonical tables on both hosts; no existing row > 5% worse vs the
      post-spec-26 baseline
- [ ] Untimed inventory recorded before any component is named
- [ ] Attribution estimates reported with ranges; residual named, additivity not claimed
- [ ] Verdict + follow-up recommendation in `docs/parity-measurement.md`
- [ ] Benchmark-only; test / difftest / ReleaseSafe / ReleaseFast green
