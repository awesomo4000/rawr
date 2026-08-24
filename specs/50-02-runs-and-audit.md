<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 50-02: Two-host runs, artifact audit, scratch comparison

Toplevel: [50-realdata-comparison.md](50-realdata-comparison.md).
Gated on: [50-01](50-01-harness-and-protocol.md).

Produces the comparison. **No board row moves and nothing is gated on the outcome.**

## 1. Runs

Full matrix on **M4 and Zen 4**: three datasets × seven operations × two implementations, **one process
per cell**, ≥5 processes each, `ReleaseFast`, native CPU.

## 2. Artifact audit — before reading any number

Confirm from the emitted artifacts, not from the harness source:

- **corpus fingerprint identical** across every implementation and process per dataset;
- **source cardinality totals identical**; **container histograms identical within each implementation**
  across repetitions;
- **semantic digests match** rawr ↔ CRoaring, and each implementation agrees with itself;
- **one process per cell** — the run manifest shows no cell sharing a process;
- allocator pairing, host, and denominators present in the header.

**Any failure here invalidates the timings.** Audit first, interpret second.

## 3. Explicit comparison with the contaminated scratch run

The §1 table in the toplevel came from a run with **three known defects**: all seven operations in one
process (allocator conditioning), unequal construction paths, and shell-glob corpus ordering.

**Report, per operation and dataset, the clean result beside the scratch result**, and state plainly
whether each of the three preliminary claims survives:

- rawr ahead on AND and XOR;
- rawr behind on pairwise OR and ANDNOT on the denser datasets;
- n-way union at parity.

**No direction is required.** A valid measurement may overturn any of them, and **"the OR/ANDNOT gap was
an artifact" is a first-class outcome** — arguably the most valuable one, since it would retire a lead
rather than send work after a phantom.

## 4. What the outcome feeds

- **If the OR/ANDNOT gap survives**, it becomes a candidate for its own spec. **Not investigated here** —
  chasing a finding inside the tool that produced it is how tools stop being trustworthy.
- **If it does not survive**, record that the scratch result was a methodology artifact and close the
  lead.

Either way, record the **container histograms** alongside: if the two sides started from materially
different representations, that shapes what any follow-up would even be about.

## Acceptance

- Full matrix run on both hosts per §1.
- **§2 audit performed and recorded before interpretation**, all checks passing.
- Results reported per operation and dataset: µs, time ÷ denominator, and **ratios from aggregate
  medians**.
- **§3 scratch comparison explicit**, each of the three preliminary claims marked survived / overturned /
  inconclusive.
- Container histograms and corpus fingerprints recorded in the artifacts.
- **No board row moves; no production change; nothing gated on the result.**
- Existing suites and checks green.
- Outcome recorded in the spec, including "artifact" if that is what it is.

## Estimate

**S/M** — the harness exists; this is running it, auditing it, and writing it up honestly.
