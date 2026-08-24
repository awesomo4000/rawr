<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 50-02: Two-host runs, artifact audit, scratch comparison

Toplevel: [50-realdata-comparison.md](50-realdata-comparison.md).
Gated on: [50-01](50-01-harness-and-protocol.md).

Produces the comparison. **No board row moves and nothing is gated on the outcome.**

## 1. Runs

Full matrix on **M4 and Zen 4**: three datasets × seven operations × two implementations.

**Each process executes exactly one cell; at least five independent processes execute each cell.**
*(An earlier draft said "one process per cell" and then required five per cell — contradictory.)*

**Mechanically required, per host:** **42 cells** (3 × 7 × 2) and, at `RUNS=5`, **exactly 210 process
results**. The controller validates these counts against its worker manifest and **fails on any
shortfall** — an omitted operation must not yield a plausible partial report.

`ReleaseFast`, native CPU.

## 2. Artifact audit — before reading any number

Confirm from the emitted artifacts, not from the harness source:

- **corpus fingerprint identical** across every implementation and process per dataset;
- **source cardinality totals identical**; **container histograms identical within each implementation**
  across repetitions;
- **semantic digests match** rawr ↔ CRoaring, and each implementation agrees with itself;
- **one process per cell** — the run manifest shows no cell sharing a process;
- allocator pairing, host, and denominators present in the header.

### 2.1 Cross-host, not just within-host

The above checks run inside each host's controller. **Also compare across M4 and Zen 4:**

- **corpus fingerprints identical** — both hosts must have loaded the same corpus;
- **semantic digests identical** — both hosts must have computed the same results;
- **source cardinality totals identical**;
- **per-implementation container histograms identical**.

A per-host audit passing twice does not establish that the two hosts measured the same thing.

**Any failure here invalidates the timings.** Audit first, interpret second.

## 3. Explicit comparison with the contaminated scratch run

The §1 table in the toplevel came from a run with known methodology problems. **Record precisely what
changed and what did not** — an earlier draft listed construction as a corrected defect, which is wrong:

| scratch run | this harness | status |
| --- | --- | --- |
| all seven operations in **one process** | **one cell per process** | **corrected** |
| corpus order from **shell glob** | **bytewise sort**, pinned, fingerprinted | **pinned** — order may be unchanged, but it is now deterministic and verified rather than inherited |
| rawr `fromSorted` vs CRoaring `create` + `add_many` | **the same two paths** | **NOT corrected** — pinned in `50-01` §2 and made **observable** via container histograms |

**So unequal construction remains a live confound**, not a resolved one. If the histograms show materially
different container representations, the OR/ANDNOT comparison is between two different starting states and
must be reported that way.

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
