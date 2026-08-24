<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 50: Real-data comparison harness (no vendored data)

**Goal.** Make the rawr-vs-CRoaring comparison on the standard real-world corpora **reproducible from a
clean checkout**, without committing any dataset to the repository.

**Not a parity-board addition.** This is an exploratory harness. It gates nothing and moves no board row.

## 1. Why

Every row on the parity board is **synthetic**. The `real-roaring-datasets` corpora are **used by the
Roaring benchmark ecosystem**, which makes them the closest thing to a shared baseline — and rawr has
never been run on them. *(An earlier draft said they are what CRoaring and Java "tune against"; no source
here establishes that.)*

**A scratch run produced a result worth being able to repeat** (M4, ReleaseFast, rawr/SMP vs
CRoaring/libc, 3 process reps × 7 internal — **preliminary, not board-grade**):

| op | uscensus2000 | census1881 | wikileaks |
|---|---:|---:|---:|
| successive AND | **0.31x** | **0.64x** | **0.68x** |
| successive OR | **0.59x** | **1.85x** | **2.60x** |
| successive ANDNOT | **0.73x** | **2.04x** | **1.85x** |
| successive XOR | **0.71x** | **0.80x** | **0.47x** |
| total union (n-way) | 1.32x | 1.02x | 1.10x |
| toArray | **0.62x** | **0.86x** | 1.01x |
| serialize+deserialize | **0.53x** | 0.98x | **0.72x** |

rawr wins most operations. **Two density-dependent losses — pairwise OR and ANDNOT on the denser sets —
while n-way union sits at 1.02–1.10x.**

**That table may be contaminated, which is itself an argument for this spec.** The scratch run executed
**all seven operations in one process, in order**, so AND conditioned the allocator before OR ran. §4's
per-operation process isolation exists precisely because the campaign has measured that effect before
(spec 35: 1.155x vs 1.727x from process history alone). **Treat the OR/ANDNOT gap as a lead to verify,
not a result.**

## 2. Data — downloaded, never committed

**No dataset file may enter the repository.** The corpora are ~180 MB zipped, and **the upstream
repository states no dataset license as of revision `929d8088817840f43ffaa8592b49373b5a2d43b2`
(2021-06-25)** — the phrasing matters: this is the state of that revision, not a claim about the data's
legal status generally.

### 2.1 `scripts/fetch-realdata.sh`

- downloads from `RoaringBitmap/real-roaring-datasets` **pinned to commit
  `929d8088817840f43ffaa8592b49373b5a2d43b2`** — a branch URL is not reproducible;
- **downloads to a temporary file, verifies the pinned SHA-256, then atomically renames** into
  `misc/realdata/`. Verification happens **before extraction**, so an interrupted download can never be
  mistaken for a valid corpus;
- is idempotent: skips an archive already present and verified;
- defaults to the small three; larger sets **opt-in by name**.

| archive | SHA-256 |
| --- | --- |
| `uscensus2000.zip` | `a0f9b171883154f7675c038387fa113f7d819262c02d2f672dfbbba03b013b3d` |
| `census1881.zip` | `68f4dc3a7cea6821d9cd844e027f313b5c0089c2252a3b689c0f6949e5d3c9a3` |
| `wikileaks-noquotes.zip` | `012d941bbd2c3fb85452233a9b82be6eb3ab4b324719425b876d30423279be99` |

`census-income`, `weather_sept_85`, `dimension_*` are opt-in — `weather_sept_85` alone is 30 MB zipped.

**Record provenance in `docs/`**: source repository, the pinned commit, that the repository states no
dataset license at that revision, and that this is why the data is fetched rather than vendored.

### 2.2 Corpus ordering is load-bearing

**Adjacent-pair operations mean bitmap order changes what is measured.** Pin it:

- **sort archive entry names bytewise** — do not rely on shell glob order or filesystem order *(the
  scratch run used shell glob, which orders `csv0, csv1, csv10, csv100, …`)*;
- **require exactly 200 entries** per dataset, fail otherwise;
- **validate each file is ascending, unique `u32`** on load;
- emit an **ordered corpus fingerprint** and report it in the header, so two runs can be shown to have
  operated on the same sequence.

**Format** (established by inspection; upstream documents the archives as bitmap-testing data but not
their internal text format): one file per bitmap, single line, comma-separated ascending `u32`.

## 3. The harness

`src/bench_realdata.zig` + build step `bench-realdata`, wired exactly like `bench_croaring` — same
`addTranslatedCImport` and `addBenchmarkPlatformShim` pattern. Dev tooling; **not** in `.paths`.

**Operations, identical on both sides:**

| op | rawr | CRoaring |
| --- | --- | --- |
| successive AND / OR / ANDNOT / XOR | pairwise `bitwiseAnd/Or/Difference/Xor` over `i, i+1` | `roaring_bitmap_and/or/andnot/xor` |
| total union | `orMany` | `roaring_bitmap_or_many` |
| toArray | `toArray` into a caller buffer | `roaring_bitmap_to_uint32_array` |
| serialize + deserialize | `serialize` → `deserialize` | `portable_serialize` → `portable_deserialize_safe` |

Results are constructed and freed inside the timed region on both sides, so allocator behaviour is part
of what is measured — that is the point.

### 3.1 Allocators — stated, not incidental

**rawr uses `std.heap.smp_allocator`; CRoaring uses its default libc `malloc`** (no memory hooks
installed). This is a **cross-allocator comparison by construction**, and it is **this project's canonical
comparison pairing** — rawr/SMP versus CRoaring/default-libc, as the parity board uses. *(Not "the
allocator it would actually be deployed with": rawr takes a caller-supplied allocator, so it has no
inherent one.)*

Say so in the report header. A reader must not mistake it for a same-allocator measurement.

*(Owner direction, 2026-08-24: focus on SMP; libc-side rawr is not of interest here.)*

## 4. Protocol

**Fresh process per `(implementation, dataset, operation)`** — *not* per implementation/dataset. Running
several operations in one process lets earlier ones condition SMP allocator state, which is the exact
artifact spec 35 measured (1.155x where canonical read 1.727x). One operation per process, always.

**Per-process timing, pinned:**

- **1 warmup cycle, then 7 timed cycles**; take the **true median** of the 7.
- **≥5 processes** per cell; report the **median of process medians** and the **full range**.
- `ReleaseFast`, native CPU, host recorded in the header.

**Denominators differ per operation and must be reported explicitly** — a per-op total is not comparable
across rows without them:

| operation | operations per cycle |
| --- | --- |
| successive AND / OR / ANDNOT / XOR | **199** (adjacent pairs over 200 bitmaps) |
| toArray, serialize+deserialize | **200** |
| total union | **1** |

Report **both** total cycle time and time ÷ denominator.

**Setup boundaries, pinned:**

- **Outside** timing: corpus load and parse, bitmap construction, caller output buffers, the pointer array
  passed to `orMany` / `or_many`.
- **Inside** timing: result allocation and teardown for every operation that produces one — allocator
  behaviour is part of what is being compared.

## 5. Correctness gate — per operation, not aggregate

*(An earlier draft compared only total-union cardinality, total value count, and total serialized bytes.
Those **do not validate pairwise OR or ANDNOT at all** — 199 results per row went unchecked.)*

**Emit a deterministic semantic digest for every operation**, computed **outside** timing, covering for
each result: its **boundary** (which pair produced it), its **cardinality**, and its **ordered values**.
Compare digests between implementations. A mismatch **fails the run**.

**Serialized bytes are reported, not required to match.** Equivalent sets have multiple valid portable
encodings, and rawr and CRoaring may legitimately choose different container representations — the same
constraint established in spec 46-00. Instead, **each implementation deserializes its own output and
semantically validates it**, and the byte counts are reported side by side as data.

## 6. Out of scope

- Adding rows to the parity board — this gates nothing.
- Vendoring data, or any redistribution of the corpora.
- Investigating the OR/ANDNOT gap. **That is a finding this harness produced, and it deserves its own
  spec** rather than being chased inside a measurement tool.
- 64-bit corpora (`10-21-bench64`'s territory).

## 7. Acceptance

- `scripts/fetch-realdata.sh` downloads, **verifies pinned SHA-256s**, is idempotent, and defaults to the
  small three; larger sets opt-in by name.
- **No dataset file is committed**; `misc/realdata/` confirmed gitignored.
- Provenance recorded in `docs/`: pinned commit, and that the repository states no dataset license at
  that revision.
- **Fetch verifies before extraction**, via temp file + atomic rename.
- **Corpus ordering pinned per §2.2** — bytewise entry sort, exactly 200 entries, ascending-`u32`
  validation, ordered fingerprint reported.
- `bench-realdata` builds via the existing CRoaring wiring; not in `.paths`.
- All §3 operations implemented identically on both sides.
- **§5 per-operation semantic digests** computed outside timing and compared; a mismatch **fails the
  run**. Serialized bytes reported, not required equal; each side self-validates its own output.
- **§4 isolation honoured**: one process per (implementation, dataset, operation); warmup/cycle/median
  policy and per-operation denominators reported.
- Protocol per §4; **allocator pairing stated in the report header**.
- **Measurements are valid per §4 and §5.** **No required direction of result.** *(An earlier draft
  demanded the scratch run's direction be reproduced — that would make the acceptance criterion a
  pre-committed conclusion, and a correct measurement is free to overturn a preliminary one.)* **Record
  any disagreement with the scratch table explicitly**, including the possibility that per-operation
  isolation removes the OR/ANDNOT gap.
- No board row moves; existing suites and checks green.

## 8. Estimate

**S** — the harness exists in scratch form and builds; the work is the fetch script, provenance, the
correctness gate, and the protocol.
