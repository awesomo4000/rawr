<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 50: Real-data comparison harness (no vendored data)

**Goal.** Make the rawr-vs-CRoaring comparison on the standard real-world corpora **reproducible from a
clean checkout**, without committing any dataset to the repository.

**Not a parity-board addition.** This is an exploratory harness. It gates nothing and moves no board row.

## 1. Why

Every row on the parity board is **synthetic**. The standard `real-roaring-datasets` corpora are what
CRoaring and the Java implementation tune against, which makes them the closest thing to a shared
baseline — and rawr has never been run on them.

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
while n-way union sits at 1.02–1.10x.** That contrast is the reason this harness should exist rather than
stay in a scratch directory.

## 2. Data — downloaded, never committed

**No dataset file may enter the repository.** The upstream README states **no license**, and the corpora
are ~180 MB zipped.

`scripts/fetch-realdata.sh`:

- downloads from `RoaringBitmap/real-roaring-datasets` into **`misc/realdata/`** (`misc/` is already
  gitignored);
- **verifies a pinned SHA-256 per archive** and fails loudly on mismatch — corpus drift must not silently
  change results;
- is idempotent: skips an archive already present and verified;
- takes an optional dataset list, defaulting to the small three.

Pinned digests for the initial set:

| archive | SHA-256 |
| --- | --- |
| `uscensus2000.zip` | `a0f9b171883154f7675c038387fa113f7d819262c02d2f672dfbbba03b013b3d` |
| `census1881.zip` | `68f4dc3a7cea6821d9cd844e027f313b5c0089c2252a3b689c0f6949e5d3c9a3` |
| `wikileaks-noquotes.zip` | `012d941bbd2c3fb85452233a9b82be6eb3ab4b324719425b876d30423279be99` |

Larger sets (`census-income`, `weather_sept_85`, `dimension_*`) are **opt-in by name**, not default —
`weather_sept_85` alone is 30 MB zipped.

**Record provenance in `docs/`**: source repository, that no license is stated, and that this is why the
data is fetched rather than vendored.

**Format** (established by inspection, since upstream does not document it): one file per bitmap, a
single line, comma-separated ascending `u32`. 200 bitmaps per dataset.

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
installed). This is a **cross-allocator comparison by construction** — each library on the allocator it
would actually be deployed with, matching how the parity board pairs rawr/SMP against CRoaring/libc.

Say so in the report header. A reader must not mistake it for a same-allocator measurement.

*(Owner direction, 2026-08-24: focus on SMP; libc-side rawr is not of interest here.)*

## 4. Protocol

- **Fresh process per (implementation, dataset)** — never both implementations in one process.
- **≥5 process runs**, report **median and full range**. *(The scratch run used 3; that is why its numbers
  are marked preliminary.)*
- `ReleaseFast`, native CPU, host recorded in the header.
- Report **µs per operation** plus the **rawr ÷ CRoaring ratio**.

## 5. Correctness gate

**Both implementations must agree**, checked outside timing, and the run **fails** if they do not:

- total-union cardinality;
- total value count from `toArray`;
- total serialized bytes.

All three matched exactly in the scratch run across all three datasets. This is what makes the timing
comparison like-for-like rather than a comparison of two different computations.

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
- Provenance recorded in `docs/`, including the absent license.
- `bench-realdata` builds via the existing CRoaring wiring; not in `.paths`.
- All §3 operations implemented identically on both sides.
- **§5 correctness gate enforced and passing**, with a failure actually failing the run.
- Protocol per §4; **allocator pairing stated in the report header**.
- Results reproduce the scratch run's direction: rawr ahead on AND/XOR, behind on dense pairwise
  OR/ANDNOT, near parity on n-way union.
- No board row moves; existing suites and checks green.

## 8. Estimate

**S** — the harness exists in scratch form and builds; the work is the fetch script, provenance, the
correctness gate, and the protocol.
