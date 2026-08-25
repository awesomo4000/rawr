<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 50-00: Pinned fetcher, provenance, deterministic loader

Toplevel: [50-realdata-comparison.md](50-realdata-comparison.md).

**No measurement.** This chunk makes the corpus reproducible and provably identical between runs. Every
number `50-01` and `50-02` produce depends on it, so it is verified before any timing exists.

## 1. `scripts/fetch-realdata.sh`

- Downloads from `RoaringBitmap/real-roaring-datasets` **pinned to commit
  `929d8088817840f43ffaa8592b49373b5a2d43b2`** — a branch URL is not reproducible.
- **Temp file → verify pinned SHA-256 → atomic rename.** Verification precedes extraction.
- **Extract to a temporary directory, then atomically rename it**, with the temp directory **under
  `misc/realdata/`** so the rename is same-filesystem and therefore actually atomic. Atomic download
  alone leaves interrupted *extraction* looking valid.
- **Portable hashing:** `sha256sum` when present, else `shasum -a 256` (macOS).
- Idempotent: skip an archive already present and verified.

**Manifest — closed. An unknown name is rejected, not fetched.**

| archive | SHA-256 | entries |
| --- | --- | --- |
| `uscensus2000.zip` | `a0f9b171883154f7675c038387fa113f7d819262c02d2f672dfbbba03b013b3d` | 200 |
| `census1881.zip` | `68f4dc3a7cea6821d9cd844e027f313b5c0089c2252a3b689c0f6949e5d3c9a3` | 200 |
| `wikileaks-noquotes.zip` | `012d941bbd2c3fb85452233a9b82be6eb3ab4b324719425b876d30423279be99` | 200 |

Adding a dataset later means **adding a manifest row** — name, digest, expected entry count. There is no
open-ended argument.

## 2. Provenance — `docs/`

Source repository, **pinned commit**, and that **the repository states no dataset license as of that
revision** (phrased as the state of that revision, not a claim about the data's legal status). Record that
this is why the corpus is fetched rather than vendored.

## 3. Deterministic loader

Adjacent-pair operations make ordering load-bearing, so it is pinned rather than inherited from the shell
or filesystem:

- **Bytewise sort of archive entry names.** *(The scratch run used shell glob, which orders
  `csv0, csv1, csv10, csv100, …`.)*
- **Exactly 200 entries** per dataset; fail otherwise.
- **Validate each file is ascending, unique `u32`** on load.
- Format: one file per bitmap, single line, comma-separated ascending `u32`.

## 4. Corpus fingerprint

**FNV-1a 64, little-endian framed** — for each bitmap in order: `u32` ordinal, `u64` value count, then
each `u32` value ascending. No `std.hash` default that may change across Zig versions, no pointer or
address input, no host-endian writes.

Emitted in the report header. `50-01`'s controller enforces it.

## Acceptance

- Fetcher per §1: pinned commit, verify-before-extract, temp-file **and** temp-directory atomic renames
  under `misc/realdata/`, portable hashing, idempotent, **rejects unknown names**.
- **No dataset file committed**; `misc/realdata/` confirmed gitignored.
- Provenance per §2 in `docs/`.
- Loader per §3 — bytewise ordering, 200-entry check, ascending-`u32` validation.
- Fingerprint per §4, stable across two separate processes on the same host.
- **Negative controls — each must seed the defect the guard actually catches.** A guard never seen to
  fail is not known to work, and a control that exercises the wrong guard proves nothing.

  **Run every control against a disposable copy of the archive and extraction — never against the
  accepted cached corpus.** Mutating the cache would leave a corrupted corpus behind for later runs, and
  a measurement harness must not damage its own inputs to test itself.

  | seeded defect | guard that must fire |
  | --- | --- |
  | corrupt a byte in a downloaded archive | **archive SHA-256 check** (§1) — *not* the corpus fingerprint |
  | remove an entry from an extracted corpus | **entry-count check** (§3) |
  | **disable or reverse the bytewise sort** in the loader | **corpus fingerprint** (§4) |
  | make a file non-ascending or non-unique | **input validation** (§3) |

  *(An earlier draft proposed "hand-reorder → fingerprint changes". That is not a valid control: the
  loader bytewise-sorts entry names, so changing archive or filesystem enumeration order should correctly
  have **no** effect. The defect to seed is the sort itself.)*
- No board row moves; existing suites and checks green.
- **No timing produced.**

## Verification record — implemented, reviewed, ACCEPTED (`0056165`)

Added `scripts/fetch-realdata.sh`, `src/realdata_corpus.zig`, `src/check_realdata_corpus.zig`,
`docs/realdata-benchmarks.md`, build steps `check-realdata-loader` and `realdata-corpus-check`.
546 lines, no production-library change.

**Every guard exercised here against seeded defects, on disposable copies:**

| control | result |
| --- | --- |
| corrupt an archive byte | **`archive SHA-256 mismatch`**, exit 1 — the **archive** guard, not the corpus fingerprint ✓ |
| remove a corpus entry | **`error.UnexpectedEntryCount`** ✓ |
| non-ascending values | **`error.ValuesNotStrictlyAscending`** ✓ |
| duplicate value | **`error.ValuesNotStrictlyAscending`** ✓ — strict ascent subsumes uniqueness |
| reverse the bytewise sort | fingerprint changes on all three datasets; `OrderMutationNotDetected` if it ever does not ✓ |
| unknown dataset name | rejected with usage, exit 2 — **not fetched** ✓ |
| fingerprint across two fresh processes | **identical** ✓ |
| `misc/realdata` | gitignored; **`git ls-files misc/` empty** ✓ |

The sort control is built into the loader as a `.reverse_bytewise` order option rather than bolted on —
so the guard's own failure mode is permanently testable, which is what the spec asked for and what an
external one-off check cannot provide.

**Independent cross-validation:** the loader's value counts — **5,985 / 1,003,861 / 275,355** — match
exactly the Python parse performed before any of this code existed. Two independent readers of the same
archives agree.

**Corpus fingerprints (M4):**

| dataset | entries | values | fingerprint |
| --- | ---: | ---: | --- |
| uscensus2000 | 200 | 5,985 | `0x3dda62df585f1b25` |
| census1881 | 200 | 1,003,861 | `0x03d40da10e217e89` |
| wikileaks-noquotes | 200 | 275,355 | `0x0140d2d90eaca255` |

Provenance in `docs/realdata-benchmarks.md` states the pinned commit and phrases the licensing as the
state of that revision — *"not a broader claim about the legal status of the underlying data"* — which is
the distinction the spec asked for.

**No timing produced**, per scope.

## Estimate

**S** — a shell script, a loader, and a hash.
