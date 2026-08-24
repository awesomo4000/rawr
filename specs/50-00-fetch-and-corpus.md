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
- **Negative controls:** corrupt an archive byte → digest check fails; remove an entry → entry-count check
  fails; hand-reorder → fingerprint changes. A guard never seen to fail is not known to work.
- No board row moves; existing suites and checks green.
- **No timing produced.**

## Estimate

**S** — a shell script, a loader, and a hash.
