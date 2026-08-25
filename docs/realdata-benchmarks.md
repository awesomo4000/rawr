<!-- SPDX-License-Identifier: MPL-2.0 -->

# External real-data benchmark corpora

Rawr's exploratory real-data benchmark uses selected archives from the
[`RoaringBitmap/real-roaring-datasets`](https://github.com/RoaringBitmap/real-roaring-datasets)
repository at commit `929d8088817840f43ffaa8592b49373b5a2d43b2` (06/25/2021).

The upstream repository states no dataset license at that revision. This describes the repository at the
pinned revision; it is not a broader claim about the legal status of the underlying data. Rawr therefore
does not vendor or redistribute these corpora. Each user downloads them directly from the upstream
repository for local benchmarking.

The initial corpus manifest is intentionally closed:

| archive | SHA-256 | entries |
| --- | --- | ---: |
| `uscensus2000.zip` | `a0f9b171883154f7675c038387fa113f7d819262c02d2f672dfbbba03b013b3d` | 200 |
| `census1881.zip` | `68f4dc3a7cea6821d9cd844e027f313b5c0089c2252a3b689c0f6949e5d3c9a3` | 200 |
| `wikileaks-noquotes.zip` | `012d941bbd2c3fb85452233a9b82be6eb3ab4b324719425b876d30423279be99` | 200 |

The upstream repository identifies the ZIP files as bitmap-testing data but does not document their
internal text format. Inspection of the pinned archives shows one bitmap per file, encoded as a single
comma-separated sequence of ascending decimal `u32` values.

Run `./scripts/fetch-realdata.sh` to download the three archives into the gitignored
`misc/realdata/` directory. The script verifies every archive before extracting it. Dataset files are
development inputs only: they are not part of rawr's Zig package or public library API.
