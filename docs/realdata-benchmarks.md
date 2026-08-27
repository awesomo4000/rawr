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

After fetching, `./scripts/run-realdata-bench.sh` runs each implementation, dataset, and operation in a
fresh process. It compares rawr with `std.heap.smp_allocator` against CRoaring with its default libc
allocator. This is intentionally a cross-allocator comparison and matches rawr's canonical benchmark
pairing. Run `./scripts/check-realdata-protocol.sh` to exercise the controller's seeded-failure guards
without producing benchmark results.

## Canonical exploratory result (08/25/2026)

The first canonical run used five fresh processes per tuple on macOS/Apple M4 and Linux/AMD Zen 4. Each
process performed one warmup cycle and seven timed cycles; the tables report the median of the five
process medians. `rawr / CRoaring` below is computed from those aggregate medians. A ratio below `1.0x`
means rawr was faster. These are exploratory workload results, not parity-board gates.

The emitted artifacts passed both per-host controller validation and a cross-host audit. Both hosts
reported the same 42-cell manifest, corpus fingerprints, source cardinalities, semantic digests, and
per-implementation container histograms. Run the same audit with:

```bash
./scripts/audit-realdata-hosts.sh <m4-artifact-prefix> <zen4-artifact-prefix>
```

Construction produced identical aggregate source-container histograms on both implementations and hosts:

| dataset | array | bitset | run | source cardinality | fingerprint |
| --- | ---: | ---: | ---: | ---: | --- |
| `uscensus2000` | 2,221 | 0 | 0 | 5,985 | `0x3dda62df585f1b25` |
| `census1881` | 1,459 | 5 | 0 | 1,003,861 | `0x03d40da10e217e89` |
| `wikileaks-noquotes` | 1,892 | 0 | 0 | 275,355 | `0x0140d2d90eaca255` |

### Apple M4

| dataset | operation | rawr cycle (us) | CR cycle (us) | rawr ns/op | CR ns/op | rawr / CR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `uscensus2000` | AND | 6.000 | 11.000 | 30.2 | 55.3 | 0.545x |
| `uscensus2000` | OR | 81.000 | 139.000 | 407.0 | 698.5 | 0.583x |
| `uscensus2000` | ANDNOT | 39.000 | 100.000 | 196.0 | 502.5 | 0.390x |
| `uscensus2000` | XOR | 75.000 | 124.000 | 376.9 | 623.1 | 0.605x |
| `uscensus2000` | total union | 585.000 | 450.000 | 585000.0 | 450000.0 | 1.300x |
| `uscensus2000` | toArray | 4.000 | 6.000 | 20.0 | 30.0 | 0.667x |
| `uscensus2000` | serialize + deserialize | 54.000 | 88.000 | 270.0 | 440.0 | 0.614x |
| `census1881` | AND | 20.000 | 27.000 | 100.5 | 135.7 | 0.741x |
| `census1881` | OR | 365.000 | 244.000 | 1834.2 | 1226.1 | 1.496x |
| `census1881` | ANDNOT | 279.000 | 168.000 | 1402.0 | 844.2 | 1.661x |
| `census1881` | XOR | 242.000 | 371.000 | 1216.1 | 1864.3 | 0.652x |
| `census1881` | total union | 751.000 | 640.000 | 751000.0 | 640000.0 | 1.173x |
| `census1881` | toArray | 268.000 | 292.000 | 1340.0 | 1460.0 | 0.918x |
| `census1881` | serialize + deserialize | 264.000 | 161.000 | 1320.0 | 805.0 | 1.640x |
| `wikileaks-noquotes` | AND | 82.000 | 117.000 | 412.1 | 587.9 | 0.701x |
| `wikileaks-noquotes` | OR | 644.000 | 249.000 | 3236.2 | 1251.3 | 2.586x |
| `wikileaks-noquotes` | ANDNOT | 441.000 | 220.000 | 2216.1 | 1105.5 | 2.005x |
| `wikileaks-noquotes` | XOR | 188.000 | 501.000 | 944.7 | 2517.6 | 0.375x |
| `wikileaks-noquotes` | total union | 342.000 | 330.000 | 342000.0 | 330000.0 | 1.036x |
| `wikileaks-noquotes` | toArray | 77.000 | 92.000 | 385.0 | 460.0 | 0.837x |
| `wikileaks-noquotes` | serialize + deserialize | 66.000 | 141.000 | 330.0 | 705.0 | 0.468x |

### AMD Zen 4

| dataset | operation | rawr cycle (us) | CR cycle (us) | rawr ns/op | CR ns/op | rawr / CR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `uscensus2000` | AND | 8.225 | 4.448 | 41.3 | 22.4 | 1.849x |
| `uscensus2000` | OR | 113.993 | 135.182 | 572.8 | 679.3 | 0.843x |
| `uscensus2000` | ANDNOT | 122.137 | 67.025 | 613.8 | 336.8 | 1.822x |
| `uscensus2000` | XOR | 111.578 | 135.473 | 560.7 | 680.8 | 0.824x |
| `uscensus2000` | total union | 933.159 | 1767.914 | 933159.0 | 1767914.0 | 0.528x |
| `uscensus2000` | toArray | 3.877 | 6.642 | 19.4 | 33.2 | 0.584x |
| `uscensus2000` | serialize + deserialize | 65.562 | 76.963 | 327.8 | 384.8 | 0.852x |
| `census1881` | AND | 16.100 | 14.246 | 80.9 | 71.6 | 1.130x |
| `census1881` | OR | 416.336 | 758.954 | 2092.1 | 3813.8 | 0.549x |
| `census1881` | ANDNOT | 309.547 | 520.400 | 1555.5 | 2615.1 | 0.595x |
| `census1881` | XOR | 266.817 | 915.606 | 1340.8 | 4601.0 | 0.291x |
| `census1881` | total union | 588.306 | 939.781 | 588306.0 | 939781.0 | 0.626x |
| `census1881` | toArray | 182.210 | 71.573 | 911.0 | 357.9 | 2.546x |
| `census1881` | serialize + deserialize | 1378.939 | 488.220 | 6894.7 | 2441.1 | 2.824x |
| `wikileaks-noquotes` | AND | 80.510 | 70.161 | 404.6 | 352.6 | 1.148x |
| `wikileaks-noquotes` | OR | 812.274 | 302.874 | 4081.8 | 1522.0 | 2.682x |
| `wikileaks-noquotes` | ANDNOT | 479.714 | 117.789 | 2410.6 | 591.9 | 4.073x |
| `wikileaks-noquotes` | XOR | 323.984 | 418.359 | 1628.1 | 2102.3 | 0.774x |
| `wikileaks-noquotes` | total union | 295.721 | 266.576 | 295721.0 | 266576.0 | 1.109x |
| `wikileaks-noquotes` | toArray | 49.111 | 21.951 | 245.6 | 109.8 | 2.237x |
| `wikileaks-noquotes` | serialize + deserialize | 111.688 | 84.197 | 558.4 | 421.0 | 1.327x |

### Scratch comparison

The scratch run was M4-only, so its ratios are compared directly with the clean M4 run. The clean Zen 4
column tests whether the direction generalizes to the second host.

| dataset | operation | scratch M4 | clean M4 | clean Zen 4 |
| --- | --- | ---: | ---: | ---: |
| `uscensus2000` | AND | 0.31x | 0.545x | 1.849x |
| `uscensus2000` | OR | 0.59x | 0.583x | 0.843x |
| `uscensus2000` | ANDNOT | 0.73x | 0.390x | 1.822x |
| `uscensus2000` | XOR | 0.71x | 0.605x | 0.824x |
| `uscensus2000` | total union | 1.32x | 1.300x | 0.528x |
| `uscensus2000` | toArray | 0.62x | 0.667x | 0.584x |
| `uscensus2000` | serialize + deserialize | 0.53x | 0.614x | 0.852x |
| `census1881` | AND | 0.64x | 0.741x | 1.130x |
| `census1881` | OR | 1.85x | 1.496x | 0.549x |
| `census1881` | ANDNOT | 2.04x | 1.661x | 0.595x |
| `census1881` | XOR | 0.80x | 0.652x | 0.291x |
| `census1881` | total union | 1.02x | 1.173x | 0.626x |
| `census1881` | toArray | 0.86x | 0.918x | 2.546x |
| `census1881` | serialize + deserialize | 0.98x | 1.640x | 2.824x |
| `wikileaks-noquotes` | AND | 0.68x | 0.701x | 1.148x |
| `wikileaks-noquotes` | OR | 2.60x | 2.586x | 2.682x |
| `wikileaks-noquotes` | ANDNOT | 1.85x | 2.005x | 4.073x |
| `wikileaks-noquotes` | XOR | 0.47x | 0.375x | 0.774x |
| `wikileaks-noquotes` | total union | 1.10x | 1.036x | 1.109x |
| `wikileaks-noquotes` | toArray | 1.01x | 0.837x | 2.237x |
| `wikileaks-noquotes` | serialize + deserialize | 0.72x | 0.468x | 1.327x |

The preliminary claims resolve as follows:

- **AND and XOR: inconclusive as one combined claim.** XOR remained faster on every dataset and host.
  AND remained faster on M4 but was slower on all three Zen 4 datasets.
- **Dense OR and ANDNOT losses: inconclusive and host-conditioned.** Both losses survived on M4 and on
  Zen 4 `wikileaks-noquotes`, but the Zen 4 `census1881` rows reversed direction.
- **N-way union at parity: overturned as a general claim.** The clean M4 ratios span 1.036x-1.300x;
  Zen 4 spans 0.528x-1.109x, with rawr materially faster on two datasets.

The scratch OR/ANDNOT lead was therefore not a pure process-history artifact: it survives cleanly on M4
and for `wikileaks-noquotes` on both hosts. It is not explained by different aggregate source-container
histograms. The host reversals also show that density alone is not a sufficient explanation. Any
optimization investigation belongs in a separate experiment.
