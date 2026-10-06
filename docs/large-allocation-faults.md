<!-- SPDX-License-Identifier: MPL-2.0 -->

# Large-allocation diagnosis, spec 52-01

## Outcome

Measured on Linux x86_64 under KVM and Linux aarch64, using Zig 0.16.0.
This is a diagnosis, not a library change or an allocator recommendation.

The bare reproducer resolves a timing increase at SMP's pooled ceiling on both
hosts. At alignment 1, 32,768 bytes uses the pool; 32,769 bytes maps and unmaps
on every cycle. Full writes then incur minor faults. At 4 MiB, direct
PageAllocator and SMP have overlapping timing ranges. Explicit glibc tunables
induce per-cycle mapping, faults, and a slowdown in the libc control too.

These observations establish mapping churn and support a substantial cost of
release and re-touch. They do **not** isolate fault-handler time from kernel
zeroing, writes, mapping, unmapping, or address-dependent effects.

Production replay has a narrower conclusion. The x86_64 KVM `serialize` split
reproduces. Its `toArrayAlloc` split remains unresolved after the permitted
retry. Neither aarch64 row exceeds the 10% range gate. Historical WSL2 numbers
were not remeasured and remain unexplained by this experiment.

## Reproduce

Requires Linux, glibc, Zig 0.16.0, Python 3, and `strace`. These are
repository-only development tools, outside the package allowlist.

```sh
python3 scripts/run-large-alloc.py --self-test
zig build bench-large-alloc -Dcpu=native
# Primary Linux/glibc host, including induction:
python3 scripts/run-large-alloc.py --skip-build --induce
# Second Linux host, without induction:
python3 scripts/run-large-alloc.py --skip-build
# Reformat retained evidence without measuring:
python3 scripts/run-large-alloc.py --report-only misc/large-alloc-RUN
```

The default is five fresh processes per cell, three warmups and 21 timed
samples per process. Tables below report milliseconds per cycle as the median
of process medians with their full range, not a confidence interval. Bare
batch counts are calibrated in disposable processes and passed into fresh
workers. Every authoritative bare timed sample exceeded 1 ms. Production
replay keeps canonical batching at one and uses the existing corpus preparation,
operation, validation, and cleanup entry points. Validation follows timing.

Untraced runs supply times and `getrusage` deltas. Separate fresh `strace` runs
with the same batch counts supply mapping counts. Do not combine traced times
with authoritative times. Counter source on both hosts is
`RAWR_RESIDENCY_FAULT_LINUX_RUSAGE`; all recorded major-fault deltas were zero.
Counter calls surround the timed operation region but are outside its timer.

The bare worker imports no rawr or CRoaring implementation. Disassembly on
both hosts confirms a call to `compiler_rt.memset` with byte `0xA5` and the
runtime payload length. Stores survive optimization. This is not a claim that
the bare fill and production serialization use identical write instructions.
Pointer-offset min/max accounting stays inside the bare operation on all arms;
small no-write times include that diagnostic overhead.

Retained buffers are filled before warmup and freed after timing. Marker
phases distinguish startup, initialization, warmup, timed work, post-timing
validation, and reporting/final deferred teardown. The `cleanup` marker begins
validation; `done` begins reporting and final deferred teardown. Neither phase
contributes to the timed mapping count.

The parser passed known counts of 1, 1, 2, 3, and 0 mapping/unmapping pairs in
the five marked phases. Missing, duplicate, and out-of-order marker controls
fail. An always-zero result fails the positive oracle. Pooled timed cells had
zero mappings. Worker tuple, count, sample-order, and median mutations fail too.

## Environment and allocation route

| property | x86_64 KVM | aarch64 Linux |
| --- | --- | --- |
| CPU | AMD EPYC, 2 guest CPUs | Raspberry Pi, 4 Cortex-A76 CPUs |
| kernel | 6.8.0-146-generic | 6.18.34+rpt-rpi-2712 |
| glibc | 2.39 | 2.41 |
| base page bytes | 4,096 | 16,384 |
| transparent huge pages | enabled policy `madvise`, defrag `madvise` | sysfs policy files unavailable |
| build | ReleaseFast, native CPU | ReleaseFast, native CPU |

Source audit of the installed Zig on both hosts gives `min_class=3`,
`slab_len=max(page_size_max,65536)=65536`, and `size_class_count=13`.
Thus 32,768 bytes is class 12; 32,769 bytes is class 13 and calls
`PageAllocator.map`. Large frees unmap. This is source-established, not inferred
from the timing discontinuity. Higher requested alignment can change the route.

Bare requests use alignment 1. Production `serialize` requests 2,524,082 bytes
at alignment 1; `toArrayAlloc` requests 3,999,572 bytes at alignment 4.
The latter reflects the random corpus's distinct values, not exactly one
million outputs. Length/alignment/pointer observations use an additional real
production invocation after timing, not a wrapper around the timed allocator.

Every record evaluates `ceil((pointer_page_offset + bytes) / page_size)`.
Bare offsets cover warmup and timed allocations. Replay offsets are post-timing
observations only. The report calls this `full-payload-pages`, an address-span
prediction, not a predicted fault count or actual touches for `no_write`.
At 32,769 bytes, page-aligned SMP spans 9 pages on x86_64 and 3 on aarch64.
At 4 MiB it spans 1,024 and 256 respectively.

## Production replay

| host / row / attempt | rawr SMP ms [min,max] | rawr libc ms [min,max] | range verdict |
| --- | ---: | ---: | --- |
| x86_64 serialize | 4.049 [3.753,5.229] | 1.539 [1.137,1.746] | present |
| x86_64 toArrayAlloc, first | 10.272 [8.202,12.457] | 5.974 [3.328,7.502] | unresolved |
| x86_64 toArrayAlloc, retry | 8.605 [5.282,9.466] | 3.647 [2.770,6.480] | unresolved |
| aarch64 serialize, first | 6.728 [6.702,7.116] | 6.306 [6.266,6.639] | unresolved |
| aarch64 serialize, retry | 6.847 [6.824,6.878] | 6.276 [6.268,6.286] | absent |
| aarch64 toArrayAlloc | 8.710 [8.663,8.759] | 8.112 [8.089,8.252] | absent |

The rule is `SMP_min/libc_max > 1.10` for present and
`SMP_max/libc_min <= 1.10` for absent. All other cases get one retry and then
remain unresolved. Absent means absent under this threshold, not equal cost.

In the separate traces, every SMP replay operation mapped/unmapped once and
libc did neither during timing. SMP recorded 617/977 minor faults per operation
on x86_64 for serialize/toArrayAlloc, and 155/245 on aarch64; libc recorded zero.
Those counts and the bare controls support release/re-touch as a contributor
to the reproduced KVM serialization gap. They do not quantify its entire cost.
The ARM results show why fault counts alone are insufficient to predict a
greater-than-10% production gap.

## Bare full-write sweep

| bytes | x86_64 SMP ms [min,max] | x86_64 libc ms [min,max] | aarch64 SMP ms [min,max] | aarch64 libc ms [min,max] |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | .001635 [.001592,.001719] | .002053 [.001616,.002121] | .000893 [.000893,.000894] | .000927 [.000926,.001172] |
| 16,384 | .005985 [.005823,.006328] | .006434 [.006104,.006757] | .003457 [.003457,.003458] | .003488 [.003487,.003491] |
| 32,768 | .013436 [.012695,.018396] | .012801 [.012000,.014081] | .006874 [.006872,.006876] | .006914 [.006905,.009222] |
| 32,769 | .057382 [.044682,.065200] | .013434 [.011350,.014881] | .013150 [.013089,.013320] | .006914 [.006904,.006917] |
| 65,536 | .077878 [.072924,.082615] | .041692 [.040743,.042206] | .021118 [.021085,.021498] | .013748 [.013744,.013752] |
| 262,144 | .310851 [.264687,.316723] | .104105 [.100933,.169621] | .080443 [.080161,.105764] | .054752 [.054733,.054777] |
| 1,048,576 | .972617 [.962228,1.002438] | .433679 [.402659,.455382] | .304985 [.304813,.305321] | .218950 [.218906,.218955] |
| 4,194,304 | 4.845187 [4.202952,5.641162] | 1.729914 [1.556315,1.928880] | 1.412814 [1.394805,1.415593] | .875542 [.875333,.875721] |
| 16,777,216 | 19.642740 [18.683894,21.048130] | 6.694348 [6.361083,6.960645] | 6.084871 [6.074483,6.115242] | 3.501707 [3.501651,3.503781] |

SMP's adjacent boundary ranges separate on both hosts. Its timed mapping count
changes from zero to one map/unmap per cycle. Minor faults across the sweep are
`0,0,0,9,16,64,256,1024,4096` on x86_64 and
`0,0,0,3,4,16,64,256,1024` on aarch64. Default libc has zero timed mappings and
zero median minor faults throughout this sweep. This describes observed reuse,
not proof of a particular malloc-internal threshold transition. Only mmap and
munmap were traced; absence of these is not absence of all OS calls.

## Fixed 4 MiB controls

| host / arm | full write ms [min,max] | no payload write ms [min,max] | retained ms [min,max] |
| --- | ---: | ---: | ---: |
| x86_64 SMP | 4.845187 [4.202952,5.641162] | .009321 [.009237,.009876] | 2.158191 [2.113669,2.182952] |
| x86_64 PageAllocator | 4.287167 [4.049513,5.634431] | .008887 [.008381,.010535] | 2.279551 [2.017421,2.419620] |
| x86_64 libc | 1.729914 [1.556315,1.928880] | .000033 [.000029,.000039] | 1.713569 [1.574051,1.752077] |
| aarch64 SMP | 1.412814 [1.394805,1.415593] | .002027 [.002020,.002045] | .875536 [.875228,.875827] |
| aarch64 PageAllocator | 1.397879 [1.387129,1.401787] | .002028 [.002011,.002032] | .875271 [.875203,.875790] |
| aarch64 libc | .875542 [.875333,.875721] | .000057 [.000056,.000071] | .875549 [.875388,.875672] |

SMP and direct PageAllocator full-write ranges overlap on both hosts, with
1,024/256 minor faults and one map/unmap each cycle. No-write preserves mapping
but removes these payload faults; its time is much smaller. Retention removes
all timed mappings and faults. Retained ARM ranges overlap; **retained x86_64
SMP still trails libc with separated ranges**. Do not claim the entire gap
vanishes, or infer equality from the ARM overlap.

On x86_64, the prescribed induction environment was:

```text
MALLOC_MMAP_THRESHOLD_=16384
MALLOC_TRIM_THRESHOLD_=0
MALLOC_TOP_PAD_=0
```

Induced libc at 4 MiB records one map/unmap per cycle. Full write costs
4.107319 [3.723760,4.140810] ms with 1,025 minor faults. No-write costs
.014398 [.013903,.017578] ms with one minor fault. Default libc full write
is 1.729914 [1.556315,1.928880] ms with zero faults. The intervention changes
threshold policy and adaptation together, so this is evidence of release and
re-touch cost, not an isolated measurement of fault latency. The extra libc
fault can include allocator metadata. Induction was run across all nine sizes;
4 KiB remained unmapped, while 16 KiB and larger mapped per cycle.

## Evidence and validation

Source baseline was `5762ab7` plus this diagnosis implementation. Raw logs,
per-sample counters, traces, calibrated batches, retries, and provenance are in
gitignored `misc/large-alloc-x86/` and `misc/large-alloc-arm/`. There are 79 and
61 cells, respectively, with 720 authoritative fresh timing processes including
the two paired retries. Calibration and syscall traces are separate processes.

Worker SHA-256 values:

```text
x86_64 bare   c7ceeebf1a19e36380f1ddd06b2f0688c38cab5223a69c521f39a715c401b1c4
x86_64 replay 5c747ede4f6267d617279ed9a50f2fbcffe50abab2ca60bd9438450ec99c77f0
aarch64 bare  9fe811f7f85dc13c7f22198ff93ceb94e1579bf91072c0e861dc5da13b1abf59
aarch64 replay 812a4f8c684b0bfcf765515e7e0eedeea4eaee258ef986799158d65812714858
```

Both installations have these Zig source SHA-256 values:

```text
SmpAllocator.zig  7fd45dd7c8df7a4cd160d794f262bcf6057ecccc8d718552125db0eb5351065b
PageAllocator.zig ecec98154814d9494bf607c10f7d7bee29d747d3c29d2c71e1c9fcd8c8645801
```

Verification completed:

- ReleaseFast workers and all measurement/parser controls on both hosts.
- x86_64 normal `zig build`, `test`, `test64`, `difftest`, `difftest64`,
  `check-32`, `check-docs`, and `check-package`.
- aarch64 `test`, `test64`, `check-docs`, `check-package`, and
  `check-portability`: 18 cells, zero broken/not-targetable cells, both build
  options pass. Package allowlist remains 33 files.
- Each host's unit suites report 488 passed and four skipped. The two x86_64
  differential suites each passed 1,000 cases.
- A separate ReleaseSafe bare build passed SMP write, libc retained, and
  PageAllocator no-write smoke cases.
- Disassembly confirms the bare writes on both architectures. No CRoaring
  implementation symbols occur in the bare worker.

No production source, API, allocator policy, or canonical board row changed.
Spec 37's ordering of many output buffers cannot explain a single output
allocation. This experiment does not exclude other layout/residency effects,
and the remaining x86_64 retained-buffer difference is left unresolved.
