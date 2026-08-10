# SmpAllocator Free-Order Investigation

Working notes. Two candidate designs, the measurements behind them, and enough
detail to write specs from. Nothing here is mapped onto an existing codebase.

**Status: the pathology is unreproduced.** Everything measured below was run on
x86_64-linux with 4 KiB pages, where the effect is absent at 8 KiB object size.
The originating observation is on Apple M4 (aarch64-macos, 16 KiB pages). Treat
the designs as mechanisms awaiting validation, not as established fixes.

---

## 1. Originating observation

- Workload: bulk teardown of ~20,000 objects of 8192 bytes each (~160 MiB).
- Single-threaded.
- Allocator: `std.heap.smp_allocator`.
- Symptom: the free loop was substantially slower than the same loop under
  libc malloc; roughly 50% of profile time attributed to free.
- Addresses returned by successive allocations were observed to be
  non-sequential.
- Sorting the pointer array by address before the free loop closed the gap.
- Reproduced on Apple M4. Not observed on AMD Zen4.

The critical inference: **sorting is an ordering-only transformation.** It
cannot change instruction counts, lock behaviour, syscall counts, or allocator
control flow. If sorting fixes it, the cost is memory-system locality —
cache misses, TLB misses, or prefetcher behaviour — and nothing else.

---

## 2. How SmpAllocator works

Source: `lib/std/heap/SmpAllocator.zig`, 224 lines as of Zig 0.16.0.

### 2.1 Structure

```zig
cpu_count: u32,
threads: [max_thread_count]Thread,   // max_thread_count = 128

const Thread = struct {
    _: void align(std.atomic.cache_line) = {},   // false-sharing pad
    mutex: std.atomic.Mutex = .unlocked,
    next_addrs: [size_class_count]usize = @splat(0),  // bump pointer per class
    frees:      [size_class_count]usize = @splat(0),  // freelist head per class
};

threadlocal var thread_index: u32 = 0;
```

Global singleton. `cpu_count` is `min(std.Thread.getCpuCount(), 128)`, computed
lazily and cached.

### 2.2 Geometry

```
slab_len        = max(page_size_max, 64 * 1024)
min_class       = log2(@sizeOf(usize))        = 3
size_class_count = log2(slab_len) - min_class = 13
slotSize(class) = 1 << (class + 3)            // 8 .. 32768
max_alloc_search = 1
```

`slab_len` is 65536 on both x86_64-linux (page_size_max 4096) and
aarch64-macos (page_size_max 16384). **Slab geometry is identical on both
platforms** — this was checked, and rules out slab size as the Darwin
differentiator.

For an 8192-byte allocation:

```
sizeClassIndex(8192, align) = max(64 - clz(8191), alignBits, 3) - 3
                            = max(13, 0, 3) - 3 = 10
slotSize(10) = 8192
slots per slab = 65536 / 8192 = 8
```

**Eight allocations per slab.** A fresh 64 KiB slab is mapped every 8th
allocation on a cold path. This is the worst small-class ratio in the whole
size-class range short of 16384/32768.

### 2.3 Allocation path

```zig
fn alloc(len, alignment) ?[*]u8 {
    class = sizeClassIndex(len, alignment);
    if (class >= size_class_count) return PageAllocator.map(len, alignment);  // large

    t = Thread.lock();
    outer: while (true) {
        if (t.frees[class] != 0) {                  // 1. pop freelist (LIFO)
            node = @ptrFromInt(t.frees[class]);
            t.frees[class] = node.*;
            return node;
        }
        if (t.next_addrs[class] % slab_len != 0) {  // 2. bump within slab
            t.next_addrs[class] += slot_size;
            return old_next_addr;
        }
        if (search_count >= max_alloc_search) {     // 3. map a fresh slab
            slab = PageAllocator.map(slab_len, .fromByteUnits(slab_len));
            t.next_addrs[class] = @intFromPtr(slab) + slot_size;
            return slab;
        }
        // 4. rotate to the next thread slot and retry
        t.unlock();
        index = (index + 1) % cpu_count;
        t = lock(threads[index]);
        thread_index = index;
        search_count += 1;
        continue :outer;
    }
}
```

### 2.4 Free path

```zig
fn free(memory, alignment) void {
    class = sizeClassIndex(memory.len, alignment);
    if (class >= size_class_count) return PageAllocator.unmap(memory);  // large

    const node: *usize = @ptrCast(memory.ptr);
    const t = Thread.lock();
    defer t.unlock();
    node.* = t.frees[class];          // <-- intrusive store into the freed block
    t.frees[class] = @intFromPtr(node);
}
```

**This is the entire mechanism of interest.** `free` writes one word into the
first 8 bytes of the block being freed. If that cache line is cold, the store
triggers a read-for-ownership: the CPU fetches a full 64-byte line from memory
in order to write 8 bytes of it. Freeing N cold blocks costs N such fetches,
and their cost depends entirely on the address pattern.

Small-class slabs are **never unmapped**. Memory returns to the OS only at
process exit.

### 2.5 Slot rotation happens single-threaded

Important and non-obvious. With `max_alloc_search = 1`, on the first pass
through `alloc` where both the freelist is empty and the bump pointer is at a
slab boundary, `search_count` is 0, so the allocator takes branch 4: it
unlocks the current slot, advances `thread_index`, locks a different slot, and
retries. Only on the second pass does it map.

Consequence: **a single thread migrates across all `cpu_count` slots**, each
with an independent bump pointer and freelist per class. On a 10-core M4 that
is up to 10 interleaved bump streams for one class in a single-threaded
program. On a 1-vCPU machine `(index + 1) % 1 == 0` and this degenerates to a
single stream — which is precisely why the test VM cannot exercise it.

This gives a mechanism for address scatter that requires no OS-specific
behaviour at all, and it should be ruled in or out before pursuing the mmap
hint theory.

### 2.6 Backing store and the address hint chain

SmpAllocator has no memory source of its own; both paths call
`std.heap.PageAllocator`. The relevant part of `PageAllocator.map`:

```zig
const page_aligned_len = alignForward(n, page_size);
const max_drop_len     = alignment_bytes -| page_size;
const overalloc_len    = page_aligned_len + max_drop_len;

hint = switch (stack_direction) {
    .down => ((prev_hint - page_aligned_len) & ~(alignment_bytes - 1)) - max_drop_len,
    .up   => (prev_hint + (alignment_bytes - 1)) & ~(alignment_bytes - 1),
};

slice = mmap(hint, overalloc_len, RW, PRIVATE|ANONYMOUS);
result_ptr = alignPointer(slice.ptr, alignment_bytes);
munmap(head slack);
if (tail slack) munmap(tail slack);

new_hint = result_ptr + (if .down then 0 else page_aligned_len);
cmpxchgStrong(&addr_hint, prev_hint, new_hint, .monotonic, .monotonic);
```

`stack_direction` is `.down` on every arch except hppa, so both Linux and
macOS use the downward chain. `enable_hints` is true everywhere except OpenBSD.

For a 64 KiB slab with 64 KiB alignment on 16 KiB pages:

```
page_aligned_len = 65536
max_drop_len     = 65536 - 16384 = 49152
overalloc_len    = 114688                    // 112 KiB requested
hint             = prev - 65536 - 49152 = prev - 114688
```

If the kernel returns exactly the hinted address, `alignPointer` rounds
`prev - 114688` up to `prev - 65536`, the head trim is exactly 49152, the tail
trim is zero, and the new slab abuts the previous one. One `mmap` + one
`munmap` per slab, perfectly contiguous descending. Measured on Linux: span
equal to footprint to within rounding (ratio 1.00x).

**The entire scheme is load-bearing on the kernel honouring the hint
verbatim.** Linux's top-down allocator has an explicit fast path: non-zero
hint, no `MAP_FIXED`, range free and in bounds returns exactly that address
with no search. Darwin routes a non-`MAP_FIXED` hint through `vm_map_enter` as
an upward gap search using the address as a floor. When the range is free the
two agree; when it isn't they diverge, and Mach takes the first gap at or
above the floor — which in a downward-walking chain is often inside territory
already carved up by the trim `munmap`s.

The chain has no memory. After a miss, `addr_hint` becomes wherever you
actually landed. On Linux one miss is self-healing. If misses are systematic
(the walk repeatedly hits the dyld shared cache, libmalloc's reserved zones,
or your own trim holes) the chain never re-establishes.

**Status: unverified hypothesis.** No Darwin host was available.

### 2.7 The hint chain is process-global

`addr_hint` is a single global shared by every `PageAllocator` consumer in the
process. Any unrelated page-level allocation interleaved with slab mapping
perturbs it. An early version of the harness that allocated a 128 MiB eviction
buffer partway through the measurement produced span ratios of 1.55x–4.07x on
Linux; moving that allocation out of the interleaved path restored 1.00x.

**This is a testable alternative hypothesis that does not require any Darwin
misbehaviour.** If the real workload interleaves other page-level allocations
between container allocations, the chain breaks on every platform.

---

## 3. Upstream status

Commit history of `lib/std/heap/SmpAllocator.zig` (18 commits total).

0.16.0 is tag `24fdd5b7a`, 2026-04-13.

**Since 0.16.0 — one commit, no semantic change:**

| commit | date | subject |
|---|---|---|
| `69a26976e` | 2026-07-17 | `zig fmt` — `@intFromEnum` → `@backingInt` mechanical rewrite |

**During the 0.16.0 cycle (shipped in 0.16.0):**

| commit | date | subject |
|---|---|---|
| `255aeb57b` | 2026-02-01 | introduce `std.atomic.Mutex`, use it in `heap.SmpAllocator` |
| `5c59a4623` | 2026-02-11 | `PageAllocator`: fix alignments > page size ignored in remap/resize |
| `674416021` | 2026-02-10 | zig libc: implement malloc (SmpAllocator-backed; musl mallocng deleted) |

**Original implementation, Feb 2025** — `51c4ffa41` through `42dbd35d3`. The
last of these, *"back to simple free"*, reverted an experiment
(`1754e014f`, *"rotate on free sometimes"*) that rotated slots on free. Nobody
has touched the free path since.

**Relevant open issue:** #36269 (2026-07-22) — `std.heap.SafeAllocator` and
`std.heap.SmpAllocator` spin without bound on locks. Different failure mode
from this one, but worth tracking.

---

## 4. Measurements

### 4.1 Test environment

```
Intel Xeon @ 2.80GHz (family 6, model 85, stepping 7 — Cascade Lake), KVM guest
1 vCPU        <-- significant limitation, see 2.5
L1d 32 KiB / L2 1 MiB / L3 33 MiB
4 KiB pages, no THP tuning applied
4 GiB RAM
Linux 6.18.5, Zig 0.16.0, -O ReleaseFast, -lc
```

Harness conventions used throughout:

- Pointer array allocated from `page_allocator`, outside the allocator under test.
- Every block written once after allocation to fault it in.
- L3 flushed between the alloc phase and the timed loop by streaming a
  128 MiB buffer (except where noted).
- Timing via `clock_gettime(MONOTONIC)`. `std.time.Timer` does not exist in
  Zig 0.16 — timing moved behind the `Io` interface.
- Best-of-N reported, N between 7 and 15.

### 4.2 Free-order sensitivity by object size

128 MiB footprint per measurement. `T-` columns are `touch` (a single volatile
store into each block, no allocator call — the pure locality probe). `F-`
columns are `free`. All values ns per object.

```
    obj       n obj/pg  slots | T-alloc  T-sort  T-shuf |  F-sort  F-shuf |  libc-F | shuf/sort
                                                                            touch   free
     64 2097152  64.00   1024 |   7.65    7.62    16.58 |   13.87   42.05 |   65.22 | 2.18x  3.03x
    256  524288  16.00    256 |  17.00   16.75    16.99 |   30.45   38.49 |  182.87 | 1.01x  1.26x
   1024  131072   4.00     64 |  14.24   14.54    16.72 |   36.80   38.63 |  127.82 | 1.15x  1.05x
   2048   65536   2.00     32 |  16.50   15.96    16.45 |   37.87   38.35 |  120.18 | 1.03x  1.01x
   4096   32768   1.00     16 |  16.85   17.13    20.14 |   41.85   42.30 |  107.06 | 1.18x  1.01x
   8192   16384   0.50      8 |  17.33   17.80    24.27 |   41.50   44.90 |  110.79 | 1.36x  1.08x
  16384    8192   0.25      4 |  20.82   19.24    28.15 |   38.92   47.18 |  107.92 | 1.46x  1.21x
  32768    4096   0.13      2 |  24.82   24.46    38.92 |   37.64   58.58 |  151.55 | 1.59x  1.56x
```

Span/footprint ratio measured 1.00x at every size — the hint chain held.

Findings:

1. **At 8192 bytes the free-order effect is 1.08x.** Not the reported pathology.
2. The effect is large at 64 bytes (3.03x) and reappears at 32768 (1.56x). It
   is **U-shaped, not monotonic in objects-per-page.** An earlier hypothesis
   that objects-per-page alone predicts the effect is not supported by this
   data; the large-object rise is unexplained (candidate: slab count and TLB
   coverage per slab, not investigated).
3. **glibc free is 2.5x–4x slower than smp free at every size.** Whatever is
   happening on macOS is a property of Apple's libmalloc, not of libc in
   general. Do not generalise from glibc.
4. A plausible partial mechanism for the 8 KiB trough on x86: the stride
   already exceeds a 4 KiB page, and Intel's L2 streamer does not prefetch
   across page boundaries, so sorted order buys nothing. On 16 KiB pages
   8192 bytes is 2 objects per page and the prefetcher can still work — which
   would make sorted order valuable on the M4 and worthless on x86. This is
   consistent with the observation but does not explain the 32768 datapoint.

### 4.3 Cost of sorting vs benefit of sorting

No eviction pass in this measurement, so absolute free numbers run lower than
in 4.2. Ratios are what matter. `obj = 8192`, ns per object.

```
      n | free-shuf free-sort    delta |       pdq     radix
   4096 |     10.34     10.51    -0.18 |     80.38      9.63
  20480 |     23.50     23.72    -0.23 |     86.98     14.68
```

- Benefit of sorting on this platform: **-0.23 ns/op**. Zero, slightly negative.
- `std.sort.pdq` on `[]u8` slices: **86.98 ns/op**. At n=20480 that is 1.78 ms
  of sorting against a 0.48 ms free loop — a 3.7x regression on teardown.
- 3-pass LSD radix on the same slices: 14.68 ns/op, roughly 6x cheaper than
  pdq. Comparison sort on 16-byte elements with an indirect comparator is a
  bad fit; radix wins easily.

**Benchmark design warning:** measuring sorted-teardown against
scattered-teardown as a single number hides this completely. Sort cost and
free delta must be separate columns.

### 4.4 Metric and reorder costs

`obj = 8192`, `n = 20480`, ns per object. Free loop baseline 53.36.

```
  free loop (baseline)               : 53.36
  travel scan (full)                 :  0.58
  inversion scan (full)              :  0.42
  inversion scan (1-in-64 sampled)   :  0.01
  1-pass 256-bucket partition        :  6.01   <-- ineffective, see below
  1-pass 4096-bucket, span-adaptive  : 11.26
  3-pass radix on []u8 slices        : 14.68
  3-pass radix on u32 indices        : 20.87   <-- worse; gather permute costs more
                                                   than it saves
  std.sort.pdq                       : 86.98
```

**The 256-bucket result is a trap worth recording.** Bucketing on fixed
pointer bits [16:24) gives 256 buckets covering 16 MiB of address range. A
160 MiB span aliases through them ~10 times, and travel after the partition
was 6856x — completely unchanged from shuffled. The bucket shift must be
derived from the measured span:

```zig
const span  = hi - lo + 1;
const shift = @max(0, 64 - @clz(span) - log2(nbuckets));
```

With 4096 span-adaptive buckets, travel dropped from 6857x to **3.65x**.

### 4.5 Does the reorder actually help?

`obj = 8192`, `n = 20480`, with eviction, ns per object.

```
  free, shuffled              : 56.46
  free, after 1-pass bucket   : 54.54
  free, after full pdq sort   : 56.26
  1-pass bucket cost          : 11.26
```

On this platform, **no**. The reorder is inside noise, and it costs 11.26
ns/op to obtain. This is the central negative result of the investigation.

### 4.6 Presortedness metrics compared

`obj = 8192`, `n = 20480`. Travel normalised by footprint.

```
sequence                           travel   inversions
fresh bump (round 1)                2.75x        12.5%
shuffled                         6857.39x        49.6%
after 256-bucket (aliased)       6856.56x        49.6%
LIFO reuse after sorted free     6856.56x        50.4%
```

The last row is the important one. See §5.2.

### 4.7 Early exit

Threshold 10x, budget checked once per 64-element chunk.

```
  scattered input: verdict=true   exited after     65/20480 elems    108 ns total
  ordered input:   verdict=false  scanned      20480/20480 elems    8999 ns total
```

---

## 5. Idea A — adaptive reorder before bulk free

### 5.1 Shape

At the point of bulk teardown, measure how scattered the pointer sequence is.
If it exceeds a threshold, reorder it into approximate address order and then
free. Otherwise free as-is.

The gate must be cheap enough that paying it on every teardown is negligible,
and the reorder must be cheap enough that paying it when triggered is a net
win. §4.3–4.5 give the budget: the free loop is ~53 ns/op, so a gate costing
<1 ns/op is free, and a reorder costing ~11 ns/op needs to save more than 11
ns/op to be worth it.

### 5.2 Metric selection

Three candidates were considered.

**(a) Inversion count / `Runs`.** Count adjacent descents:

```zig
var inv: u32 = 0;
var prev: usize = 0;
for (items) |p| {
    const a = @intFromPtr(p.ptr);
    inv += @intFromBool(a < prev);
    prev = a;
}
```

This is Knuth's `Runs` measure minus one — a standard presortedness statistic.
Cost 0.42 ns/op.

**Rejected.** It measures sortedness, and sortedness is not what the memory
system cares about. A strictly descending sequence has perfect locality and
scores 100% inverted.

This is not hypothetical. SmpAllocator's freelist is LIFO, so sort-then-free
pushes blocks on ascending and pops them descending. Row 4 of §4.6 shows a
post-sorted-free reuse sequence scoring 50.4% inversions and 6856x travel —
the inversion counter would flag an already-optimal sequence as disordered on
every round, forever. **Any direction-sensitive metric is unusable in a system
with a LIFO freelist.**

**(b) Travel / total variation.** Sum absolute distance between consecutive
addresses:

```zig
fn travel(items: []const []u8) u64 {
    var t: u64 = 0;
    var prev = @intFromPtr(items[0].ptr);
    for (items[1..]) |p| {
        const a = @intFromPtr(p.ptr);
        t += if (a > prev) a - prev else prev - a;
        prev = a;
    }
    return t;
}
// ratio = t / (items.len * obj_size)
```

Cost 0.58 ns/op. Direction-agnostic. Normalised so that 1.0 is a clean linear
sweep in either direction, and a uniform random permutation over a span of S
object-widths gives roughly S/3.

Dynamic range on real data: 2.75x (fresh bump) to 6857x (shuffled). Three
orders of magnitude of separation.

The closest named relative in the literature is `Osc`, the oscillation measure
of Levcopoulos and Petersson, but `Osc` counts line-plot crossings and travel
sums edge magnitudes, so they are not the same function.

**Selected.**

**(c) Page-locality fraction.** Count the fraction of consecutive pairs
landing in the same page:

```zig
local += @intFromBool((a ^ prev) >> page_shift == 0);
```

Same cost as travel. Arguably a closer proxy for the actual hardware
behaviour — "what fraction of my frees hit a page I just touched" — and
immune to travel's outlier sensitivity (a clean sweep with fifty huge jumps
scores badly under travel despite 99.8% of accesses being local).

**Not selected, but should be measured alongside travel on the target
platform.** Whichever tracks the timing better wins. This is an open question
that the M4 data will settle.

### 5.3 Early exit

Travel accumulates monotonically, so partial travel can be compared against
the **full-sequence budget**:

```zig
const budget = threshold * items.len * obj_size;
// ... accumulate t ...
if (t > budget) return .scattered;
```

Soundness: if partial travel already exceeds what the whole sequence was
allowed, the final ratio must exceed the threshold, since every remaining term
is non-negative. No false positives. It is a one-sided test — you can conclude
"definitely scattered" early but never "definitely ordered" early, because a
clean prefix says nothing about the tail.

Self-calibrating: steps average roughly `span/3`, so the number of elements
needed to blow a `threshold × n × obj` budget is approximately
`threshold × n / ratio`. At ratio 6857 and threshold 10 that is ~30 elements.

Check the budget once per chunk (64 or 256 elements) rather than per element,
so the inner loop stays branch-free and vectorizable. Exit is delayed by at
most one chunk. Measured: exit at element 65 of 20480 with a 64-element chunk.

Resulting cost asymmetry — the desirable shape:

| case | gate cost | then |
|---|---|---|
| scattered | 108 ns (0.005 ns/op) | pay the reorder you wanted |
| ordered | 8999 ns (0.44 ns/op) | skip the reorder entirely |

The full scan is only ever paid in the case where it saves you 230 µs–1.8 ms
of pointless reordering. 9 µs against a ~1.1 ms teardown is 0.8%.

### 5.4 Reorder mechanism

Never `std.sort.pdq`. You do not need total order — you need consecutive frees
to land on the same page or slab, which a counting partition provides at a
fraction of the cost and in O(n).

```zig
const nbuckets = 4096;

fn bucket(items: [][]u8, scratch: [][]u8, counts: []u32, offs: []u32) void {
    // pass 1: span
    var lo: usize = maxInt(usize);
    var hi: usize = 0;
    for (items) |p| { const v = @intFromPtr(p.ptr); lo = @min(lo, v); hi = @max(hi, v); }

    const span = hi - lo + 1;
    const shift: u6 = @intCast(@max(0, 64 - @clz(span) - 12));  // 12 = log2(4096)

    // pass 2: histogram
    @memset(counts, 0);
    for (items) |p| counts[(@intFromPtr(p.ptr) - lo) >> shift] += 1;

    // prefix sum
    var sum: u32 = 0;
    for (counts, offs) |c, *o| { o.* = sum; sum += c; }

    // pass 3: scatter
    for (items) |p| {
        const b = (@intFromPtr(p.ptr) - lo) >> shift;
        scratch[offs[b]] = p;
        offs[b] += 1;
    }
    @memcpy(items, scratch);
}
```

Cost 11.26 ns/op at n=20480. Reduces travel from 6857x to 3.65x. Requires
`n` slots of scratch plus 2 × `nbuckets` × 4 bytes of counters (32 KiB at
4096 buckets — sizing choice trades L1 residency against bucket resolution).

The alternative, 3-pass LSD radix on pointer bits [12:36), costs 14.68 ns/op
and produces true sorted order. Marginally more expensive, no measured benefit
over the partition. Radix on u32 indices with a final gather permute is
*worse* (20.87) — the permute costs more than the narrower elements save.

### 5.5 Fused variant

The separate gate pass can be eliminated: free while scanning, and if the
budget blows at element k, stop, reorder `items[k..]`, and free the rest.

- Good case: zero extra passes over the array.
- Bad case: k frees performed in the wrong order, where k ≈ 65.

Saves 9 µs per teardown at the cost of meaningful complexity. Only worth it if
teardown is genuinely hot.

### 5.6 Sampling variants

To make the ordered case cheaper than 0.44 ns/op:

**Strided sampling (every 64th pointer) — hazardous.** Costs 0.01 ns/op. But
a sequence that is locally scrambled within 64-element windows while globally
ascending samples as perfectly ordered, and is exactly as bad for the
prefetcher as a full shuffle. Local structure is what the hardware cares
about; strided sampling is blind to it.

**Windowed sampling — preferred.** Take ~16 contiguous runs of 64 elements
spread through the array, accumulate travel within each run, normalise per
run. Roughly 1024 elements examined, ~0.03 ns/op amortised, and it measures
local structure. Untested.

### 5.7 Threshold calibration

The threshold is the one parameter that cannot be derived from theory. 10x is
a placeholder chosen because it sits three orders of magnitude below the
shuffled ratio and well above the ~2.75x a clean bump-allocated sweep
produces.

Calibration procedure, to be run on each target platform:

1. Generate pointer sequences at controlled travel ratios (interpolate between
   sorted and shuffled by shuffling within windows of varying size).
2. For each ratio, measure free-loop time with and without the reorder.
3. Find the crossover ratio where `reorder_cost + free_reordered <
   free_as_is`. That is the threshold.
4. If no crossover exists at any ratio, the reorder never pays on that
   platform and the gate should be compiled out.

Step 4 is the outcome observed on x86_64-linux at 8192 bytes.

### 5.8 Test specification sketch

**Correctness**

| id | property |
|---|---|
| A-C1 | Reorder is a permutation: same multiset of pointers in, out. No duplicates, no drops. Verify with a sorted-copy comparison. |
| A-C2 | Every pointer is freed exactly once regardless of gate outcome. Verify with a shadow bitmap. |
| A-C3 | `travel()` returns 0 for `n == 1`; does not read `items[0]` for `n == 0`. |
| A-C4 | Budget arithmetic does not overflow: `threshold * n * obj_size` in u64 with n=10^7, obj=32768, threshold=1000. |
| A-C5 | `shift` computation is correct for span 1, span = 2^63, and span < nbuckets. |
| A-C6 | Bucket partition with `nbuckets` larger than `n` produces correct output. |
| A-C7 | Aliasing regression: partition on a 160 MiB span must reduce travel. Assert post-partition travel below some bound; this catches the fixed-shift bug from §4.4. |

**Behavioural**

| id | property |
|---|---|
| A-B1 | Sorted-ascending input: gate returns "ordered". |
| A-B2 | Sorted-descending input: gate returns "ordered". This is the LIFO trap; an inversion-based gate fails it. |
| A-B3 | Uniform shuffle over a large span: gate returns "scattered", exits within 2 chunks. |
| A-B4 | Locally-shuffled-within-page but globally ascending: travel should stay low; confirms the metric does not over-trigger. |
| A-B5 | Globally-ascending with k huge jumps: documents travel's outlier sensitivity. Compare against page-locality metric. |
| A-B6 | Gate is idempotent: reorder, then re-gate, must return "ordered". |

**Performance regression**

| id | property |
|---|---|
| A-P1 | Gate cost on ordered input < 2% of free-loop cost. |
| A-P2 | Gate cost on scattered input < 0.05% of free-loop cost. |
| A-P3 | End-to-end teardown with the gate is never slower than without it by more than the gate cost, on any platform in CI. |
| A-P4 | Reorder cost scales linearly in n across 10^3 .. 10^6. |

**Platform matrix**

Minimum: aarch64-macos (16 KiB pages), x86_64-linux (4 KiB), aarch64-linux
(4 KiB and 16 KiB kernels if available). The aarch64-linux 4 KiB case
separates "is it ARM" from "is it 16 KiB pages" from "is it Darwin" — a
three-way discrimination none of the current data can make.

### 5.9 Risks and open questions

- **The payoff is unmeasured.** Every configuration tested shows zero benefit.
- Travel vs page-locality has not been compared against actual timing.
- The 32768-byte rise in §4.2 is unexplained and may indicate the model is
  wrong.
- Single-vCPU test machine could not exercise slot rotation (§2.5).
- Thresholds are platform-specific and will need re-calibration as hardware
  changes — a maintenance liability for a library.
- If the root cause turns out to be the hint chain (§2.6) or interleaved
  PageAllocator use (§2.7), this entire design treats a symptom.

---

## 6. Idea B — per-owner fixed-size pool

### 6.1 Motivation

Idea A is a per-teardown patch with a platform-dependent payoff. This is a
structural change that makes free order stop being a concept.

The key observation: 8192-byte bitset containers are **exactly one fixed
size**. No size classes, no fragmentation, no coalescing, no split/merge. That
is the ideal case for the cheapest possible allocator design.

### 6.2 Design

A pool owned by each bitmap, backed by the caller's `Allocator`:

- Pool requests **chunks** from the caller's allocator. A chunk is
  `C * 8192` bytes plus a header, aligned to its own size so that
  `chunk_base = ptr & ~(chunk_size - 1)`.
- Chunk header holds: an occupancy bitmap (`C` bits), a live count, and a
  link into the pool's partial-chunk list.
- `alloc`: find a chunk with a free slot, `@ctz` the inverted bitmap, set the
  bit, increment count, return `base + slot * 8192`.
- `free`: mask the pointer to find the chunk, compute
  `slot = (ptr - base) >> 13`, clear the bit, decrement count. **The
  container payload is never touched.**
- When a chunk's count hits zero, either release it to the caller's allocator
  or retain one spare chunk to avoid thrash.
- Teardown: release all chunks. `n/C` allocator calls instead of `n`.

### 6.3 Why this eliminates the problem

`free` touches only the chunk header. Freeing N objects scattered across K
chunks touches K cache lines, not N. At C=256, freeing 20000 objects in
random order touches 79 header lines instead of 20000 cold payload lines.

Free order becomes irrelevant **structurally**, not conditionally. It holds on
Darwin, on x86, and on hardware that does not exist yet. No thresholds, no
calibration, no platform matrix.

This is the same out-of-band-metadata principle that musl's mallocng and
several other modern allocators use, scoped to the one case where it is
trivially correct.

### 6.4 Secondary wins

- Construction drops N allocator round-trips to N/C.
- Container payloads within a chunk are contiguous by construction, which
  helps iteration and not just teardown.
- The pool is immune to whatever the underlying allocator does with addresses,
  including the hint-chain question entirely.

### 6.5 Freeable resolution — what actually changes

Logical and physical free decouple.

**Reuse resolution stays per-container.** A freed slot is immediately
available to the next promotion within the same pool. During mutation
(array→bitset conversion, container drops) the caller's allocator is never
touched. This is strictly better than per-container malloc/free.

**Release-to-allocator resolution becomes per-chunk.** A chunk can only go
back when every slot in it is free. One live container pins the whole chunk.

This only matters in one workload shape: a long-lived bitmap that sheds most
of its bitsets, never re-adds any, and needs the RSS back before destruction.
At teardown everything is released regardless, so chunk granularity costs
nothing there. Build-once-query-many and churny-but-steady-size workloads are
unaffected.

Worst-case retention is `occupancy × chunk_size`. C is the dial:

| C | chunk size | allocator calls for 20k | worst-case bloat |
|---|---|---|---|
| 8 | 64 KiB | 2500 | 8x |
| 32 | 256 KiB | 625 | 32x |
| 256 | 2 MiB | 79 | 256x |

### 6.6 Compaction

A general-purpose allocator cannot compact because it does not know who holds
pointers into its blocks. **This pool can, because there is exactly one
reference to each container** — the owner's container array.

```
relocate(container):
    dst = find_free_slot_in_a_fuller_chunk()
    memcpy(dst, src, 8192)
    owner_array[index_of(container)] = dst
    clear_bit(src)
```

8 KiB copy plus one pointer store. That converts the fragmentation objection
from a hard constraint into a policy question.

Suggested trigger: on a bulk-remove path, if live slots drop below 50% of
allocated slots *and* allocated size exceeds some absolute floor, compact.
Amortised by only checking on operations that can shrink the pool.

Compaction can be added later without changing the interface. Do not build it
first.

### 6.7 Constraints and hazards

**Container transfer between owners.** If any operation moves a container from
one bitmap into another by transferring pointer ownership — some in-place
union implementations do this — a per-owner pool forbids it. The container
lives in owner A's chunk and cannot be re-homed without an 8 KiB copy. **This
is the constraint most likely to kill the design; audit for it before
proceeding.**

**Only fixed-size containers qualify.** Variable-size containers stay on the
caller's allocator and retain whatever free-order behaviour they have. The
pool does not make the problem disappear globally, only for the type that
dominates teardown by byte count.

**Pointer provenance.** Chunk-base-by-masking requires the chunk allocation to
be aligned to its own size. A caller's allocator can provide that via an
alignment request, but over-aligned allocation may itself trigger the
over-map-and-trim path in §2.6. Alternative: store the chunk pointer in a
header immediately preceding each slot (8 bytes overhead per container,
1/1024 of payload) and skip the alignment requirement. Measure both.

**Allocator failure mid-chunk.** A chunk allocation failure must not leave the
pool in a state where previously allocated containers are unreachable.

### 6.8 Test specification sketch

**Correctness**

| id | property |
|---|---|
| B-C1 | Alloc/free of every slot in a chunk, in every permutation for small C, leaves the bitmap consistent. |
| B-C2 | Chunk is released exactly when the last slot is freed, never earlier. |
| B-C3 | `chunk_base(ptr)` is correct for the first and last slot of a chunk. |
| B-C4 | Slot index round-trips: `slot_of(ptr_of(slot)) == slot` for all slots. |
| B-C5 | Double-free of a slot is detected (bit already clear) under a debug build. |
| B-C6 | Allocator failure on chunk request propagates as `error.OutOfMemory` with no leak and no corruption. |
| B-C7 | Pool teardown releases exactly the chunks it allocated — verify against a counting allocator wrapper. |
| B-C8 | Compaction preserves container contents byte-for-byte and updates every owner reference. |
| B-C9 | Compaction is a no-op when already compact. |

**Behavioural**

| id | property |
|---|---|
| B-B1 | Allocator call count for constructing and destroying n containers is `⌈n/C⌉ × 2` within a small constant, not `n × 2`. |
| B-B2 | Free in random order costs the same as free in address order, within noise. **This is the whole point of the design; it is the primary acceptance test.** |
| B-B3 | Peak RSS after shedding 90% of containers without compaction matches the predicted worst-case bloat. |
| B-B4 | Same scenario with compaction enabled returns RSS to within a small factor of live footprint. |

**Performance**

| id | property |
|---|---|
| B-P1 | Teardown of 20k containers beats the per-container-free baseline on every platform in the matrix. |
| B-P2 | Per-container alloc/free in steady-state mutation is no slower than the baseline allocator's. |
| B-P3 | Chunk size sweep (C = 8, 32, 64, 256) measured for teardown time and worst-case bloat, to pick a default. |

---

## 7. Prior art

Searched, not exhaustively.

**Address-ordering free operations for cache reasons** is documented, but on
the allocator's side rather than the caller's. US patent 6539464 ("Memory
allocator for multithread environment") describes occasionally sorting free
block lists by increasing or decreasing block start address, on the reasoning
that this improves locality of memory references and reduces cache misses in
both application code and the allocator's own routines. Mark-sweep collectors
sweeping the heap in address order are the same physical insight again.
Address-ordered free lists go back to Knuth's first-fit analysis, though there
the motivation is coalescing and fragmentation rather than cache.

**Presortedness measures** are standard. Knuth's `Runs` is the adjacent-descent
count used in §5.2(a). `Osc`, the oscillation measure, was introduced by
Levcopoulos and Petersson for adaptive heapsort. Estivill-Castro and Wood's
survey places roughly a dozen such measures in a partial order by dominance.
Branching on a cheap presortedness statistic is exactly what Timsort and
pdqsort already do — applied to comparison count rather than cache misses.

**Out-of-band allocator metadata** is the basis of musl's mallocng and several
other modern allocators.

The specific combination in Idea A — caller-side, adaptive, early-exiting,
partition-rather-than-sort, targeting the prefetcher rather than the
comparison count — was not found under any name. That is weak evidence; two
searches is not a literature review, and this is plausibly folklore in game
engines and database engines without a paper attached.

---

## 8. Recommended order of work

1. **Run the reproducer on the M4.** Everything downstream depends on numbers
   that do not exist yet. Record: the shuf/sort ratio for `touch` and `free`
   separately, and the span/footprint ratio.
2. **Discriminate the mechanism.** Three hypotheses, all testable:
   - span >> 1.0 → hint chain broken (§2.6), or interleaved PageAllocator use
     (§2.7). Test single-threaded with no other page-level allocation.
   - span ≈ 1.0 but touch is order-sensitive → pure page-size/prefetcher
     effect. Idea A is the right treatment.
   - touch order-insensitive but free order-sensitive → the intrusive freelist
     is the cost. Idea B is the right treatment.
3. **Check the libc column.** If Apple's free is order-insensitive while smp's
   is not, that confirms out-of-band metadata as the differentiator and points
   straight at Idea B.
4. **Audit for container ownership transfer** (§6.7) before committing to
   Idea B.
5. Only then build. Idea B if it is viable — it is the structural fix and it
   wins on teardown by construction rather than by threshold. Idea A only if
   B is blocked by ownership transfer, and only with the calibration procedure
   in §5.7 producing a real crossover.

---

## Appendix — reproducer

Cross-compiles clean to aarch64-macos. Build:
`zig build-exe smp_free_order.zig -O ReleaseFast -lc`

```zig
const std = @import("std");
const builtin = @import("builtin");

const Order = enum { alloc_order, sorted, shuffled };
const Which = enum { smp, libc };
const Mode = enum { touch, free };

const rounds = 9;
const footprint: usize = 128 << 20;
const obj_sizes = [_]usize{ 64, 256, 1024, 2048, 4096, 8192, 16384, 32768 };

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn ltPtr(_: void, a: []u8, b: []u8) bool {
    return @intFromPtr(a.ptr) < @intFromPtr(b.ptr);
}

var evict_buf: []u8 = &.{};
fn evict() void {
    if (evict_buf.len == 0) {
        evict_buf = std.heap.page_allocator.alloc(u8, 128 << 20) catch return;
        @memset(evict_buf, 1);
    }
    var i: usize = 0;
    var acc: usize = 0;
    while (i < evict_buf.len) : (i += 64) acc +%= evict_buf[i];
    std.mem.doNotOptimizeAway(acc);
}

fn span(ptrs: []const []u8) f64 {
    var lo: usize = std.math.maxInt(usize);
    var hi: usize = 0;
    for (ptrs) |p| {
        lo = @min(lo, @intFromPtr(p.ptr));
        hi = @max(hi, @intFromPtr(p.ptr) + p.len);
    }
    return @as(f64, @floatFromInt(hi - lo)) /
        @as(f64, @floatFromInt(ptrs.len * ptrs[0].len));
}

fn run(w: Which, n: usize, obj: usize, order: Order, mode: Mode, span_out: ?*f64) !f64 {
    const a: std.mem.Allocator = switch (w) {
        .smp => std.heap.smp_allocator,
        .libc => std.heap.c_allocator,
    };
    const ptrs = try std.heap.page_allocator.alloc([]u8, n);
    defer std.heap.page_allocator.free(ptrs);

    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const rand = prng.random();

    var best: u64 = std.math.maxInt(u64);
    for (0..rounds) |r| {
        for (ptrs) |*p| {
            p.* = try a.alloc(u8, obj);
            p.*[0] = 1;
        }
        if (r == 0) if (span_out) |s| {
            s.* = span(ptrs);
        };

        switch (order) {
            .alloc_order => {},
            .sorted => std.sort.pdq([]u8, ptrs, {}, ltPtr),
            .shuffled => rand.shuffle([]u8, ptrs),
        }

        evict();

        const t0 = nowNs();
        switch (mode) {
            .touch => for (ptrs) |p| {
                @as(*volatile usize, @ptrCast(@alignCast(p.ptr))).* = 0;
            },
            .free => for (ptrs) |p| a.free(p),
        }
        const dt = nowNs() - t0;

        if (mode == .touch) for (ptrs) |p| a.free(p);
        best = @min(best, dt);
    }
    return @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(n));
}

pub fn main() !void {
    const page = std.heap.pageSize();
    std.debug.print(
        \\target      = {s}-{s}
        \\page size   = {d} B   (std.heap.page_size_max = {d})
        \\smp slab    = {d} B
        \\footprint   = {d} MiB per measurement
        \\
        \\
    , .{
        @tagName(builtin.cpu.arch), @tagName(builtin.os.tag),
        page,                       std.heap.page_size_max,
        @max(std.heap.page_size_max, 64 * 1024),
        footprint >> 20,
    });

    std.debug.print(
        "{s:>7} {s:>7} {s:>6} {s:>6} | {s:>8} {s:>8} {s:>8} | {s:>7} {s:>7} | {s:>8}\n",
        .{ "obj", "n", "obj/pg", "slots", "T-alloc", "T-sort", "T-shuf", "F-sort", "F-shuf", "libc-F" },
    );

    for (obj_sizes) |obj| {
        const n = footprint / obj;
        var sp: f64 = 0;
        const ta = try run(.smp, n, obj, .alloc_order, .touch, &sp);
        const ts = try run(.smp, n, obj, .sorted, .touch, null);
        const th = try run(.smp, n, obj, .shuffled, .touch, null);
        const fs = try run(.smp, n, obj, .sorted, .free, null);
        const fh = try run(.smp, n, obj, .shuffled, .free, null);
        const lf = try run(.libc, n, obj, .shuffled, .free, null);

        const slab = @max(std.heap.page_size_max, 64 * 1024);
        std.debug.print(
            "{d:>7} {d:>7} {d:>6.2} {d:>6} | {d:>8.2} {d:>8.2} {d:>8.2} | {d:>7.2} {d:>7.2} | {d:>8.2}   span={d:.2}x  shuf/sort: touch {d:.2}x free {d:.2}x\n",
            .{
                obj,   n,  @as(f64, @floatFromInt(page)) / @as(f64, @floatFromInt(obj)),
                slab / obj, ta, ts, th, fs, fh, lf, sp, th / ts, fh / fs,
            },
        );
    }
}
```

### Interpreting the output

- **`touch` shuf/sort large, `free` shuf/sort similar** → pure cache/TLB
  locality. Idea A treats it.
- **`touch` shuf/sort ≈ 1.0, `free` shuf/sort large** → the intrusive freelist
  is the cost. Idea B treats it.
- **`span` >> 1.0** → hint chain broken; that is an upstream bug and the
  proximate cause.
- **`libc-F` order-insensitive while smp is not** → confirms out-of-band
  metadata as the differentiator.
