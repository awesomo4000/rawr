# Spec 10-15: `Roaring64Bitmap` bulk ops with locality context

Phase-2 parity chunk of [64-bit Roaring](10-roaring64.md). Throughput variants of
add/remove/contains that reuse a cached bucket location across calls, for
monotonic or clustered access patterns.

## Features

| rawr 64-bit (new) | CRoaring | Semantics |
|---|---|---|
| `addBulk(ctx, value) !void` | `roaring64_bitmap_add_bulk` | add, reusing `ctx`'s cached bucket |
| `containsBulk(ctx, value) bool` | `roaring64_bitmap_contains_bulk` | contains, reusing `ctx` |
| `removeBulk(ctx, value) !void` | `roaring64_bitmap_remove_bulk` | remove, reusing `ctx` |
| `BulkContext` | `roaring64_bulk_context_t` | caches the last-touched bucket (`hi` + index) |

## Context type + invalidation contract (pin this down)

CRoaring's `roaring64_bulk_context_t` caches the last container and is
**invalidated by any non-bulk mutation** (a plain `add`/`remove`/set-op between
bulk calls leaves the cached pointer dangling — undefined behavior in CRoaring).

rawr should choose the **safer** of two contracts and state it explicitly:

- **(preferred) validating context** — `BulkContext` stores the cached `hi` key
  **plus a structure-version counter**, not a raw bucket pointer. Each bulk call
  checks the version; if the bitmap mutated since, it falls back to a normal
  bucket lookup and re-caches. This makes interleaving bulk and non-bulk calls
  safe (just slower on the miss), avoiding CRoaring's dangling-pointer footgun.
- (alternative) match CRoaring — cache the index, document "invalidated by any
  non-bulk mutation," UB otherwise. Only choose this if the version check is
  measurably too costly (unlikely — it's a `u32` compare).

**Recommend the validating context.** Never cache a raw `*Bucket` pointer — the
bucket array reallocs on growth, so a pointer would dangle even across bulk calls;
cache the `hi` + index and re-validate against `size`/version.

## Wrapper decls

```c
typedef struct roaring64_bulk_context_s { /* opaque to us */ uint8_t high_bytes[8]; void *leaf; } roaring64_bulk_context_t;
void roaring64_bitmap_add_bulk(roaring64_bitmap_t *r, roaring64_bulk_context_t *ctx, uint64_t val);
bool roaring64_bitmap_contains_bulk(const roaring64_bitmap_t *r, roaring64_bulk_context_t *ctx, uint64_t val);
void roaring64_bitmap_remove_bulk(roaring64_bitmap_t *r, roaring64_bulk_context_t *ctx, uint64_t val);
```

> Confirm the exact `roaring64_bulk_context_t` layout against `vendor/roaring.h`
> when adding the wrapper decl; the oracle only needs a zero-initialized context
> passed by pointer.

## Tests / oracle

- Inline: monotonic ascending `addBulk` run (same-key clustering hits the cache);
  interleave a plain `add`/`remove` between bulk calls and assert **correct
  results** (the validating context must not corrupt or dangle); `containsBulk`
  after `addBulk`; `removeBulk` prunes emptied buckets.
- `difftest64`: run a bulk sequence on rawr with `BulkContext` and the same
  sequence on CRoaring with a zeroed `roaring64_bulk_context_t`; assert
  `assertAgreement`. Only feed CRoaring's context bulk calls (respect its
  invalidation rule on the oracle side).

## Acceptance

- `BulkContext` caches `hi` + index/version, never a raw bucket pointer; the
  invalidation contract is documented and (preferred) made safe via version check.
- Bulk ops agree with the CRoaring `*_bulk` functions in `difftest64`.
- Green; no 32-bit regression.
