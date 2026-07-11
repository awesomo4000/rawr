// Minimal CRoaring wrapper for Zig translate-c bindings.
// Only exposes the portable serialization API we need for interop testing.

#ifndef CROARING_WRAPPER_H
#define CROARING_WRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Opaque bitmap type
typedef struct roaring_bitmap_s roaring_bitmap_t;
typedef struct roaring64_bitmap_s roaring64_bitmap_t;

// Creation and destruction
roaring_bitmap_t *roaring_bitmap_create(void);
roaring_bitmap_t *roaring_bitmap_copy(const roaring_bitmap_t *r);
void roaring_bitmap_free(const roaring_bitmap_t *r);

// Basic operations
void roaring_bitmap_add(roaring_bitmap_t *r, uint32_t x);
bool roaring_bitmap_add_checked(roaring_bitmap_t *r, uint32_t x);
void roaring_bitmap_add_many(roaring_bitmap_t *r, size_t n_args, const uint32_t *vals);
void roaring_bitmap_add_range(roaring_bitmap_t *r, uint64_t min, uint64_t max);
void roaring_bitmap_remove_many(roaring_bitmap_t *r, size_t n_args, const uint32_t *vals);
bool roaring_bitmap_contains(const roaring_bitmap_t *r, uint32_t x);
uint64_t roaring_bitmap_get_cardinality(const roaring_bitmap_t *r);
bool roaring_bitmap_is_empty(const roaring_bitmap_t *r);
uint32_t roaring_bitmap_minimum(const roaring_bitmap_t *r);
uint32_t roaring_bitmap_maximum(const roaring_bitmap_t *r);
void roaring_bitmap_to_uint32_array(const roaring_bitmap_t *r, uint32_t *ans);
void roaring_bitmap_remove_range_closed(roaring_bitmap_t *r, uint32_t lo, uint32_t hi);
uint64_t roaring_bitmap_range_cardinality_closed(const roaring_bitmap_t *r, uint32_t lo, uint32_t hi);
bool roaring_bitmap_contains_range_closed(const roaring_bitmap_t *r, uint32_t lo, uint32_t hi);

// Set operations
roaring_bitmap_t *roaring_bitmap_and(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_or(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_xor(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_andnot(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_flip_closed(const roaring_bitmap_t *r, uint32_t lo, uint32_t hi);
roaring_bitmap_t *roaring_bitmap_or_many(size_t number, const roaring_bitmap_t **rs);
roaring_bitmap_t *roaring_bitmap_or_many_heap(uint32_t number, const roaring_bitmap_t **rs);
roaring_bitmap_t *roaring_bitmap_xor_many(size_t number, const roaring_bitmap_t **rs);
roaring_bitmap_t *roaring_bitmap_lazy_or(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2, bool bitsetconversion);
void roaring_bitmap_lazy_or_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2, bool bitsetconversion);
roaring_bitmap_t *roaring_bitmap_lazy_xor(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_lazy_xor_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_repair_after_lazy(roaring_bitmap_t *r);
uint64_t roaring_bitmap_and_cardinality(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
uint64_t roaring_bitmap_or_cardinality(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
uint64_t roaring_bitmap_xor_cardinality(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
uint64_t roaring_bitmap_andnot_cardinality(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
double roaring_bitmap_jaccard_index(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
bool roaring_bitmap_intersect(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
bool roaring_bitmap_intersect_with_range(const roaring_bitmap_t *r, uint64_t x, uint64_t y);
bool roaring_bitmap_equals(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
bool roaring_bitmap_is_subset(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
bool roaring_bitmap_is_strict_subset(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
uint64_t roaring_bitmap_rank(const roaring_bitmap_t *r, uint32_t x);
void roaring_bitmap_rank_many(const roaring_bitmap_t *r, const uint32_t *begin, const uint32_t *end, uint64_t *ans);
bool roaring_bitmap_select(const roaring_bitmap_t *r, uint32_t rank, uint32_t *element);
int64_t roaring_bitmap_get_index(const roaring_bitmap_t *r, uint32_t x);

// In-place set operations
void roaring_bitmap_and_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_or_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_xor_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_andnot_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_flip_inplace_closed(roaring_bitmap_t *r, uint32_t lo, uint32_t hi);

// Optimization
bool roaring_bitmap_run_optimize(roaring_bitmap_t *r);

// Portable serialization (RoaringFormatSpec)
size_t roaring_bitmap_portable_size_in_bytes(const roaring_bitmap_t *r);
size_t roaring_bitmap_portable_serialize(const roaring_bitmap_t *r, char *buf);
roaring_bitmap_t *roaring_bitmap_portable_deserialize_safe(const char *buf, size_t maxbytes);

// Iteration callback
typedef bool (*roaring_iterator)(uint32_t value, void *param);
bool roaring_iterate(const roaring_bitmap_t *r, roaring_iterator iterator, void *ptr);

// 64-bit bitmap lifecycle / identity operations
roaring64_bitmap_t *roaring64_bitmap_create(void);
void roaring64_bitmap_free(roaring64_bitmap_t *r);
roaring64_bitmap_t *roaring64_bitmap_copy(const roaring64_bitmap_t *r);
void roaring64_bitmap_add(roaring64_bitmap_t *r, uint64_t x);
void roaring64_bitmap_add_many(roaring64_bitmap_t *r, size_t n, const uint64_t *vals);
bool roaring64_bitmap_remove_checked(roaring64_bitmap_t *r, uint64_t x);
bool roaring64_bitmap_contains(const roaring64_bitmap_t *r, uint64_t x);
uint64_t roaring64_bitmap_get_cardinality(const roaring64_bitmap_t *r);
bool roaring64_bitmap_is_empty(const roaring64_bitmap_t *r);
uint64_t roaring64_bitmap_minimum(const roaring64_bitmap_t *r);
uint64_t roaring64_bitmap_maximum(const roaring64_bitmap_t *r);
void roaring64_bitmap_to_uint64_array(const roaring64_bitmap_t *r, uint64_t *out);
bool roaring64_bitmap_equals(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
roaring64_bitmap_t *roaring64_bitmap_and(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
roaring64_bitmap_t *roaring64_bitmap_or(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
roaring64_bitmap_t *roaring64_bitmap_xor(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
roaring64_bitmap_t *roaring64_bitmap_andnot(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
void roaring64_bitmap_and_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
void roaring64_bitmap_or_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
void roaring64_bitmap_xor_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
void roaring64_bitmap_andnot_inplace(roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
uint64_t roaring64_bitmap_and_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
uint64_t roaring64_bitmap_or_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
uint64_t roaring64_bitmap_xor_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
uint64_t roaring64_bitmap_andnot_cardinality(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
bool roaring64_bitmap_intersect(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
bool roaring64_bitmap_is_subset(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
bool roaring64_bitmap_is_strict_subset(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
double roaring64_bitmap_jaccard_index(const roaring64_bitmap_t *r1, const roaring64_bitmap_t *r2);
uint64_t roaring64_bitmap_rank(const roaring64_bitmap_t *r, uint64_t x);
bool roaring64_bitmap_select(const roaring64_bitmap_t *r, uint64_t rank, uint64_t *element);
bool roaring64_bitmap_get_index(const roaring64_bitmap_t *r, uint64_t x, uint64_t *out_index);
void roaring64_bitmap_add_range_closed(roaring64_bitmap_t *r, uint64_t min, uint64_t max);
void roaring64_bitmap_remove_range_closed(roaring64_bitmap_t *r, uint64_t min, uint64_t max);
uint64_t roaring64_bitmap_range_closed_cardinality(const roaring64_bitmap_t *r, uint64_t min, uint64_t max);
bool roaring64_bitmap_contains_range(const roaring64_bitmap_t *r, uint64_t min, uint64_t max);
bool roaring64_bitmap_intersect_with_range(const roaring64_bitmap_t *r, uint64_t min, uint64_t max);
roaring64_bitmap_t *roaring64_bitmap_flip_closed(const roaring64_bitmap_t *r, uint64_t min, uint64_t max);
void roaring64_bitmap_flip_closed_inplace(roaring64_bitmap_t *r, uint64_t min, uint64_t max);
bool roaring64_bitmap_run_optimize(roaring64_bitmap_t *r);
size_t roaring64_bitmap_shrink_to_fit(roaring64_bitmap_t *r);
void roaring64_bitmap_clear(roaring64_bitmap_t *r);
size_t roaring64_bitmap_portable_size_in_bytes(const roaring64_bitmap_t *r);
size_t roaring64_bitmap_portable_serialize(const roaring64_bitmap_t *r, char *buf);
roaring64_bitmap_t *roaring64_bitmap_portable_deserialize_safe(const char *buf, size_t maxbytes);

#endif // CROARING_WRAPPER_H
