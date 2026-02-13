// Minimal CRoaring wrapper for Zig @cImport
// Only exposes the portable serialization API we need for interop testing.

#ifndef CROARING_WRAPPER_H
#define CROARING_WRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Opaque bitmap type
typedef struct roaring_bitmap_s roaring_bitmap_t;

// Creation and destruction
roaring_bitmap_t *roaring_bitmap_create(void);
roaring_bitmap_t *roaring_bitmap_copy(const roaring_bitmap_t *r);
void roaring_bitmap_free(const roaring_bitmap_t *r);

// Basic operations
void roaring_bitmap_add(roaring_bitmap_t *r, uint32_t x);
bool roaring_bitmap_add_checked(roaring_bitmap_t *r, uint32_t x);
void roaring_bitmap_add_range(roaring_bitmap_t *r, uint64_t min, uint64_t max);
bool roaring_bitmap_contains(const roaring_bitmap_t *r, uint32_t x);
uint64_t roaring_bitmap_get_cardinality(const roaring_bitmap_t *r);
bool roaring_bitmap_is_empty(const roaring_bitmap_t *r);

// Set operations
roaring_bitmap_t *roaring_bitmap_and(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_or(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_xor(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
roaring_bitmap_t *roaring_bitmap_andnot(const roaring_bitmap_t *r1, const roaring_bitmap_t *r2);

// In-place set operations
void roaring_bitmap_and_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_or_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_xor_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);
void roaring_bitmap_andnot_inplace(roaring_bitmap_t *r1, const roaring_bitmap_t *r2);

// Optimization
bool roaring_bitmap_run_optimize(roaring_bitmap_t *r);

// Portable serialization (RoaringFormatSpec)
size_t roaring_bitmap_portable_size_in_bytes(const roaring_bitmap_t *r);
size_t roaring_bitmap_portable_serialize(const roaring_bitmap_t *r, char *buf);
roaring_bitmap_t *roaring_bitmap_portable_deserialize_safe(const char *buf, size_t maxbytes);

// Iteration callback
typedef bool (*roaring_iterator)(uint32_t value, void *param);
bool roaring_iterate(const roaring_bitmap_t *r, roaring_iterator iterator, void *ptr);

#endif // CROARING_WRAPPER_H
