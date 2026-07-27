// SPDX-License-Identifier: MPL-2.0

#ifndef CROARING_RANGE_ATTRIB_H
#define CROARING_RANGE_ATTRIB_H

#if defined(RAWR_CR_RANGE_ATTRIB_INTERNAL)
#include "../vendor/roaring.h"
#else
#include "croaring_wrapper.h"
#endif

typedef struct rawr_cr_clone_inventory {
    uint64_t containers;
    uint64_t arrays;
    uint64_t bitsets;
    uint64_t runs;
    uint64_t clone_allocations;
    uint64_t clone_requested_bytes;
    uint64_t copied_bytes;
} rawr_cr_clone_inventory;

rawr_cr_clone_inventory rawr_cr_range_clone_inventory(
    const roaring_bitmap_t *bitmap
);

#endif
