// SPDX-License-Identifier: MPL-2.0

#ifndef CROARING_ITERATE_DIAG_H
#define CROARING_ITERATE_DIAG_H

#if defined(RAWR_CR_ITERATE_INTERNAL)
#include "../vendor/roaring.h"
#else
#include "croaring_wrapper.h"
#endif

typedef struct rawr_cr_iterate_result {
    uint64_t count;
    uint64_t sum;
} rawr_cr_iterate_result;

typedef struct rawr_cr_container_counts {
    uint32_t arrays;
    uint32_t bitsets;
    uint32_t runs;
} rawr_cr_container_counts;

rawr_cr_iterate_result rawr_cr_iterate_pull(const roaring_bitmap_t *bitmap);
rawr_cr_iterate_result rawr_cr_iterate_push(const roaring_bitmap_t *bitmap);

size_t rawr_cr_iterate_pull_values(
    const roaring_bitmap_t *bitmap,
    uint32_t *output,
    size_t capacity
);

size_t rawr_cr_iterate_push_values(
    const roaring_bitmap_t *bitmap,
    uint32_t *output,
    size_t capacity
);

rawr_cr_container_counts rawr_cr_iterate_container_counts(
    const roaring_bitmap_t *bitmap
);

#endif
