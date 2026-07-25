// SPDX-License-Identifier: MPL-2.0

#ifndef CROARING_SELECT_DIAG_H
#define CROARING_SELECT_DIAG_H

#if defined(RAWR_CR_SELECT_INTERNAL)
#include "../vendor/roaring.h"
#else
#include "croaring_wrapper.h"
#endif

typedef struct rawr_cr_select_result {
    uint64_t count;
    uint64_t sum;
} rawr_cr_select_result;

typedef struct rawr_cr_select_container_counts {
    uint32_t arrays;
    uint32_t bitsets;
    uint32_t runs;
} rawr_cr_select_container_counts;

rawr_cr_select_result rawr_cr_select_loop(
    const roaring_bitmap_t *bitmap,
    const uint32_t *queries,
    size_t query_count
);

rawr_cr_select_container_counts rawr_cr_select_counts(
    const roaring_bitmap_t *bitmap
);

#endif
