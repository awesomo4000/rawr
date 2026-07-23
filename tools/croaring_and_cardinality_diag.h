// SPDX-License-Identifier: MPL-2.0

#ifndef CROARING_AND_CARDINALITY_DIAG_H
#define CROARING_AND_CARDINALITY_DIAG_H

#if defined(RAWR_CR_AND_CARD_INTERNAL)
#include "../vendor/roaring.h"
#else
#include "croaring_wrapper.h"
#endif

int32_t rawr_cr_and_card_gallop(
    const uint16_t *small,
    size_t small_len,
    const uint16_t *large,
    size_t large_len
);

int32_t rawr_cr_and_card_dispatch(
    const uint16_t *first,
    size_t first_len,
    const uint16_t *second,
    size_t second_len
);

bool rawr_cr_and_card_all_arrays(
    const roaring_bitmap_t *bitmap,
    size_t expected_containers,
    int32_t expected_cardinality
);

#endif
