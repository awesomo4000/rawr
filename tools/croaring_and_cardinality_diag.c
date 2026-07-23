// SPDX-License-Identifier: MPL-2.0

#define RAWR_CR_AND_CARD_INTERNAL 1
#include "croaring_and_cardinality_diag.h"

// Compile the unmodified amalgamation into this benchmark-only translation unit
// so the probes can call internal container routines directly.
#include "../vendor/roaring.c"

#if defined(_MSC_VER)
#define RAWR_CR_NOINLINE __declspec(noinline)
#else
#define RAWR_CR_NOINLINE __attribute__((noinline))
#endif

RAWR_CR_NOINLINE int32_t rawr_cr_and_card_gallop(
    const uint16_t *small,
    size_t small_len,
    const uint16_t *large,
    size_t large_len
) {
    return intersect_skewed_uint16_cardinality(
        small,
        small_len,
        large,
        large_len
    );
}

RAWR_CR_NOINLINE int32_t rawr_cr_and_card_dispatch(
    const uint16_t *first,
    size_t first_len,
    const uint16_t *second,
    size_t second_len
) {
    array_container_t first_container = {
        .cardinality = (int32_t)first_len,
        .capacity = (int32_t)first_len,
        .array = (uint16_t *)first,
    };
    array_container_t second_container = {
        .cardinality = (int32_t)second_len,
        .capacity = (int32_t)second_len,
        .array = (uint16_t *)second,
    };
    return array_container_intersection_cardinality(
        &first_container,
        &second_container
    );
}

bool rawr_cr_and_card_all_arrays(
    const roaring_bitmap_t *bitmap,
    size_t expected_containers,
    int32_t expected_cardinality
) {
    if (bitmap->high_low_container.size != (int32_t)expected_containers) {
        return false;
    }
    for (int32_t i = 0; i < bitmap->high_low_container.size; i++) {
        uint8_t type = bitmap->high_low_container.typecodes[i];
        const container_t *container = container_unwrap_shared(
            bitmap->high_low_container.containers[i],
            &type
        );
        if (type != ARRAY_CONTAINER_TYPE) return false;
        const array_container_t *array = (const array_container_t *)container;
        if (array->cardinality != expected_cardinality) return false;
    }
    return true;
}
