// SPDX-License-Identifier: MPL-2.0

#define RAWR_CR_RANGE_ATTRIB_INTERNAL 1
#include "croaring_range_attrib.h"

rawr_cr_clone_inventory rawr_cr_range_clone_inventory(
    const roaring_bitmap_t *bitmap
) {
    rawr_cr_clone_inventory result = {0};
    const roaring_array_t *array = &bitmap->high_low_container;

    result.containers = (uint64_t)array->size;
    // roaring_bitmap_copy allocates the bitmap and one combined top-level array.
    result.clone_allocations = 2;
    result.clone_requested_bytes = sizeof(roaring_bitmap_t) +
        (size_t)array->size *
            (sizeof(container_t *) + sizeof(uint16_t) + sizeof(uint8_t));
    result.copied_bytes =
        (size_t)array->size * (sizeof(uint16_t) + sizeof(uint8_t));

    for (int32_t i = 0; i < array->size; ++i) {
        switch (array->typecodes[i]) {
            case ARRAY_CONTAINER_TYPE: {
                const array_container_t *container =
                    const_CAST_array(array->containers[i]);
                result.arrays++;
                result.clone_allocations += 2;
                result.clone_requested_bytes += sizeof(*container) +
                    (size_t)container->capacity * sizeof(uint16_t);
                result.copied_bytes +=
                    (size_t)container->cardinality * sizeof(uint16_t);
                break;
            }
            case BITSET_CONTAINER_TYPE: {
                const bitset_container_t *container =
                    const_CAST_bitset(array->containers[i]);
                result.bitsets++;
                result.clone_allocations += 2;
                result.clone_requested_bytes += sizeof(*container) +
                    BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t);
                result.copied_bytes +=
                    BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t);
                break;
            }
            case RUN_CONTAINER_TYPE: {
                const run_container_t *container =
                    const_CAST_run(array->containers[i]);
                result.runs++;
                result.clone_allocations += 2;
                result.clone_requested_bytes += sizeof(*container) +
                    (size_t)container->capacity * sizeof(rle16_t);
                result.copied_bytes +=
                    (size_t)container->n_runs * sizeof(rle16_t);
                break;
            }
            default:
                break;
        }
    }

    return result;
}
