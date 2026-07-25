// SPDX-License-Identifier: MPL-2.0

#define RAWR_CR_SELECT_INTERNAL 1
#include "croaring_select_diag.h"

rawr_cr_select_result rawr_cr_select_loop(
    const roaring_bitmap_t *bitmap,
    const uint32_t *queries,
    size_t query_count
) {
    rawr_cr_select_result result = {0, 0};
    for (size_t i = 0; i < query_count; ++i) {
        uint32_t value;
        if (roaring_bitmap_select(bitmap, queries[i], &value)) {
            result.count++;
            result.sum += value;
        }
    }
    return result;
}

rawr_cr_select_container_counts rawr_cr_select_counts(
    const roaring_bitmap_t *bitmap
) {
    roaring_statistics_t stats;
    roaring_bitmap_statistics(bitmap, &stats);
    rawr_cr_select_container_counts result = {
        stats.n_array_containers,
        stats.n_bitset_containers,
        stats.n_run_containers,
    };
    return result;
}
