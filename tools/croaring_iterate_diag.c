// SPDX-License-Identifier: MPL-2.0

#define RAWR_CR_ITERATE_INTERNAL 1
#include "croaring_iterate_diag.h"

typedef struct rawr_cr_push_sum_context {
    rawr_cr_iterate_result result;
} rawr_cr_push_sum_context;

static bool rawr_cr_push_sum(uint32_t value, void *opaque) {
    rawr_cr_push_sum_context *context = opaque;
    context->result.count++;
    context->result.sum += value;
    return true;
}

rawr_cr_iterate_result rawr_cr_iterate_pull(const roaring_bitmap_t *bitmap) {
    rawr_cr_iterate_result result = {0, 0};
    roaring_uint32_iterator_t iterator;
    roaring_iterator_init(bitmap, &iterator);
    while (iterator.has_value) {
        result.count++;
        result.sum += iterator.current_value;
        roaring_uint32_iterator_advance(&iterator);
    }
    return result;
}

rawr_cr_iterate_result rawr_cr_iterate_push(const roaring_bitmap_t *bitmap) {
    rawr_cr_push_sum_context context = {{0, 0}};
    roaring_iterate(bitmap, rawr_cr_push_sum, &context);
    return context.result;
}

size_t rawr_cr_iterate_pull_values(
    const roaring_bitmap_t *bitmap,
    uint32_t *output,
    size_t capacity
) {
    size_t count = 0;
    roaring_uint32_iterator_t iterator;
    roaring_iterator_init(bitmap, &iterator);
    while (iterator.has_value) {
        if (count >= capacity) return SIZE_MAX;
        output[count++] = iterator.current_value;
        roaring_uint32_iterator_advance(&iterator);
    }
    return count;
}

typedef struct rawr_cr_push_values_context {
    uint32_t *output;
    size_t capacity;
    size_t count;
    bool overflow;
} rawr_cr_push_values_context;

static bool rawr_cr_push_value(uint32_t value, void *opaque) {
    rawr_cr_push_values_context *context = opaque;
    if (context->count >= context->capacity) {
        context->overflow = true;
        return false;
    }
    context->output[context->count++] = value;
    return true;
}

size_t rawr_cr_iterate_push_values(
    const roaring_bitmap_t *bitmap,
    uint32_t *output,
    size_t capacity
) {
    rawr_cr_push_values_context context = {output, capacity, 0, false};
    roaring_iterate(bitmap, rawr_cr_push_value, &context);
    return context.overflow ? SIZE_MAX : context.count;
}

rawr_cr_container_counts rawr_cr_iterate_container_counts(
    const roaring_bitmap_t *bitmap
) {
    roaring_statistics_t stats;
    roaring_bitmap_statistics(bitmap, &stats);
    rawr_cr_container_counts counts = {
        stats.n_array_containers,
        stats.n_bitset_containers,
        stats.n_run_containers,
    };
    return counts;
}
