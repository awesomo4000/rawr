// SPDX-License-Identifier: MPL-2.0

#ifndef CROARING_LAZY_ATTRIBUTION_H
#define CROARING_LAZY_ATTRIBUTION_H

#if defined(RAWR_CR_ATTR_INTERNAL)
#include "../vendor/roaring.h"
#else
#include "croaring_wrapper.h"
#endif

typedef struct rawr_cr_attr_context rawr_cr_attr_context;

typedef struct rawr_cr_attr_counts {
    size_t left_keys;
    size_t right_keys;
    size_t shared_keys;
    size_t left_only_keys;
    size_t right_only_keys;
    size_t non_array_shared_keys;
    size_t bitsets_created;
    size_t bytes_cleared;
} rawr_cr_attr_counts;

typedef struct rawr_cr_attr_materialization_counts {
    size_t before_array;
    size_t before_bitset;
    size_t before_run;
    size_t after_array;
    size_t after_bitset;
    size_t after_run;
} rawr_cr_attr_materialization_counts;

rawr_cr_attr_context *rawr_cr_attr_context_create(
    const roaring_bitmap_t *left,
    const roaring_bitmap_t *right
);
void rawr_cr_attr_context_free(rawr_cr_attr_context *context);
rawr_cr_attr_counts rawr_cr_attr_get_counts(const rawr_cr_attr_context *context);
bool rawr_cr_attr_get_materialization_counts(
    const rawr_cr_attr_context *context,
    rawr_cr_attr_materialization_counts *counts
);

bool rawr_cr_attr_alloc_headers(rawr_cr_attr_context *context);
void rawr_cr_attr_free_headers(rawr_cr_attr_context *context);
bool rawr_cr_attr_alloc_words(rawr_cr_attr_context *context);
void rawr_cr_attr_free_words(rawr_cr_attr_context *context);
bool rawr_cr_attr_create_bitsets(rawr_cr_attr_context *context);
void rawr_cr_attr_free_bitsets(rawr_cr_attr_context *context);
bool rawr_cr_attr_shared_pipeline(rawr_cr_attr_context *context);
void rawr_cr_attr_dirty_words(rawr_cr_attr_context *context);
void rawr_cr_attr_zero_words(rawr_cr_attr_context *context);
void rawr_cr_attr_accumulate_first(rawr_cr_attr_context *context);
void rawr_cr_attr_accumulate_second(rawr_cr_attr_context *context);
bool rawr_cr_attr_clone_unique(rawr_cr_attr_context *context);
void rawr_cr_attr_free_clones(rawr_cr_attr_context *context);
uint64_t rawr_cr_attr_merge_append(rawr_cr_attr_context *context);

void rawr_cr_attr_zero_probe(uint64_t *words);
void rawr_cr_attr_accumulate_probe(
    uint64_t *words,
    const uint16_t *values,
    size_t count
);
void rawr_cr_attr_call_zig_probes(
    uint64_t *words,
    const uint16_t *values,
    size_t count
);

#endif
