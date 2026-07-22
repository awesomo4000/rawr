// SPDX-License-Identifier: MPL-2.0

#define RAWR_CR_ATTR_INTERNAL 1
#include "croaring_lazy_attribution.h"

// Compile the unmodified amalgamation into this benchmark-only translation unit
// so the probes below can call its internal container routines directly.
#include "../vendor/roaring.c"

typedef struct rawr_cr_attr_pair {
    const array_container_t *first;
    const array_container_t *second;
} rawr_cr_attr_pair;

typedef struct rawr_cr_attr_unique {
    const container_t *container;
    uint8_t type;
} rawr_cr_attr_unique;

#if defined(_MSC_VER)
#define RAWR_CR_NOINLINE __declspec(noinline)
#else
#define RAWR_CR_NOINLINE __attribute__((noinline))
#endif

struct rawr_cr_attr_context {
    const roaring_bitmap_t *left;
    const roaring_bitmap_t *right;
    rawr_cr_attr_counts counts;
    rawr_cr_attr_pair *shared;
    rawr_cr_attr_unique *unique;
    bitset_container_t **bitsets;
    container_t **clones;
    uint16_t *output_keys;
    container_t **output_containers;
    uint8_t *output_types;
};

static size_t rawr_cr_attr_bitset_alignment(void) {
#if CROARING_IS_X64
    return (croaring_hardware_support() & ROARING_SUPPORTS_AVX512) ? 64 : 32;
#else
    return 32;
#endif
}

static void rawr_cr_attr_release_partial(rawr_cr_attr_context *context) {
    if (context == NULL) return;
    free(context->shared);
    free(context->unique);
    free(context->bitsets);
    free(context->clones);
    free(context->output_keys);
    free(context->output_containers);
    free(context->output_types);
    free(context);
}

rawr_cr_attr_context *rawr_cr_attr_context_create(
    const roaring_bitmap_t *left,
    const roaring_bitmap_t *right
) {
    rawr_cr_attr_context *context = calloc(1, sizeof(*context));
    if (context == NULL) return NULL;
    context->left = left;
    context->right = right;
    context->counts.left_keys = (size_t)left->high_low_container.size;
    context->counts.right_keys = (size_t)right->high_low_container.size;

    const size_t max_shared = context->counts.left_keys < context->counts.right_keys
        ? context->counts.left_keys
        : context->counts.right_keys;
    const size_t max_unique = context->counts.left_keys + context->counts.right_keys;
    context->shared = malloc(max_shared * sizeof(*context->shared));
    context->unique = malloc(max_unique * sizeof(*context->unique));
    context->bitsets = calloc(max_shared, sizeof(*context->bitsets));
    context->clones = calloc(max_unique, sizeof(*context->clones));
    context->output_keys = malloc(max_unique * sizeof(*context->output_keys));
    context->output_containers = malloc(max_unique * sizeof(*context->output_containers));
    context->output_types = malloc(max_unique * sizeof(*context->output_types));
    if ((max_shared != 0 && (context->shared == NULL || context->bitsets == NULL)) ||
        (max_unique != 0 && (context->unique == NULL || context->clones == NULL ||
                            context->output_keys == NULL || context->output_containers == NULL ||
                            context->output_types == NULL))) {
        rawr_cr_attr_release_partial(context);
        return NULL;
    }

    size_t i = 0;
    size_t j = 0;
    size_t shared_index = 0;
    size_t unique_index = 0;
    while (i < context->counts.left_keys && j < context->counts.right_keys) {
        const uint16_t left_key = left->high_low_container.keys[i];
        const uint16_t right_key = right->high_low_container.keys[j];
        if (left_key < right_key) {
            context->unique[unique_index++] = (rawr_cr_attr_unique){
                .container = left->high_low_container.containers[i],
                .type = left->high_low_container.typecodes[i],
            };
            context->counts.left_only_keys++;
            i++;
        } else if (left_key > right_key) {
            context->unique[unique_index++] = (rawr_cr_attr_unique){
                .container = right->high_low_container.containers[j],
                .type = right->high_low_container.typecodes[j],
            };
            context->counts.right_only_keys++;
            j++;
        } else {
            uint8_t left_type = left->high_low_container.typecodes[i];
            uint8_t right_type = right->high_low_container.typecodes[j];
            const container_t *left_container = container_unwrap_shared(
                left->high_low_container.containers[i],
                &left_type
            );
            const container_t *right_container = container_unwrap_shared(
                right->high_low_container.containers[j],
                &right_type
            );
            if (left_type != ARRAY_CONTAINER_TYPE || right_type != ARRAY_CONTAINER_TYPE) {
                context->counts.non_array_shared_keys++;
            } else {
                context->shared[shared_index++] = (rawr_cr_attr_pair){
                    .first = (const array_container_t *)left_container,
                    .second = (const array_container_t *)right_container,
                };
            }
            context->counts.shared_keys++;
            i++;
            j++;
        }
    }
    while (i < context->counts.left_keys) {
        context->unique[unique_index++] = (rawr_cr_attr_unique){
            .container = left->high_low_container.containers[i],
            .type = left->high_low_container.typecodes[i],
        };
        context->counts.left_only_keys++;
        i++;
    }
    while (j < context->counts.right_keys) {
        context->unique[unique_index++] = (rawr_cr_attr_unique){
            .container = right->high_low_container.containers[j],
            .type = right->high_low_container.typecodes[j],
        };
        context->counts.right_only_keys++;
        j++;
    }

    context->counts.bitsets_created = context->counts.shared_keys;
    context->counts.bytes_cleared = context->counts.shared_keys *
        BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t);
    return context;
}

void rawr_cr_attr_context_free(rawr_cr_attr_context *context) {
    if (context == NULL) return;
    rawr_cr_attr_free_clones(context);
    rawr_cr_attr_free_bitsets(context);
    rawr_cr_attr_release_partial(context);
}

rawr_cr_attr_counts rawr_cr_attr_get_counts(const rawr_cr_attr_context *context) {
    return context->counts;
}

bool rawr_cr_attr_alloc_headers(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        context->bitsets[i] = roaring_malloc(sizeof(bitset_container_t));
        if (context->bitsets[i] == NULL) return false;
        context->bitsets[i]->words = NULL;
        context->bitsets[i]->cardinality = 0;
    }
    return true;
}

void rawr_cr_attr_free_headers(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        if (context->bitsets[i] != NULL) {
            roaring_free(context->bitsets[i]);
            context->bitsets[i] = NULL;
        }
    }
}

bool rawr_cr_attr_alloc_words(rawr_cr_attr_context *context) {
    const size_t alignment = rawr_cr_attr_bitset_alignment();
    const size_t bytes = BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t);
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        context->bitsets[i]->words = roaring_aligned_malloc(alignment, bytes);
        if (context->bitsets[i]->words == NULL) return false;
    }
    return true;
}

void rawr_cr_attr_free_words(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        if (context->bitsets[i] != NULL && context->bitsets[i]->words != NULL) {
            roaring_aligned_free(context->bitsets[i]->words);
            context->bitsets[i]->words = NULL;
        }
    }
}

bool rawr_cr_attr_create_bitsets(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        context->bitsets[i] = bitset_container_create();
        if (context->bitsets[i] == NULL) return false;
    }
    return true;
}

void rawr_cr_attr_free_bitsets(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        if (context->bitsets[i] != NULL) {
            bitset_container_free(context->bitsets[i]);
            context->bitsets[i] = NULL;
        }
    }
}

bool rawr_cr_attr_shared_pipeline(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        bitset_container_t *bitset = bitset_container_from_array(context->shared[i].first);
        if (bitset == NULL) return false;
        context->bitsets[i] = bitset;
        bitset_set_list(
            bitset->words,
            context->shared[i].second->array,
            (size_t)context->shared[i].second->cardinality
        );
        bitset->cardinality = BITSET_UNKNOWN_CARDINALITY;
    }
    return true;
}

void rawr_cr_attr_dirty_words(rawr_cr_attr_context *context) {
    const size_t bytes = BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t);
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        memset(context->bitsets[i]->words, 0xA5, bytes);
        context->bitsets[i]->cardinality = 0;
    }
}

RAWR_CR_NOINLINE void rawr_cr_attr_zero_probe(uint64_t *words) {
    memset(words, 0, BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t));
}

void rawr_cr_attr_zero_words(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        rawr_cr_attr_zero_probe(context->bitsets[i]->words);
        context->bitsets[i]->cardinality = 0;
    }
}

RAWR_CR_NOINLINE void rawr_cr_attr_accumulate_probe(
    uint64_t *words,
    const uint16_t *values,
    size_t count
) {
    bitset_set_list(words, values, count);
}

extern void rawr_lazy_attr_zero_probe(uint64_t *words);
extern void rawr_lazy_attr_accumulate_probe(
    uint64_t *words,
    const uint16_t *values,
    size_t count
);

void rawr_cr_attr_call_zig_probes(
    uint64_t *words,
    const uint16_t *values,
    size_t count
) {
    rawr_lazy_attr_zero_probe(words);
    rawr_lazy_attr_accumulate_probe(words, values, count);
}

void rawr_cr_attr_accumulate_first(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        const array_container_t *source = context->shared[i].first;
        for (int32_t j = 0; j < source->cardinality; j++) {
            bitset_container_set(context->bitsets[i], source->array[j]);
        }
    }
}

void rawr_cr_attr_accumulate_second(rawr_cr_attr_context *context) {
    for (size_t i = 0; i < context->counts.shared_keys; i++) {
        const array_container_t *source = context->shared[i].second;
        rawr_cr_attr_accumulate_probe(
            context->bitsets[i]->words,
            source->array,
            (size_t)source->cardinality
        );
        context->bitsets[i]->cardinality = BITSET_UNKNOWN_CARDINALITY;
    }
}

bool rawr_cr_attr_clone_unique(rawr_cr_attr_context *context) {
    const size_t count = context->counts.left_only_keys + context->counts.right_only_keys;
    for (size_t i = 0; i < count; i++) {
        context->clones[i] = container_clone(
            context->unique[i].container,
            context->unique[i].type
        );
        if (context->clones[i] == NULL) return false;
    }
    return true;
}

void rawr_cr_attr_free_clones(rawr_cr_attr_context *context) {
    const size_t count = context->counts.left_only_keys + context->counts.right_only_keys;
    for (size_t i = 0; i < count; i++) {
        if (context->clones[i] != NULL) {
            container_free(context->clones[i], context->unique[i].type);
            context->clones[i] = NULL;
        }
    }
}

uint64_t rawr_cr_attr_merge_append(rawr_cr_attr_context *context) {
    const roaring_array_t *left = &context->left->high_low_container;
    const roaring_array_t *right = &context->right->high_low_container;
    size_t i = 0;
    size_t j = 0;
    size_t out = 0;
    uint64_t checksum = 0;
    while (i < (size_t)left->size && j < (size_t)right->size) {
        const uint16_t left_key = left->keys[i];
        const uint16_t right_key = right->keys[j];
        const roaring_array_t *source;
        size_t source_index;
        if (left_key < right_key) {
            source = left;
            source_index = i++;
        } else if (left_key > right_key) {
            source = right;
            source_index = j++;
        } else {
            source = left;
            source_index = i;
            i++;
            j++;
        }
        context->output_keys[out] = source->keys[source_index];
        context->output_containers[out] = source->containers[source_index];
        context->output_types[out] = source->typecodes[source_index];
        checksum += context->output_keys[out] + context->output_types[out];
        out++;
    }
    while (i < (size_t)left->size) {
        context->output_keys[out] = left->keys[i];
        context->output_containers[out] = left->containers[i];
        context->output_types[out] = left->typecodes[i];
        checksum += context->output_keys[out] + context->output_types[out];
        out++;
        i++;
    }
    while (j < (size_t)right->size) {
        context->output_keys[out] = right->keys[j];
        context->output_containers[out] = right->containers[j];
        context->output_types[out] = right->typecodes[j];
        checksum += context->output_keys[out] + context->output_types[out];
        out++;
        j++;
    }
    return checksum + out;
}
