// SPDX-License-Identifier: MPL-2.0

#include "../vendor/roaring.h"

#define RAWR_CROARING_ARRAY_INTERNAL_BUILD 1
#include "croaring_array_attribution.h"

#include <stddef.h>

enum { RAWR_ARRAY_MAX_CARDINALITY = 4096 };

static uint64_t hash_byte(uint64_t hash, uint8_t byte) {
    return (hash ^ byte) * UINT64_C(1099511628211);
}

static uint64_t hash_u64(uint64_t hash, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash = hash_byte(hash, (uint8_t)(value >> shift));
    }
    return hash;
}

static uint64_t hash_u16(uint64_t hash, uint16_t value) {
    hash = hash_byte(hash, (uint8_t)value);
    return hash_byte(hash, (uint8_t)(value >> 8));
}

static uint64_t hash_result(uint64_t hash, size_t pair_index,
                            const uint16_t *values, size_t count) {
    hash = hash_u64(hash, pair_index);
    hash = hash_u64(hash, count);
    for (size_t index = 0; index < count; ++index) {
        hash = hash_u16(hash, values[index]);
    }
    return hash;
}

bool rawr_cr_array_runtime_has_avx2(void) {
#if CROARING_IS_X64
    return (croaring_hardware_support() & ROARING_SUPPORTS_AVX2) != 0;
#else
    return false;
#endif
}

static size_t run_scalar(const rawr_cr_array_pair_t *pair,
                         rawr_cr_array_operation_t operation,
                         uint16_t *output) {
    if (operation == RAWR_CR_ARRAY_UNION) {
        return union_uint16(pair->left, pair->left_len, pair->right,
                            pair->right_len, output);
    }
    return (size_t)difference_uint16(
        pair->left, (int)pair->left_len, pair->right, (int)pair->right_len,
        output);
}

static size_t run_production(const rawr_cr_array_pair_t *pair,
                             rawr_cr_array_operation_t operation,
                             uint16_t *output, size_t output_capacity,
                             bool *storage_unchanged) {
    if (operation == RAWR_CR_ARRAY_UNION) {
        return fast_union_uint16(pair->left, pair->left_len, pair->right,
                                 pair->right_len, output);
    }

    array_container_t left = {
        .cardinality = (int32_t)pair->left_len,
        .capacity = (int32_t)pair->left_len,
        .array = (uint16_t *)pair->left,
    };
    array_container_t right = {
        .cardinality = (int32_t)pair->right_len,
        .capacity = (int32_t)pair->right_len,
        .array = (uint16_t *)pair->right,
    };
    array_container_t out = {
        .cardinality = 0,
        .capacity = (int32_t)output_capacity,
        .array = output,
    };
    array_container_andnot(&left, &right, &out);
    *storage_unchanged = *storage_unchanged && out.array == output &&
                         out.capacity == (int32_t)output_capacity;
    return (size_t)out.cardinality;
}

static bool run_allocating(const rawr_cr_array_pair_t *pair,
                           rawr_cr_array_operation_t operation,
                           uint16_t **values, size_t *count,
                           uint64_t *allocation_calls,
                           array_container_t **owned) {
    const uint32_t capacity = operation == RAWR_CR_ARRAY_UNION
                                  ? pair->left_len + pair->right_len
                                  : pair->left_len;
    array_container_t *out =
        array_container_create_given_capacity((int32_t)capacity);
    if (out == NULL) return false;
    *allocation_calls += capacity == 0 ? 1 : 2;

    array_container_t left = {
        .cardinality = (int32_t)pair->left_len,
        .capacity = (int32_t)pair->left_len,
        .array = (uint16_t *)pair->left,
    };
    array_container_t right = {
        .cardinality = (int32_t)pair->right_len,
        .capacity = (int32_t)pair->right_len,
        .array = (uint16_t *)pair->right,
    };
    if (operation == RAWR_CR_ARRAY_UNION) {
        array_container_union(&left, &right, out);
    } else {
        array_container_andnot(&left, &right, out);
    }
    *values = out->array;
    *count = (size_t)out->cardinality;
    *owned = out;
    return true;
}

bool rawr_cr_array_attribution_run(const rawr_cr_array_pair_t *pairs,
                                   size_t pair_count,
                                   rawr_cr_array_operation_t operation,
                                   rawr_cr_array_arm_t arm,
                                   bool digest_outputs,
                                   rawr_cr_array_result_t *result) {
    if (result == NULL || (pair_count != 0 && pairs == NULL)) return false;
    if (operation != RAWR_CR_ARRAY_UNION &&
        operation != RAWR_CR_ARRAY_DIFFERENCE) {
        return false;
    }
    if (arm != RAWR_CR_ARRAY_SCALAR && arm != RAWR_CR_ARRAY_PRODUCTION &&
        arm != RAWR_CR_ARRAY_ALLOCATING) {
        return false;
    }

    uint16_t output[RAWR_ARRAY_MAX_CARDINALITY];
    rawr_cr_array_result_t measured = {
        .checksum = 0,
        .digest = UINT64_C(14695981039346656037),
        .pair_count = pair_count,
        .input_elements = 0,
        .allocation_calls = 0,
        .branch = RAWR_CR_ARRAY_BRANCH_NOT_APPLICABLE,
        .outputs_distinct = true,
        .output_storage_unchanged = true,
    };

    if (arm == RAWR_CR_ARRAY_SCALAR) {
        measured.branch = RAWR_CR_ARRAY_BRANCH_SCALAR;
    } else if (arm == RAWR_CR_ARRAY_PRODUCTION) {
        measured.branch = rawr_cr_array_runtime_has_avx2()
                              ? RAWR_CR_ARRAY_BRANCH_AVX2
                              : RAWR_CR_ARRAY_BRANCH_SCALAR;
    }

    for (size_t index = 0; index < pair_count; ++index) {
        const rawr_cr_array_pair_t *pair = &pairs[index];
        const size_t capacity = operation == RAWR_CR_ARRAY_UNION
                                    ? pair->left_len + pair->right_len
                                    : pair->left_len;
        if (capacity > RAWR_ARRAY_MAX_CARDINALITY) return false;
        if (output == pair->left || output == pair->right) {
            measured.outputs_distinct = false;
            return false;
        }

        uint16_t *values = output;
        size_t count = 0;
        array_container_t *owned = NULL;
        if (arm == RAWR_CR_ARRAY_SCALAR) {
            count = run_scalar(pair, operation, output);
        } else if (arm == RAWR_CR_ARRAY_PRODUCTION) {
            count = run_production(pair, operation, output, capacity,
                                   &measured.output_storage_unchanged);
        } else if (!run_allocating(pair, operation, &values, &count,
                                   &measured.allocation_calls, &owned)) {
            return false;
        }

        measured.input_elements += pair->left_len + pair->right_len;
        measured.checksum += count;
        if (digest_outputs) {
            measured.digest = hash_result(measured.digest, index, values, count);
        }
        if (owned != NULL) array_container_free(owned);
    }

    if (arm != RAWR_CR_ARRAY_ALLOCATING && measured.allocation_calls != 0) {
        return false;
    }
    if (arm == RAWR_CR_ARRAY_PRODUCTION &&
        !measured.output_storage_unchanged) {
        return false;
    }
    *result = measured;
    return true;
}
