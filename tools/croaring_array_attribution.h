// SPDX-License-Identifier: MPL-2.0

#ifndef RAWR_CROARING_ARRAY_ATTRIBUTION_H
#define RAWR_CROARING_ARRAY_ATTRIBUTION_H

#ifndef RAWR_CROARING_ARRAY_INTERNAL_BUILD
#include "croaring_wrapper.h"
#endif

typedef struct rawr_cr_array_pair_s {
    const uint16_t *left;
    uint32_t left_len;
    const uint16_t *right;
    uint32_t right_len;
} rawr_cr_array_pair_t;

typedef enum rawr_cr_array_operation_e {
    RAWR_CR_ARRAY_UNION = 0,
    RAWR_CR_ARRAY_DIFFERENCE = 1,
} rawr_cr_array_operation_t;

typedef enum rawr_cr_array_arm_e {
    RAWR_CR_ARRAY_SCALAR = 0,
    RAWR_CR_ARRAY_PRODUCTION = 1,
    RAWR_CR_ARRAY_ALLOCATING = 2,
} rawr_cr_array_arm_t;

typedef enum rawr_cr_array_branch_e {
    RAWR_CR_ARRAY_BRANCH_SCALAR = 0,
    RAWR_CR_ARRAY_BRANCH_AVX2 = 1,
    RAWR_CR_ARRAY_BRANCH_NOT_APPLICABLE = 2,
} rawr_cr_array_branch_t;

typedef struct rawr_cr_array_result_s {
    uint64_t checksum;
    uint64_t digest;
    uint64_t pair_count;
    uint64_t input_elements;
    uint64_t allocation_calls;
    rawr_cr_array_branch_t branch;
    bool outputs_distinct;
    bool output_storage_unchanged;
} rawr_cr_array_result_t;

bool rawr_cr_array_attribution_run(const rawr_cr_array_pair_t *pairs,
                                   size_t pair_count,
                                   rawr_cr_array_operation_t operation,
                                   rawr_cr_array_arm_t arm,
                                   bool digest_outputs,
                                   rawr_cr_array_result_t *result);

bool rawr_cr_array_runtime_has_avx2(void);

#endif
