// SPDX-License-Identifier: MPL-2.0

#ifndef RAWR_BENCH_RESIDENCY_DIAG_H
#define RAWR_BENCH_RESIDENCY_DIAG_H

#include "croaring_bench_diag.h"

#include <stddef.h>
#include <stdint.h>

typedef struct rawr_residency_fault_snapshot_s {
    uint64_t primary;
    uint64_t major;
    uint64_t cow;
    uint32_t valid;
    uint32_t source;
} rawr_residency_fault_snapshot_t;

enum {
    RAWR_RESIDENCY_FAULT_NONE = 0,
    RAWR_RESIDENCY_FAULT_LINUX_RUSAGE = 1,
    RAWR_RESIDENCY_FAULT_DARWIN_TASK_EVENTS = 2,
};

enum {
    RAWR_RESIDENCY_CACHE_NONE = 0,
    RAWR_RESIDENCY_CACHE_LINUX_L3 = 1,
    RAWR_RESIDENCY_CACHE_DARWIN_L3 = 2,
    RAWR_RESIDENCY_CACHE_DARWIN_PERF_L2 = 3,
    RAWR_RESIDENCY_CACHE_DARWIN_L2 = 4,
};

int rawr_residency_fault_snapshot(rawr_residency_fault_snapshot_t *snapshot);
size_t rawr_residency_page_size(void);
uint64_t rawr_residency_last_level_cache_size(void);
uint32_t rawr_residency_cache_source(void);

#endif
