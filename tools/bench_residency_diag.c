// SPDX-License-Identifier: MPL-2.0

#include "bench_residency_diag.h"

#include <string.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task_info.h>
#include <sys/sysctl.h>
#include <unistd.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

int rawr_residency_fault_snapshot(rawr_residency_fault_snapshot_t *snapshot) {
    memset(snapshot, 0, sizeof(*snapshot));

#if defined(__APPLE__)
    task_events_info_data_t info;
    mach_msg_type_number_t count = TASK_EVENTS_INFO_COUNT;
    kern_return_t rc = task_info(mach_task_self(), TASK_EVENTS_INFO,
                                 (task_info_t)&info, &count);
    if (rc != KERN_SUCCESS || count < TASK_EVENTS_INFO_COUNT) {
        return 0;
    }
    snapshot->primary = (uint32_t)info.faults;
    snapshot->major = (uint32_t)info.pageins;
    snapshot->cow = (uint32_t)info.cow_faults;
    snapshot->valid = 1;
    snapshot->source = RAWR_RESIDENCY_FAULT_DARWIN_TASK_EVENTS;
    return 1;
#elif defined(__linux__)
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    snapshot->primary = (uint64_t)usage.ru_minflt;
    snapshot->major = (uint64_t)usage.ru_majflt;
    snapshot->valid = 1;
    snapshot->source = RAWR_RESIDENCY_FAULT_LINUX_RUSAGE;
    return 1;
#else
    return 0;
#endif
}

size_t rawr_residency_page_size(void) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return (size_t)info.dwPageSize;
#elif defined(_SC_PAGESIZE)
    long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? (size_t)value : 0;
#else
    return 0;
#endif
}

#if defined(__APPLE__)
static uint64_t read_sysctl_u64(const char *name) {
    uint64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, NULL, 0) != 0) {
        return 0;
    }
    return value;
}
#endif

uint64_t rawr_residency_last_level_cache_size(void) {
#if defined(__APPLE__)
    uint64_t value = read_sysctl_u64("hw.l3cachesize");
    if (value != 0) return value;
    value = read_sysctl_u64("hw.perflevel0.l2cachesize");
    if (value != 0) return value;
    return read_sysctl_u64("hw.l2cachesize");
#elif defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
    long value = sysconf(_SC_LEVEL3_CACHE_SIZE);
    return value > 0 ? (uint64_t)value : 0;
#else
    return 0;
#endif
}

uint32_t rawr_residency_cache_source(void) {
#if defined(__APPLE__)
    if (read_sysctl_u64("hw.l3cachesize") != 0) {
        return RAWR_RESIDENCY_CACHE_DARWIN_L3;
    }
    if (read_sysctl_u64("hw.perflevel0.l2cachesize") != 0) {
        return RAWR_RESIDENCY_CACHE_DARWIN_PERF_L2;
    }
    if (read_sysctl_u64("hw.l2cachesize") != 0) {
        return RAWR_RESIDENCY_CACHE_DARWIN_L2;
    }
    return RAWR_RESIDENCY_CACHE_NONE;
#elif defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
    return rawr_residency_last_level_cache_size() == 0
               ? RAWR_RESIDENCY_CACHE_NONE
               : RAWR_RESIDENCY_CACHE_LINUX_L3;
#else
    return RAWR_RESIDENCY_CACHE_NONE;
#endif
}
