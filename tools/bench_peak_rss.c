// SPDX-License-Identifier: MPL-2.0

#include "bench_peak_rss.h"

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

size_t rawr_bench_peak_rss_bytes(void) {
#if defined(_WIN32)
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return (size_t)usage.ru_maxrss;
#else
    return (size_t)usage.ru_maxrss * 1024;
#endif
#endif
}
