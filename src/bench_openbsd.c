#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

uint64_t rawr_bench_monotonic_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (uint64_t)tv.tv_sec * 1000000000ULL + (uint64_t)tv.tv_usec * 1000ULL;
    }

    return 0;
}

void *rawr_bench_malloc(size_t size) {
    return malloc(size);
}

void *rawr_bench_aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

void rawr_bench_free(void *ptr) {
    free(ptr);
}

void rawr_bench_write_stderr(const char *ptr, size_t len) {
    if (len == 0) {
        return;
    }
    (void)fwrite(ptr, 1, len, stderr);
}
