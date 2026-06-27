#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

void obsd_repro_hello(void) {
    fputs("c helper: hello\n", stderr);
    fflush(stderr);
}

void obsd_repro_mark(const char *message) {
    fputs(message, stderr);
    fputc('\n', stderr);
    fflush(stderr);
}

void obsd_repro_write(const char *ptr, size_t len) {
    fwrite(ptr, 1, len, stderr);
    fflush(stderr);
}

void obsd_repro_report_ptr(const char *label, const void *ptr, size_t len) {
    fprintf(stderr, "%s ptr=%p len=%zu\n", label, ptr, len);
    fflush(stderr);
}

void obsd_repro_report_u64(const char *label, uint64_t value) {
    fprintf(stderr, "%s=%llu\n", label, (unsigned long long)value);
    fflush(stderr);
}

void *obsd_repro_malloc(size_t size) {
    return malloc(size);
}

void obsd_repro_free(void *ptr) {
    free(ptr);
}

uint64_t obsd_repro_clock_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

uint64_t obsd_repro_timeval_us(void) {
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        return 0;
    }
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}
