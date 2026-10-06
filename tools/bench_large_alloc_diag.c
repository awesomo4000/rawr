// SPDX-License-Identifier: MPL-2.0
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>

// A dedicated descriptor keeps diagnostic markers distinct from report writes.
void rawr_large_marker(int phase) {
    static const char *const markers[] = {
        "RAWR52:init\n", "RAWR52:warmup\n", "RAWR52:timed\n",
        "RAWR52:cleanup\n", "RAWR52:done\n"
    };
    static const size_t lengths[] = {12, 14, 13, 15, 12};
    if (phase == 0) {
        int fd = open("/dev/null", O_WRONLY);
        if (fd < 0 || dup2(fd, 198) < 0) abort();
        if (fd != 198) close(fd);
    }
    if (phase < 0 || phase > 4 || write(198, markers[phase], lengths[phase]) != (ssize_t)lengths[phase]) abort();
    if (phase == 4) close(198);
}

void rawr_large_control_map(void) {
    long size = sysconf(_SC_PAGESIZE);
    if (size <= 0) abort();
    void *p = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED || munmap(p, (size_t)size) != 0) abort();
}
