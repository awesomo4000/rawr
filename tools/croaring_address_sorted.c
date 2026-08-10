// SPDX-License-Identifier: MPL-2.0

#define RAWR_CR_SORTED_INTERNAL 1
#include "croaring_address_sorted.h"

// Compile the unmodified amalgamation into this benchmark-only translation
// unit so the diagnostic can use the same bitset allocation path as CRoaring.
#include "../vendor/roaring.c"

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#define RAWR_CR_SORTED_MAX_CONTAINERS 16364

static bitset_container_t *rawr_cr_sorted_bitsets[RAWR_CR_SORTED_MAX_CONTAINERS];

static size_t rawr_cr_sorted_alignment(void) {
#if CROARING_IS_X64
    return (croaring_hardware_support() & ROARING_SUPPORTS_AVX512) ? 64 : 32;
#else
    return 32;
#endif
}

bool rawr_cr_sorted_mass_alloc(size_t count) {
    if (count > RAWR_CR_SORTED_MAX_CONTAINERS) return false;

    const size_t alignment = rawr_cr_sorted_alignment();
    const size_t bytes = BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t);
    size_t created = 0;
    for (; created < count; created++) {
        bitset_container_t *bitset = roaring_malloc(sizeof(*bitset));
        if (bitset == NULL) break;
        bitset->words = roaring_aligned_malloc(alignment, bytes);
        if (bitset->words == NULL) {
            roaring_free(bitset);
            break;
        }
        bitset->cardinality = 0;
        rawr_cr_sorted_bitsets[created] = bitset;
    }

    if (created == count) return true;
    while (created > 0) {
        created--;
        bitset_container_free(rawr_cr_sorted_bitsets[created]);
        rawr_cr_sorted_bitsets[created] = NULL;
    }
    return false;
}

void rawr_cr_sorted_mass_zero(size_t count) {
    if (count > RAWR_CR_SORTED_MAX_CONTAINERS) return;
    for (size_t i = 0; i < count; i++) {
        memset(
            rawr_cr_sorted_bitsets[i]->words,
            0,
            BITSET_CONTAINER_SIZE_IN_WORDS * sizeof(uint64_t)
        );
    }
}

void rawr_cr_sorted_mass_free(size_t count) {
    if (count > RAWR_CR_SORTED_MAX_CONTAINERS) return;
    for (size_t i = 0; i < count; i++) {
        bitset_container_free(rawr_cr_sorted_bitsets[i]);
        rawr_cr_sorted_bitsets[i] = NULL;
    }
}

size_t rawr_cr_sorted_peak_rss_bytes(void) {
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
