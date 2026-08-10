// SPDX-License-Identifier: MPL-2.0

#ifndef CROARING_ADDRESS_SORTED_H
#define CROARING_ADDRESS_SORTED_H

#if defined(RAWR_CR_SORTED_INTERNAL)
#include "../vendor/roaring.h"
#else
#include "croaring_wrapper.h"
#endif

bool rawr_cr_sorted_mass_alloc(size_t count);
void rawr_cr_sorted_mass_zero(size_t count);
void rawr_cr_sorted_mass_free(size_t count);
size_t rawr_cr_sorted_peak_rss_bytes(void);

#endif
