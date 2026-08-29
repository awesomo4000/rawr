// SPDX-License-Identifier: MPL-2.0

#ifndef RAWR_CROARING_SUPPORT_H
#define RAWR_CROARING_SUPPORT_H

enum {
    RAWR_CROARING_SUPPORTS_AVX2 = 1,
    RAWR_CROARING_SUPPORTS_AVX512 = 2,
};

int rawr_croaring_hardware_support(void);

#endif
