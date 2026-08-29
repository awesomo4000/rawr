// SPDX-License-Identifier: MPL-2.0

#include "../vendor/roaring.h"

int rawr_croaring_hardware_support(void) {
#if CROARING_IS_X64
    return croaring_hardware_support();
#else
    return 0;
#endif
}
