<!-- SPDX-License-Identifier: MPL-2.0 -->

# Third-Party Notices

## CRoaring

This repository contains generated amalgamation files from
[CRoaring](https://github.com/RoaringBitmap/CRoaring):

- `vendor/roaring.c`
- `vendor/roaring.h`

The vendored files identify themselves as CRoaring 4.5.2. CRoaring is offered
under a dual license: recipients may choose either the Apache License 2.0 or the
MIT License. Repository checkouts preserve the complete upstream dual-license
text in `vendor/LICENSE-CRoaring`; it is also available from CRoaring's
[authoritative upstream LICENSE](https://github.com/RoaringBitmap/CRoaring/blob/master/LICENSE).
The original notices remain embedded in both amalgamation files.

`vendor/roaring.c` also contains highly modified CPU feature-detection code
derived from PyTorch. Its BSD 3-Clause license and copyright notices are
preserved inline in that file.

rawr uses these CRoaring files only as a correctness oracle for validation and
differential testing, and as a reference implementation for comparative
benchmarking. They are not part of rawr's public Zig library API and are excluded
from the downstream Zig package.

The files under `vendor/` are not covered by rawr's MPL-2.0 license. They remain
governed by their respective upstream terms. References to CRoaring, PyTorch,
and their contributors identify the origin of the vendored code and do not imply
endorsement of rawr by any upstream author or organization.
