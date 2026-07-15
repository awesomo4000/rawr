<!-- SPDX-License-Identifier: MPL-2.0 -->

# Spec 01-09: Malformed input smoke test (optional)

Chunk of [`01-differential-testing.md`](01-differential-testing.md). **Optional,
last, isolated.** Untrusted input is out of scope as a feature, but a tiny
corruption sweep is a cheap way to surface real bugs (panics, OOB reads, infinite
loops) in `deserialize`. ~40 lines, needs no CRoaring. Trivially droppable.

## Dependencies

- **01-02** (generator) to produce a known-good mixed bitmap. Otherwise standalone.

## Task

1. Serialize a known-good mixed bitmap to bytes.
2. In a loop, copy the bytes and corrupt them:
   - flip a random byte,
   - truncate to a random length,
   - zero the cardinality field,
   - set a container's cardinality to `0xFFFF`.
3. Call `RoaringBitmap.deserialize` on each corrupted buffer.

The **only** acceptable outcomes are: returns a valid bitmap, or returns a Zig
error. A crash / panic / hang is a **finding**.

Use a fixed (or printed) seed for reproducibility.

## Acceptance criteria

1. A corruption sweep runs over a serialized mixed bitmap covering the four
   corruption modes above.
2. Every `deserialize` call returns either a valid bitmap or a Zig error — no
   crash/panic/hang.
3. Any non-error crash is reported (file as a bug; not necessarily fixed in this
   chunk).

## Note

Treat findings as bugs to file, not necessarily to fix now — this chunk's job is
to *surface* them cheaply.
