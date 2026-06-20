# Spec 00-03: Executable Entrypoints, Time, And Args

## Goal

Update executable entry points and runtime APIs to Zig 0.16.0 so benchmark and
utility targets build again.

## Scope

Primary files:

- `src/bench.zig`
- `src/bench_croaring.zig`
- `src/bench_allocators.zig`
- `src/validate_croaring.zig`, only for `main` signature or runtime API fallout

## Implementation Notes

Prefer full Zig 0.16.0 process initialization:

```zig
pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    _ = gpa;
    _ = io;
}
```

For args:

```zig
var args = init.minimal.args.iterate();
_ = args.skip();
while (args.next()) |arg| {
    // arg is sentinel-terminated; it can be used as a slice where needed.
}
```

For benchmarking duration measurement, replace removed `std.time.Timer` /
`std.time.nanoTimestamp` patterns with the Zig 0.16.0 `std.Io` clock/timestamp
model. Keep reported units and benchmark semantics unchanged.

For run timestamp printing, use an equivalent Zig 0.16.0 wall-clock timestamp
source. If the exact old UTC header is expensive to preserve, prefer a small
helper that keeps the existing visible format.

Do not refactor benchmark logic while changing runtime APIs.

## Dependencies

- `00-02` complete

## Validation

```bash
zig build bench
zig build bench-compare
zig build bench-alloc
zig build validate
```

Running the full benchmarks is not required for this chunk; building the
executables is enough unless compile errors only appear at runtime.

## Checklist

- [x] Update executable `main` signatures where runtime APIs are needed
- [x] Replace `std.process.args()` usage
- [x] Replace removed `std.time` timer/timestamp usage
- [x] Preserve benchmark output semantics
- [x] Benchmark and validation executables build
