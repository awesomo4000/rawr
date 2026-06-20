# Spec: Upgrade rawr to Zig 0.16.0

## Goal

Move rawr from Zig 0.15.2 to Zig 0.16.0 while preserving API behavior,
RoaringFormatSpec wire compatibility, CRoaring interop validation, and benchmark
coverage.

The local toolchain is already Zig 0.16.0. The first known compile failure is:

```text
src/serialize.zig:71:22: error: root source file struct 'std' has no member named 'io'
```

This comes from removed `std.io.fixedBufferStream`.

## Current Findings

- `build.zig` already uses modern `root_module` / `b.createModule` patterns.
- `build.zig.zon` still declares `minimum_zig_version = "0.15.2"`.
- Core data structures are mostly allocator-only and should not need broad API redesign.
- Serialization is the first blocker for `zig build test`.
- Benchmark/validation executables still use older process/time/C-import patterns.
- `@cImport` still works for now but is deprecated in Zig 0.16.0; migrate it as part of this upgrade rather than leaving a known warning path.

## Chunk Specs

Work through these in order:

1. `specs/00-01-serialization-io-api.md`
2. `specs/00-02-library-test-compat.md`
3. `specs/00-03-executable-entrypoints.md`
4. `specs/00-04-croaring-interop.md`
5. `specs/00-05-metadata-docs-final-pass.md`

## Acceptance Criteria

- `zig build test` passes with Zig 0.16.0.
- `zig build validate` passes with Zig 0.16.0.
- `zig build bench`, `zig build bench-compare`, and `zig build bench-alloc` build with Zig 0.16.0.
- `build.zig.zon` and docs state Zig 0.16.0 requirements accurately.
- CRoaring-compatible portable serialization remains byte-compatible.
- The migration keeps public bitmap APIs stable unless a chunk spec explicitly calls out an intentional change.

## Out Of Scope

- Performance tuning beyond preserving current behavior.
- Redesigning public APIs unrelated to Zig 0.16.0 compatibility.
- Expanding differential test coverage beyond what is needed to prove migration safety.

## Checklist

- [ ] Complete `00-01` serialization I/O API migration
- [ ] Complete `00-02` remaining library/test compatibility
- [ ] Complete `00-03` executable entrypoint/time/args migration
- [ ] Complete `00-04` CRoaring interop migration
- [ ] Complete `00-05` metadata/docs/final validation pass
