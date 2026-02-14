#!/bin/bash
set -e

cd "$(dirname "$0")/.."

zig build bench-alloc
mkdir -p misc

OUTFILE="misc/bench-alloc-$(date +%Y%m%d-%H%M%S).txt"
./zig-out/bin/bench_alloc --matrix 2>&1 | tee "$OUTFILE"

echo ""
echo "Saved to: $OUTFILE"
