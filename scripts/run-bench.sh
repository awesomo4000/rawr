#!/bin/bash
set -e

cd "$(dirname "$0")/.."

zig build bench -Doptimize=ReleaseFast
mkdir -p misc

OUTFILE="misc/bench-$(date +%Y%m%d-%H%M%S).txt"
./zig-out/bin/bench 2>&1 | tee "$OUTFILE"

echo ""
echo "Saved to: $OUTFILE"
