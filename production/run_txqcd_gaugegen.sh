#!/bin/bash
set -e
cd "$(dirname "$0")"

GRID_ARGS="${@:---grid 8.8.8.16}"

mkdir -p cfgs/txqcd logs

echo "Starting TXQCD gauge generation..."
echo "Grid args: $GRID_ARGS"

./gen_txqcd_cfgs $GRID_ARGS 2>&1 | tee logs/gen_txqcd_cfgs.log
