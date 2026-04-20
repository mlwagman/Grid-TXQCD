#!/bin/bash
set -e
cd "$(dirname "$0")"

GRID_ARGS="${@:---grid 8.8.8.16}"

mkdir -p cfgs/qcd logs

echo "Starting QCD gauge generation..."
echo "Grid args: $GRID_ARGS"

./gen_qcd_cfgs $GRID_ARGS 2>&1 | tee logs/gen_qcd_cfgs.log
