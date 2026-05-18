#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

SRC_DIR="$PROJECT_DIR/src"
OUTPUT_DIR="$PROJECT_DIR/outputs"

mkdir -p "$OUTPUT_DIR/models" "$OUTPUT_DIR/results"

echo "Running classification pipeline"
echo "Src:   $SRC_DIR"
echo "Out:   $OUTPUT_DIR"
echo "Args:  $@"
echo ""

python "$SRC_DIR/main.py" --output-dir "$OUTPUT_DIR" "$@"
