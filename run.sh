#!/bin/bash
set -e

IMAGE_NAME="thyroid-classification"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_DIR="$PROJECT_DIR/data"
SRC_DIR="$PROJECT_DIR/src"
OUTPUT_DIR="$PROJECT_DIR/outputs"

mkdir -p "$OUTPUT_DIR/models" "$OUTPUT_DIR/results"

echo "Running: $IMAGE_NAME"
echo "Data:  $DATA_DIR"
echo "Src:   $SRC_DIR"
echo "Out:   $OUTPUT_DIR"
echo "Args:  $@"
echo ""

docker run --rm --gpus all --shm-size=8g \
    -v "$DATA_DIR:/app/data" \
    -v "$SRC_DIR:/app/src" \
    -v "$OUTPUT_DIR:/app/outputs" \
    "$IMAGE_NAME" \
    python src/main.py --output-dir /app/outputs "$@"
