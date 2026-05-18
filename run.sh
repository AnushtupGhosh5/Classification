#!/bin/bash
set -e

IMAGE_NAME="thyroid-classification"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

DATA_DIR="$PROJECT_DIR/data"
SRC_DIR="$PROJECT_DIR/src"
OUTPUT_DIR="$PROJECT_DIR/outputs"
SCRIPT="$PROJECT_DIR/runScript.sh"

mkdir -p "$OUTPUT_DIR/models" "$OUTPUT_DIR/results"

docker run --rm --gpus all --shm-size=8g --network host \
    -v "$DATA_DIR:/app/data" \
    -v "$SRC_DIR:/app/src" \
    -v "$OUTPUT_DIR:/app/outputs" \
    -v "$SCRIPT:/app/runScript.sh" \
    "$IMAGE_NAME" \
    bash /app/runScript.sh
