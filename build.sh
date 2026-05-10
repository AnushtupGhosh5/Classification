#!/bin/bash
set -e

IMAGE_NAME="thyroid-classification"

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

echo ""
echo "Build complete!"
echo "Run with: ./run.sh --model mobilenetv2 --batch-size 32 --epochs 10 --lr 0.001"
