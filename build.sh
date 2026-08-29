#!/bin/bash
set -e

IMAGE_NAME="thyroid-classification"

echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

echo ""
echo "Build complete!"
echo "Run the configured experiment with: ./run.sh"
