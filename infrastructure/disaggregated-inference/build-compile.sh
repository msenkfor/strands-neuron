#!/bin/bash

# Build Neuron compile Docker image
# Usage:
#   ./build-compile.sh                    # Build with defaults
#   IMAGE_NAME=my-compile ./build-compile.sh
#   TP_DEGREE=32 BATCH_SIZE=8 ./build-compile.sh

set -e
trap 'echo ""; echo "Build failed! Press Enter to close..."; read' ERR

IMAGE_NAME="${IMAGE_NAME:-neuron-compile}"

# Optional build-time defaults (can always be overridden at run time via env vars)
TP_DEGREE="${TP_DEGREE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "=================================="
echo "Building Neuron Compile Image"
echo "=================================="
echo "Image Name: $IMAGE_NAME"
echo "TP Degree:  $TP_DEGREE (default, overridable at runtime)"
echo "Batch Size: $BATCH_SIZE (default, overridable at runtime)"
echo "=================================="

docker build \
    --no-cache \
    --build-arg TP_DEGREE="${TP_DEGREE}" \
    --build-arg BATCH_SIZE="${BATCH_SIZE}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile.compile .

echo "=================================="
echo "Build complete: ${IMAGE_NAME}"
echo "=================================="
