#!/bin/bash

# Build disaggregated inference router Docker image
# Usage:
#   ./build-router.sh
#   IMAGE_NAME=my-router ./build-router.sh

set -e
trap 'echo ""; echo "Build failed! Press Enter to close..."; read' ERR

IMAGE_NAME="${IMAGE_NAME:-neuron-router}"

echo "=================================="
echo "Building Disaggregated Router Image"
echo "=================================="
echo "Image Name: $IMAGE_NAME"
echo "=================================="

docker build \
    --no-cache \
    -t "${IMAGE_NAME}" \
    -f Dockerfile.router .

echo "=================================="
echo "Build complete: ${IMAGE_NAME}"
echo "=================================="
