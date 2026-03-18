#!/bin/bash

# Build disaggregated vLLM Neuron server Docker image
# Usage:
#   ./build-disagg.sh                    # Build with defaults
#   IMAGE_NAME=my-disagg ./build-disagg.sh
#   TP_DEGREE=32 BATCH_SIZE=8 ./build-disagg.sh

set -e
trap 'echo ""; echo "Build failed! Press Enter to close..."; read' ERR

IMAGE_NAME="${IMAGE_NAME:-neuron-disagg}"

# Optional build-time defaults (always overridable at runtime)
TP_DEGREE="${TP_DEGREE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "=================================="
echo "Building Disaggregated vLLM Neuron Image"
echo "=================================="
echo "Image Name: $IMAGE_NAME"
echo "TP Degree:  $TP_DEGREE (default, overridable at runtime)"
echo "Batch Size: $BATCH_SIZE (default, overridable at runtime)"
echo "=================================="

docker build \
    --build-arg TP_DEGREE="${TP_DEGREE}" \
    --build-arg BATCH_SIZE="${BATCH_SIZE}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile.disagg .

echo "=================================="
echo "Build complete: ${IMAGE_NAME}"
echo "=================================="
