#!/bin/bash
# Build the disaggregated inference router image.
# Usage: ./build-router.sh

set -e

IMAGE_NAME="${IMAGE_NAME:-neuron-router}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo "Building Router Image: ${IMAGE_NAME}"
echo "=================================="

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.router" \
    -t "${IMAGE_NAME}" \
    "${SCRIPT_DIR}"

echo "=================================="
echo "Build complete: ${IMAGE_NAME}"
echo "=================================="
