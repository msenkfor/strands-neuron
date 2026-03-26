#!/bin/bash
# Build the disaggregated inference server image.
# This installs the vLLM v0 fork (upstreaming-to-vllm) which supports
# NeuronConnector KV transfer for P/D disaggregation.
#
# Usage: ./build-server.sh
#
# Note: build context is infrastructure/ (parent directory) so that
# start-vllm.sh can be copied into the image.

set -e

IMAGE_NAME="${IMAGE_NAME:-vllm-server-disagg}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=================================="
echo "Building Disaggregated Server Image: ${IMAGE_NAME}"
echo "(installs vLLM v0 fork — this will take several minutes)"
echo "=================================="

docker build \
    -f "${SCRIPT_DIR}/Dockerfile.server" \
    -t "${IMAGE_NAME}" \
    "${INFRA_DIR}"

echo "=================================="
echo "Build complete: ${IMAGE_NAME}"
echo "=================================="
