#!/bin/bash

# Run the disaggregated inference router inside Docker
#
# Single-instance mode (prefill + decode on same machine):
#   ./run-router.sh   # defaults to 127.0.0.1 for both nodes
#
# Multi-instance mode:
#   PREFILL_IP=<prefill-machine-ip> DECODE_IP=<decode-machine-ip> ./run-router.sh
#
# The router listens on ROUTER_PORT (default 8000) and is the single
# endpoint callers should send requests to.

set -e

IMAGE_NAME="${IMAGE_NAME:-neuron-router}"
CONTAINER_NAME="${CONTAINER_NAME:-neuron-router}"

PREFILL_IP="${PREFILL_IP:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_IP="${DECODE_IP:-127.0.0.1}"
DECODE_PORT="${DECODE_PORT:-8200}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# Remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists. Removing it..."
    docker rm -f "${CONTAINER_NAME}"
fi

echo "=================================="
echo "Starting Disaggregated Inference Router"
echo "=================================="
echo "Image:    ${IMAGE_NAME}"
echo "Prefill:  ${PREFILL_IP}:${PREFILL_PORT}"
echo "Decode:   ${DECODE_IP}:${DECODE_PORT}"
echo "Router:   0.0.0.0:${ROUTER_PORT}"
echo "=================================="

# --network host so the router can reach prefill (8100) and decode (8200)
# on localhost in single-instance mode, or via host IPs in multi-instance mode.
docker run -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    -e PREFILL_IP="${PREFILL_IP}" \
    -e PREFILL_PORT="${PREFILL_PORT}" \
    -e DECODE_IP="${DECODE_IP}" \
    -e DECODE_PORT="${DECODE_PORT}" \
    -e ROUTER_PORT="${ROUTER_PORT}" \
    "${IMAGE_NAME}"
