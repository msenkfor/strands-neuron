#!/bin/bash

# Run vLLM Neuron Server
# Usage:
#   ./run.sh                              # Use default config
#   ./run.sh configs/tool-calling.env     # Use specific config profile
#   CONFIG_FILE=my-config.env ./run.sh    # Set via environment variable
#   IMAGE_NAME=my-image ./run.sh          # Use custom image name
#   CONTAINER_NAME=my-container ./run.sh  # Use custom container name
#   PORT=8081 ./run.sh                    # Override port mapping

set -e

# Configuration file (optional - first argument or CONFIG_FILE env var)
CONFIG_FILE="${1:-${CONFIG_FILE:-}}"

# Image and container names (can be overridden)
IMAGE_NAME="${IMAGE_NAME:-vllm-server-strands}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-server-strands}"

# Port mapping (default 8080, can be overridden)
PORT="${PORT:-8080}"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. You may need it for private models."
    echo "   Set it with: export HF_TOKEN=<your-token>"
fi

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists. Removing it..."
    docker rm -f "${CONTAINER_NAME}"
fi

# Build device flags for existing neuron devices
DEVICE_FLAGS=""
for i in {0..15}; do
    if [ -e "/dev/neuron${i}" ]; then
        DEVICE_FLAGS="${DEVICE_FLAGS} --device=/dev/neuron${i}"
    fi
done

# Pass EFA/Infiniband devices
for i in {0..7}; do
    if [ -e "/dev/infiniband/uverbs${i}" ]; then
        DEVICE_FLAGS="${DEVICE_FLAGS} --device=/dev/infiniband/uverbs${i}"
    fi
done

# Handle config file
ENV_FILE_FLAG=""
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Using configuration file: $CONFIG_FILE"
    ENV_FILE_FLAG="--env-file $CONFIG_FILE"

    # Extract PORT from config file if present (for port mapping)
    CONFIG_PORT=$(grep -E "^PORT=" "$CONFIG_FILE" 2>/dev/null | cut -d'=' -f2 || true)
    if [ -n "$CONFIG_PORT" ]; then
        PORT="$CONFIG_PORT"
    fi
else
    echo "Using default configuration"
fi

echo "=================================="
echo "Starting vLLM Neuron Server"
echo "=================================="
echo "Image: ${IMAGE_NAME}"
echo "Container: ${CONTAINER_NAME}"
echo "Port: ${PORT}"
echo "=================================="

docker run -it \
    -e HF_TOKEN=$HF_TOKEN \
    -e LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/efa/lib64:${LD_LIBRARY_PATH} \
    ${ENV_FILE_FLAG} \
    ${DEVICE_FLAGS} \
    --privileged \
    --shm-size=10g \
    -v /opt/amazon/efa:/opt/amazon/efa:ro \
    -p ${PORT}:${PORT} \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}

