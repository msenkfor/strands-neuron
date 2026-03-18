#!/bin/bash

# Run Neuron model compilation inside Docker
# Usage:
#   Local model:
#     HOST_MODEL_PATH=/path/to/model HOST_OUTPUT_DIR=/path/to/output ./run-compile.sh
#   HuggingFace model:
#     HF_TOKEN=<token> MODEL_PATH=meta-llama/Llama-3.1-8B HOST_OUTPUT_DIR=/path/to/output ./run-compile.sh
#   HuggingFace model with cache (avoids re-downloading):
#     HF_TOKEN=<token> MODEL_PATH=meta-llama/Llama-3.1-8B HF_CACHE_DIR=/path/to/cache HOST_OUTPUT_DIR=/path/to/output ./run-compile.sh

set -e

IMAGE_NAME="${IMAGE_NAME:-neuron-compile}"
CONTAINER_NAME="${CONTAINER_NAME:-neuron-compile}"

# HuggingFace token (required for gated/private models)
HF_TOKEN="${HF_TOKEN:-}"

# Local model path to mount (optional — omit when using a HuggingFace model ID)
HOST_MODEL_PATH="${HOST_MODEL_PATH:-}"

# HuggingFace model ID (used when HOST_MODEL_PATH is not set)
# e.g. meta-llama/Llama-3.1-8B-Instruct
MODEL_PATH="${MODEL_PATH:-}"

# Optional: host directory to use as HuggingFace cache (avoids re-downloading)
HF_CACHE_DIR="${HF_CACHE_DIR:-}"

# Output directory for compiled artifacts (required)
HOST_OUTPUT_DIR="${HOST_OUTPUT_DIR:-}"

# Runtime overrides (optional — fall back to image defaults)
TP_DEGREE="${TP_DEGREE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"

# Warn if HF_TOKEN is not set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Required for gated or private HuggingFace models."
    echo "   Set it with: export HF_TOKEN=<your-token>"
fi

# Validate: need either a local model path or a HF model ID
if [ -z "$HOST_MODEL_PATH" ] && [ -z "$MODEL_PATH" ]; then
    echo "Error: provide either HOST_MODEL_PATH (local) or MODEL_PATH (HuggingFace model ID)."
    echo "Usage: HOST_MODEL_PATH=/path/to/model HOST_OUTPUT_DIR=/path/to/output ./run-compile.sh"
    echo "   or: HF_TOKEN=<token> MODEL_PATH=meta-llama/Llama-3.1-8B HOST_OUTPUT_DIR=/path/to/output ./run-compile.sh"
    exit 1
fi
if [ -z "$HOST_OUTPUT_DIR" ]; then
    echo "Error: HOST_OUTPUT_DIR is required."
    exit 1
fi

# Remove existing container if present
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} already exists. Removing it..."
    docker rm -f "${CONTAINER_NAME}"
fi

# Discover Neuron devices on the host
DEVICE_FLAGS=""
for i in {0..15}; do
    if [ -e "/dev/neuron${i}" ]; then
        DEVICE_FLAGS="${DEVICE_FLAGS} --device=/dev/neuron${i}"
    fi
done

# Build volume and env flags
VOLUME_FLAGS="-v ${HOST_OUTPUT_DIR}:/output"
ENV_OVERRIDES="-e HF_TOKEN=${HF_TOKEN}"

if [ -n "$HOST_MODEL_PATH" ]; then
    # Local model: mount it read-only and point MODEL_PATH at /model
    VOLUME_FLAGS="${VOLUME_FLAGS} -v ${HOST_MODEL_PATH}:/model:ro"
    ENV_OVERRIDES="${ENV_OVERRIDES} -e MODEL_PATH=/model"
else
    # HuggingFace model ID: pass it directly
    ENV_OVERRIDES="${ENV_OVERRIDES} -e MODEL_PATH=${MODEL_PATH}"
fi

if [ -n "$HF_CACHE_DIR" ]; then
    VOLUME_FLAGS="${VOLUME_FLAGS} -v ${HF_CACHE_DIR}:/root/.cache/huggingface"
fi

if [ -n "$TP_DEGREE" ]; then ENV_OVERRIDES="${ENV_OVERRIDES} -e TP_DEGREE=${TP_DEGREE}"; fi
if [ -n "$BATCH_SIZE" ]; then ENV_OVERRIDES="${ENV_OVERRIDES} -e BATCH_SIZE=${BATCH_SIZE}"; fi

echo "=================================="
echo "Running Neuron Model Compilation"
echo "=================================="
echo "Image:       ${IMAGE_NAME}"
if [ -n "$HOST_MODEL_PATH" ]; then
    echo "Model:       ${HOST_MODEL_PATH} -> /model"
else
    echo "Model:       ${MODEL_PATH} (HuggingFace)"
fi
if [ -n "$HF_CACHE_DIR" ]; then echo "HF Cache:    ${HF_CACHE_DIR}"; fi
echo "Output Dir:  ${HOST_OUTPUT_DIR} -> /output"
if [ -n "$TP_DEGREE" ]; then echo "TP Degree:   ${TP_DEGREE}"; fi
if [ -n "$BATCH_SIZE" ]; then echo "Batch Size:  ${BATCH_SIZE}"; fi
echo "=================================="

docker run -it \
    --name "${CONTAINER_NAME}" \
    ${VOLUME_FLAGS} \
    ${DEVICE_FLAGS} \
    ${ENV_OVERRIDES} \
    --cap-add SYS_ADMIN \
    --cap-add IPC_LOCK \
    "${IMAGE_NAME}"
