#!/bin/bash

# Run disaggregated vLLM Neuron server inside Docker
#
# Single-instance mode (both roles on one device):
#   Terminal 1 — prefill:
#     SEND=1 SINGLE_INSTANCE=1 \
#       HF_TOKEN=<token> MODEL_PATH=meta-llama/Llama-3.3-70B-Instruct \
#       HF_CACHE_DIR=/home/ubuntu/.cache/huggingface \
#       HOST_COMPILED_MODEL_PATH=/home/ubuntu/compiled-models/di_traced_model_tp32_b8 \
#       ./run-disagg.sh
#   Terminal 2 — decode: same without SEND=1
#
# Multi-instance mode (separate machines):
#   Set NEURON_SEND_IP/NEURON_RECV_IP to the actual machine IPs.

set -e

IMAGE_NAME="${IMAGE_NAME:-neuron-disagg}"

# Role selection
SEND="${SEND:-0}"
SINGLE_INSTANCE="${SINGLE_INSTANCE:-0}"

# Derive container name from role to allow both to run simultaneously
if [ "$SEND" = "1" ]; then
    ROLE_NAME="prefill"
    DEFAULT_CONTAINER="neuron-disagg-prefill"
else
    ROLE_NAME="decode"
    DEFAULT_CONTAINER="neuron-disagg-decode"
fi
CONTAINER_NAME="${CONTAINER_NAME:-$DEFAULT_CONTAINER}"

# HuggingFace config
HF_TOKEN="${HF_TOKEN:-}"
MODEL_PATH="${MODEL_PATH:-}"          # HF model ID e.g. meta-llama/Llama-3.3-70B-Instruct
HF_CACHE_DIR="${HF_CACHE_DIR:-}"      # host HF cache dir — mounted at /root/.cache/huggingface

# Local model path (alternative to HF model ID)
HOST_MODEL_PATH="${HOST_MODEL_PATH:-}"

# Compiled artifacts (required)
HOST_COMPILED_MODEL_PATH="${HOST_COMPILED_MODEL_PATH:-}"

# Shared directory for KV cache transfer (required for single-instance mode)
HOST_KV_CACHE_DIR="${HOST_KV_CACHE_DIR:-}"

# Runtime overrides (optional)
TP_DEGREE="${TP_DEGREE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"

# Validate: need a model source
if [ -z "$HOST_MODEL_PATH" ] && [ -z "$MODEL_PATH" ]; then
    echo "Error: provide either HOST_MODEL_PATH (local) or MODEL_PATH (HuggingFace model ID)."
    exit 1
fi
if [ -z "$HOST_COMPILED_MODEL_PATH" ]; then
    echo "Error: HOST_COMPILED_MODEL_PATH is required."
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Required for gated or private HuggingFace models."
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
VOLUME_FLAGS="-v ${HOST_COMPILED_MODEL_PATH}:/compiled-model:ro"
ENV_OVERRIDES="-e HF_TOKEN=${HF_TOKEN}"

if [ -n "$HOST_MODEL_PATH" ]; then
    VOLUME_FLAGS="${VOLUME_FLAGS} -v ${HOST_MODEL_PATH}:/model:ro"
    ENV_OVERRIDES="${ENV_OVERRIDES} -e MODEL_PATH=/model"
else
    # HF model ID — mount the cache so vLLM finds it without re-downloading
    ENV_OVERRIDES="${ENV_OVERRIDES} -e MODEL_PATH=${MODEL_PATH}"
    if [ -n "$HF_CACHE_DIR" ]; then
        VOLUME_FLAGS="${VOLUME_FLAGS} -v ${HF_CACHE_DIR}:/root/.cache/huggingface"
    fi
fi

if [ -n "$HOST_KV_CACHE_DIR" ]; then
    mkdir -p "$HOST_KV_CACHE_DIR"
    VOLUME_FLAGS="${VOLUME_FLAGS} -v ${HOST_KV_CACHE_DIR}:/kv-cache"
    ENV_OVERRIDES="${ENV_OVERRIDES} -e KV_CACHE_PATH=/kv-cache"
fi

if [ -n "$TP_DEGREE" ]; then ENV_OVERRIDES="${ENV_OVERRIDES} -e TP_DEGREE=${TP_DEGREE}"; fi
if [ -n "$BATCH_SIZE" ]; then ENV_OVERRIDES="${ENV_OVERRIDES} -e BATCH_SIZE=${BATCH_SIZE}"; fi

echo "=================================="
echo "Starting Disaggregated vLLM Server ($ROLE_NAME)"
echo "=================================="
echo "Image:              ${IMAGE_NAME}"
echo "Container:          ${CONTAINER_NAME}"
if [ -n "$HOST_MODEL_PATH" ]; then
    echo "Model:              ${HOST_MODEL_PATH} -> /model"
else
    echo "Model:              ${MODEL_PATH} (HuggingFace)"
    if [ -n "$HF_CACHE_DIR" ]; then echo "HF Cache:           ${HF_CACHE_DIR}"; fi
fi
echo "Compiled Artifacts: ${HOST_COMPILED_MODEL_PATH} -> /compiled-model"
if [ -n "$HOST_KV_CACHE_DIR" ]; then echo "KV Cache Dir:       ${HOST_KV_CACHE_DIR} -> /kv-cache"; fi
echo "Single Instance:    ${SINGLE_INSTANCE}"
if [ -n "$TP_DEGREE" ]; then echo "TP Degree:          ${TP_DEGREE}"; fi
if [ -n "$BATCH_SIZE" ]; then echo "Batch Size:         ${BATCH_SIZE}"; fi
echo "=================================="

docker run -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    ${VOLUME_FLAGS} \
    ${DEVICE_FLAGS} \
    -e SEND="${SEND}" \
    -e SINGLE_INSTANCE="${SINGLE_INSTANCE}" \
    ${ENV_OVERRIDES} \
    --cap-add SYS_ADMIN \
    --cap-add IPC_LOCK \
    "${IMAGE_NAME}"
