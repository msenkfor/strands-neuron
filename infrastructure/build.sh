#!/bin/bash

# Build vLLM Neuron Server Docker Image
# Usage:
#   ./build.sh                              # Build with default config
#   ./build.sh configs/tool-calling.env     # Build with specific config
#   CONFIG_FILE=my-config.env ./build.sh    # Set config via environment variable
#   IMAGE_NAME=my-image ./build.sh          # Override image name

set -e
trap 'echo ""; echo "Build failed! Press Enter to close..."; read' ERR

# Configuration file (optional - first argument or CONFIG_FILE env var)
CONFIG_FILE="${1:-${CONFIG_FILE:-}}"

# Image name (can be overridden)
IMAGE_NAME="${IMAGE_NAME:-vllm-server-strands}"

# Load configuration from file if provided
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    set -a  # automatically export all variables
    source "$CONFIG_FILE"
    set +a
fi

# Default values (can be overridden via config file or environment variables)
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${PORT:-8080}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-llama3_json}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

# Speculative decoding config
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
OVERRIDE_NEURON_CONFIG="${OVERRIDE_NEURON_CONFIG:-}"
NEURON_CONFIG_FLAG="${NEURON_CONFIG_FLAG:-additional-config}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"

echo "=================================="
echo "Building vLLM Neuron Server Image"
echo "=================================="
echo "Image Name: $IMAGE_NAME"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Max Sequences: $MAX_NUM_SEQS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Tool Calling: $ENABLE_TOOL_CALLING"
if [ "$ENABLE_TOOL_CALLING" = "true" ]; then
    echo "Tool Parser: $TOOL_CALL_PARSER"
fi
if [ -n "$SPECULATIVE_CONFIG" ]; then
    echo "Speculative Config: $SPECULATIVE_CONFIG"
fi
if [ -n "$OVERRIDE_NEURON_CONFIG" ]; then
    echo "Neuron Config Flag: --$NEURON_CONFIG_FLAG"
    echo "Override Neuron Config: $OVERRIDE_NEURON_CONFIG"
fi
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "=================================="

docker build \
    --build-arg MODEL="${MODEL}" \
    --build-arg PORT="${PORT}" \
    --build-arg MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    --build-arg MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    --build-arg TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    --build-arg ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING}" \
    --build-arg TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
    --build-arg ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
    --build-arg NEURON_CONFIG_FLAG="${NEURON_CONFIG_FLAG}" \
    --build-arg VLLM_USE_V1="${VLLM_USE_V1}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile .

echo "=================================="
echo "Build complete: ${IMAGE_NAME}"
echo "=================================="

