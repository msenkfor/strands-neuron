#!/bin/bash
# vLLM Neuron Server Startup Script
# This script provides flexible configuration for vLLM on AWS Neuron instances
#
# Configuration Priority (highest to lowest):
#   1. Environment variables passed to container
#   2. --env-file passed to docker run
#   3. CONFIG_FILE environment variable (path to config file inside container)
#   4. Build-time defaults baked into the image

set -e

# =============================================================================
# Load Config File (if specified)
# =============================================================================
# CONFIG_FILE can point to a config file mounted inside the container
# This allows runtime configuration without rebuilding the image
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    set -a  # automatically export all variables
    source "$CONFIG_FILE"
    set +a
fi

# =============================================================================
# Core Configuration (Required)
# =============================================================================
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${PORT:-8080}"

# =============================================================================
# Model Configuration
# =============================================================================
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# =============================================================================
# Tool Calling Configuration
# =============================================================================
# Enable tool calling support
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-true}"
# Parser: llama3_json, hermes, mistral, etc.
# See: https://docs.vllm.ai/en/latest/features/tool_calling/
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-llama3_json}"

# =============================================================================
# Performance Configuration
# =============================================================================
# Prefix caching can improve performance for repeated prompts
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

# =============================================================================
# Advanced Configuration (--additional-config)
# =============================================================================
# Neuron config overrides (JSON format)
# Example: '{"override_neuron_config":{"enable_bucketing":false}}'
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"
# Strip surrounding quotes if present (handles docker --env-file including literal quotes)
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG#\'}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG%\'}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG#\"}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG%\"}"

# =============================================================================
# Speculative Decoding (--speculative-config)
# =============================================================================
# Speculative decoding config (JSON format, empty to disable)
# Example: '{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 7, "max_model_len": 2048, "method": "eagle"}'
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
# Strip surrounding quotes if present
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG#\'}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG%\'}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG#\"}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG%\"}"

# =============================================================================
# KV Cache Transfer (Optional - for distributed setups)
# =============================================================================
# Enable KV cache transfer between instances
ENABLE_KV_TRANSFER="${ENABLE_KV_TRANSFER:-false}"
KV_CONNECTOR="${KV_CONNECTOR:-NeuronConnector}"
KV_ROLE="${KV_ROLE:-kv_producer}"
KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-2e11}"
ETCD="${ETCD:-}"  # etcd address for coordination

# =============================================================================
# Build Command
# =============================================================================

echo "=================================="
echo "Starting vLLM Neuron Server"
echo "=================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Max Sequences: $MAX_NUM_SEQS"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Tool Calling: $ENABLE_TOOL_CALLING"
if [ "$ENABLE_TOOL_CALLING" = "true" ]; then
    echo "Tool Parser: $TOOL_CALL_PARSER"
fi
if [ -n "$ADDITIONAL_CONFIG" ]; then
    echo "Additional Config: $ADDITIONAL_CONFIG"
fi
if [ -n "$SPECULATIVE_CONFIG" ]; then
    echo "Speculative Config: $SPECULATIVE_CONFIG"
fi
echo "VLLM_USE_V1: $VLLM_USE_V1"
echo "=================================="

# Build command as array to properly handle JSON arguments
CMD_ARRAY=(
    "python3" "-m" "vllm.entrypoints.openai.api_server"
    "--model" "$MODEL"
    "--port" "$PORT"
    "--max-num-seqs" "$MAX_NUM_SEQS"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
)

# Add tool calling support
if [ "$ENABLE_TOOL_CALLING" = "true" ]; then
    CMD_ARRAY+=("--enable-auto-tool-choice")
    CMD_ARRAY+=("--tool-call-parser" "$TOOL_CALL_PARSER")
fi

# Add prefix caching
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    CMD_ARRAY+=("--enable-prefix-caching")
else
    CMD_ARRAY+=("--no-enable-prefix-caching")
fi

# Add KV cache transfer configuration
if [ "$ENABLE_KV_TRANSFER" = "true" ] && [ -n "$ETCD" ]; then
    KV_CONFIG="{\"kv_connector\":\"$KV_CONNECTOR\",\"kv_role\":\"$KV_ROLE\",\"kv_buffer_size\":$KV_BUFFER_SIZE,\"etcd\":\"$ETCD\"}"
    CMD_ARRAY+=("--kv-transfer-config" "$KV_CONFIG")
fi

# Add additional config (neuron overrides)
if [ -n "$ADDITIONAL_CONFIG" ] && [ "$ADDITIONAL_CONFIG" != "{}" ]; then
    CMD_ARRAY+=("--additional-config" "$ADDITIONAL_CONFIG")
fi

# Add speculative decoding config (separate argument)
if [ -n "$SPECULATIVE_CONFIG" ] && [ "$SPECULATIVE_CONFIG" != "{}" ]; then
    CMD_ARRAY+=("--speculative-config" "$SPECULATIVE_CONFIG")
fi
# Execute the command
echo "Executing: ${CMD_ARRAY[@]}"
echo "=================================="
exec "${CMD_ARRAY[@]}"
