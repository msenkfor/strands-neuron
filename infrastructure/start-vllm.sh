#!/bin/bash
# vLLM Neuron Server Startup Script
#
# Configuration priority (highest to lowest):
#   1. -e VAR=value flags on docker run
#   2. --env-file passed to docker run
#   3. CONFIG_FILE sourced inside the container
#   4. Defaults below

set -e

# =============================================================================
# Load Config File (if specified)
# =============================================================================
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    set -a
    source "$CONFIG_FILE"
    set +a
fi

# =============================================================================
# Core Configuration
# =============================================================================
MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${PORT:-8080}"

# =============================================================================
# Model Configuration
# =============================================================================
# Accept both MAX_NUM_SEQS and VLLM_BATCH (tutorial alias)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${VLLM_BATCH:-4}}"
# Accept both MAX_MODEL_LEN and MAX_LEN (tutorial alias)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-${MAX_LEN:-1024}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

# =============================================================================
# Device
# =============================================================================
# Set VLLM_DEVICE=neuron for Neuron instances (required for disaggregated inference)
VLLM_DEVICE="${VLLM_DEVICE:-}"

# =============================================================================
# Tool Calling Configuration
# =============================================================================
ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-llama3_json}"

# =============================================================================
# Performance Configuration
# =============================================================================
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"

# =============================================================================
# Speculative Decoding
# =============================================================================
# Max model length for speculative decoding (required for disaggregated inference)
SPECULATIVE_MAX_MODEL_LEN="${SPECULATIVE_MAX_MODEL_LEN:-}"

# Speculative decoding config (JSON, empty to disable)
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG#\'}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG%\'}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG#\"}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG%\"}"

# =============================================================================
# Neuron Config Overrides
# =============================================================================
# --override-neuron-config: pass '{}' for disaggregated inference
OVERRIDE_NEURON_CONFIG="${OVERRIDE_NEURON_CONFIG:-}"

# --additional-config: for other neuron overrides (JSON)
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG:-}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG#\'}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG%\'}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG#\"}"
ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG%\"}"

# =============================================================================
# KV Cache Transfer (disaggregated inference)
# =============================================================================
ENABLE_KV_TRANSFER="${ENABLE_KV_TRANSFER:-false}"
KV_CONNECTOR="${KV_CONNECTOR:-NeuronConnector}"
KV_ROLE="${KV_ROLE:-kv_producer}"       # kv_producer (prefill) or kv_consumer (decode)
KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-2e11}"
ETCD="${ETCD:-}"                         # Required: <proxy-ip>:8989

# =============================================================================
# Build Command
# =============================================================================

echo "=================================="
echo "Starting vLLM Neuron Server"
echo "=================================="
echo "Model:              $MODEL"
echo "Port:               $PORT"
echo "Max Sequences:      $MAX_NUM_SEQS"
echo "Max Model Length:   $MAX_MODEL_LEN"
echo "Tensor Parallel:    $TENSOR_PARALLEL_SIZE"
if [ -n "$VLLM_DEVICE" ]; then
    echo "Device:             $VLLM_DEVICE"
fi
echo "Tool Calling:       $ENABLE_TOOL_CALLING"
if [ "$ENABLE_KV_TRANSFER" = "true" ]; then
    echo "KV Transfer:        role=$KV_ROLE etcd=$ETCD"
fi
echo "=================================="

CMD_ARRAY=(
    "python3" "-m" "vllm.entrypoints.openai.api_server"
    "--model" "$MODEL"
    "--port" "$PORT"
    "--max-num-seqs" "$MAX_NUM_SEQS"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
)

# Device backend (e.g., neuron)
if [ -n "$VLLM_DEVICE" ]; then
    CMD_ARRAY+=("--device" "$VLLM_DEVICE")
fi

# Tool calling
if [ "$ENABLE_TOOL_CALLING" = "true" ]; then
    CMD_ARRAY+=("--enable-auto-tool-choice")
    CMD_ARRAY+=("--tool-call-parser" "$TOOL_CALL_PARSER")
fi

# Prefix caching
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    CMD_ARRAY+=("--enable-prefix-caching")
else
    CMD_ARRAY+=("--no-enable-prefix-caching")
fi

# Speculative max model length (required for disaggregated inference)
if [ -n "$SPECULATIVE_MAX_MODEL_LEN" ]; then
    CMD_ARRAY+=("--speculative-max-model-len" "$SPECULATIVE_MAX_MODEL_LEN")
fi

# KV cache transfer (disaggregated inference)
if [ "$ENABLE_KV_TRANSFER" = "true" ]; then
    if [ -z "$ETCD" ]; then
        echo "Error: ETCD must be set when ENABLE_KV_TRANSFER=true (e.g. ETCD=<proxy-ip>:8989)"
        exit 1
    fi
    KV_CONFIG="{\"kv_connector\":\"$KV_CONNECTOR\",\"kv_role\":\"$KV_ROLE\",\"kv_buffer_size\":$KV_BUFFER_SIZE,\"etcd\":\"$ETCD\"}"
    CMD_ARRAY+=("--kv-transfer-config" "$KV_CONFIG")
fi

# Neuron config override (pass '{}' for disaggregated inference)
if [ -n "$OVERRIDE_NEURON_CONFIG" ]; then
    CMD_ARRAY+=("--override-neuron-config" "$OVERRIDE_NEURON_CONFIG")
fi

# Additional config (neuron overrides)
if [ -n "$ADDITIONAL_CONFIG" ] && [ "$ADDITIONAL_CONFIG" != "{}" ]; then
    CMD_ARRAY+=("--additional-config" "$ADDITIONAL_CONFIG")
fi

# Speculative decoding config
if [ -n "$SPECULATIVE_CONFIG" ] && [ "$SPECULATIVE_CONFIG" != "{}" ]; then
    CMD_ARRAY+=("--speculative-config" "$SPECULATIVE_CONFIG")
fi

echo "Executing: ${CMD_ARRAY[@]}"
echo "=================================="
exec "${CMD_ARRAY[@]}"
