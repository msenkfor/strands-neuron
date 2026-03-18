#!/bin/bash
# Disaggregated vLLM server entrypoint
# Runs inside the disagg Docker container.
#
# Roles:
#   SEND=1  -> prefill node  (kv_producer, port 8100, Neuron cores 0-31 if SINGLE_INSTANCE=1)
#   SEND=0  -> decode node   (kv_consumer, port 8200, Neuron cores 32-63 if SINGLE_INSTANCE=1)
#
# KV cache is transferred via a shared directory (KV_CACHE_PATH).
# Both containers must mount the same host directory at that path.

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
TP_DEGREE="${TP_DEGREE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL_PATH="${MODEL_PATH:-/model}"
COMPILED_MODEL_PATH="${COMPILED_MODEL_PATH:-/compiled-model}"

# =============================================================================
# Disaggregation Configuration
# =============================================================================
SEND="${SEND:-0}"
SINGLE_INSTANCE="${SINGLE_INSTANCE:-0}"

# Shared directory for KV cache transfer between prefill and decode
KV_CACHE_PATH="${KV_CACHE_PATH:-/kv-cache}"

# =============================================================================
# Neuron Runtime Environment Variables
# =============================================================================
export NEURON_COMPILED_ARTIFACTS="$COMPILED_MODEL_PATH"
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2

# =============================================================================
# Role Selection: prefill (SEND=1) vs decode (SEND=0)
# =============================================================================
if [ "$SEND" = "1" ]; then
    PORT=8100
    ROLE="prefill (kv_producer)"
    KV_ROLE="kv_producer"
    if [ "$SINGLE_INSTANCE" = "1" ]; then
        export NEURON_RT_VISIBLE_CORES=0-31
    fi
else
    PORT=8200
    ROLE="decode (kv_consumer)"
    KV_ROLE="kv_consumer"
    if [ "$SINGLE_INSTANCE" = "1" ]; then
        export NEURON_RT_VISIBLE_CORES=32-63
    fi
fi

TRANSFER_CONFIG='{
    "kv_connector": "SharedStorageConnector",
    "kv_role": "'"$KV_ROLE"'",
    "kv_connector_extra_config": {"shared_storage_path": "'"$KV_CACHE_PATH"'"}
}'

echo "=================================="
echo "Starting Disaggregated vLLM Server"
echo "=================================="
echo "Role:               $ROLE"
echo "Port:               $PORT"
echo "Model Path:         $MODEL_PATH"
echo "Compiled Artifacts: $COMPILED_MODEL_PATH"
echo "KV Cache Path:      $KV_CACHE_PATH"
echo "TP Degree:          $TP_DEGREE"
echo "Batch Size:         $BATCH_SIZE"
if [ "$SINGLE_INSTANCE" = "1" ]; then
    echo "Neuron Cores:       $NEURON_RT_VISIBLE_CORES"
fi
echo "=================================="

python3 -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --max-num-seqs "$BATCH_SIZE" \
      --max-model-len 16384 \
      --tensor-parallel-size "$TP_DEGREE" \
      --no-enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser llama3_json \
      --kv-transfer-config "$TRANSFER_CONFIG" \
      --additional-config '{"max_prompt_length": 8192}' \
      --port "$PORT"
