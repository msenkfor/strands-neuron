#!/bin/bash
# Compile entrypoint for Neuron inference_demo
# Runs inside the compile Docker container.
# All flags mirror compile.sh exactly — override via environment variables.

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
# Core Configuration (Required)
# =============================================================================
TP_DEGREE="${TP_DEGREE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MODEL_PATH="${MODEL_PATH:-/model}"
MODEL_PATH="${MODEL_PATH%/}"   # strip trailing slash
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

# =============================================================================
# Resolve HuggingFace model ID to local path
# inference_demo only accepts local paths — HF model IDs must be downloaded first.
# If HF_CACHE_DIR is mounted at /root/.cache/huggingface this is a no-op on re-runs.
# =============================================================================
if [[ "$MODEL_PATH" != /* ]]; then
    echo "Downloading HuggingFace model: $MODEL_PATH"
    MODEL_PATH=$(python3 -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download('$MODEL_PATH', token=os.environ.get('HF_TOKEN'))
print(path)
")
    echo "Resolved model path: $MODEL_PATH"
fi

# =============================================================================
# Derived Paths (matches compile.sh naming convention)
# =============================================================================
export COMPILED_MODEL_PATH="${OUTPUT_DIR}/di_traced_model_tp${TP_DEGREE}_b${BATCH_SIZE}/"

echo "=================================="
echo "Compiling Model for Neuron"
echo "=================================="
echo "Model Path:          $MODEL_PATH"
echo "TP Degree:           $TP_DEGREE"
echo "Batch Size:          $BATCH_SIZE"
echo "Compiled Model Path: $COMPILED_MODEL_PATH"
echo "=================================="

inference_demo \
   --model-type llama \
   --task-type causal-lm \
   run \
   --model-path "$MODEL_PATH" \
   --compiled-model-path "$COMPILED_MODEL_PATH" \
   --torch-dtype bfloat16 \
   --tp-degree "$TP_DEGREE" \
   --batch-size "$BATCH_SIZE" \
   --ctx-batch-size 1 \
   --tkg-batch-size "$BATCH_SIZE" \
   --is-continuous-batching \
   --max-context-length 16384 \
   --seq-len 16384 \
   --on-device-sampling \
   --fused-qkv \
   --global-topk 256 --dynamic \
   --top-k 50 --top-p 0.9 --temperature 0.7 \
   --do-sample \
   --sequence-parallel-enabled \
   --qkv-kernel-enabled \
   --attn-kernel-enabled \
   --mlp-kernel-enabled \
   --cc-pipeline-tiling-factor 1 \
   --pad-token-id 2 \
   --logical-neuron-cores 2 \
   --context-encoding-buckets 256 512 1024 2048 4096 8192 16384 \
   --token-generation-buckets 512 1024 2048 4096 8192 16384 \
   --apply-seq-ids-mask \
   --enable-bucketing \
   --prompt "test prompt" \
   --save-sharded-checkpoint \
   --attn-block-tkg-nki-kernel-enabled \
   --attn-block-tkg-nki-kernel-cache-update \
   --k-cache-transposed \
   --async-mode \
   --compile-only

echo "=================================="
echo "Compilation complete: $COMPILED_MODEL_PATH"
echo "=================================="
