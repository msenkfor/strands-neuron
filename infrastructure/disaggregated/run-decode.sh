#!/bin/bash
# Run the decode server for disaggregated inference.
# Uses the AWS Neuron vLLM DLC directly — no custom image build required.
#
# Usage:
#   ETCD=<proxy-ip>:8989 HF_TOKEN=<token> ./run-decode.sh
#   ETCD=<proxy-ip>:8989 HF_TOKEN=<token> NEURON_COMPILED_ARTIFACTS=/path ./run-decode.sh

set -e

DLC_IMAGE="${DLC_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py311-sdk2.26.1-ubuntu22.04}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-decode}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/../configs/disaggregated-decode.env}"
START_SCRIPT="${SCRIPT_DIR}/../start-vllm.sh"

# Load config to get PORT for the port mapping
if [ -f "$CONFIG_FILE" ]; then
    set -a; source "$CONFIG_FILE"; set +a
fi
PORT="${PORT:-8200}"

if [ -z "$ETCD" ]; then
    echo "Error: ETCD must be set to <proxy-ip>:8989"
    echo "  Example: ETCD=10.0.0.1:8989 HF_TOKEN=hf_... ./run-decode.sh"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set."
fi

DEVICE_FLAGS=""
for i in {0..15}; do
    if [ -e "/dev/neuron${i}" ]; then
        DEVICE_FLAGS="${DEVICE_FLAGS} --device=/dev/neuron${i}"
    fi
done

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "=================================="
echo "Starting Decode Server"
echo "Image:  ${DLC_IMAGE}"
echo "Config: ${CONFIG_FILE}"
echo "Port:   ${PORT}"
echo "ETCD:   ${ETCD}"
echo "=================================="

docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --device /dev/infiniband/uverbs0 \
    --shm-size=10g \
    -p "${PORT}:${PORT}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    --env-file "${CONFIG_FILE}" \
    -e ETCD="${ETCD}" \
    -e VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference" \
    -e NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT="45645" \
    -e NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED="1" \
    -e NEURON_RT_VISIBLE_CORES="0-31" \
    ${NEURON_COMPILED_ARTIFACTS:+-e NEURON_COMPILED_ARTIFACTS="${NEURON_COMPILED_ARTIFACTS}"} \
    -v "${START_SCRIPT}:/app/start-vllm.sh:ro" \
    "${DLC_IMAGE}" \
    bash -c "python -m vllm.entrypoints.openai.api_server \
    --model \$MODEL \
    --max-num-seqs \$MAX_NUM_SEQS \
    ${DEVICE_FLAGS} \
    --max-model-len \$MAX_MODEL_LEN \
    --tensor-parallel-size 32 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{\"kv_connector\":\"NeuronConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":2e11,\"etcd\":\"\$ETCD\", \"neuron_core_offset\":0, \"kv_buffer_device\":\"cpu\",\"kv_rank\":1}' \
    --port \$PORT"
