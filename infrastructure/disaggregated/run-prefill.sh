#!/bin/bash
# Run the prefill server for disaggregated inference.
# Uses the AWS Neuron vLLM DLC directly — no custom image build required.
#
# Usage:
#   ETCD=<proxy-ip>:8989 HF_TOKEN=<token> ./run-prefill.sh
#   ETCD=<proxy-ip>:8989 HF_TOKEN=<token> NEURON_COMPILED_ARTIFACTS=/path ./run-prefill.sh

set -e

DLC_IMAGE="${DLC_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.1-ubuntu22.04}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-prefill}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/../configs/disaggregated-prefill.env}"
START_SCRIPT="${SCRIPT_DIR}/../start-vllm.sh"

# Load config to get PORT for the port mapping
if [ -f "$CONFIG_FILE" ]; then
    set -a; source "$CONFIG_FILE"; set +a
fi
PORT="${PORT:-8100}"

if [ -z "$ETCD" ]; then
    echo "Error: ETCD must be set to <proxy-ip>:8989"
    echo "  Example: ETCD=10.0.0.1:8989 HF_TOKEN=hf_... ./run-prefill.sh"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set."
fi

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "=================================="
echo "Starting Prefill Server"
echo "Image:  ${DLC_IMAGE}"
echo "Config: ${CONFIG_FILE}"
echo "Port:   ${PORT}"
echo "ETCD:   ${ETCD}"
echo "=================================="

docker run -d \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --device /dev/infiniband/uverbs0 \
    --shm-size=10g \
    -p "${PORT}:${PORT}" \
    -e HF_TOKEN="${HF_TOKEN}" \
    --env-file "${CONFIG_FILE}" \
    -e ETCD="${ETCD}" \
    ${NEURON_COMPILED_ARTIFACTS:+-e NEURON_COMPILED_ARTIFACTS="${NEURON_COMPILED_ARTIFACTS}"} \
    -v "${START_SCRIPT}:/app/start-vllm.sh:ro" \
    "${DLC_IMAGE}" \
    /app/start-vllm.sh
