#!/bin/bash
# Run the etcd service and neuron-proxy-server for disaggregated inference.
# Run this on your proxy instance (e.g., m5.xlarge) — not on the Neuron instances.
#
# Usage:
#   ./run-router.sh
#
# After this starts, set ETCD=<this-host-ip>:8989 on your prefill/decode instances.

set -e

DLC_IMAGE="${DLC_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.7.2-neuronx-py310-sdk2.24.1-ubuntu22.04}"
ETCD_PORT="${ETCD_PORT:-8989}"
PROXY_PORT="${PROXY_PORT:-8000}"
HOST_IP="${HOST_IP:-$(hostname -i | awk '{print $1}')}"

# Remove old containers
docker rm -f etcd proxy 2>/dev/null || true

echo "=================================="
echo "Starting etcd"
echo "Host IP: ${HOST_IP} | Port: ${ETCD_PORT}"
echo "=================================="

docker run -d \
    --name etcd \
    --shm-size=10g \
    --privileged \
    -p "${ETCD_PORT}:${ETCD_PORT}" \
    -e ETCD_IP="${HOST_IP}" \
    ubuntu:22.04 \
    bash -c "apt-get update && apt-get install -y etcd && \
             exec etcd \
               --data-dir=/etcd-data \
               --listen-client-urls=http://0.0.0.0:${ETCD_PORT} \
               --advertise-client-urls=http://\$ETCD_IP:${ETCD_PORT} \
               --listen-peer-urls=http://127.0.0.1:21323 \
               --initial-advertise-peer-urls=http://127.0.0.1:21323 \
               --initial-cluster=default=http://127.0.0.1:21323 \
               --name=default"

echo "=================================="
echo "Starting neuron-proxy-server"
echo "Proxy port: ${PROXY_PORT}"
echo "=================================="

echo "=================================="
echo "Proxy endpoint: http://${HOST_IP}:${PROXY_PORT}"
echo "ETCD endpoint:  ${HOST_IP}:${ETCD_PORT}"
echo ""
echo "On each prefill/decode instance, run:"
echo "  export ETCD=${HOST_IP}:${ETCD_PORT}"
echo "=================================="

docker run --rm \
    --name proxy \
    --shm-size=10g \
    --privileged \
    -p "${PROXY_PORT}:${PROXY_PORT}" \
    -e ETCD_IP="${HOST_IP}" \
    -e ETCD_PORT="${ETCD_PORT}" \
    "${DLC_IMAGE}" \
    bash -c "exec neuron-proxy-server --etcd \$ETCD_IP:\$ETCD_PORT"
