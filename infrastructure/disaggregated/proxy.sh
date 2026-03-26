
set -e

DLC_IMAGE="${DLC_IMAGE:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.7.2-neuronx-py310-sdk2.24.1-ubuntu22.04}"
ETCD_PORT="${ETCD_PORT:-8989}"
PROXY_PORT="${PROXY_PORT:-8000}"
# HOST_IP=$(hostname -i | awk '{print $1}')
HOST_IP="${HOST_IP:-$(hostname -i | awk '{print $1}')}"

# Remove old containers
docker rm -f etcd proxy 2>/dev/null || true

# Start etcd
docker run -it \
  --name etcd \
  --shm-size=10g \
  --privileged \
  -p 8989:8989 \
  -e ETCD_IP=$HOST_IP \
  ubuntu:22.04 \
  bash -c "apt-get update && apt-get install -y etcd && \
           exec etcd \
             --data-dir=/etcd-data \
             --listen-client-urls=http://0.0.0.0:8989 \
             --advertise-client-urls=http://\$ETCD_IP:8989 \
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


# Start proxy
docker run -it \
  --name proxy \
  --shm-size=10g \
  --privileged \
  -p 8000:8000 \
  -e ETCD_IP=$HOST_IP \
  -e ETCD_PORT=8989 \
  public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.1-ubuntu22.04 \
  bash -c "exec neuron-proxy-server --etcd \$ETCD_IP:\$ETCD_PORT"