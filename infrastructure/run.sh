#!/bin/bash

# Check if container with name vllm-server-strands is already running
if docker ps -a --format '{{.Names}}' | grep -q '^vllm-server-strands$'; then
    echo "Container vllm-server-strands already exists. Removing it..."
    docker rm -f vllm-server-strands
fi

docker run -it \
    -e HF_TOKEN=$HF_TOKEN \
    --device=/dev/neuron0 \
    --device=/dev/neuron1 \
    --device=/dev/neuron2 \
    --device=/dev/neuron3 \
    --device=/dev/neuron4 \
    --device=/dev/neuron5 \
    --device=/dev/neuron6 \
    --device=/dev/neuron7 \
    --device=/dev/neuron8 \
    --device=/dev/neuron9 \
    --device=/dev/neuron10 \
    --device=/dev/neuron11 \
    --device=/dev/neuron12 \
    --device=/dev/neuron13 \
    --device=/dev/neuron14 \
    --device=/dev/neuron15 \
    --cap-add SYS_ADMIN \
    --cap-add IPC_LOCK \
    -p 8080:8080 \
    --name vllm-server-strands \
    vllm-server-strands

