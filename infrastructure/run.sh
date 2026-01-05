#!/bin/bash

# Build device flags for existing neuron devices
DEVICE_FLAGS=""
for i in {0..15}; do
    if [ -e "/dev/neuron${i}" ]; then
        DEVICE_FLAGS="${DEVICE_FLAGS} --device=/dev/neuron${i}"
    fi
done

docker run -it \
    -e HF_TOKEN=$HF_TOKEN \
    ${DEVICE_FLAGS} \
    --cap-add SYS_ADMIN \
    --cap-add IPC_LOCK \
    -p 8080:8080 \
    --name vllm-server-strands \
    vllm-server-strands

