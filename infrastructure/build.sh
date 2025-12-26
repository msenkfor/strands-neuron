#!/bin/bash

# Default values (can be overridden via environment variables)
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8080}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"

docker build \
    --build-arg MODEL="${MODEL}" \
    --build-arg PORT="${PORT}" \
    --build-arg MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    --build-arg MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    --build-arg TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    -t vllm-server-strands \
    -f Dockerfile .

