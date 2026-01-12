FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.11.0-neuronx-py312-sdk2.27.0-ubuntu24.04

# Build arguments
ARG MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ARG PORT=8080
ARG MAX_NUM_SEQS=4
ARG MAX_MODEL_LEN=1024
ARG TENSOR_PARALLEL_SIZE=8

# Set environment variables from build args
ENV MODEL="${MODEL}"
ENV PORT="${PORT}"
ENV MAX_NUM_SEQS="${MAX_NUM_SEQS}"
ENV MAX_MODEL_LEN="${MAX_MODEL_LEN}"
ENV TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}"

EXPOSE ${PORT}

# Launch vLLM OpenAI-compatible server with tool choice support
# Note: For Llama 3.1+ models, use --tool-call-parser llama3_json
# For Hermes models, use --tool-call-parser hermes
CMD ["sh", "-c", "python -m vllm.entrypoints.openai.api_server --model=\"${MODEL}\" --max-num-seqs=${MAX_NUM_SEQS} --max-model-len=${MAX_MODEL_LEN} --tensor-parallel-size=${TENSOR_PARALLEL_SIZE} --port=${PORT} --enable-auto-tool-choice --tool-call-parser llama3_json --no-enable-prefix-caching --additional-config='{\"override_neuron_config\":{\"enable_bucketing\":false}}'"]
