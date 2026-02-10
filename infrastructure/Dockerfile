FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.11.0-neuronx-py312-sdk2.27.0-ubuntu24.04

# Copy startup script and make it executable
COPY start-vllm.sh /app/start-vllm.sh
RUN chmod +x /app/start-vllm.sh

# Create configs directory for mounting config files
RUN mkdir -p /app/configs

# Build arguments for baking defaults into the image
ARG MODEL="mistralai/Mistral-7B-Instruct-v0.3"
ARG PORT="8080"
ARG MAX_NUM_SEQS="4"
ARG MAX_MODEL_LEN="1024"
ARG TENSOR_PARALLEL_SIZE="8"
ARG ENABLE_TOOL_CALLING="true"
ARG TOOL_CALL_PARSER="llama3_json"
ARG ENABLE_PREFIX_CACHING="false"
ARG NEURON_CONFIG_FLAG="additional-config"
ARG VLLM_USE_V1="1"

# Default configuration (can be overridden via environment variables or config file)
# See vllm-config.env for all available options
# Priority: Runtime env vars > --env-file > CONFIG_FILE mount > Build-time defaults
ENV MODEL="${MODEL}"
ENV PORT="${PORT}"
ENV MAX_NUM_SEQS="${MAX_NUM_SEQS}"
ENV MAX_MODEL_LEN="${MAX_MODEL_LEN}"
ENV TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}"
ENV ENABLE_TOOL_CALLING="${ENABLE_TOOL_CALLING}"
ENV TOOL_CALL_PARSER="${TOOL_CALL_PARSER}"
ENV ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}"
ENV OVERRIDE_NEURON_CONFIG='{"override_neuron_config":{"enable_bucketing":false}}'
# Flag name for neuron config: "override-neuron-config" (newer vLLM) or "additional-config" (older)
ENV NEURON_CONFIG_FLAG="${NEURON_CONFIG_FLAG}"

# Speculative decoding config (JSON format, empty to disable)
ENV SPECULATIVE_CONFIG=""

# vLLM version control (set to 0 for speculative decoding compatibility)
ENV VLLM_USE_V1="${VLLM_USE_V1}"

# Optional: Path to config file inside container (can be mounted at runtime)
# If set, the startup script will source this file
ENV CONFIG_FILE=""


# KV cache transfer (optional - for distributed setups)
ENV ENABLE_KV_TRANSFER="false"
ENV KV_CONNECTOR="NeuronConnector"
ENV KV_ROLE="kv_producer"
ENV KV_BUFFER_SIZE="2e11"
ENV ETCD=""

EXPOSE ${PORT}

# Launch vLLM server using the startup script
# The script provides flexible configuration through environment variables
#
# Tool Calling Configuration:
#   - Correctly configured with --enable-auto-tool-choice and --tool-call-parser
#   - Supports llama3_json, hermes, mistral parsers
#
# Parallel Tool Calling Limitations:
#   - Llama 3.1 models: Only support SINGLE tool calls at once
#   - Llama 4 models: Support parallel tool calls
#   - Alternative models: Granite 3.1, xLAM support parallel calls
#
# Usage:
#   1. Use default config: docker run <image>
#   2. Override via env vars: docker run -e MAX_NUM_SEQS=8 <image>
#   3. Use config file: docker run --env-file configs/tool-calling.env <image>
#   4. Mount config file: docker run -v ./my-config.env:/app/configs/config.env -e CONFIG_FILE=/app/configs/config.env <image>
CMD ["/app/start-vllm.sh"]
