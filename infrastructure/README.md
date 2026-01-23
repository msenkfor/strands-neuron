# strands-neuron

AWS [vLLM on Neuron infrastructure](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/vllm/index.html) provider for Strands Agents SDK.

## Prerequisites

### Launch an EC2 instance with Neuron DLAMI

Before getting started, launch an EC2 instance using the AWS Neuron Deep Learning AMI (DLAMI) for Ubuntu 22.04.

See the official setup instructions: [Multi-Framework Ubuntu 22 Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami)

## Quick Start

```bash
cd infrastructure
```

### 1. Pull the vLLM Neuron image from [AWS Neuron Deep Learning Containers](https://github.com/aws-neuron/deep-learning-containers/tree/main?tab=readme-ov-file#vllm-inference-neuronx)

```bash
. pull.sh
```

### 2. Build the Docker image

```bash
# Default build
./build.sh

# Build with a specific configuration
./build.sh configs/tool-calling.env

# Build with environment variable
CONFIG_FILE=configs/speculative-decoding.env ./build.sh
```

### 3. Run the server

The vLLM server supports flexible configuration. Choose the method that works best for you:

#### Option A: Default Configuration

```bash
HF_TOKEN=<your-token> ./run.sh
```

#### Option B: Use a Configuration Profile (Recommended)

We provide optimized profiles in the `configs/` directory:

```bash
# Basic configuration (no tool calling)
HF_TOKEN=<your-token> ./run.sh configs/basic.env

# Tool calling enabled (Strands agents) - Recommended
HF_TOKEN=<your-token> ./run.sh configs/tool-calling.env

# High throughput (production workloads)
HF_TOKEN=<your-token> ./run.sh configs/high-throughput.env

# Distributed setup with KV cache transfer
HF_TOKEN=<your-token> ./run.sh configs/distributed-kv.env

# Speculative decoding for improved latency
HF_TOKEN=<your-token> ./run.sh configs/speculative-decoding.env
```

#### Option C: Use Custom Configuration

```bash
# Copy and customize the template
cp vllm-config.env my-config.env
# Edit my-config.env with your settings
HF_TOKEN=<your-token> ./run.sh my-config.env
```

#### Option D: Override Individual Settings

```bash
HF_TOKEN=<your-token> \
MAX_NUM_SEQS=8 \
MODEL="meta-llama/Llama-3.1-70B-Instruct" \
./run.sh
```

### 4. Test the endpoint

```bash
. test.sh
```

## Build Configuration

The `build.sh` script accepts configuration in multiple ways:

### Build with Config File

```bash
# Pass config file as argument
./build.sh configs/tool-calling.env

# Or set via environment variable
CONFIG_FILE=configs/speculative-decoding.env ./build.sh
```

### Build with Individual Overrides

```bash
MODEL="Qwen/Qwen3-32B" \
TENSOR_PARALLEL_SIZE=64 \
MAX_MODEL_LEN=6400 \
./build.sh
```

### Build Arguments

The following values can be baked into the image at build time:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model ID from HuggingFace |
| `PORT` | `8080` | Server port |
| `MAX_NUM_SEQS` | `4` | Maximum concurrent sequences |
| `MAX_MODEL_LEN` | `2048` | Maximum sequence length |
| `TENSOR_PARALLEL_SIZE` | `8` | Number of Neuron cores |
| `ENABLE_TOOL_CALLING` | `true` | Enable tool/function calling |
| `TOOL_CALL_PARSER` | `llama3_json` | Tool call parser type |
| `ENABLE_PREFIX_CACHING` | `false` | Enable prefix caching |
| `VLLM_USE_V1` | `1` | vLLM V1 engine (required for this version) |

## Run Configuration

### Runtime Environment Variables

These can be passed at runtime via `--env-file` or `-e` flags:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Model ID from HuggingFace |
| `PORT` | `8080` | Server port |
| `MAX_NUM_SEQS` | `4` | Maximum concurrent sequences (batch size) |
| `MAX_MODEL_LEN` | `1024` | Maximum sequence length |
| `TENSOR_PARALLEL_SIZE` | `8` | Number of Neuron cores for tensor parallelism |
| `ENABLE_TOOL_CALLING` | `true` | Enable tool/function calling support |
| `TOOL_CALL_PARSER` | `llama3_json` | Parser: `llama3_json`, `hermes`, `mistral`, etc. |
| `ENABLE_PREFIX_CACHING` | `false` | Cache prefixes for repeated prompts |
| `ADDITIONAL_CONFIG` | (empty) | Neuron config overrides (JSON) |
| `SPECULATIVE_CONFIG` | (empty) | Speculative decoding config (JSON) |
| `VLLM_USE_V1` | `1` | vLLM V1 engine control |
| `ENABLE_KV_TRANSFER` | `false` | Enable KV cache transfer (distributed) |
| `KV_CONNECTOR` | `NeuronConnector` | KV connector type |
| `KV_ROLE` | `kv_producer` | KV role: `kv_producer` or `kv_consumer` |
| `KV_BUFFER_SIZE` | `2e11` | KV buffer size in bytes |
| `ETCD` | (empty) | etcd server address for KV coordination |

### Configuration Methods

#### Method 1: Using `--env-file` (Recommended)

```bash
docker run --env-file configs/tool-calling.env <image>
```

**Note:** When using `--env-file`, values are read literally. For JSON values, do NOT use quotes:

```env
# Correct for --env-file
ADDITIONAL_CONFIG={"override_neuron_config": {"enable_bucketing": false}}

# Incorrect - quotes will be included literally
ADDITIONAL_CONFIG='{"override_neuron_config": {"enable_bucketing": false}}'
```

#### Method 2: Using CONFIG_FILE (for bash sourcing)

Mount a config file and set `CONFIG_FILE` to source it inside the container:

```bash
docker run \
  -v ./my-config.env:/app/configs/config.env \
  -e CONFIG_FILE=/app/configs/config.env \
  <image>
```

**Note:** When using `CONFIG_FILE`, the file is sourced by bash, so use proper shell quoting:

```env
# Correct for CONFIG_FILE (bash sourcing)
ADDITIONAL_CONFIG='{"override_neuron_config": {"enable_bucketing": false}}'
```

#### Method 3: Override Environment Variables

```bash
docker run \
  -e MODEL="meta-llama/Llama-3.1-70B-Instruct" \
  -e MAX_NUM_SEQS=8 \
  -e MAX_MODEL_LEN=2048 \
  -e TENSOR_PARALLEL_SIZE=16 \
  <image>
```

## Advanced Configuration

### Speculative Decoding

Speculative decoding uses a smaller draft model to improve latency. See `configs/speculative-decoding.env` for an example.

```env
# Model configuration
MODEL=Qwen/Qwen3-32B
MAX_NUM_SEQS=1
MAX_MODEL_LEN=6400
TENSOR_PARALLEL_SIZE=64

# Neuron config for speculative decoding
ADDITIONAL_CONFIG='{"override_neuron_config": {"save_sharded_checkpoint": true, "enable_fused_speculation": true}}'

# Speculative decoding config (separate argument)
SPECULATIVE_CONFIG='{"model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 7, "max_model_len": 2048, "method": "eagle"}'

# Required for speculative decoding
VLLM_USE_V1=1
```

Reference: [AWS Neuron Speculative Decoding Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/speculative_decoding.html)

### Neuron Configuration Overrides

Use `ADDITIONAL_CONFIG` to pass neuron-specific settings:

```env
# Enable bucketing for variable sequence lengths
ADDITIONAL_CONFIG='{"override_neuron_config": {"enable_bucketing": true}}'

# Configure context encoding buckets
ADDITIONAL_CONFIG='{"override_neuron_config": {"enable_bucketing": true, "context_encoding_buckets": [256, 512, 1024, 2048]}}'
```

### KV Cache Transfer (Distributed)

For distributed setups with KV cache transfer:

```env
ENABLE_KV_TRANSFER=true
KV_CONNECTOR=NeuronConnector
KV_ROLE=kv_producer
KV_BUFFER_SIZE=2e11
ETCD=http://etcd-server:2379
```

## Tool Calling Configuration

**Parser Selection:**
- **Llama 3.1+**: Use `TOOL_CALL_PARSER=llama3_json` (default)
- **Hermes models**: Use `TOOL_CALL_PARSER=hermes`
- **Mistral models**: Use `TOOL_CALL_PARSER=mistral`
- **Other models**: Check [vLLM documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)

**Parallel Tool Calling Limitations:**
- **Llama 3.1 models**: Only support SINGLE tool calls at once
- **Llama 4 models**: Support parallel tool calls
- **Other models**: Granite 3.1, xLAM support parallel calls

If you encounter `"This model only supports single tool-calls at once!"` errors, this is a model limitation. See the [main README](../README.md#parallel-tool-calling-support) for workarounds.

## Configuration Files

Pre-configured profiles in `configs/`:

| File | Description |
|------|-------------|
| `basic.env` | Basic configuration without tool calling |
| `tool-calling.env` | Tool calling enabled (recommended for Strands agents) |
| `high-throughput.env` | Optimized for production workloads |
| `distributed-kv.env` | Distributed setup with KV cache transfer |
| `speculative-decoding.env` | Speculative decoding for improved latency |

## Troubleshooting

### Terminal closes on build failure

The build script includes error handling that pauses on failure. If the build fails, you'll see "Build failed! Press Enter to close..." and can review the error before the terminal closes.

### JSON quoting issues

- **For `--env-file`**: Don't use outer quotes around JSON values
- **For `CONFIG_FILE` (sourced)**: Use single quotes around JSON values
- **The start script** automatically strips surrounding quotes if present

### VLLM_USE_V1 assertion error

This vLLM version requires `VLLM_USE_V1=1`. If you see an assertion error about `VLLM_USE_V1`, ensure it's set to `1`.

### Speculative decoding errors

- Ensure `SPECULATIVE_CONFIG` is passed as a separate argument (not nested in `ADDITIONAL_CONFIG`)
- The neuron config should include `"enable_fused_speculation": true`
- Use `MAX_NUM_SEQS=1` for speculative decoding
