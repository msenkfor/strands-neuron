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
| `MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Model ID from HuggingFace |
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
| `MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | Model ID from HuggingFace |
| `PORT` | `8080` | Server port |
| `MAX_NUM_SEQS` | `4` | Maximum concurrent sequences (batch size) |
| `MAX_MODEL_LEN` | `1024` | Maximum sequence length |
| `TENSOR_PARALLEL_SIZE` | `8` | Number of Neuron cores for tensor parallelism |
| `ENABLE_TOOL_CALLING` | `true` | Enable tool/function calling support |
| `TOOL_CALL_PARSER` | `llama3_json` | Parser: `llama3_json`, `hermes`, `mistral`, etc. |
| `ENABLE_PREFIX_CACHING` | `false` | Cache prefixes for repeated prompts |
| `OVERRIDE_NEURON_CONFIG` | (empty) | Neuron config overrides (JSON) |
| `SPECULATIVE_CONFIG` | (empty) | Speculative decoding config (JSON) |
| `VLLM_USE_V1` | `0` | vLLM V1 engine; must be `0` for Neuron in the 0.9.1 image |
| `ENABLE_KV_TRANSFER` | `false` | Enable KV cache transfer (disaggregated inference) |
| `KV_CONNECTOR` | `NeuronConnector` | KV connector type |
| `KV_ROLE` | `kv_producer` | KV role: `kv_producer` or `kv_consumer` |
| `KV_BUFFER_SIZE` | `2e11` | KV buffer size in bytes |
| `KV_NEURON_CORE_OFFSET` | `0` | Absolute host logical NeuronCore index for KV transfer |
| `ETCD` | (empty) | etcd server address (`<host-ip>:8989`) |
| `NEURON_RT_VISIBLE_CORES` | (empty) | Physical NeuronCore range for this worker (e.g. `0-31`) |

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

### Disaggregated Inference (Prefill / Decode Split)

Disaggregated inference splits prefill (KV cache generation) and decode (token generation) into
separate workers connected via EFA. See [DISAGGREGATED_INFERENCE.md](../DISAGGREGATED_INFERENCE.md)
for full architecture details and troubleshooting.

> **Requires**: Two `trn2.48xlarge` instances with EFA enabled at launch, and the
> `0.9.1-neuronx-py310-sdk2.25.0-ubuntu22.04` base image (`NeuronConnector` is absent from newer images).

#### Step 1 — Launch two instances with EFA enabled

Launch two `trn2.48xlarge` instances using the Neuron DLAMI, same VPC/subnet/security group.
For each instance: **Advanced network configuration → Enable Elastic Fabric Adapter**.
The security group needs a self-referencing all-traffic inbound rule.

#### Step 2 — On the prefill instance: pull and build

```bash
cd infrastructure
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.0-ubuntu22.04
./build.sh configs/distributed-kv.env
```

#### Step 3 — On the decode instance: copy repo, pull and build

```bash
# From the prefill instance
scp -r ~/strands-neuron ubuntu@<decode-ip>:~/strands-neuron

# On the decode instance
cd strands-neuron/infrastructure
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.0-ubuntu22.04
./build.sh configs/kv-consumer.env
```

#### Step 4 — On the prefill instance: start etcd

```bash
export HOST_IP=$(hostname -I | awk '{print $1}')

docker rm -f etcd 2>/dev/null || true

docker run -d --name etcd \
  -p 8989:8989 \
  quay.io/coreos/etcd:v3.5.0 \
  etcd \
  --listen-client-urls http://0.0.0.0:8989 \
  --advertise-client-urls http://${HOST_IP}:8989
```

#### Step 5 — On the prefill instance: start prefill worker (Terminal 1)

```bash
HF_TOKEN=<your-token> CONTAINER_NAME=vllm-prefill ./run.sh configs/distributed-kv.env
```

First run compiles the model — expect **20–40 minutes**. Wait for:
```
INFO: Application startup complete.
```

#### Step 6 — On the decode instance: start decode worker (Terminal 1)

```bash
HF_TOKEN=<your-token> CONTAINER_NAME=vllm-decode ./run.sh configs/kv-consumer.env
```

Wait for `Application startup complete.`

#### Step 7 — On the prefill instance: start the proxy server (Terminal 2)

The official `neuron-proxy-server` (bundled in the vLLM image) handles request routing with the correct `request_id` encoding — prefill receives decode's address and decode receives prefill's address, so each worker knows where to push/pull the KV cache via EFA. It discovers workers automatically via etcd.

```bash
export HOST_IP=$(hostname -I | awk '{print $1}')

docker rm -f proxy 2>/dev/null || true

docker run -d \
  --name proxy \
  --network=host \
  -e ETCD_IP=${HOST_IP} \
  -e ETCD_PORT=8989 \
  vllm-server-strands \
  bash -c "neuron-proxy-server --etcd \$ETCD_IP:\$ETCD_PORT"
```

The proxy listens on port **8000**.

#### Step 8 — Test

Send requests to the proxy — do **not** call prefill or decode directly:

```bash
export HOST_IP=$(hostname -I | awk '{print $1}')

curl http://${HOST_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

The proxy:
1. Discovers prefill and decode workers via etcd
2. Sends prefill the request with decode's address in the `request_id` (so prefill knows where to push KV cache)
3. Sends decode the request with prefill's address in the `request_id` (so decode knows where to pull KV cache from)
4. Streams the response from decode back to the client

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
| `distributed-kv.env` | Disaggregated inference — prefill / KV producer worker |
| `kv-consumer.env` | Disaggregated inference — decode / KV consumer worker |
| `speculative-decoding.env` | Speculative decoding for improved latency |

## Troubleshooting

### Terminal closes on build failure

The build script includes error handling that pauses on failure. If the build fails, you'll see "Build failed! Press Enter to close..." and can review the error before the terminal closes.

### JSON quoting issues

- **For `--env-file`**: Don't use outer quotes around JSON values
- **For `CONFIG_FILE` (sourced)**: Use single quotes around JSON values
- **The start script** automatically strips surrounding quotes if present

### VLLM_USE_V1 error

- **Speculative decoding** (`speculative-decoding.env`): requires `VLLM_USE_V1=1`
- **Disaggregated inference** (`distributed-kv.env` / `kv-consumer.env`): requires `VLLM_USE_V1=0` — the 0.9.1 Neuron image does not support the V1 engine on Neuron

### Speculative decoding errors

- Ensure `SPECULATIVE_CONFIG` is passed as a separate argument (not nested in `OVERRIDE_NEURON_CONFIG`)
- The neuron config should include `"enable_fused_speculation": true`
- Use `MAX_NUM_SEQS=1` for speculative decoding

### Disaggregated inference: `No matching NIC found for referred NIC BDF`

The instance was launched with insufficient EFA NICs. Each NeuronDevice requires an adjacent EFA
NIC (by PCI BDF). A `trn2.48xlarge` needs ~8 EFA NICs for full coverage of all 16 NeuronDevices.
Use two separate instances (one per worker), each launched with EFA enabled.
See the [Disaggregated Inference guide](../DISAGGREGATED_INFERENCE.md) for details.
