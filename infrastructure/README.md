# strands-neuron

AWS [vLLM on Neuron infrastructure](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/vllm/index.html) provider for Strands Agents SDK.

## Quick Start

```bash
cd infrastructure
```

### 1. Pull the vLLM Neuron image

```bash
. pull.sh
```

### 2. Build the Docker image

```bash
. build.sh
```

Customize with environment variables:

```bash
MODEL="meta-llama/Llama-3.1-8B-Instruct" PORT=8080 . build.sh
```

### 3. Run the server

Use your `huggingface token` token for HF_TOKEN:

```bash
HF_TOKEN=<> . run.sh
```

### 4. Test the endpoint

```bash
. test.sh
```
