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
