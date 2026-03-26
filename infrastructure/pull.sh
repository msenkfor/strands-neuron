aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.7.2-neuronx-py310-sdk2.24.1-ubuntu22.04
