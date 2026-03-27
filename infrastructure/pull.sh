aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py311-sdk2.26.1-ubuntu22.04
