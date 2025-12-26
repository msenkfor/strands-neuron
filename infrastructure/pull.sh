aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.11.0-neuronx-py312-sdk2.27.0-ubuntu24.04
