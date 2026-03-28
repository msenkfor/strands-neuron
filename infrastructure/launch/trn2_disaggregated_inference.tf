# =============================================================================
# trn2.48xlarge Disaggregated Inference - EFA Network Configuration
# =============================================================================
# Deploys a prefill + decode node pair with 16 EFA interfaces each,
# cluster placement group, and SSH access via EIP on prefill node only.
#
# Usage:
#   terraform init
#   terraform apply -var="key_name=my-key" -var="region=us-east-2"
#
# Requirements:
#   - AWS CLI configured with sufficient IAM permissions
#   - An existing EC2 key pair in the target region
#   - Capacity reservation or EC2 Capacity Block for trn2.48xlarge
# =============================================================================

terraform {
  required_version = ">= 1.3.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

# =============================================================================
# Variables
# =============================================================================

variable "region" {
  description = "AWS region. trn2.48xlarge is currently only available in us-east-2."
  type        = string
  default     = "us-east-2"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH access."
  type        = string
}

variable "instance_type" {
  description = "Instance type. Must be trn2.48xlarge for 16-card EFA config."
  type        = string
  default     = "trn2.48xlarge"
}

variable "neuron_dlami_name_filter" {
  description = "Name filter for the Neuron Deep Learning AMI. Adjust if a newer version is available."
  type        = string
  default     = "Deep Learning AMI Neuron*Ubuntu*22.04*"
}

variable "volume_size_gb" {
  description = "Root EBS volume size in GB."
  type        = number
  default     = 512
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to SSH to the prefill node. Defaults to 0.0.0.0/0 — restrict this in production."
  type        = string
  default     = "0.0.0.0/0"
}

variable "capacity_reservation_id" {
  description = "Optional EC2 Capacity Reservation or Capacity Block ID for trn2.48xlarge."
  type        = string
  default     = ""
}

# =============================================================================
# Provider
# =============================================================================

provider "aws" {
  region = var.region
}

# =============================================================================
# Data Sources
# =============================================================================

# Latest Neuron DLAMI (Ubuntu 22.04)
data "aws_ami" "neuron_dlami" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = [var.neuron_dlami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# Default VPC
data "aws_vpc" "default" {
  default = true
}

# Subnets in default VPC — pick the first available in the region
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# =============================================================================
# Placement Group
# =============================================================================

resource "aws_placement_group" "trn2_cluster" {
  name     = "trn2-disaggregated-inference"
  strategy = "cluster"
}

# =============================================================================
# Security Group
# =============================================================================

resource "aws_security_group" "efa" {
  name        = "trn2-efa-sg"
  description = "EFA security group for trn2 disaggregated inference"
  vpc_id      = data.aws_vpc.default.id

  # SSH from specified CIDR (prefill node only gets EIP, decode is accessed via ProxyJump)
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  # All traffic within the security group (required for EFA inter-node communication)
  ingress {
    description = "All intra-cluster traffic (EFA)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All intra-cluster traffic (EFA)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  tags = {
    Name = "trn2-efa-sg"
  }
}

# =============================================================================
# Launch Template
# Encodes the 16-card EFA network configuration:
#   Card 0, Device 0 → efa (EFA+ENA, primary interface, gets IP)
#   Cards 1-15, Device 1 → efa-only (no IP, RDMA-only for KV cache transfer)
# =============================================================================

resource "aws_launch_template" "trn2" {
  name                   = "trn2-disaggregated-inference"
  image_id               = data.aws_ami.neuron_dlami.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  update_default_version = true

  placement {
    group_name = aws_placement_group.trn2_cluster.name
  }

  # Primary interface: EFA+ENA on network card 0, device index 0
  # This is the only interface with a routable IP address
  network_interfaces {
    network_card_index    = 0
    device_index          = 0
    interface_type        = "efa"
    subnet_id             = data.aws_subnets.default.ids[0]
    security_groups       = [aws_security_group.efa.id]
    delete_on_termination = true
  }

  # EFA-only interfaces for network cards 1-15
  # DeviceIndex=1 for all secondary cards (trn2/p5 pattern)
  # No IP assigned — used exclusively for EFA/RDMA KV cache transfers
  dynamic "network_interfaces" {
    for_each = range(1, 16)
    content {
      network_card_index    = network_interfaces.value
      device_index          = 1
      interface_type        = "efa-only"
      subnet_id             = data.aws_subnets.default.ids[0]
      security_groups       = [aws_security_group.efa.id]
      delete_on_termination = true
    }
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.volume_size_gb
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  # Required for EFA
  ena_support = true

  dynamic "capacity_reservation_specification" {
    for_each = var.capacity_reservation_id != "" ? [1] : []
    content {
      capacity_reservation_target {
        capacity_reservation_id = var.capacity_reservation_id
      }
    }
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "trn2-inference"
    }
  }

  tag_specifications {
    resource_type = "volume"
    tags = {
      Name = "trn2-inference-root"
    }
  }

  tags = {
    Name = "trn2-disaggregated-inference"
  }
}

# =============================================================================
# Instances
# =============================================================================

resource "aws_instance" "prefill" {
  launch_template {
    id      = aws_launch_template.trn2.id
    version = "$Latest"
  }

  tags = {
    Name = "trn2-prefill"
    Role = "prefill"
  }
}

resource "aws_instance" "decode" {
  launch_template {
    id      = aws_launch_template.trn2.id
    version = "$Latest"
  }

  tags = {
    Name = "trn2-decode"
    Role = "decode"
  }
}

# =============================================================================
# Elastic IP — prefill node only
# Decode node is accessed via SSH ProxyJump through prefill
# =============================================================================

resource "aws_eip" "prefill" {
  domain = "vpc"

  tags = {
    Name = "trn2-prefill-eip"
  }
}

resource "aws_eip_association" "prefill" {
  instance_id   = aws_instance.prefill.id
  allocation_id = aws_eip.prefill.id
}

# =============================================================================
# Outputs
# =============================================================================

output "prefill_public_ip" {
  description = "Public IP of prefill node. SSH directly to this."
  value       = aws_eip.prefill.public_ip
}

output "prefill_private_ip" {
  description = "Private IP of prefill node."
  value       = aws_instance.prefill.private_ip
}

output "decode_private_ip" {
  description = "Private IP of decode node. Access via ProxyJump through prefill."
  value       = aws_instance.decode.private_ip
}

output "ami_used" {
  description = "Neuron DLAMI resolved and used for both instances."
  value       = data.aws_ami.neuron_dlami.name
}

output "ssh_config" {
  description = "Paste this into ~/.ssh/config for easy access to both nodes."
  value       = <<-EOT
    Host trn2-prefill
        HostName ${aws_eip.prefill.public_ip}
        User ubuntu
        IdentityFile ~/.ssh/${var.key_name}.pem

    Host trn2-decode
        HostName ${aws_instance.decode.private_ip}
        User ubuntu
        IdentityFile ~/.ssh/${var.key_name}.pem
        ProxyJump trn2-prefill
  EOT
}
