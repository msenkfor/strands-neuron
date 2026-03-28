# trn2.48xlarge Disaggregated Inference — Terraform Setup

This Terraform configuration deploys a prefill + decode node pair on AWS EC2 `trn2.48xlarge` instances, fully configured for disaggregated LLM inference using AWS Neuron SDK and EFA networking.

---

## What Gets Deployed

| Resource | Details |
|---|---|
| 2× `trn2.48xlarge` instances | One prefill node, one decode node |
| 16 EFA interfaces per instance | 1× EFA+ENA (primary) + 15× EFA-only |
| Cluster placement group | Ensures non-blocking 3.2 Tbps EFA bandwidth between nodes |
| Security group | Port 22 SSH + full intra-cluster EFA traffic |
| Elastic IP | Attached to prefill node only |
| Neuron DLAMI | Latest Ubuntu 22.04 AMI with Neuron SDK + EFA pre-installed |

---

## Prerequisites

### 1. AWS CLI and Terraform
```bash
# Verify both are installed
aws --version       # >= 2.x
terraform --version # >= 1.3.0
```

### 2. IAM Permissions
Your AWS credentials need the following permissions:
- `ec2:RunInstances`
- `ec2:DescribeInstances`
- `ec2:AllocateAddress` / `ec2:AssociateAddress`
- `ec2:CreateSecurityGroup` / `ec2:AuthorizeSecurityGroupIngress`
- `ec2:CreatePlacementGroup`
- `ec2:CreateLaunchTemplate`
- `ec2:DescribeImages`
- `ec2:DescribeSubnets` / `ec2:DescribeVpcs`

### 3. EC2 Key Pair
You need an existing key pair in `us-east-2` (Ohio). If you don't have one:
```bash
aws ec2 create-key-pair \
  --key-name trn2-key \
  --region us-east-2 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/trn2-key.pem

chmod 400 ~/.ssh/trn2-key.pem
```

### 4. Capacity Reservation
`trn2.48xlarge` instances are not available on-demand — you need an **EC2 Capacity Block** reservation. Purchase one via the AWS Console under EC2 → Capacity Blocks, then note the reservation ID.

> Without a Capacity Block, `terraform apply` will fail with an `InsufficientInstanceCapacity` error.

---

## Configuration

All variables are in `trn2_disaggregated_inference.tf`. The key ones:

| Variable | Default | Description |
|---|---|---|
| `region` | `us-east-2` | Must be us-east-2 — only region with trn2 availability |
| `key_name` | *(required)* | Your EC2 key pair name |
| `ssh_ingress_cidr` | `0.0.0.0/0` | Restrict to your IP in production |
| `capacity_reservation_id` | `""` | Your Capacity Block ID |
| `volume_size_gb` | `512` | Root EBS size — increase for large model checkpoints |
| `neuron_dlami_name_filter` | `Deep Learning AMI Neuron*Ubuntu*22.04*` | AMI name filter |

---

## Launching

### Step 1 — Initialize Terraform
```bash
terraform init
```

### Step 2 — Preview the plan
```bash
terraform plan \
  -var="key_name=trn2-key" \
  -var="ssh_ingress_cidr=$(curl -s https://checkip.amazonaws.com)/32" \
  -var="capacity_reservation_id=cr-xxxxxxxxxxxxxxxxx"
```

### Step 3 — Apply
```bash
terraform apply \
  -var="key_name=trn2-key" \
  -var="ssh_ingress_cidr=$(curl -s https://checkip.amazonaws.com)/32" \
  -var="capacity_reservation_id=cr-xxxxxxxxxxxxxxxxx"
```

Type `yes` when prompted. The apply takes 3–5 minutes. Both instances take additional time to finish booting as Neuron drivers initialize.

### Step 4 — Configure SSH
Terraform outputs a ready-to-use SSH config block. Copy it into `~/.ssh/config`:

```
Host trn2-prefill
    HostName <prefill-EIP>
    User ubuntu
    IdentityFile ~/.ssh/trn2-key.pem

Host trn2-decode
    HostName <decode-private-ip>
    User ubuntu
    IdentityFile ~/.ssh/trn2-key.pem
    ProxyJump trn2-prefill
```

Then connect:
```bash
ssh trn2-prefill   # direct SSH via EIP
ssh trn2-decode    # tunnels automatically through prefill
```

---

## Network Architecture

```
Your Laptop
     │
     │ SSH (port 22)
     ▼
┌─────────────────────────────┐
│  trn2-prefill               │
│  Public IP: <EIP>           │
│  Private IP: 172.31.x.x     │
│                             │
│  card 0  → EFA+ENA  ──────┐ │
│  card 1  → EFA-only       │ │
│  card 2  → EFA-only       │ │  3.2 Tbps EFA
│  ...                      │ │  KV cache transfer
│  card 15 → EFA-only       │ │
└───────────────────────│───┘ │
                        │
┌───────────────────────▼─────┐
│  trn2-decode                │
│  Private IP: 172.31.x.x     │
│                             │
│  card 0  → EFA+ENA          │
│  card 1  → EFA-only         │
│  ...                        │
│  card 15 → EFA-only         │
└─────────────────────────────┘

Both instances in the same:
  - Cluster Placement Group
  - Subnet (same AZ)
  - Security Group (EFA self-referencing rule)
```

### Why EFA-only on cards 1–15?
EFA-only interfaces do not get a routable IP address. This:
- Conserves IP space (only 2 private IPs used across 32 total interfaces)
- Avoids Linux routing issues (source IP mismatches, hostname resolution problems) that occur when an instance has many routable ENIs
- Is the AWS-recommended pattern for trn2 and p5 instance families

---

## Decode Node Internet Access

The decode node has no public IP and no outbound internet by default. This means it cannot `git clone`, `pip install`, or reach any external endpoints. You have two options:

---

### Option A — NAT Gateway (recommended)

Gives the decode node outbound internet without exposing it publicly. This is the cleanest solution and the right long-term architecture.

Add the following to `trn2_disaggregated_inference.tf` and run `terraform apply`:

```hcl
resource "aws_eip" "nat" {
  domain = "vpc"
  tags = {
    Name = "trn2-nat-eip"
  }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = data.aws_subnets.default.ids[0]

  tags = {
    Name = "trn2-nat-gateway"
  }

  depends_on = [data.aws_subnets.default]
}

resource "aws_route_table" "private" {
  vpc_id = data.aws_vpc.default.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "trn2-private-route-table"
  }
}

resource "aws_route_table_association" "decode" {
  subnet_id      = data.aws_subnets.default.ids[0]
  route_table_id = aws_route_table.private.id
}
```

> NAT Gateways cost ~$0.045/hr plus data transfer charges. Negligible relative to trn2 instance cost, but tear it down when not in use.

---

### Option B — Temporary EIP on decode node

Attach an EIP directly to the decode node just long enough to install dependencies, then remove it. No Terraform changes needed.

```bash
# Allocate EIP
EIP=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)

# Get decode node's primary ENI (network card 0)
ENI=$(aws ec2 describe-instances \
  --filters "Name=private-ip-address,Values=<decode-private-ip>" \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.NetworkCardIndex==`0`].NetworkInterfaceId' \
  --output text)

# Attach EIP
aws ec2 associate-address --allocation-id $EIP --network-interface-id $ENI
```

Once dependencies are installed, disassociate and release to avoid charges:

```bash
aws ec2 disassociate-address --allocation-id $EIP
aws ec2 release-address --allocation-id $EIP
```

---

### Why not just rsync from prefill to decode?

You can clone on prefill and rsync to decode, but it requires SSH agent forwarding which has its own complexity:

```bash
# On your laptop — start SSH agent and add key
eval $(ssh-agent -s)
ssh-add ~/.ssh/my-key.pem        # Linux
ssh-add --apple-use-keychain ~/.ssh/my-key.pem  # macOS

# Connect to prefill with agent forwarding
ssh -A Trn2-prefill

# From inside prefill — rsync to decode using private IP
rsync -avz vllm-neuron/ ubuntu@<decode-private-ip>:~/vllm-neuron/
```

Add `ForwardAgent yes` to your SSH config to make this permanent:

```
Host Trn2-prefill
    HostName <prefill-EIP>
    User ubuntu
    IdentityFile ~/.ssh/my-key.pem
    ForwardAgent yes
    AddKeysToAgent yes
```

> Do not copy your private key onto the prefill instance. Always use agent forwarding instead.

---

## SSH Configuration

The SSH config is case-sensitive. The `Host` entry and any `ProxyJump` reference must use the **exact same casing**. The recommended config for your `~/.ssh/config` on your laptop:

```
Host Trn2-prefill
    HostName <prefill-EIP>
    User ubuntu
    IdentityFile ~/.ssh/my-key.pem
    ForwardAgent yes
    AddKeysToAgent yes

Host Trn2-decode
    HostName <decode-private-ip>
    User ubuntu
    IdentityFile ~/.ssh/my-key.pem
    ProxyJump Trn2-prefill
```

Replace `<prefill-EIP>` and `<decode-private-ip>` with the real values from Terraform output:

```bash
terraform output prefill_public_ip
terraform output decode_private_ip
```

To work on both nodes simultaneously, open two terminal windows and SSH to each independently. The ProxyJump on the decode entry handles the tunnel automatically in the background — the prefill terminal does not need to be open.

---

## Running Disaggregated Inference

Once both instances are up and you've SSH'd in, the high-level flow using AWS Neuron NxD Inference:

### On both nodes — verify EFA
```bash
fi_info -p efa | grep "provider:"
# Should show 16 entries: efa_0 through efa_15
```

### On prefill node — start prefill server
```bash
MODEL_PATH=/path/to/your/model

python -m neuronx_distributed_inference.models.llama.prefill_server \
  --model-path $MODEL_PATH \
  --tensor-parallel-size 32 \
  --port 8100
```

### On decode node — start decode server
```bash
python -m neuronx_distributed_inference.models.llama.decode_server \
  --model-path $MODEL_PATH \
  --tensor-parallel-size 32 \
  --port 8200
```

### On prefill node — start the router
```bash
pip install quart

neuron-proxy-server \
  --prefill-ip <prefill-private-ip> \
  --decode-ip <decode-private-ip> \
  --prefill-port 8100 \
  --decode-port 8200
```

Refer to the [AWS Neuron Disaggregated Inference tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/disaggregated-inference-tutorial.html) for model compilation steps and full benchmarking instructions.

---

## Teardown

```bash
terraform destroy \
  -var="key_name=trn2-key" \
  -var="capacity_reservation_id=cr-xxxxxxxxxxxxxxxxx"
```

> The Capacity Block reservation itself is not managed by Terraform and must be cancelled separately in the AWS Console if no longer needed.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `InsufficientInstanceCapacity` | No Capacity Block | Purchase a Capacity Block in us-east-2 |
| `ssh: connect to host ... Operation timed out` | No public IP or missing IGW route | Confirm EIP is associated; check VPC main route table has `0.0.0.0/0 → igw-xxx` |
| `Could not resolve hostname Trn2-decode` | SSH config casing mismatch or placeholder not replaced | Ensure `Host` and `ProxyJump` use identical casing; replace `172.31.x.x` with actual private IP from `terraform output decode_private_ip` |
| `Permission denied (publickey)` when rsyncing from prefill to decode | Key not forwarded to prefill session | Use `ssh -A Trn2-prefill` or add `ForwardAgent yes` to SSH config; never copy your private key onto the instance |
| `Could not open a connection to your authentication agent` | SSH agent not running on laptop | Run `eval $(ssh-agent -s)` then `ssh-add ~/.ssh/my-key.pem` before connecting |
| `Failed to connect to github.com` on decode node | Decode node has no outbound internet | Add NAT Gateway (Option A) or temporary EIP (Option B) — see Decode Node Internet Access section |
| `fi_info -p efa` shows fewer than 16 devices | EFA not enabled or interfaces not attached | Verify launch template network interface config; check instance was launched via CLI/Terraform not console |
| Neuron driver errors on boot | Instance still initializing | Wait 3–5 minutes after SSH access before running Neuron commands |