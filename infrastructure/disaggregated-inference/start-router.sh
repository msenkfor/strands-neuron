#!/bin/bash
# Disaggregated inference router (proxy server) entrypoint
# Forwards requests to both the prefill and decode vLLM servers,
# then returns the combined response to the caller.
#
# The router listens on ROUTER_PORT (default 8000).
# Prefill server is expected on PREFILL_IP:PREFILL_PORT (default 127.0.0.1:8100)
# Decode server is expected on DECODE_IP:DECODE_PORT  (default 127.0.0.1:8200)

set -e

# =============================================================================
# Load Config File (if specified)
# =============================================================================
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    set -a
    source "$CONFIG_FILE"
    set +a
fi

# =============================================================================
# Configuration
# =============================================================================
PREFILL_IP="${PREFILL_IP:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_IP="${DECODE_IP:-127.0.0.1}"
DECODE_PORT="${DECODE_PORT:-8200}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

echo "=================================="
echo "Starting Disaggregated Inference Router"
echo "=================================="
echo "Prefill: ${PREFILL_IP}:${PREFILL_PORT}"
echo "Decode:  ${DECODE_IP}:${DECODE_PORT}"
echo "Router:  0.0.0.0:${ROUTER_PORT}"
echo "=================================="

python3 /app/neuron_proxy_server.py \
    --prefill-ip "$PREFILL_IP" \
    --decode-ip "$DECODE_IP" \
    --prefill-port "$PREFILL_PORT" \
    --decode-port "$DECODE_PORT"
