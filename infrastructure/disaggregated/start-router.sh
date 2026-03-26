#!/bin/bash
# Disaggregated inference router entrypoint.
# Forwards each request to both the prefill (port 8100) and decode (port 8200)
# vLLM servers, stitching their responses into a single stream for the caller.

set -e

if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    set -a
    source "$CONFIG_FILE"
    set +a
fi

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

exec python3 /app/neuron_proxy_server.py \
    --prefill-ip  "$PREFILL_IP" \
    --prefill-port "$PREFILL_PORT" \
    --decode-ip   "$DECODE_IP" \
    --decode-port  "$DECODE_PORT"
