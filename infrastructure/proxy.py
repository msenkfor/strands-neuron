#!/usr/bin/env python3
"""
Disaggregated inference proxy for vLLM NeuronConnector.

Routes each request to both prefill (kv_producer) and decode (kv_consumer)
simultaneously using a shared request_id that encodes the prefill address.
The response is returned from the decode worker.

Usage:
    pip install fastapi uvicorn httpx
    python proxy.py

    # Or with custom ports:
    PREFILL_URL=http://localhost:8080 DECODE_URL=http://localhost:8082 python proxy.py

Then point your NeuronModel at http://localhost:8000/v1
"""

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PREFILL_URL = os.environ.get("PREFILL_URL", "http://localhost:8080")
DECODE_URL = os.environ.get("DECODE_URL", "http://localhost:8082")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))

# Prefill host:port extracted for the request_id (must be the API address, not ZMQ)
PREFILL_HOST = os.environ.get("PREFILL_HOST", "172.31.31.191")
PREFILL_API_PORT = os.environ.get("PREFILL_API_PORT", "8080")

app = FastAPI()


def make_request_id() -> str:
    """Build a request_id that encodes the prefill address.

    NeuronConnector parses this on the decode side to find prefill's ZMQ server
    at prefill_host:(prefill_api_port + 1).

    Format: <uuid>_<prefill_host>:<prefill_api_port>
    """
    return f"{uuid.uuid4()}_{PREFILL_HOST}:{PREFILL_API_PORT}"


async def _forward(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    headers: dict,
    content: bytes,
    stream: bool,
) -> httpx.Response:
    request = client.build_request(method, url, headers=headers, content=content)
    return await client.send(request, stream=stream)


async def _drain(response: httpx.Response) -> None:
    """Consume and discard a streaming response."""
    try:
        async for _ in response.aiter_bytes():
            pass
    except Exception:
        pass
    finally:
        await response.aclose()


async def _stream_decode(client: httpx.AsyncClient, response: httpx.Response) -> AsyncIterator[bytes]:
    """Yield chunks from the decode response, then close client."""
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str) -> Response:
    body = await request.body()
    method = request.method

    # Pass-through non-completion endpoints (models list, health, etc.)
    is_completion = path in ("v1/chat/completions", "v1/completions")

    # Build forwarding headers (strip host)
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    if not is_completion or method != "POST":
        # Forward directly to prefill for non-inference endpoints
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)) as client:
            resp = await _forward(
                client, method,
                f"{PREFILL_URL}/{path}",
                forward_headers, body, stream=False,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    # --- Disaggregated inference path ---
    request_id = make_request_id()
    forward_headers["x-request-id"] = request_id
    logger.info("Routing request_id=%s to prefill+decode", request_id)

    prefill_url = f"{PREFILL_URL}/{path}"
    decode_url = f"{DECODE_URL}/{path}"

    # Build prefill body: max_tokens=1 so prefill only generates the first token,
    # transfers KV cache to decode, then is immediately free for the next request.
    # Decode gets the original body with the full max_tokens.
    try:
        prefill_body = json.dumps({**json.loads(body), "max_tokens": 1, "stream": False}).encode()
    except (json.JSONDecodeError, ValueError):
        prefill_body = body

    # Client is NOT used as a context manager here — it must stay open until
    # streaming is complete. It is closed inside _stream_decode or explicitly below.
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None))

    prefill_task = asyncio.create_task(
        _forward(client, "POST", prefill_url, forward_headers, prefill_body, stream=False)
    )
    decode_task = asyncio.create_task(
        _forward(client, "POST", decode_url, forward_headers, body, stream=True)
    )

    prefill_resp, decode_resp = await asyncio.gather(prefill_task, decode_task)

    # Prefill response is a small non-streaming response (max_tokens=1) — close it directly
    asyncio.create_task(prefill_resp.aclose())

    content_type = decode_resp.headers.get("content-type", "")
    wants_stream = (
        "text/event-stream" in request.headers.get("accept", "")
        or content_type.startswith("text/event-stream")
    )

    if wants_stream:
        return StreamingResponse(
            _stream_decode(client, decode_resp),
            status_code=decode_resp.status_code,
            headers={
                k: v for k, v in decode_resp.headers.items()
                if k.lower() not in ("transfer-encoding",)
            },
            media_type=content_type or "text/event-stream",
        )
    else:
        try:
            content = await decode_resp.aread()
        finally:
            await decode_resp.aclose()
            await client.aclose()
        return Response(
            content=content,
            status_code=decode_resp.status_code,
            headers={
                k: v for k, v in decode_resp.headers.items()
                if k.lower() not in ("transfer-encoding",)
            },
        )


if __name__ == "__main__":
    import uvicorn
    logger.info("Disaggregated proxy starting on port %s", PROXY_PORT)
    logger.info("  Prefill: %s (request_id encodes %s:%s)", PREFILL_URL, PREFILL_HOST, PREFILL_API_PORT)
    logger.info("  Decode:  %s", DECODE_URL)
    logger.info("  Point NeuronModel at: http://localhost:%s/v1", PROXY_PORT)
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
