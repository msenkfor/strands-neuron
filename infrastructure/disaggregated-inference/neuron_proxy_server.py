# SPDX-License-Identifier: Apache-2.0
# Source: https://github.com/aws-neuron/upstreaming-to-vllm/blob/neuron-2.24-vllm-v0.7.2/vllm/neuron_immediate_first_token_proxy_server.py

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from builtins import anext

from quart import Quart, make_response, request

from vllm.logger import init_logger

# skip aiohttp from isort since it conflicts with yapf
import aiohttp  # isort: skip
'''
A Proxy Server for Disaggregated Inference that has the following features
1. Immediate return of token(s) from prefill server
2. Handles both streaming and non-streaming use cases
3. Forwards requests to both prefill and decode immediately before processing
   prefill output
4. Makes responses from the prefill and decode server appear as if they are
   coming from a single server
'''

# Configure logging
logger = init_logger(__name__)

# Default to WARNING, can be changed to DEBUG for more verbose logging
logger.setLevel(logging.WARNING)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)


async def forward_request(url, data, request_id, request_type="unknown"):
    logger.debug("Starting %s request to %s", request_type, url)
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                "X-Request-Id": f"{request_id}",
            }
            logger.debug("%s request attempting connection", request_type)
            try:
                async with session.post(url=url, json=data,
                                        headers=headers) as response:
                    logger.debug("%s request connected", request_type)
                    yield "connection acquired"
                    if response.status == 200:
                        async for chunk in response.content:
                            if chunk:
                                if isinstance(chunk, bytes):
                                    chunk = chunk.decode('utf-8')
                                yield chunk
                    else:
                        error_text = await response.text()
                        logger.error("Error from %s (%s): %s", url,
                                     request_type, error_text)
                        yield json.dumps({
                            "error":
                            f"Request failed with status {response.status}",
                            "details": error_text
                        }) + '\n'
            except Exception as e:
                logger.error("Request failed for %s: %s", request_type, str(e))
                yield json.dumps({"error": str(e)}) + '\n'
    except Exception as e:
        logger.error("Session creation failed for %s: %s", request_type,
                     str(e))
        yield json.dumps({"error": str(e)}) + '\n'


async def handle_prefill_response(prefill_response, streaming, endpoint,
                                  request_id, request_time):
    logger.debug("Request at %s: Starting prefill", request_time)
    async for chunk in prefill_response:
        if chunk and streaming:
            if "[DONE]" in chunk:
                continue
            chunk = replace_key_in_data_string(chunk, "id", request_id)
            chunk = replace_key_in_data_string(chunk, "finish_reason", None)
            logger.debug("prefill chunk %s", chunk)
            yield chunk
        else:
            pass


async def handle_decode_response(decode_response, streaming, endpoint,
                                 request_id, request_time):
    logger.debug("Request at %s: Starting decode", request_time)
    seen_indices: set[str] = set()
    async for chunk in decode_response:
        if chunk and streaming:
            index = read_key_from_data_string(chunk, "index")
            logger.debug("index: %s, seen_indices: %s", index, seen_indices)
            if index in seen_indices:
                chunk = replace_key_in_data_string(chunk, "id", request_id)
                logger.debug("decode chunk %s", chunk)
                yield chunk
            else:
                seen_indices.add(index)
                logger.debug("skipping decode chunk %s", chunk)
        elif chunk:
            yield chunk


@app.route('/v1/models', methods=['GET'])
async def handle_models():
    prefill_url = f"http://{app.args.prefill_ip}:{app.args.prefill_port}/v1/models"
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(prefill_url) as response:
                data = await response.json()
                return data, response.status
    except Exception as e:
        return {"error": str(e)}, 503


@app.route('/v1/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
async def handle_request():
    endpoint = request.path
    request_time = str(time.time())
    logger.debug("Processing request at %s", request_time)
    original_request_data = await request.get_json()
    streaming = original_request_data.get('stream', False)
    prefill_request = original_request_data.copy()
    prefill_request['max_tokens'] = 1

    prefill_ip = app.args.prefill_ip
    prefill_port = app.args.prefill_port
    decode_ip = app.args.decode_ip
    decode_port = app.args.decode_port
    uid = uuid.uuid4()
    prefill_request_id = f"cmpl-{uid}"
    decode_request_id = f"cmpl-{uid}"

    prefill_url = f"http://{prefill_ip}:{prefill_port}{endpoint}"
    decode_url = f"http://{decode_ip}:{decode_port}{endpoint}"

    if not streaming:
        # Non-streaming: fire both requests, discard prefill, return decode JSON
        try:
            prefill_response = forward_request(prefill_url, prefill_request,
                                               prefill_request_id, "prefill")
            decode_response = forward_request(decode_url, original_request_data,
                                              decode_request_id, "decode")

            prefill_task = asyncio.create_task(anext(prefill_response))
            decode_task = asyncio.create_task(anext(decode_response))

            # Wait for both connections to be established
            await asyncio.gather(prefill_task, decode_task)

            # Drain prefill in background (KV cache transfer needs it to complete)
            async def drain(gen):
                async for _ in gen:
                    pass
            asyncio.create_task(drain(prefill_response))

            # Collect full decode response body
            body_parts = []
            async for chunk in decode_response:
                if chunk:
                    body_parts.append(chunk)
            body = "".join(body_parts)

            try:
                data = json.loads(body)
                # Normalize the request id
                data["id"] = decode_request_id
                return data, 200
            except json.JSONDecodeError:
                logger.error("Failed to parse decode response: %s", body)
                return {"error": "Invalid JSON from decode server", "body": body}, 502

        except Exception as e:
            logger.exception("Error in non-streaming request at %s", request_time)
            return {"error": str(e), "timestamp": request_time}, 500

    async def streaming_responses(original_request_data, prefill_request):
        try:
            logger.info("Routing prefill request %s to %s", prefill_request_id,
                        prefill_url)
            prefill_response = forward_request(prefill_url, prefill_request,
                                               prefill_request_id, "prefill")
            logger.info("Routing decode request %s to %s", decode_request_id,
                        decode_url)
            decode_response = forward_request(decode_url,
                                              original_request_data,
                                              decode_request_id, "decode")

            prefill_task = asyncio.create_task(anext(prefill_response))
            decode_task = asyncio.create_task(anext(decode_response))

            await prefill_task
            async for chunk in handle_prefill_response(prefill_response,
                                                       streaming, endpoint,
                                                       prefill_request_id,
                                                       request_time):
                yield chunk

            await decode_task
            async for chunk in handle_decode_response(decode_response,
                                                      streaming, endpoint,
                                                      decode_request_id,
                                                      request_time):
                yield chunk

        except Exception as e:
            logger.exception("Error in request at %s", request_time)
            yield json.dumps({
                "error": str(e),
                "timestamp": request_time
            }) + '\n'

    response = await make_response(
        streaming_responses(original_request_data, prefill_request), {
            'Content-Type': 'text/event-stream',
            'Transfer-Encoding': 'chunked'
        })
    response.timeout = None
    return response


def replace_key_in_data_string(data_string, key_to_replace, new_value):
    trailing_newline = '\n' if data_string.endswith('\n') else ''
    data_string = data_string.rstrip()
    if data_string.startswith("data: "):
        json_string = data_string[6:]
        prefix = "data: "
    else:
        json_string = data_string
        prefix = ""

    def replace_nested(obj, key, value):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    obj[k] = value
                else:
                    replace_nested(v, key, value)
        elif isinstance(obj, list):
            for item in obj:
                replace_nested(item, key, value)

    try:
        data = json.loads(json_string)
        replace_nested(data, key_to_replace, new_value)
        modified_json = json.dumps(data, separators=(',', ':'))
        return f"{prefix}{modified_json}{trailing_newline}"
    except json.JSONDecodeError as e:
        logger.debug(
            "JSON decode error: %s, may be because there is no JSON "
            "in this string", e)
        return data_string


def read_key_from_data_string(data_string, key_to_read):
    data_string = data_string.strip()
    if data_string.startswith("data: "):
        json_string = data_string[6:]
    else:
        json_string = data_string

    def find_nested(obj, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key:
                    return v
                elif isinstance(v, (dict, list)):
                    result = find_nested(v, key)
                    if result is not None:
                        return result
        elif isinstance(obj, list):
            for item in obj:
                result = find_nested(item, key)
                if result is not None:
                    return result
        return None

    try:
        data = json.loads(json_string)
        result = find_nested(data, key_to_read)
        return result
    except json.JSONDecodeError as e:
        logger.debug(
            "JSON decode error: %s, may be because there is no JSON "
            "in this string", e)
        return None


def enable_debug_logging():
    """Call this function to enable debug logging"""
    logger.setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(
        description=
        'Proxy server for prefill and decode servers that immediately returns '
        'the first token.')
    parser.add_argument(
        '--prefill-ip',
        default='localhost',
        help='IP address for prefill server (default: localhost)')
    parser.add_argument(
        '--decode-ip',
        default='localhost',
        help='IP address for decode server (default: localhost)')
    parser.add_argument('--prefill-port',
                        type=int,
                        default=8100,
                        help='Port for prefill server (default: 8100)')
    parser.add_argument('--decode-port',
                        type=int,
                        default=8200,
                        help='Port for decode server (default: 8200)')
    args = parser.parse_args()
    app.args = args
    app.run(host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()
