"""Neuron model provider implementation."""

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar, Union, cast

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.streaming import StopReason, StreamEvent
from strands.types.tools import ToolChoice, ToolSpec
from strands.models._validation import validate_config_keys, warn_on_tool_choice_not_supported
from strands.models.model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NeuronModel(Model):
    """Neuron model provider implementation."""

    class NeuronConfig(TypedDict, total=False):
        """Configuration for NeuronModel using OpenAI Chat Completion API."""

        # Required
        model_id: str
        
        # OpenAI Chat Completion parameters
        audio: Optional[Dict[str, Any]]
        frequency_penalty: Optional[float]
        logit_bias: Optional[Dict[str, int]]
        logprobs: Optional[bool]
        max_completion_tokens: Optional[int]
        metadata: Optional[Dict[str, str]]
        modalities: Optional[List[str]]
        n: Optional[int]
        parallel_tool_calls: Optional[bool]
        prediction: Optional[Dict[str, Any]]
        presence_penalty: Optional[float]
        prompt_cache_key: Optional[str]
        prompt_cache_retention: Optional[str]
        reasoning_effort: Optional[str]
        response_format: Optional[Dict[str, Any]]
        safety_identifier: Optional[str]
        service_tier: Optional[str]
        stop: Optional[Union[str, List[str]]]
        stop_sequences: Optional[List[str]]  # Internal alias for stop (for backwards compatibility)
        store: Optional[bool]
        stream_options: Optional[Dict[str, Any]]
        temperature: Optional[float]
        tool_choice: Optional[Union[str, Dict[str, Any]]]
        tools: Optional[List[Dict[str, Any]]]
        top_logprobs: Optional[int]
        top_p: Optional[float]
        verbosity: Optional[str]
        web_search_options: Optional[Dict[str, Any]]
        
        # Client configuration
        api_key: Optional[str]
        base_url: Optional[str]
        
        # vLLM server capabilities
        support_tool_choice_auto: Optional[bool]  # Set True if vLLM has --enable-auto-tool-choice flag
        
        # Additional arguments (for any future parameters)
        additional_args: Optional[Dict[str, Any]]

    def __init__(self, config: NeuronConfig):
        """Initialize the NeuronModel with the given configuration."""
        validate_config_keys(config, self.NeuronConfig)
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not config.get("model_id"):
            raise ValueError("model_id is required")
        
        # Check if the OpenAI API server is running
        if config.get("base_url"):
            self._check_server_availability(config["base_url"])
        
        self.logger.info("Initializing NeuronModel with model: %s", config["model_id"])

    def _get_client(self) -> AsyncOpenAI:
        """Get configured AsyncOpenAI client."""
        return AsyncOpenAI(
            api_key=self.config.get("api_key", "EMPTY"),
            base_url=self.config.get("base_url", "http://localhost:8080/v1"),
        )

    def _extract_usage(self, usage: Any) -> dict[str, int]:
        """Extract usage tokens from usage object."""
        return {
            "inputTokens": getattr(usage, "prompt_tokens", 0),
            "outputTokens": getattr(usage, "completion_tokens", 0),
            "totalTokens": getattr(usage, "total_tokens", 0),
        }

    def _create_tool_block_start(self, tool_name: str, tool_use_id: str) -> StreamEvent:
        """Create a contentBlockStart event for tool use."""
        return {
            "contentBlockStart": {
                "start": {"toolUse": {"name": tool_name, "toolUseId": tool_use_id}}
            }
        }

    def _create_tool_block_delta(self, args_chunk: str) -> StreamEvent:
        """Create a contentBlockDelta event for tool use."""
        return {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": args_chunk}}
            }
        }

    def _initialize_tool_call_accumulator(self, tool_call_delta: Any) -> dict[str, Any]:
        """Initialize a tool call accumulator for tracking streaming tool calls."""
        return {
            "id": getattr(tool_call_delta, "id", None),
            "type": getattr(tool_call_delta, "type", "function"),
            "function": {
                "name": None,
                "arguments": "",
            }
        }

    def _check_server_availability(self, base_url: str, timeout: float = 5.0) -> None:
        """Check if the OpenAI API server is available.
        
        Args:
            base_url: The base URL of the OpenAI API server
            timeout: Request timeout in seconds
        """
        try:
            # Try to reach the models endpoint or base URL
            with httpx.Client(timeout=timeout) as client:
                # Most OpenAI-compatible servers have a /v1/models endpoint
                test_url = f"{base_url.rstrip('/')}/models"
                response = client.get(test_url)
                
                if response.status_code == 200:
                    self.logger.info("Successfully connected to OpenAI API server at %s", base_url)
                else:
                    self.logger.warning(
                        "OpenAI API server at %s returned status %d. Server may not be fully available.",
                        base_url,
                        response.status_code,
                    )
        except httpx.ConnectError:
            self.logger.warning(
                "Could not connect to OpenAI API server at %s. "
                "Please ensure the server is running and accessible.",
                base_url,
            )
        except httpx.TimeoutException:
            self.logger.warning(
                "Connection to OpenAI API server at %s timed out after %.1f seconds. "
                "Server may be slow or unresponsive.",
                base_url,
                timeout,
            )
        except Exception as e:
            self.logger.warning(
                "Error checking OpenAI API server at %s: %s. Proceeding anyway.",
                base_url,
                str(e),
            )

    @override
    def update_config(self, **model_config: Unpack[NeuronConfig]) -> None:  # type: ignore[override]
        validate_config_keys(model_config, self.NeuronConfig)
        
        # Check if base_url is being updated
        if "base_url" in model_config and model_config["base_url"]:
            self._check_server_availability(model_config["base_url"])
        
        self.config.update(model_config)

    @override
    def get_config(self) -> NeuronConfig:
        return self.config

    def _format_request_message_contents(self, role: str, content: ContentBlock) -> list[dict[str, Any]]:
        if "text" in content:
            return [{"role": role, "content": content["text"]}]
        if "image" in content:
            return [{"role": role, "images": [content["image"]["source"]["bytes"]]}]
        if "document" in content:
            doc = content["document"]
            name = doc.get("name", "document")
            fmt = doc.get("format", "unknown")
            text = f"[Attached document: {name} ({fmt})]"
            return [{"role": role, "content": text}]
        if "toolUse" in content:
            tool_use = content["toolUse"]
            # Use 'name' if available, otherwise fall back to 'toolUseId'
            tool_name = tool_use.get("name", tool_use["toolUseId"])
            return [
                {
                    "role": role,
                    "tool_calls": [
                        {
                            "id": tool_use["toolUseId"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_use["input"]) if isinstance(tool_use["input"], dict) else tool_use["input"],
                            }
                        }
                    ],
                }
            ]
        if "toolResult" in content:
            tool_result_block = content["toolResult"]
            tool_use_id = tool_result_block["toolUseId"]

            # Combine all tool result contents into a single message
            # OpenAI expects one tool message per tool_call_id
            result_parts = []
            has_images = False
            images = []

            for tool_result in tool_result_block["content"]:
                if "json" in tool_result:
                    result_parts.append(json.dumps(tool_result["json"]))
                elif "text" in tool_result:
                    result_parts.append(tool_result["text"])
                elif "image" in tool_result:
                    has_images = True
                    images.append(tool_result["image"]["source"]["bytes"])

            # Create a single tool message
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_use_id,
            }

            # Add content if we have text/json results
            if result_parts:
                tool_message["content"] = "\n".join(result_parts)
            else:
                # OpenAI requires content field, use empty string if no text
                tool_message["content"] = ""

            # Add images if present (some OpenAI-compatible APIs support this)
            if has_images:
                tool_message["images"] = images

            return [tool_message]
        raise TypeError(f"Unsupported content type: {next(iter(content))}")

    def _format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        system_message = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return system_message + [
            formatted_message
            for message in messages
            for content in message["content"]
            for formatted_message in self._format_request_message_contents(message["role"], content)
        ]

    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Return a dictionary suitable for OpenAI Async client."""
        request: dict[str, Any] = {
            "messages": self._format_request_messages(messages, system_prompt),
            "model": self.config["model_id"],
            "stream": stream,
        }
        
        direct_params = [
            "audio",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_completion_tokens",
            "metadata",
            "modalities",
            "n",
            "parallel_tool_calls",
            "prediction",
            "presence_penalty",
            "prompt_cache_key",
            "prompt_cache_retention",
            "reasoning_effort",
            "response_format",
            "safety_identifier",
            "service_tier",
            "store",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "verbosity",
            "web_search_options",
        ]
        
        # Add all parameters that are present in config
        for param in direct_params:
            if (value := self.config.get(param)) is not None:
                request[param] = value
        
        # Handle special cases
        
        # stop parameter (supports both 'stop' and 'stop_sequences' for backwards compatibility)
        if stop_value := (self.config.get("stop") or self.config.get("stop_sequences")):
            request["stop"] = stop_value
        
        # stream_options only included when streaming
        if stream and self.config.get("stream_options"):
            request["stream_options"] = self.config["stream_options"]
        
        # Handle tool_specs (convert to OpenAI tools format)
        if tool_specs:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["inputSchema"]["json"],
                    }
                }
                for t in tool_specs
            ]
        
        # Add any additional arguments (these can override defaults)
        if additional_args := self.config.get("additional_args"):
            request.update(additional_args)
        
        return request

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format vLLM/OpenAI response events into standardized message chunks.

        Args:
            event: A response event from the vLLM/OpenAI model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "text":
                    return {"contentBlockStart": {"start": {}}}

                tool_name = event["data"].function.name
                return {"contentBlockStart": {"start": {"toolUse": {"name": tool_name, "toolUseId": tool_name}}}}

            case "content_delta":
                if event["data_type"] == "text":
                    return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

                tool_arguments = event["data"].function.arguments
                return {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_arguments)}}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                reason: StopReason
                if event["data"] == "tool_use":
                    reason = "tool_use"
                elif event["data"] == "length":
                    reason = "max_tokens"
                else:
                    reason = "end_turn"

                return {"messageStop": {"stopReason": reason}}

            case "metadata":
                usage = event.get("usage")
                usage_dict = self._extract_usage(usage) if usage else {
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "totalTokens": 0,
                }

                return {
                    "metadata": {
                        "usage": usage_dict,
                        "metrics": {
                            "latencyMs": event.get("latency_ms", 0),
                        },
                    },
                }

            case _:
                raise RuntimeError(f"Unknown chunk_type: {event['chunk_type']}")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: Optional[List[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        warn_on_tool_choice_not_supported(tool_choice)

        request = self.format_request(messages, tool_specs, system_prompt, stream=True)
        client = self._get_client()

        tool_requested = False
        finish_reason: str | None = None
        usage_data = None
        start_time = time.time()

        # Track tool calls as they stream in (OpenAI sends them incrementally)
        tool_calls_accumulator: dict[int, dict[str, Any]] = {}

        # Track whether we've started a text content block
        text_block_started = False
        has_content = False

        yield self.format_chunk({"chunk_type": "message_start"})

        stream_response = await client.chat.completions.create(**request)
        async for chunk in stream_response:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                # Start text content block on first text delta
                if not text_block_started:
                    yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
                    text_block_started = True

                has_content = True
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": delta.content})

            if delta.tool_calls:
                # Close text content block if it was started before tool calls
                if text_block_started:
                    yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
                    text_block_started = False

                for tool_call_delta in delta.tool_calls:
                    index = tool_call_delta.index

                    # Initialize tool call accumulator for this index if first time
                    if index not in tool_calls_accumulator:
                        tool_calls_accumulator[index] = self._initialize_tool_call_accumulator(tool_call_delta)

                        # Emit contentBlockStart when we first see this tool call
                        if tool_call_delta.function and tool_call_delta.function.name:
                            tool_name = tool_call_delta.function.name
                            tool_calls_accumulator[index]["function"]["name"] = tool_name
                            tool_use_id = getattr(tool_call_delta, "id", tool_name)
                            yield self._create_tool_block_start(tool_name, tool_use_id)
                            tool_requested = True

                    # Accumulate arguments
                    if tool_call_delta.function and tool_call_delta.function.arguments:
                        args_chunk = tool_call_delta.function.arguments
                        tool_calls_accumulator[index]["function"]["arguments"] += args_chunk

                        # Emit contentBlockDelta with the new chunk of arguments
                        yield self._create_tool_block_delta(args_chunk)

            # Capture finish reason
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data = chunk.usage

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Close text content block if it's still open
        if text_block_started:
            yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

        # Emit contentBlockStop for each tool call
        for _ in tool_calls_accumulator:
            yield {"contentBlockStop": {}}

        stop_reason = "tool_use" if tool_requested else self._convert_finish_reason(finish_reason)
        yield self.format_chunk({"chunk_type": "message_stop", "data": stop_reason})

        yield self.format_chunk({
            "chunk_type": "metadata",
            "usage": usage_data,
            "latency_ms": latency_ms
        })
    
    def _convert_finish_reason(self, finish_reason: str | None) -> StopReason:
        """Convert OpenAI finish_reason to Strands StopReason."""
        if not finish_reason:
            return "end_turn"
        
        reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
            "function_call": "tool_use",
        }
        
        return reason_map.get(finish_reason, "end_turn")

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        """Get structured output from the model using tool calling.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Dictionary with the structured output.

        Raises:
            ValueError: If the model fails to return a valid tool call or the output cannot be parsed.
        """
        # Convert Pydantic model to tool specification
        tool_spec: ToolSpec = {
            "name": output_model.__name__,
            "description": f"Return a {output_model.__name__}",
            "inputSchema": {"json": output_model.model_json_schema()},
        }
        
        # Build request with tool specification and force tool use
        request = self.format_request(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            stream=False
        )
        request["tool_choice"] = {"type": "function", "function": {"name": tool_spec["name"]}}
        
        # Create client and make request
        client = self._get_client()
        
        try:
            response = await client.chat.completions.create(**request)
        except Exception as e:
            raise ValueError(f"Failed to get structured output from model: {e}") from e
        
        # Extract and validate the tool call response
        parsed: T | None = None
        
        # Check for multiple choices
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the model response.")
        
        # Find the first choice with tool_calls
        for choice in response.choices:
            message = choice.message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                
                # Validate tool name matches expected
                if tool_call.function.name != tool_spec["name"]:
                    raise ValueError(
                        f"Expected tool '{tool_spec['name']}', got '{tool_call.function.name}'"
                    )
                
                # Parse and validate the tool arguments
                try:
                    # Try to parse the arguments string as JSON first
                    args_dict = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments

                    # Some models might wrap the response in extra structure, try to extract the actual data
                    # If the response has 'parameters' or 'properties' field, use that instead
                    if isinstance(args_dict, dict):
                        if "parameters" in args_dict and isinstance(args_dict["parameters"], dict):
                            args_dict = args_dict["parameters"]
                        elif "properties" in args_dict and isinstance(args_dict["properties"], dict):
                            args_dict = args_dict["properties"]

                    parsed = output_model.model_validate(args_dict)
                except Exception as e:
                    # Provide more context in the error message
                    raise ValueError(
                        f"Failed to validate output model: {e}\n"
                        f"Raw arguments received: {tool_call.function.arguments}"
                    ) from e
                break
        
        # Yield the parsed output
        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid tool use or tool use input was found in the model response.")