"""Neuron-hosted vLLM model provider using the OpenAI-compatible API surface.

- Docs: https://platform.openai.com/docs/overview
"""

import base64
import json
import importlib.util
import logging
import mimetypes
import httpx
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypedDict, TypeVar, cast, Optional

import openai
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from pydantic import BaseModel, ValidationError
from typing_extensions import Unpack, override
from openai import AsyncOpenAI

from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolResult, ToolSpec, ToolUse
from strands.models._validation import validate_config_keys
from strands.models.model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NeuronServerUnavailable(RuntimeError):
    """Raised when the Neuron-backed vLLM server cannot be reached."""


class NeuronModel(Model):
    """Model provider for vLLM running on AWS Neuron using the OpenAI-compatible API."""

    class NeuronConfig(TypedDict, total=False):
        """Configuration for NeuronModel.

        Attributes:
            model_id: Model ID to use (e.g., "mistralai/Mistral-7B-Instruct-v0.3").
            base_url: Base URL for the vLLM server (default: "http://localhost:8080/v1").
            api_key: API key for authentication (default: "EMPTY" for local vLLM).
            params: Additional model parameters (e.g., temperature, max_tokens).
            streaming: Whether to use streaming mode (default: True). Set to False to work around
                vLLM streaming bugs with certain models (e.g., Mistral tool calls).
        """

        model_id: str
        base_url: Optional[str]
        api_key: Optional[str]
        params: Optional[dict[str, Any]]
        streaming: Optional[bool]


    def __init__(self, config: NeuronConfig):
        """Initialize the NeuronModel with the given configuration.

        Args:
            config: Configuration dictionary for the Neuron model.

        Raises:
            ValueError: If model_id is not provided.
        """
        validate_config_keys(config, self.NeuronConfig)

        if not config.get("model_id"):
            raise ValueError("model_id is required")

        self._validate_hardware()
        logger.info("Initializing NeuronModel with model: %s", config["model_id"])

        self.config = {
            "model_id": config["model_id"],
            "base_url": config.get("base_url", "http://localhost:8080/v1"),
            "api_key": config.get("api_key", "EMPTY"),
            "params": config.get("params", {}),
            "streaming": config.get("streaming", True),
        }
        self._check_server_online()
    
    def _validate_hardware(self) -> None:
        """Validate that Neuron hardware or libraries are available."""
        if importlib.util.find_spec("torch_neuronx") is not None:
            logger.info("Neuron hardware validation passed")
        else:
            logger.warning("Neuron libraries not available - running in compatibility mode")

    @override
    def update_config(self, **model_config: Unpack[NeuronConfig]) -> None:
        """Update the Neuron model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.NeuronConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> NeuronConfig:
        """Get the Neuron model configuration.

        Returns:
            The Neuron model configuration.
        """
        return cast(NeuronModel.NeuronConfig, self.config)

    def _check_server_online(self) -> None:
        """Best-effort check that the backing server is reachable.

        Logs a friendly warning instead of raising, so offline environments (e.g., unit tests)
        continue to work, but users still get a clear signal when the endpoint is down.
        """
        base_url = self.config["base_url"].rstrip("/")
        probe_urls = [f"{base_url}/models", base_url]
        last_error: Exception | None = None

        for url in probe_urls:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code < 500:
                    return
                last_error = RuntimeError(f"HTTP {resp.status_code}")
            except Exception as exc:  # noqa: BLE001 - we want a friendly message regardless of exception type
                last_error = exc

        logger.warning(
            "Neuron server not reachable at %s (last error: %s). "
            "Requests may fail until the service is available.",
            base_url,
            last_error or "unknown error",
        )

    def _raise_server_unavailable(self, exc: Exception) -> "NoReturn":  # type: ignore[name-defined]
        """Raise a user-friendly exception when the server is unreachable."""
        base_url = self.config["base_url"]
        msg = (
            f"Neuron server not reachable at {base_url}. "
            "Start the server or set OPENAI_API_BASE_URL/NEURON_VLLM_MODEL_ID to a reachable endpoint. "
            f"Original error: {exc}"
        )
        raise NeuronServerUnavailable(msg) from None

    @classmethod
    def format_request_message_content(cls, content: ContentBlock, **kwargs: Any) -> dict[str, Any]:
        """Format an OpenAI compatible content block.

        Args:
            content: Message content.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            OpenAI compatible content block.

        Raises:
            TypeError: If the content block type cannot be converted to an OpenAI-compatible format.
        """
        if "document" in content:
            mime_type = mimetypes.types_map.get(f".{content['document']['format']}", "application/octet-stream")
            file_data = base64.b64encode(content["document"]["source"]["bytes"]).decode("utf-8")
            return {
                "file": {
                    "file_data": f"data:{mime_type};base64,{file_data}",
                    "filename": content["document"]["name"],
                },
                "type": "file",
            }

        if "image" in content:
            mime_type = mimetypes.types_map.get(f".{content['image']['format']}", "application/octet-stream")
            image_data = base64.b64encode(content["image"]["source"]["bytes"]).decode("utf-8")

            return {
                "image_url": {
                    "detail": "auto",
                    "format": mime_type,
                    "url": f"data:{mime_type};base64,{image_data}",
                },
                "type": "image_url",
            }

        if "text" in content:
            return {"text": content["text"], "type": "text"}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    @classmethod
    def format_request_message_tool_call(cls, tool_use: ToolUse, **kwargs: Any) -> dict[str, Any]:
        """Format an OpenAI compatible tool call.

        Args:
            tool_use: Tool use requested by the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            OpenAI compatible tool call.
        """
        return {
            "function": {
                "arguments": json.dumps(tool_use["input"]),
                "name": tool_use["name"],
            },
            "id": tool_use["toolUseId"],
            "type": "function",
        }

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult, **kwargs: Any) -> dict[str, Any]:
        """Format an OpenAI compatible tool message.

        Args:
            tool_result: Tool result collected from a tool execution.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            OpenAI compatible tool message.
        """
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": [cls.format_request_message_content(content) for content in contents],
        }

    @classmethod
    def _split_tool_message_images(cls, tool_message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Split a tool message into text-only tool message and optional user message with images.

        OpenAI API restricts images to user role messages only. This method extracts any image
        content from a tool message and returns it separately as a user message.

        Args:
            tool_message: A formatted tool message that may contain images.

        Returns:
            A tuple of (tool_message_without_images, user_message_with_images_or_None).
        """
        if tool_message.get("role") != "tool":
            return tool_message, None

        content = tool_message.get("content", [])
        if not isinstance(content, list):
            return tool_message, None

        text_content = []
        image_content = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                image_content.append(item)
            else:
                text_content.append(item)

        if not image_content:
            return tool_message, None

        logger.warning(
            "tool_call_id=<%s> | Moving image from tool message to a new user message for OpenAI compatibility",
            tool_message["tool_call_id"],
        )

        text_content.append(
            {
                "type": "text",
                "text": (
                    "Tool successfully returned an image. The image is being provided in the following user message."
                ),
            }
        )

        tool_message_clean = {
            "role": "tool",
            "tool_call_id": tool_message["tool_call_id"],
            "content": text_content,
        }

        user_message_with_images = {"role": "user", "content": image_content}

        return tool_message_clean, user_message_with_images

    @classmethod
    def _format_request_tool_choice(cls, tool_choice: ToolChoice | None) -> dict[str, Any]:
        """Format a tool choice for the OpenAI-compatible payload.

        OpenAI's SDK uses literal strings for tool choice instead of constants.

        Args:
            tool_choice: Tool choice configuration from the Strands schema.

        Returns:
            OpenAI compatible tool choice format.
        """
        if not tool_choice:
            return {}

        match tool_choice:
            case {"auto": _}:
                return {"tool_choice": "auto"}
            case {"any": _}:
                return {"tool_choice": "required"}
            case {"tool": {"name": tool_name}}:
                return {"tool_choice": {"type": "function", "function": {"name": tool_name}}}
            case _:
                return {"tool_choice": "auto"}

    @classmethod
    def _format_system_messages(
        cls,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format system messages for OpenAI-compatible providers.

        Args:
            system_prompt: System prompt to provide context to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted system messages.
        """
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        return [
            {"role": "system", "content": content["text"]}
            for content in system_prompt_content or []
            if "text" in content
        ]

    @classmethod
    def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
        """Format regular messages for OpenAI-compatible providers.

        Args:
            messages: List of message objects to be processed by the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            List of formatted messages.
        """
        formatted_messages = []

        for message in messages:
            contents = message["content"]

            if any("reasoningContent" in content for content in contents):
                logger.warning(
                    "reasoningContent is not supported in multi-turn conversations with the Chat Completions API."
                )

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse", "reasoningContent"])
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)

            for tool_msg in formatted_tool_messages:
                tool_msg_clean, user_msg_with_images = cls._split_tool_message_images(tool_msg)
                formatted_messages.append(tool_msg_clean)
                if user_msg_with_images:
                    formatted_messages.append(user_msg_with_images)

        return formatted_messages

    @classmethod
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Format an OpenAI compatible messages array.

        Args:
            messages: List of message objects to be processed by the model.
            system_prompt: System prompt to provide context to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            An OpenAI compatible messages array.
        """
        formatted_messages = cls._format_system_messages(system_prompt, system_prompt_content=system_prompt_content)
        formatted_messages.extend(cls._format_regular_messages(messages))

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    def format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the OpenAI-compatible payload sent to the Neuron-backed vLLM server.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            An OpenAI compatible chat streaming request.

        Raises:
            TypeError: If a message contains a content block type that cannot be converted to an OpenAI-compatible
                format.
        """
        streaming = self.config.get("streaming", True)
        request = {
            "messages": self.format_request_messages(
                messages, system_prompt, system_prompt_content=system_prompt_content
            ),
            "model": self.config["model_id"],
            "stream": streaming,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **(self._format_request_tool_choice(tool_choice)),
            **cast(dict[str, Any], self.config.get("params", {})),
        }
        if streaming:
            request["stream_options"] = {"include_usage": True}
        return request

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        """Translate a streaming event from the OpenAI client into the SDK chunk format.

        Args:
            event: A response event from the OpenAI compatible model.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as chunk_type is controlled in the stream method.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockStart": {
                            "start": {
                                "toolUse": {
                                    "name": event["data"].function.name,
                                    "toolUseId": event["data"].id,
                                }
                            }
                        }
                    }

                return {"contentBlockStart": {"start": {}}}

            case "content_delta":
                if event["data_type"] == "tool":
                    return {
                        "contentBlockDelta": {"delta": {"toolUse": {"input": event["data"].function.arguments or ""}}}
                    }

                if event["data_type"] == "reasoning_content":
                    return {"contentBlockDelta": {"delta": {"reasoningContent": {"text": event["data"]}}}}

                return {"contentBlockDelta": {"delta": {"text": event["data"]}}}

            case "content_stop":
                return {"contentBlockStop": {}}

            case "message_stop":
                match event["data"]:
                    case "tool_calls":
                        return {"messageStop": {"stopReason": "tool_use"}}
                    case "length":
                        return {"messageStop": {"stopReason": "max_tokens"}}
                    case _:
                        return {"messageStop": {"stopReason": "end_turn"}}

            case "metadata":
                return {
                    "metadata": {
                        "usage": {
                            "inputTokens": event["data"].prompt_tokens,
                            "outputTokens": event["data"].completion_tokens,
                            "totalTokens": event["data"].total_tokens,
                        },
                        "metrics": {
                            "latencyMs": 0,
                        },
                    },
                }

            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[Any]:
        """Yield a short-lived AsyncOpenAI client pointed at the Neuron vLLM endpoint.

        A new client is created per request to avoid connection sharing in the asyncio
        event loop. For more details, see https://github.com/encode/httpx/discussions/2959.

        Yields:
            Client: An OpenAI-compatible client instance.
        """
        client = AsyncOpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"],
        )
        try:
            yield client
        finally:
            await client.close()

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation responses from the Neuron-hosted vLLM model.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Formatted message chunks from the model.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by OpenAI (rate limits).
        """
        logger.debug("formatting request")
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)
        logger.debug("formatted request=<%s>", request)

        logger.debug("invoking model")

        async with self._get_client() as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    logger.warning("OpenAI threw context window overflow error")
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except (openai.APIConnectionError, httpx.ConnectError) as e:
                self._raise_server_unavailable(e)
            except openai.RateLimitError as e:
                logger.warning("OpenAI threw rate limit error")
                raise ModelThrottledException(str(e)) from e

            logger.debug("got response from model")

            streaming = self.config.get("streaming", True)

            if streaming:
                yield self.format_chunk({"chunk_type": "message_start"})
                tool_calls: dict[int, list[Any]] = {}
                data_type = None
                finish_reason = None
                event = None

                async for event in response:
                    if not getattr(event, "choices", None):
                        continue
                    choice = event.choices[0]

                    if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                        chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                        for chunk in chunks:
                            yield chunk
                        yield self.format_chunk(
                            {
                                "chunk_type": "content_delta",
                                "data_type": data_type,
                                "data": choice.delta.reasoning_content,
                            }
                        )

                    if choice.delta.content:
                        chunks, data_type = self._stream_switch_content("text", data_type)
                        for chunk in chunks:
                            yield chunk
                        yield self.format_chunk(
                            {"chunk_type": "content_delta", "data_type": data_type, "data": choice.delta.content}
                        )

                    for tool_call in choice.delta.tool_calls or []:
                        tool_calls.setdefault(tool_call.index, []).append(tool_call)

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                        if data_type:
                            yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                        break

                for tool_deltas in tool_calls.values():
                    yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_deltas[0]})

                    for tool_delta in tool_deltas:
                        yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_delta})

                    yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

                yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason or "end_turn"})

                async for event in response:
                    _ = event

                if event and hasattr(event, "usage") and event.usage:
                    yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})
            else:
                yield self.format_chunk({"chunk_type": "message_start"})

                if not response.choices:
                    yield self.format_chunk({"chunk_type": "message_stop", "data": "end_turn"})
                    return

                choice = response.choices[0]
                message = choice.message

                if message.content:
                    yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
                    yield self.format_chunk(
                        {"chunk_type": "content_delta", "data_type": "text", "data": message.content}
                    )
                    yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})

                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        class MockToolCallDelta:
                            def __init__(self, tc: Any):
                                self.id = tc.id
                                self.type = tc.type
                                self.index = 0
                                self.function = tc.function

                        mock_delta = MockToolCallDelta(tool_call)
                        yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": mock_delta})
                        yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": mock_delta})
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

                yield self.format_chunk({"chunk_type": "message_stop", "data": choice.finish_reason or "end_turn"})

                if response.usage:
                    yield self.format_chunk({"chunk_type": "metadata", "data": response.usage})

        logger.debug("finished streaming response from model")

    def _stream_switch_content(self, data_type: str, prev_data_type: str | None) -> tuple[list[StreamEvent], str]:
        """Handle switching to a new content stream.

        Args:
            data_type: The next content data type.
            prev_data_type: The previous content data type.

        Returns:
            Tuple containing:
            - Stop block for previous content and the start block for the next content.
            - Next content data type.
        """
        chunks = []
        if data_type != prev_data_type:
            if prev_data_type is not None:
                chunks.append(self.format_chunk({"chunk_type": "content_stop", "data_type": prev_data_type}))
            chunks.append(self.format_chunk({"chunk_type": "content_start", "data_type": data_type}))

        return chunks, data_type

    @override
    async def structured_output(
        self, output_model: type[T], prompt: Messages, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        A new AsyncOpenAI client is created per call to avoid connection sharing
        in the asyncio event loop (see https://github.com/encode/httpx/discussions/2959).
        Expects exactly one choice in the response and raises if multiple are returned.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the request is throttled by OpenAI (rate limits).
        """
        async with self._get_client() as client:
            try:
                response: ParsedChatCompletion = await client.beta.chat.completions.parse(
                    model=self.get_config()["model_id"],
                    messages=self.format_request(prompt, system_prompt=system_prompt)["messages"],
                    response_format=output_model,
                )
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    logger.warning("OpenAI threw context window overflow error")
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except (openai.APIConnectionError, httpx.ConnectError) as e:
                self._raise_server_unavailable(e)
            except openai.RateLimitError as e:
                logger.warning("OpenAI threw rate limit error")
                raise ModelThrottledException(str(e)) from e
            except ValidationError as e:
                logger.warning("OpenAI response failed structured parse; retrying via tool call fallback: %s", e)
                parsed = await self._structured_output_via_tools(
                    client,
                    output_model,
                    self.format_request(prompt, system_prompt=system_prompt)["messages"],
                )
                yield {"output": parsed}
                return

        parsed: T | None = None
        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the OpenAI response.")

        for choice in response.choices:
            if isinstance(choice.message.parsed, output_model):
                parsed = choice.message.parsed
                break

        if parsed:
            yield {"output": parsed}
        else:
            raise ValueError("No valid tool use or tool use input was found in the OpenAI response.")

    async def _structured_output_via_tools(
        self, client: AsyncOpenAI, output_model: type[T], messages: Messages
    ) -> T:
        """Fallback structured output using tool calls when direct parse fails.

        Many providers support function calling even when `response_format` is not honored.
        We synthesize a single-function tool with the Pydantic schema and force a call to it.
        """
        tool_schema = output_model.model_json_schema()
        tool_name = output_model.__name__
        tool_spec = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Return a JSON object that matches the {tool_name} schema.",
                    "parameters": tool_schema,
                },
            }
        ]

        try:
            response = await client.chat.completions.create(
                model=self.get_config()["model_id"],
                messages=messages,
                tools=tool_spec,
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )
        except (openai.APIConnectionError, httpx.ConnectError) as e:
            self._raise_server_unavailable(e)

        if len(response.choices) > 1:
            raise ValueError("Multiple choices found in the OpenAI response.")

        choice = response.choices[0]
        if not choice.message.tool_calls:
            raise ValueError("No tool call returned in structured output fallback.")

        arguments = choice.message.tool_calls[0].function.arguments
        try:
            coerced = self._coerce_tool_arguments(arguments)
            return output_model.model_validate(coerced)
        except ValidationError as exc:
            logger.error("Tool call arguments failed validation against %s: %s", tool_name, exc)
            raise
        except ValueError as exc:
            logger.error("Tool call arguments could not be coerced for %s: %s", tool_name, exc)
            raise

    @staticmethod
    def _extract_json_object(raw: str) -> str | None:
        """Extract the first JSON object substring from a string."""
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        return None

    def _coerce_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        """Normalize tool call arguments into a JSON-serializable dict."""
        if isinstance(arguments, dict):
            if "arguments" in arguments:
                return self._coerce_tool_arguments(arguments["arguments"])
            return arguments

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return self._coerce_tool_arguments(parsed)
            except json.JSONDecodeError:
                cleaned = self._extract_json_object(arguments)
                if cleaned and cleaned != arguments:
                    parsed = json.loads(cleaned)
                    return self._coerce_tool_arguments(parsed)
                raise ValueError(f"Unable to parse tool arguments string: {arguments}")

        raise ValueError(f"Unsupported tool call argument type: {type(arguments)}")