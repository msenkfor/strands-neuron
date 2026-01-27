import base64
import json
import unittest.mock
from typing import AsyncGenerator, List, TypeVar

import pydantic
import pytest

from strands_neuron import NeuronModel
import strands_neuron.neuron as neuron_module
from strands.types.content import Messages

T = TypeVar("T")


async def _agenerator(items: list[T]) -> AsyncGenerator[T, None]:
    """Create an async generator from a list."""
    for item in items:
        yield item


async def _alist(async_gen: AsyncGenerator[T, None]) -> List[T]:
    """Collect all items from an async generator into a list."""
    return [item async for item in async_gen]


@pytest.fixture
def agenerator():
    """Fixture that returns the agenerator helper function."""
    return _agenerator


@pytest.fixture
def alist():
    """Fixture that returns the alist helper function."""
    return _alist


@pytest.fixture
def neuron_client(monkeypatch: pytest.MonkeyPatch) -> unittest.mock.Mock:
    mock_client_cls = unittest.mock.Mock()
    mock_client = unittest.mock.AsyncMock()
    mock_client.chat.completions.create = unittest.mock.AsyncMock()
    mock_client.beta.chat.completions.parse = unittest.mock.AsyncMock()
    mock_client.close = unittest.mock.AsyncMock()
    mock_client_cls.return_value = mock_client

    monkeypatch.setattr(neuron_module, "AsyncOpenAI", mock_client_cls)
    return mock_client


@pytest.fixture
def model_id() -> str:
    return "m1"


@pytest.fixture
def model(model_id: str) -> NeuronModel:
    return NeuronModel({"model_id": model_id})


@pytest.fixture
def model_with_stream_options(model_id: str) -> NeuronModel:
    return NeuronModel({"model_id": model_id})


@pytest.fixture
def messages() -> Messages:
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt() -> str:
    return "s1"


@pytest.fixture
def test_output_model_cls() -> type[pydantic.BaseModel]:
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__model_configs(model_id: str) -> None:
    params = {"temperature": 0.25}
    model = NeuronModel({"model_id": model_id, "params": params})

    cfg = model.get_config()
    assert cfg["model_id"] == model_id
    assert cfg["params"] == params
    # Extra, unsupported keys should not be present
    assert "max_completion_tokens" not in cfg


def test_update_config(model: NeuronModel, model_id: str) -> None:
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model: NeuronModel, messages: Messages, model_id: str) -> None:
    tru_request = model.format_request(messages)
    exp_messages = [{"role": "user", "content": [{"text": "test", "type": "text"}]}]

    assert tru_request["messages"] == exp_messages
    assert tru_request["model"] == model_id
    assert tru_request["stream"] is True
    assert tru_request["stream_options"] == {"include_usage": True}
    assert tru_request["tools"] == []


def test_format_request_with_override(model: NeuronModel, messages: Messages, model_id: str) -> None:
    model.update_config(model_id=model_id)
    tru_request = model.format_request(messages, tool_specs=None)
    assert tru_request["model"] == model_id
    assert tru_request["messages"][0]["content"][0]["text"] == "test"


def test_format_request_with_system_prompt(
    model: NeuronModel, messages: Messages, model_id: str, system_prompt: str
) -> None:
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"text": "test", "type": "text"}]},
    ]

    assert tru_request["messages"] == exp_messages
    assert tru_request["model"] == model_id


def test_format_request_with_image(model: NeuronModel, model_id: str) -> None:
    image_bytes = b"base64encodedimage"
    messages: Messages = [
        {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": image_bytes}}}]}
    ]

    tru_request = model.format_request(messages)
    content_block = tru_request["messages"][0]["content"][0]

    assert tru_request["model"] == model_id
    assert content_block["type"] == "image_url"
    assert content_block["image_url"]["format"] == "image/png"
    assert content_block["image_url"]["url"].endswith(base64.b64encode(image_bytes).decode("utf-8"))


def test_format_request_with_tool_use(model: NeuronModel, model_id: str) -> None:
    messages: Messages = [
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "calculator", "name": "calculator", "input": {"expression": "2+2"}}}
            ],
        }
    ]

    tru_request = model.format_request(messages)
    message = tru_request["messages"][0]
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["id"] == "calculator"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"expression": "2+2"}'


def test_format_request_with_tool_result(model: NeuronModel, model_id: str) -> None:
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "calculator",
                        "status": "success",
                        "content": [
                            {"text": "4"},
                            {"image": {"format": "png", "source": {"bytes": b"image"}}},
                            {"json": ["4"]},
                        ],
                    },
                },
                {
                    "text": "see results",
                },
            ],
        },
    ]

    tru_request = model.format_request(messages)
    tool_msgs = [msg for msg in tru_request["messages"] if msg["role"] == "tool"]
    user_msgs = [msg for msg in tru_request["messages"] if msg["role"] == "user"]

    assert tool_msgs, "expected a tool role message"
    tool_msg = tool_msgs[0]
    assert tool_msg["tool_call_id"] == "calculator"
    assert any(block.get("type") == "text" for block in tool_msg["content"])

    # Images are moved to a follow-up user message for OpenAI compatibility
    assert any(any(block.get("type") == "image_url" for block in msg.get("content", [])) for msg in user_msgs)


def test_format_request_with_unsupported_type(model: NeuronModel) -> None:
    messages: Messages = [
        {
            "role": "user",
            "content": [{"unsupported": {}}],
        },
    ]

    with pytest.raises(TypeError, match="unsupported type"):
        model.format_request(messages)


def test_format_request_with_tool_specs(model: NeuronModel, messages: Messages, model_id: str) -> None:
    tool_specs = [
        {
            "name": "calculator",
            "description": "Calculate mathematical expressions",
            "inputSchema": {
                "json": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            },
        }
    ]

    tru_request = model.format_request(messages, tool_specs)
    tool = tru_request["tools"][0]

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculator"
    assert tru_request["messages"][0]["content"][0]["text"] == "test"


def test_format_request_with_inference_config(model: NeuronModel, messages: Messages, model_id: str) -> None:
    inference_config = {
        "max_completion_tokens": 1,
        "stop_sequences": ["stop"],
        "temperature": 1.0,
        "top_p": 1.0,
    }

    model.update_config(params=inference_config)
    tru_request = model.format_request(messages)
    assert tru_request["model"] == model_id
    assert tru_request["max_completion_tokens"] == 1
    assert tru_request["stop_sequences"] == ["stop"]


def test_format_chunk_message_start(model: NeuronModel) -> None:
    event = {"chunk_type": "message_start"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStart": {"role": "assistant"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_start_text(model: NeuronModel) -> None:
    event = {"chunk_type": "content_start", "data_type": "text"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_start_tool(model: NeuronModel) -> None:
    mock_function = unittest.mock.Mock()
    mock_function.function.name = "calculator"
    mock_function.id = "calculator"

    event = {"chunk_type": "content_start", "data_type": "tool", "data": mock_function}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calculator"}}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_delta_text(model: NeuronModel) -> None:
    event = {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_delta_tool(model: NeuronModel) -> None:
    event = {
        "chunk_type": "content_delta",
        "data_type": "tool",
        "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments='{"expression": "2+2"}')),
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_stop(model: NeuronModel) -> None:
    event = {"chunk_type": "content_stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStop": {}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_end_turn(model: NeuronModel) -> None:
    event = {"chunk_type": "message_stop", "data": "stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "end_turn"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_tool_use(model: NeuronModel) -> None:
    event = {"chunk_type": "message_stop", "data": "tool_calls"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "tool_use"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata(model: NeuronModel) -> None:
    # Test with usage data
    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30
    event = {
        "chunk_type": "metadata",
        "data": mock_usage,
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "metadata": {
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30,
            },
            "metrics": {
                "latencyMs": 0,
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata_without_usage(model: NeuronModel) -> None:
    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 0
    mock_usage.completion_tokens = 0
    mock_usage.total_tokens = 0

    event = {"chunk_type": "metadata", "data": mock_usage}

    tru_chunk = model.format_chunk(event)
    assert tru_chunk["metadata"]["usage"]["totalTokens"] == 0


def test_format_chunk_other(model: NeuronModel) -> None:
    event = {"chunk_type": "other"}

    with pytest.raises(RuntimeError, match="unknown type"):
        model.format_chunk(event)


@pytest.mark.asyncio
async def test_stream(
    neuron_client: unittest.mock.Mock,
    model_with_stream_options: NeuronModel,
    agenerator,
    alist,
) -> None:
    model = model_with_stream_options
    mock_chunk = unittest.mock.Mock()
    mock_choice = unittest.mock.Mock()
    mock_delta = unittest.mock.Mock()
    mock_delta.content = "Hello"
    mock_delta.tool_calls = None
    mock_delta.reasoning_content = None
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = "stop"
    mock_chunk.choices = [mock_choice]
    mock_chunk.usage = unittest.mock.Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    neuron_client.chat.completions.create.return_value = agenerator([mock_chunk])

    messages: Messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    # Find the first text delta
    assert any(event.get("contentBlockDelta", {}).get("delta", {}).get("text") == "Hello" for event in tru_events)
    # Ensure we emit a messageStop
    assert any(event.get("messageStop", {}).get("stopReason") for event in tru_events)
    # Check metadata structure exists
    metadata_events = [event for event in tru_events if "metadata" in event]
    assert metadata_events
    assert "usage" in metadata_events[0]["metadata"]
    assert "metrics" in metadata_events[0]["metadata"]

    expected_request = {
        "messages": [{"role": "user", "content": [{"text": "Hello", "type": "text"}]}],
        "model": "m1",
        "stream": True,
        "tools": [],
        "stream_options": {"include_usage": True},
    }
    neuron_client.chat.completions.create.assert_awaited_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_with_tool_calls(
    neuron_client: unittest.mock.Mock,
    model_with_stream_options: NeuronModel,
    agenerator,
    alist,
) -> None:
    model = model_with_stream_options
    # Simulate OpenAI's incremental tool call streaming
    # Chunk 1: Text content
    chunk1 = unittest.mock.Mock()
    choice1 = unittest.mock.Mock()
    delta1 = unittest.mock.Mock()
    delta1.content = "I'll calculate that for you"
    delta1.tool_calls = None
    choice1.delta = delta1
    choice1.finish_reason = None
    chunk1.choices = [choice1]
    chunk1.usage = None
    
    # Chunk 2: Tool call start (with name)
    chunk2 = unittest.mock.Mock()
    choice2 = unittest.mock.Mock()
    delta2 = unittest.mock.Mock()
    tool_call_start = unittest.mock.Mock()
    tool_call_start.index = 0
    tool_call_start.id = "call_123"
    tool_call_start.type = "function"
    tool_call_start.function = unittest.mock.Mock()
    tool_call_start.function.name = "calculator"
    tool_call_start.function.arguments = None
    delta2.content = None
    delta2.tool_calls = [tool_call_start]
    choice2.delta = delta2
    choice2.finish_reason = None
    chunk2.choices = [choice2]
    chunk2.usage = None
    
    # Chunk 3: Tool call arguments
    chunk3 = unittest.mock.Mock()
    choice3 = unittest.mock.Mock()
    delta3 = unittest.mock.Mock()
    tool_call_args = unittest.mock.Mock()
    tool_call_args.index = 0
    tool_call_args.function = unittest.mock.Mock()
    tool_call_args.function.name = None
    tool_call_args.function.arguments = '{"expression": "2+2"}'
    delta3.content = None
    delta3.tool_calls = [tool_call_args]
    choice3.delta = delta3
    choice3.finish_reason = "tool_calls"
    chunk3.choices = [choice3]
    chunk3.usage = None

    neuron_client.chat.completions.create.return_value = agenerator([chunk1, chunk2, chunk3])

    messages: Messages = [{"role": "user", "content": [{"text": "Calculate 2+2"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)

    # Basic structural checks
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    assert tru_events[1] == {"contentBlockStart": {"start": {}}}
    message_stop_events = [e for e in tru_events if "messageStop" in e]
    assert message_stop_events
    assert message_stop_events[0]["messageStop"]["stopReason"] == "tool_use"

    # One toolUse start with expected name/id
    tool_starts = [e for e in tru_events if e.get("contentBlockStart", {}).get("start", {}).get("toolUse") is not None]
    assert len(tool_starts) == 1
    tool_use = tool_starts[0]["contentBlockStart"]["start"]["toolUse"]
    assert tool_use["name"] == "calculator"
    assert tool_use["toolUseId"] == "call_123"

    # One toolUse delta with expected input
    tool_deltas = [e for e in tru_events if "contentBlockDelta" in e and "toolUse" in e["contentBlockDelta"]["delta"]]
    assert len(tool_deltas) >= 1
    assert tool_deltas[-1]["contentBlockDelta"]["delta"]["toolUse"]["input"] == '{"expression": "2+2"}'

    # One text delta with the assistant message
    text_deltas = [e for e in tru_events if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]]
    assert len(text_deltas) == 1
    assert text_deltas[0]["contentBlockDelta"]["delta"]["text"] == "I'll calculate that for you"

    expected_request = {
        "messages": [{"role": "user", "content": [{"text": "Calculate 2+2", "type": "text"}]}],
        "model": "m1",
        "stream": True,
        "tools": [],
        "stream_options": {"include_usage": True},
    }
    neuron_client.chat.completions.create.assert_awaited_once_with(**expected_request)


@pytest.mark.asyncio
async def test_structured_output(
    neuron_client: unittest.mock.Mock,
    model: NeuronModel,
    test_output_model_cls: type[pydantic.BaseModel],
    alist,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: Messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    # Mock the tool call response
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function.name = test_output_model_cls.__name__
    mock_tool_call.function.arguments = '{"name": "John", "age": 30}'
    
    mock_message = unittest.mock.Mock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.parsed = test_output_model_cls(name="John", age=30)
    
    mock_choice = unittest.mock.Mock()
    mock_choice.message = mock_message
    
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]
    
    # Mock the chat.completions.create call
    neuron_client.beta.chat.completions.parse.return_value = mock_response

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    # Should have only the output event
    assert len(events) == 1
    tru_result = events[0]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result
    
    # Verify the request was made correctly
    call_kwargs = neuron_client.beta.chat.completions.parse.call_args.kwargs
    assert call_kwargs["model"] == "m1"
    assert call_kwargs["response_format"] == test_output_model_cls
