import json
import unittest.mock
from typing import AsyncGenerator, List, TypeVar

import httpx
import pydantic
import pytest

from strands_neuron import NeuronModel
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
    from strands_neuron import neuron

    mock_client_cls = unittest.mock.Mock()
    mock_client = unittest.mock.AsyncMock()
    mock_client.chat.completions.create = unittest.mock.AsyncMock()
    mock_client_cls.return_value = mock_client

    monkeypatch.setattr(neuron, "AsyncOpenAI", mock_client_cls)
    return mock_client


@pytest.fixture
def model_id() -> str:
    return "m1"


@pytest.fixture
def model(model_id: str) -> NeuronModel:
    return NeuronModel({"model_id": model_id})


@pytest.fixture
def model_with_stream_options(model_id: str) -> NeuronModel:
    return NeuronModel({
        "model_id": model_id,
        "stream_options": {"include_usage": True}
    })


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
    model = NeuronModel({"model_id": model_id, "max_completion_tokens": 1})

    tru_max_completion_tokens = model.get_config().get("max_completion_tokens")
    exp_max_completion_tokens = 1

    assert tru_max_completion_tokens == exp_max_completion_tokens


def test_update_config(model: NeuronModel, model_id: str) -> None:
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model: NeuronModel, messages: Messages, model_id: str) -> None:
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_override(model: NeuronModel, messages: Messages, model_id: str) -> None:
    model.update_config(model_id=model_id)
    tru_request = model.format_request(messages, tool_specs=None)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(
    model: NeuronModel, messages: Messages, model_id: str, system_prompt: str
) -> None:
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "test"},
        ],
        "model": model_id,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_image(model: NeuronModel, model_id: str) -> None:
    messages: Messages = [{"role": "user", "content": [{"image": {"source": {"bytes": "base64encodedimage"}}}]}]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "images": ["base64encodedimage"]}],
        "model": model_id,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_tool_use(model: NeuronModel, model_id: str) -> None:
    messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "calculator", "input": '{"expression": "2+2"}'}}]}
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "calculator",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        }
                    }
                ],
            }
        ],
        "model": model_id,
        "stream": True,
    }

    assert tru_request == exp_request


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
                            {"image": {"source": {"bytes": b"image"}}},
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
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "tool_call_id": "calculator",
                "content": "4\n" + '["4"]',
                "images": [b"image"],
            },
            {
                "role": "user",
                "content": "see results",
            },
        ],
        "model": model_id,
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_unsupported_type(model: NeuronModel) -> None:
    messages: Messages = [
        {
            "role": "user",
            "content": [{"unsupported": {}}],
        },
    ]

    with pytest.raises(TypeError, match="Unsupported content type: unsupported"):
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
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate mathematical expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                }
            }
        ],
    }

    assert tru_request == exp_request


def test_format_request_with_inference_config(model: NeuronModel, messages: Messages, model_id: str) -> None:
    inference_config = {
        "max_completion_tokens": 1,
        "stop_sequences": ["stop"],
        "temperature": 1.0,
        "top_p": 1.0,
    }

    model.update_config(**inference_config)
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "temperature": inference_config["temperature"],
        "top_p": inference_config["top_p"],
        "max_completion_tokens": inference_config["max_completion_tokens"],
        "stop": inference_config["stop_sequences"],
        "stream": True,
    }

    assert tru_request == exp_request


def test_format_request_with_additional_args(model: NeuronModel, messages: Messages, model_id: str) -> None:
    additional_args = {"o1": 1}

    model.update_config(additional_args=additional_args)
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "stream": True,
        "o1": 1,
    }

    assert tru_request == exp_request


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
        "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments={"expression": "2+2"})),
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"expression": "2+2"})}}}}

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
    event = {"chunk_type": "message_stop", "data": "tool_use"}

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
        "usage": mock_usage,
        "latency_ms": 150,
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
                "latencyMs": 150,
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata_without_usage(model: NeuronModel) -> None:
    # Test without usage data (defaults to 0)
    event = {
        "chunk_type": "metadata",
        "latency_ms": 100,
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "metadata": {
            "usage": {
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 0,
            },
            "metrics": {
                "latencyMs": 100,
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_other(model: NeuronModel) -> None:
    event = {"chunk_type": "other"}

    with pytest.raises(RuntimeError, match="Unknown chunk_type: other"):
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
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = "stop"
    mock_chunk.choices = [mock_choice]

    neuron_client.chat.completions.create.return_value = agenerator([mock_chunk])

    messages: Messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {"latencyMs": 0}}},
    ]

    # Check structure (comparing exact latency is flaky due to timing)
    assert len(tru_events) == len(exp_events)
    assert tru_events[0] == exp_events[0]
    assert tru_events[1] == exp_events[1]
    assert tru_events[2] == exp_events[2]
    assert tru_events[3] == exp_events[3]
    assert tru_events[4] == exp_events[4]
    # Check metadata structure exists
    assert "metadata" in tru_events[5]
    assert "usage" in tru_events[5]["metadata"]
    assert "metrics" in tru_events[5]["metadata"]

    expected_request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "m1",
        "stream": True,
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
    assert tru_events[-2]["messageStop"]["stopReason"] == "tool_use"

    # One toolUse start with expected name/id
    tool_starts = [e for e in tru_events if e.get("contentBlockStart", {}).get("start", {}).get("toolUse") is not None]
    assert len(tool_starts) == 1
    tool_use = tool_starts[0]["contentBlockStart"]["start"]["toolUse"]
    assert tool_use["name"] == "calculator"
    assert tool_use["toolUseId"] == "call_123"

    # One toolUse delta with expected input
    tool_deltas = [e for e in tru_events if "contentBlockDelta" in e and "toolUse" in e["contentBlockDelta"]["delta"]]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["contentBlockDelta"]["delta"]["toolUse"]["input"] == '{"expression": "2+2"}'

    # One text delta with the assistant message
    text_deltas = [e for e in tru_events if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]]
    assert len(text_deltas) == 1
    assert text_deltas[0]["contentBlockDelta"]["delta"]["text"] == "I'll calculate that for you"

    expected_request = {
        "messages": [{"role": "user", "content": "Calculate 2+2"}],
        "model": "m1",
        "stream": True,
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
    
    mock_choice = unittest.mock.Mock()
    mock_choice.message = mock_message
    
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]
    
    # Mock the chat.completions.create call
    neuron_client.chat.completions.create.return_value = mock_response

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    # Should have only the output event
    assert len(events) == 1
    tru_result = events[0]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result
    
    # Verify the request was made correctly
    call_kwargs = neuron_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "m1"
    assert call_kwargs["stream"] is False
    assert "tools" in call_kwargs
    assert call_kwargs["tool_choice"]["type"] == "function"
    assert call_kwargs["tool_choice"]["function"]["name"] == test_output_model_cls.__name__


# Server availability check tests


def test_check_server_availability_success(monkeypatch: pytest.MonkeyPatch, model_id: str) -> None:
    """Test successful connection to OpenAI API server."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 200
    
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_response
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    model = NeuronModel({"model_id": model_id, "base_url": "http://localhost:8080/v1"})
    
    # Verify the client was called with correct URL
    mock_client.__enter__.return_value.get.assert_called_once_with("http://localhost:8080/v1/models")


def test_check_server_availability_connection_error(
    monkeypatch: pytest.MonkeyPatch, model_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test graceful handling of connection error."""
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.side_effect = httpx.ConnectError("Connection refused")
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    # Should not raise an exception
    model = NeuronModel({"model_id": model_id, "base_url": "http://localhost:8080/v1"})
    
    assert model is not None
    assert "Could not connect to OpenAI API server" in caplog.text


def test_check_server_availability_timeout(
    monkeypatch: pytest.MonkeyPatch, model_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test graceful handling of timeout."""
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.side_effect = httpx.TimeoutException("Request timeout")
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    # Should not raise an exception
    model = NeuronModel({"model_id": model_id, "base_url": "http://localhost:8080/v1"})
    
    assert model is not None
    assert "timed out" in caplog.text


def test_check_server_availability_non_200_status(
    monkeypatch: pytest.MonkeyPatch, model_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test handling of non-200 status code."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 404
    
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_response
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    model = NeuronModel({"model_id": model_id, "base_url": "http://localhost:8080/v1"})
    
    assert model is not None
    assert "returned status 404" in caplog.text


def test_check_server_availability_generic_exception(
    monkeypatch: pytest.MonkeyPatch, model_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test handling of generic exception."""
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.side_effect = Exception("Something went wrong")
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    # Should not raise an exception
    model = NeuronModel({"model_id": model_id, "base_url": "http://localhost:8080/v1"})
    
    assert model is not None
    assert "Error checking OpenAI API server" in caplog.text


def test_init_without_base_url(monkeypatch: pytest.MonkeyPatch, model_id: str) -> None:
    """Test that server check is not performed when base_url is not provided."""
    mock_client_cls = unittest.mock.Mock()
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    model = NeuronModel({"model_id": model_id})
    
    # Verify httpx.Client was never called
    mock_client_cls.assert_not_called()
    assert model is not None


def test_update_config_with_base_url_success(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    """Test that server check is performed when updating base_url."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 200
    
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_response
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    # Create model without base_url
    model = NeuronModel({"model_id": model_id})
    mock_client_cls.reset_mock()
    
    # Update config with base_url
    model.update_config(base_url="http://localhost:8080/v1")
    
    # Verify the server check was performed
    mock_client_cls.assert_called_once()
    mock_client.__enter__.return_value.get.assert_called_once_with("http://localhost:8080/v1/models")


def test_update_config_with_base_url_connection_error(
    monkeypatch: pytest.MonkeyPatch, model_id: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Test graceful handling when updating base_url with connection error."""
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.side_effect = httpx.ConnectError("Connection refused")
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    model = NeuronModel({"model_id": model_id})
    caplog.clear()
    
    # Should not raise an exception
    model.update_config(base_url="http://localhost:8080/v1")
    
    assert "Could not connect to OpenAI API server" in caplog.text
    assert model.get_config()["base_url"] == "http://localhost:8080/v1"


def test_update_config_without_base_url(monkeypatch: pytest.MonkeyPatch, model: NeuronModel) -> None:
    """Test that server check is not performed when updating other config parameters."""
    mock_client_cls = unittest.mock.Mock()
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    model.update_config(temperature=0.7, max_completion_tokens=100)
    
    # Verify httpx.Client was never called
    mock_client_cls.assert_not_called()


def test_check_server_availability_url_formatting(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    """Test that trailing slashes in base URL are handled correctly."""
    mock_response = unittest.mock.Mock()
    mock_response.status_code = 200
    
    mock_client = unittest.mock.MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_response
    
    mock_client_cls = unittest.mock.Mock(return_value=mock_client)
    monkeypatch.setattr("httpx.Client", mock_client_cls)
    
    # Test with trailing slash
    model = NeuronModel({"model_id": model_id, "base_url": "http://localhost:8080/v1/"})
    
    # Verify the URL is correctly formatted without double slashes
    mock_client.__enter__.return_value.get.assert_called_once_with("http://localhost:8080/v1/models")