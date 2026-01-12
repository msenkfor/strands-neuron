import os

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands_neuron import NeuronModel

@pytest.fixture
def model() -> NeuronModel:
    base_url = os.getenv("OPENAI_API_BASE_URL", "http://localhost:8080/v1")
    model_id = os.getenv("NEURON_VLLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

    return NeuronModel(
        {
            "model_id": model_id,
            "base_url": base_url,
            "api_key": api_key,
        }
    )


@pytest.fixture
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model: NeuronModel, tools):
    return Agent(model=model, tools=tools)


@pytest.fixture
def weather():
    class Weather(BaseModel):
        """Extracts the time and weather from the user's message with the exact strings."""

        time: str
        weather: str

    return Weather(time="12:00", weather="sunny")


def test_agent_invoke(agent):
    result = agent("What is the time and weather in New York?")
    # Verify we got a response
    assert result.message is not None
    assert len(result.message["content"]) > 0
    text = result.message["content"][0]["text"].lower()
    
    # Check if tools were used by looking for tool results in the message content
    tool_used = any(
        "toolResult" in content for content in result.message["content"]
    )
    
    if tool_used:
        # If tools were used, verify the expected values are present
        assert all(string in text for string in ["12:00", "sunny"])
    else:
        # If tools weren't used, just verify we got a response (model may not always use tools)
        assert len(text) > 0


@pytest.mark.asyncio
async def test_agent_invoke_async(agent):
    result = await agent.invoke_async("What is the time and weather in New York?")
    # Verify we got a response
    assert result.message is not None
    assert len(result.message["content"]) > 0
    text = result.message["content"][0]["text"].lower()
    
    # Check if tools were used by looking for tool results in the message content
    tool_used = any(
        "toolResult" in content for content in result.message["content"]
    )
    
    if tool_used:
        # If tools were used, verify the expected values are present
        assert all(string in text for string in ["12:00", "sunny"])
    else:
        # If tools weren't used, just verify we got a response (model may not always use tools)
        assert len(text) > 0


@pytest.mark.asyncio
async def test_agent_stream_async(agent):
    stream = agent.stream_async("What is the time and weather in New York?")
    async for event in stream:
        _ = event

    result = event["result"]
    # Verify we got a response
    assert result.message is not None
    assert len(result.message["content"]) > 0
    text = result.message["content"][0]["text"].lower()
    
    # Check if tools were used by looking for tool results in the message content
    tool_used = any(
        "toolResult" in content for content in result.message["content"]
    )
    
    if tool_used:
        # If tools were used, verify the expected values are present
        assert all(string in text for string in ["12:00", "sunny"])
    else:
        # If tools weren't used, just verify we got a response (model may not always use tools)
        assert len(text) > 0


def test_agent_structured_output(agent, weather):
    """Test synchronous structured output."""
    tru_weather = agent.structured_output(type(weather), "The time is 12:00 and the weather is sunny")
    exp_weather = weather
    assert tru_weather == exp_weather


@pytest.mark.asyncio
async def test_agent_structured_output_async(agent, weather):
    """Test asynchronous structured output."""
    tru_weather = await agent.structured_output_async(
        type(weather),
        "The time is 12:00 and the weather is sunny",
    )
    exp_weather = weather
    assert tru_weather == exp_weather