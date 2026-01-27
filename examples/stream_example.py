"""Example demonstrating NeuronModel streaming capabilities."""

import asyncio

from strands_neuron import NeuronModel


async def stream_with_system_prompt():
    """Stream with system prompt."""
    print("System Prompt Example")
    print("System: You are a helpful coding assistant.")
    print("Prompt: Explain Python list comprehension")
    print("Response: ", end="")
    
    model = NeuronModel(
        config={
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "base_url": "http://localhost:8080/v1",
            "api_key": "EMPTY",
            "params": {
                "temperature": 0.5,
            }
        }
    )
    
    messages = [{"role": "user", "content": [{"text": "Explain Python list comprehension"}]}]
    
    async for event in model.stream(
        messages,
        system_prompt="You are a helpful coding assistant. Be concise."
    ):
        # Extract text from contentBlockDelta events
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    print("\n")


async def stream_simple():
    """Simple streaming example without system prompt."""
    print("\nSimple Streaming Example")
    print("Prompt: What is machine learning?")
    print("Response: ", end="")
    
    model = NeuronModel(
        config={
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "base_url": "http://localhost:8080/v1",
            "api_key": "EMPTY",
        }
    )
    
    messages = [{"role": "user", "content": [{"text": "What is machine learning?"}]}]
    
    async for event in model.stream(messages):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    print("\n")


async def stream_with_conversation():
    """Stream with a multi-turn conversation."""
    print("\nConversation Example")
    
    model = NeuronModel(
        config={
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "base_url": "http://localhost:8080/v1",
            "api_key": "EMPTY",
        }
    )
    
    # First turn
    messages = [{"role": "user", "content": [{"text": "What is the capital of France?"}]}]
    print("User: What is the capital of France?")
    print("Assistant: ", end="")
    
    async for event in model.stream(messages):
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                print(delta["text"], end="", flush=True)
    print("\n")


async def main():
    """Run all streaming examples."""
    try:
        await stream_with_system_prompt()
        await stream_simple()
        await stream_with_conversation()
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure the vLLM server is running on http://localhost:8080/v1")


if __name__ == "__main__":
    asyncio.run(main())

