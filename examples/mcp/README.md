# MCP Example with NeuronModel

This example demonstrates how to use Model Context Protocol (MCP) with NeuronModel to create an agent that can use tools from an MCP server.

## Prerequisites

1. **vLLM Neuron server must be running** - See the [infrastructure README](../../infrastructure/README.md) for setup instructions
2. **MCP server must be running** - The MCP server needs to be started before running the example

## Setup

### Step 1: Start the vLLM Neuron Server

Follow the instructions in the [infrastructure README](../../infrastructure/README.md) to start the vLLM server. The example expects the server to be running on `http://localhost:8080/v1`.

### Step 2: Start the MCP Server

**Important: You must start the MCP server before running the client example.**

Run the MCP server in a separate terminal:

```bash
python mcp-server.py
```

The server will start on `http://localhost:8000/mcp/` and provide calculator tools that can be used by the agent.

### Step 3: Run the Example

Once both servers are running, execute the client example:

```bash
python mcp-example.py
```

## How It Works

1. **MCP Server** (`mcp-server.py`): 
   - Provides calculator tools (add, subtract, multiply, divide)
   - Runs on `http://localhost:8000/mcp/`
   - Must be running before the client connects

2. **MCP Client** (`mcp-example.py`):
   - Connects to the MCP server to retrieve available tools
   - Creates a NeuronModel instance pointing to the vLLM server
   - Creates an Agent that can use the MCP tools
   - Demonstrates using the agent with various math queries

## Configuration

### MCP Server URL

The default MCP server URL is `http://localhost:8000/mcp/`. To change it, modify the URL in `mcp-example.py`:

```python
def create_streamable_http_transport():
   return streamablehttp_client("http://your-server:port/mcp/")
```

### vLLM Server URL

The default vLLM server URL is `http://localhost:8080/v1`. To change it, modify the `openai_api_base` in the NeuronModel config:

```python
model = NeuronModel(
    config={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "openai_api_base": "http://your-server:port/v1",
        "openai_api_key": "EMPTY",
    }
)
```

## Troubleshooting

### Connection Errors

If you see `httpx.ConnectError: All connection attempts failed`, check:

1. **MCP Server is running**: Make sure `mcp-server.py` is running and accessible at `http://localhost:8000/mcp/`
2. **vLLM Server is running**: Verify the vLLM server is running at `http://localhost:8080/v1`
3. **Port conflicts**: Ensure ports 8000 and 8080 are not in use by other applications

### MCP Client Initialization Error

If you see `MCPClientInitializationError`, the MCP server may not be ready or accessible. Wait a few seconds after starting the server before running the client.

