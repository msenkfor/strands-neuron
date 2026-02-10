from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient
from strands_neuron import NeuronModel

def create_streamable_http_transport():
   return streamablehttp_client("http://localhost:8000/mcp/")

streamable_http_mcp_client = MCPClient(create_streamable_http_transport)

# Use the MCP server in a context manager
with streamable_http_mcp_client:
    # Get the tools from the MCP server
    tools = streamable_http_mcp_client.list_tools_sync()
    
    model = NeuronModel(
    config={
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_url": "http://localhost:8080/v1",
        "api_key": "EMPTY",
    }
    )

    # Create an agent with the MCP tools
    agent = Agent(tools=tools, model=model)
    response = agent("What is 125 plus 375?")
    print(response)
    response = agent("If I have 1000 and spend 246, how much do I have left?")
    print(response)
    response = agent("What is 24 multiplied by 7 divided by 3?")
    print(response)