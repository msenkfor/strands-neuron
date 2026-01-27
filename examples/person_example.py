"""Example demonstrating structured output using agent.structured_output() method."""
from strands import Agent
from strands_neuron import NeuronModel
from pydantic import BaseModel, Field


class PersonInfo(BaseModel):
    """Person information model."""
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title")


# Initialize the model
# Note: streaming=False is a workaround for vLLM streaming bugs with Mistral tool calls
model = NeuronModel(
    config={
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_url": "http://localhost:8080/v1",
        "api_key": "EMPTY",
        "streaming": False,  # Disable streaming to avoid vLLM tool call parsing bugs
    }
)

agent = Agent(model=model, structured_output_model=PersonInfo)
result = agent("John Smith is a 30-year-old engineer.")

person_info: PersonInfo = result.structured_output
print(f"Name: {person_info.name}")      
print(f"Age: {person_info.age}")      
print(f"Job: {person_info.occupation}") 