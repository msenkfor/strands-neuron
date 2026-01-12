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
model = NeuronModel(
    config={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "base_url": "http://localhost:8080/v1",
        "api_key": "EMPTY",
    }
)

agent = Agent(model=model, structured_output_model=PersonInfo)
result = agent("John Smith is a 30-year-old engineer.")

person_info: PersonInfo = result.structured_output
print(f"Name: {person_info.name}")      
print(f"Age: {person_info.age}")      
print(f"Job: {person_info.occupation}") 