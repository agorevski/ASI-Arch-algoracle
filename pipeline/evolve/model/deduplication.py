from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class DeduplicationOutput(BaseModel):
    name: str
    motivation: str


# Deduplication Agent
deduplication = Agent(
    name="Innovation Diversifier",
    instructions=Config.load_agent("innovation_diversifier"),
    output_type=DeduplicationOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_EVOLVER,
    tools=[read_code_file, write_code_file]
)
