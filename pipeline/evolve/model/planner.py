from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class PlannerOutput(BaseModel):
    name: str
    motivation: str


# Planning Agent
planner = Agent(
    name="Architecture Designer",
    instructions=Config.load_agent("architecture_designer"),
    output_type=PlannerOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_PLANNER,
    tools=[read_code_file, write_code_file]
)
