from agents import Agent
from pydantic import BaseModel
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class MotivationCheckOutput(BaseModel):
    is_repeated: bool
    repeated_index: list[int]
    judgement_reason: str


motivation_checker = Agent(
    name="Motivation Checker",
    instructions=Config.load_agent("motivation_checker"),
    output_type=MotivationCheckOutput,
    tools=[],
    model=Config.AZURE_DEPLOYMENT_MODEL_MOTIVATION_CHECKER,
)
