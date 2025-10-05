from agents import Agent
from pydantic import BaseModel
from tools import run_training_script
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class TrainingResultOutput(BaseModel):
    success: bool
    error: str


trainer = Agent(
    name="Training Runner",
    instructions=Config.load_agent("training_runner"),
    tools=[run_training_script],
    output_type=TrainingResultOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_TRAINER
)
