from agents import Agent
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config


class ModelJudgementOutput(BaseModel):
    performance_score: int
    innovation_score: int
    complexity_score: int
    weighted_final_score: float
    judgement_reason: str


model_judger = Agent(
    name="Model Judger",
    instructions=Config.load_agent("model_judger"),
    output_type=ModelJudgementOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_JUDGER,
    tools=[],
)
