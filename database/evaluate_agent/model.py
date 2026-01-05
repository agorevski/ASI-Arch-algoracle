from agents import Agent
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config_loader import Config


class ModelJudgementOutput(BaseModel):
    """Output model for model judgement evaluation results.

    Attributes:
        performance_score: Score evaluating the model's performance.
        innovation_score: Score evaluating the model's innovation.
        complexity_score: Score evaluating the model's complexity.
        weighted_final_score: Final weighted score combining all metrics.
        judgement_reason: Explanation for the judgement scores.
    """

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
