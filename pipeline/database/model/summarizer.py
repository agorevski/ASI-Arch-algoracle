from agents import Agent
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class SummaryOutput(BaseModel):
    experience: str


# Summary Agent
summarizer = Agent(
    name="Experience Synthesizer",
    instructions=Config.load_agent("summarizer"),
    output_type=SummaryOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_SUMMARIZER,
    tools=[]
)
