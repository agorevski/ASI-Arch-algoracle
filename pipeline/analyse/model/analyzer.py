from agents import Agent
from pydantic import BaseModel
from tools import read_code_file
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class AnalyzerOutput(BaseModel):
    design_evaluation: str
    experimental_results_analysis: str
    expectation_vs_reality_comparison: str
    theoretical_explanation_with_evidence: str
    synthesis_and_insights: str


analyzer = Agent(
    name="Architecture Performance Analyzer",
    instructions=Config.load_agent("architecture_performance_analyzer"),
    output_type=AnalyzerOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_ANALYZER,
    tools=[read_code_file]
)
