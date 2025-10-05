from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class CodeCheckerOutput(BaseModel):
    success: bool
    error: str


# Code Checker Agent
code_checker = Agent(
    name="Code Checker and Fixer",
    instructions=Config.load_agent("code_checker"),
    output_type=CodeCheckerOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_CHECKER,
    tools=[read_code_file, write_code_file]
)
