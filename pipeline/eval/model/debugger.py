from agents import Agent
from pydantic import BaseModel
from tools import read_code_file, write_code_file
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


class DebuggerOutput(BaseModel):
    changes_made: str


# Debugger Agent
debugger = Agent(
    name="Training Code Debugger",
    instructions=Config.load_agent("training_code_debugger"),
    output_type=DebuggerOutput,
    model=Config.AZURE_DEPLOYMENT_MODEL_DEBUGGER,
    tools=[read_code_file, write_code_file]
)
