import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Debugger_input(motivation: str, previous_error: str) -> str:
    return f"""# Debug Training Error

## Design Motivation (Must Preserve)
{motivation}

## Training Error Log (Last Few Hundred Lines)
{previous_error}

{Config.load_agent("debugger_input")}"""
