import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Debugger_input(motivation: str, previous_error: str) -> str:
    """Generate a debug training error prompt.

    Args:
        motivation: The design motivation that must be preserved during debugging.
        previous_error: The training error log containing the last few hundred lines.

    Returns:
        A formatted string containing the debug prompt with motivation,
        error log, and debugger agent configuration.
    """
    return f"""# Debug Training Error

## Design Motivation (Must Preserve)
{motivation}

## Training Error Log (Last Few Hundred Lines)
{previous_error}

{Config.load_agent("debugger_input")}"""
