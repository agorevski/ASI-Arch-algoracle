import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def CodeChecker_input(motivation: str) -> str:
    """Generate a prompt for the code checker agent to review implemented code.

    Args:
        motivation: The motivation context describing why the code was implemented.

    Returns:
        A formatted prompt string for the code checker agent.
    """
    return f"""Check the implemented code for critical issues and fix them if found.

## Motivation (for context)
{motivation}

{Config.load_agent("code_checker_input")}"""
