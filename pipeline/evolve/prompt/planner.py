import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Planner_input(context: str) -> str:
    """Generate the planner input prompt with experimental context.

    Args:
        context: The experimental context and historical evidence to include
            in the planner prompt.

    Returns:
        A formatted string containing the neural architecture evolution mission
        prompt with the provided context and loaded planner configuration.
    """
    return f"""# Neural Architecture Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

{Config.load_agent("planner_input")}"""
