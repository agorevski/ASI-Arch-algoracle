import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Planner_input(context: str) -> str:
    return f"""# Neural Architecture Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

{Config.load_agent("planner_input")}"""
