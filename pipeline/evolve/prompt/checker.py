import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def CodeChecker_input(motivation: str) -> str:
    return f"""Check the implemented code for critical issues and fix them if found.

## Motivation (for context)
{motivation}

{Config.load_agent("code_checker_input")}"""
