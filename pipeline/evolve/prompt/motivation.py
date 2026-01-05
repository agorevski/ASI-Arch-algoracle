import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Motivation_checker_input(context: str, motivation: str) -> str:
    """Generate a prompt for analyzing research motivation duplication.

    Args:
        context: Historical research context to compare against.
        motivation: The target motivation text to analyze for duplication.

    Returns:
        A formatted prompt string for the motivation duplication analysis.
    """
    return f"""
# Linear Attention Research Motivation Duplication Analysis

{Config.load_agent("motivation_checker_input_overview")}

## TARGET MOTIVATION FOR ANALYSIS
```
{motivation}
```

## HISTORICAL RESEARCH CONTEXT
{context}

{Config.load_agent("motivation_checker_input_analysis_framework")}"""
