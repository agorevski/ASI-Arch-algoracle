import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Motivation_checker_input(context: str, motivation: str) -> str:
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
