import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def Deduplication_input(context: str, repeated_motivation: str) -> str:
    return f"""
# Neural Architecture Innovation Diversification Task

## TASK OVERVIEW
**Primary Objective**: Generate breakthrough architectural code that fundamentally differs from repeated design patterns
**Innovation Scope**: Implement paradigm shifts, not incremental variations
**Deliverable Priority**: Revolutionary architecture code implementation (PRIMARY), documentation (SECONDARY)

## REPEATED PATTERN ANALYSIS
### Target for Differentiation:
```
{repeated_motivation}
```

### Pattern Recognition Task:
1. **Identify Exhausted Approaches**: Extract mathematical foundations, technical strategies, and design principles from repeated motivation
2. **Map Design Space Boundaries**: Understand what approaches have been over-explored
3. **Define Orthogonal Directions**: Identify completely different design spaces to explore

## HISTORICAL CONTEXT & EXPERIMENTAL INSIGHTS
{context}

{Config.load_agent("innovation_diversifier_framework_input")}"""
