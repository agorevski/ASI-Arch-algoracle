import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def model_judger_input(element) -> str:
    return f"""# Model Judgement Task

{Config.load_agent("model_judger_input_baseline_reference")}

## New Model Architecture to Evaluate

### Model Name: {element.name}

### Architecture Details:
'''python
{element.program}
'''

### Motivation:
{element.motivation}

### Training Performance:
{element.result['train']}

### Evaluation Results:
{element.result['test']}

{Config.load_agent("model_judger_input_evaluation_criteria_and_scoring_framework")}"""
