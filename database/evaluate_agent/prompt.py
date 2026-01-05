import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from config_loader import Config


def model_judger_input(element) -> str:
    """Generate a formatted prompt for model judgement evaluation.

    Args:
        element: An object containing model details with the following attributes:
            - name: The name of the model.
            - program: The model architecture code.
            - motivation: The motivation behind the model design.
            - result: A dict with 'train' and 'test' performance metrics.

    Returns:
        str: A formatted string prompt for evaluating the model architecture.
    """
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
