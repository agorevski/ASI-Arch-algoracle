"""
Verbose logging utilities for displaying model inputs/outputs and pipeline progress.
"""

from datetime import datetime
from typing import Any, Dict
import traceback
from utils.agent_logger import log_agent_run


def log_model_input(agent_name: str, input_data: Any) -> None:
    """Log model input data verbosely to console.

    Args:
        agent_name: The name of the agent processing the input.
        input_data: The input data being sent to the model.

    Returns:
        None
    """
    print(f"🤖 MODEL INPUT: {agent_name} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')} | Input Text: {len(input_data)} chars")


def log_model_output(agent_name: str, output_data: Any, success: bool = True) -> None:
    """Log model output data verbosely to console.

    Displays the agent output with success/failure status, timestamp, and output size.
    For successful outputs, attempts to display the final_output attribute if available,
    truncating long string values for readability.

    Args:
        agent_name: The name of the agent that produced the output.
        output_data: The output data from the model.
        success: Whether the model execution was successful. Defaults to True.

    Returns:
        None
    """
    print(f"🤖 MODEL OUTPUT: {agent_name} - {'✅ SUCCESS' if success else '❌ FAILED'} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')} | Output Size: {len(str(output_data).encode('utf-8'))} bytes")

    if success:
        try:
            if hasattr(output_data, 'final_output'):
                final_output = output_data.final_output
                print("Final Output:")
                if hasattr(final_output, '__dict__'):
                    for key, value in final_output.__dict__.items():
                        if isinstance(value, str) and len(value) > 200:
                            print(f"  {key}: {value[:100]}... [TRUNCATED - {len(value)} chars total]")
                        else:
                            print(f"  {key}: {value}")
                else:
                    print(f"  {str(final_output)[:1000]}")
            else:
                print(f"Output: {str(output_data)[:1000]}")
        except Exception:
            print(f"Output (serialization failed): {type(output_data).__name__}")
    else:
        print(f"Error: {output_data}")


def log_database_operation(operation: str, details: Dict[str, Any]) -> None:
    """Log database operations verbosely.

    Args:
        operation: The type of database operation being performed.
        details: A dictionary containing additional details about the operation.

    Returns:
        None
    """
    print(f"🗄️ DATABASE OPERATION: {operation} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')}")


def log_file_operation(operation: str, file_path: str, content_preview: str = None, size: int = None) -> None:
    """Log file operations verbosely.

    Args:
        operation: The type of file operation being performed (e.g., 'read', 'write').
        file_path: The path to the file being operated on.
        content_preview: Optional preview of the file content. Defaults to None.
        size: Optional size of the file in bytes. Defaults to None.

    Returns:
        None
    """
    print(f"📁 FILE OPERATION: {operation} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')} | 📄 File: {file_path}")
    # if size is not None:
    #     print(f"\t📊 Size: {size} bytes")
    # if content_preview:
    #     print("\t📋 Content Preview:")
    #     if len(content_preview) > 250:
    #         print(content_preview[:250])
    #         print(f"... [TRUNCATED - showing first 250 of {len(content_preview)} chars] ...")
    #         print(content_preview[-250:])
    #     else:
    #         print(content_preview)


def log_training_progress(step: str, details: str, success: bool = None) -> None:
    """Log training progress verbosely.

    Displays the training step with a status icon (green for success, red for failure,
    or rotating arrows for in-progress).

    Args:
        step: The name or description of the current training step.
        details: Additional details about the training step.
        success: Whether the training step was successful. None indicates in-progress.
            Defaults to None.

    Returns:
        None
    """
    status_icon = "🟢" if success else "🔴" if success is False else "🔄"
    print(f"{status_icon} TRAINING: {step} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    # if details:
    #     print(f"Details: {details}")


def log_error_context(error_source: str, error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with full context.

    Displays error details including type, message, optional context dictionary,
    and the full traceback for debugging purposes.

    Args:
        error_source: The source or location where the error occurred.
        error: The exception that was raised.
        context: Optional dictionary containing additional context about the error.
            Long string values are truncated for readability. Defaults to None.

    Returns:
        None
    """
    print(f"❌ ERROR in {error_source} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')} | Error Type: {type(error).__name__} | Error Message: {str(error)}")
    if context:
        print("\tContext:")
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"\t  {key}: {value[:100]}... [TRUNCATED]")
            else:
                print(f"\t  {key}: {value}")

    print("\tTraceback:")
    print(traceback.format_exc())
    print("-" * 60)


def log_pipeline_step(step_name: str, details: str = "") -> None:
    """Log pipeline step with visual separator.

    Displays the pipeline step name with rocket emoji separators and a timestamp
    for easy identification in console output.

    Args:
        step_name: The name of the pipeline step being executed.
        details: Optional additional details about the step. Defaults to empty string.

    Returns:
        None
    """
    print(f"\n{'🚀' * 3} PIPELINE STEP: {step_name} {'🚀' * 3} | ⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    # if details:
    #     print(f"Details: {details}")


# Wrapper for the existing log_agent_run function to add verbose console output
async def verbose_log_agent_run(agent_name: str, agent, input_data: Any = None, **kwargs) -> Any:
    """Execute an agent run with verbose console logging.

    Enhanced version of log_agent_run that adds verbose console output for both
    inputs and outputs. Logs the input before execution and the output (or error)
    after execution completes.

    Args:
        agent_name: The name of the agent being executed.
        agent: The agent object to execute.
        input_data: Optional input data to pass to the agent. Defaults to None.
        **kwargs: Additional keyword arguments to pass to log_agent_run.

    Returns:
        The result from the agent execution.

    Raises:
        Exception: Re-raises any exception from the agent execution after logging.
    """

    # Log input verbosely
    log_model_input(agent_name, input_data)

    try:
        # Execute the agent call with existing logging
        result = await log_agent_run(agent_name, agent, input_data, **kwargs)

        # Log successful output verbosely
        log_model_output(agent_name, result, success=True)

        return result
    except Exception as e:
        # Log failed output verbosely
        log_model_output(agent_name, str(e), success=False)
        log_error_context(f"Agent {agent_name}", e, {
            "input_data": str(input_data)[:500] if input_data else "None",
            "kwargs": str(kwargs)
        })
        raise
