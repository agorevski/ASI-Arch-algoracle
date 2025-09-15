"""
Verbose logging utilities for displaying model inputs/outputs and pipeline progress.
"""

from datetime import datetime
from typing import Any, Dict
import traceback
from utils.agent_logger import log_agent_run


def log_model_input(agent_name: str, input_data: Any) -> None:
    """Log model input data verbosely to console."""
    print(f"🤖 MODEL INPUT: {agent_name}")
    print(f"\t⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\tInput Text ({len(input_data)} chars)")


def log_model_output(agent_name: str, output_data: Any, success: bool = True) -> None:
    """Log model output data verbosely to console."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"🤖 MODEL OUTPUT: {agent_name} - {status}")
    print(f"\t⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    output_bytes = len(str(output_data).encode('utf-8'))
    print(f"\tOutput Size: {output_bytes} bytes")

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
    """Log database operations verbosely."""
    print(f"🗄️  DATABASE OPERATION: {operation}")
    print(f"\t⏰ Time: {datetime.now().strftime('%H:%M:%S')}")


def log_file_operation(operation: str, file_path: str, content_preview: str = None, size: int = None) -> None:
    """Log file operations verbosely."""
    print(f"📁 FILE OPERATION: {operation}")
    print(f"\t⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\t📄 File: {file_path}")
    if size is not None:
        print(f"\t📊 Size: {size} bytes")
    if content_preview:
        print("\t📋 Content Preview:")
        if len(content_preview) > 500:
            print(content_preview[:250])
            print(f"... [TRUNCATED - showing first 250 of {len(content_preview)} chars] ...")
            print(content_preview[-250:])
        else:
            print(content_preview)


def log_training_progress(step: str, details: str, success: bool = None) -> None:
    """Log training progress verbosely."""
    status_icon = "🟢" if success else "🔴" if success is False else "🔄"
    print(f"{status_icon} TRAINING: {step}")
    print(f"\t⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    if details:
        print(f"Details: {details}")


def log_error_context(error_source: str, error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with full context."""
    print(f"❌ ERROR in {error_source}")
    print(f"\t⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\tError Type: {type(error).__name__}")
    print(f"\tError Message: {str(error)}")

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
    """Log pipeline step with visual separator."""
    print(f"\n{'🚀' * 3} PIPELINE STEP: {step_name} {'🚀' * 3}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    if details:
        print(f"Details: {details}")


# Wrapper for the existing log_agent_run function to add verbose console output
async def verbose_log_agent_run(agent_name: str, agent, input_data: Any = None, **kwargs) -> Any:
    """
    Enhanced version of log_agent_run with verbose console output.
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
