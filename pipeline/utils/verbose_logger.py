"""
Verbose logging utilities for displaying model inputs/outputs and pipeline progress.
"""

from datetime import datetime
from typing import Any, Dict


def log_model_input(agent_name: str, input_data: Any) -> None:
    """Log model input data verbosely to console."""
    print(f"\n{'='*80}")
    print(f"🤖 MODEL INPUT: {agent_name}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    if isinstance(input_data, str):
        print(f"Input Text ({len(input_data)} chars):")
        if len(input_data) > 1000:
            print(input_data[:500])
            print(f"\n... [TRUNCATED - showing first 500 of {len(input_data)} chars] ...")
            print(input_data[-500:])
        else:
            print(input_data)
    else:
        # Try to serialize the input data
        try:
            if hasattr(input_data, '__dict__'):
                print("Input Object:")
                for key, value in input_data.__dict__.items():
                    if isinstance(value, str) and len(value) > 200:
                        print(f"  {key}: {value[:100]}... [TRUNCATED - {len(value)} chars total]")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"Input: {str(input_data)[:1000]}")
        except Exception:
            print(f"Input (serialization failed): {type(input_data).__name__}")
    print(f"{'='*80}\n")


def log_model_output(agent_name: str, output_data: Any, success: bool = True) -> None:
    """Log model output data verbosely to console."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"\n{'='*80}")
    print(f"🤖 MODEL OUTPUT: {agent_name} - {status}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")

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

    print(f"{'='*80}\n")


def log_database_operation(operation: str, details: Dict[str, Any]) -> None:
    """Log database operations verbosely."""
    print(f"\n🗄️  DATABASE OPERATION: {operation}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)

    for key, value in details.items():
        if isinstance(value, str) and len(value) > 300:
            print(f"{key}: {value[:150]}... [TRUNCATED - {len(value)} chars total]")
        elif isinstance(value, (list, tuple)) and len(value) > 5:
            print(f"{key}: [{len(value)} items] {str(value[:3])}... [TRUNCATED]")
        else:
            print(f"{key}: {value}")
    print("-" * 60)


def log_file_operation(operation: str, file_path: str, content_preview: str = None, size: int = None) -> None:
    """Log file operations verbosely."""
    print(f"\n📁 FILE OPERATION: {operation}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📄 File: {file_path}")

    if size is not None:
        print(f"📊 Size: {size} bytes")

    if content_preview:
        print("📋 Content Preview:")
        if len(content_preview) > 500:
            print(content_preview[:250])
            print(f"... [TRUNCATED - showing first 250 of {len(content_preview)} chars] ...")
            print(content_preview[-250:])
        else:
            print(content_preview)
    print("-" * 60)


def log_training_progress(step: str, details: str, success: bool = None) -> None:
    """Log training progress verbosely."""
    status_icon = "🟢" if success else "🔴" if success is False else "🔄"
    print(f"\n{status_icon} TRAINING: {step}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    if details:
        print(f"Details: {details}")
    print("-" * 40)


def log_error_context(error_source: str, error: Exception, context: Dict[str, Any] = None) -> None:
    """Log error with full context."""
    print(f"\n❌ ERROR in {error_source}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Error Type: {type(error).__name__}")
    print(f"Error Message: {str(error)}")

    if context:
        print("Context:")
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:100]}... [TRUNCATED]")
            else:
                print(f"  {key}: {value}")

    import traceback
    print("Traceback:")
    print(traceback.format_exc())
    print("-" * 60)


def log_pipeline_step(step_name: str, details: str = "") -> None:
    """Log pipeline step with visual separator."""
    print(f"\n{'🚀' * 3} PIPELINE STEP: {step_name} {'🚀' * 3}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    if details:
        print(f"Details: {details}")
    print("=" * 80)


# Wrapper for the existing log_agent_run function to add verbose console output
async def verbose_log_agent_run(agent_name: str, agent, input_data: Any = None, **kwargs) -> Any:
    """
    Enhanced version of log_agent_run with verbose console output.
    """
    from utils.agent_logger import log_agent_run

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
